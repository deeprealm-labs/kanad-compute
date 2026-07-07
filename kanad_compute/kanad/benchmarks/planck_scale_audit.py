"""Scaling audit: push the rocm-planck core + kanad analysis up the qubit ladder.

Two tiers:

  TIER A  (analysis correctness + CPU parity) — full PhysicsVQE solve with
          enable_analysis=True on real molecules, validate the attached analysis
          dict (energy decomposition, bonding, dipole) AND planck vs statevector
          energy parity. Bounded to sizes where a full iterative solve is quick.

  TIER B  (capacity / "every register") — drive the planck statevector core
          DIRECTLY on a real molecular Jordan-Wigner Hamiltonian, no optimizer:
          build a dense fixed-seed HEA state at n qubits, measure GPU VRAM
          (proves the 2^n vector is actually resident), check norm == 1 (unitary
          kernels must preserve it — a strong correctness signal at scale),
          evaluate <H> (complex128, <= 32q where 64 GB fits) and the cheap
          single-pass observable <Sum Z_i>, and parity-check vs a CPU statevector
          where it still fits (<= ~26q). H17/34q runs in complex64 (137 GB) — the
          memory headline; expectation needs c128 so only build+norm there.

Ladder = linear H_n chains (sto-3g, active_space('full')): n orbitals -> 2n qubits,
the standard hard statevector benchmark. Even n only (closed shell); H17 uses spin=1.

Run:  PYTHONPATH=<parent-of-kanad> python benchmarks/planck_scale_audit.py
Env:  SCALE_TIERB="20,24,28,32"  SCALE_TIERA="14,20"  SCALE_HH_FULL_EXPECT=30
"""
import json
import os
import subprocess
import time
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore")

OUT = "benchmarks/out/planck_scale_audit.json"
RESULTS = {"tierA": [], "tierB": []}
if os.path.exists(OUT):
    try:
        RESULTS = json.load(open(OUT))
    except Exception:
        pass


def _save():
    os.makedirs("benchmarks/out", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)


def vram_used_bytes():
    """Resident VRAM in bytes (rocm-smi on AMD, nvidia-smi fallback), else None."""
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            stderr=subprocess.DEVNULL, timeout=20).decode()
        d = json.loads(out)
        for card, v in d.items():
            for k, val in v.items():
                if "Used" in k and "VRAM" in k:
                    return int(val)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=20).decode()
        return int(out.strip().splitlines()[0]) * 1024 * 1024
    except Exception:
        return None


# ---------------------------------------------------------------- molecule ladder
def h_chain(n, r=1.0):
    from kanad.builder import MolecularBuilder
    atoms = [("H", (0.0, 0.0, i * r)) for i in range(n)]
    b = MolecularBuilder.from_atoms(atoms).basis("sto-3g")
    if n % 2:
        b = b.spin(1)
    return b.active_space("full").build()


def h2o():
    from kanad.builder import MolecularBuilder
    return (MolecularBuilder
            .from_atoms([("O", (0.0, 0.0, 0.117)),
                         ("H", (0.0, 0.757, -0.467)),
                         ("H", (0.0, -0.757, -0.467))])
            .basis("sto-3g").build())


# qubit -> system constructor (full space)
def system_for_qubits(nq):
    if nq == 14:
        return "H2O/sto-3g", h2o()
    assert nq % 2 == 0
    n = nq // 2
    return f"H{n}/sto-3g/full", h_chain(n)


# ---------------------------------------------------------------- HEA test circuit
def hea_circuit(nq, layers=2, seed=7):
    """Dense, fully-entangling fixed-seed ansatz: every amplitude becomes nonzero,
    so the statevector genuinely fills memory and every register is touched."""
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(nq)
    for _ in range(layers):
        for q in range(nq):
            qc.ry(float(rng.uniform(0, 2 * np.pi)), q)
        for q in range(nq - 1):
            qc.cx(q, q + 1)
        qc.cx(nq - 1, 0)
    for q in range(nq):
        qc.ry(float(rng.uniform(0, 2 * np.pi)), q)
    return qc


def sum_z_op(nq):
    from qiskit.quantum_info import SparsePauliOp
    return SparsePauliOp.from_list([("I" * (nq - 1 - i) + "Z" + "I" * i, 1.0) for i in range(nq)])


# ================================================================ TIER A
def tier_a(qubit_list, cpu_parity_max=20):
    """Full PhysicsVQE solve (energy parity vs CPU where it's fast enough) + run the
    standalone kanad analysis suite on the planck-backed result, validating invariants.
    The CPU statevector solve is skipped above cpu_parity_max (it gets slow: qiskit
    <H> over thousands of terms x hundreds of evals); planck always runs."""
    from kanad.solvers import PhysicsVQE
    for nq in qubit_list:
        name, sysm = system_for_qubits(nq)
        cell = {"qubits": nq, "system": name, "backends": {}}
        print(f"\n=== TIER A  {name}  ({nq}q) ===", flush=True)
        energies, hist = {}, None
        backends = ("statevector", "planck") if nq <= cpu_parity_max else ("planck",)
        for backend in backends:
            r = {}
            t = time.perf_counter()
            try:
                res = PhysicsVQE(sysm, backend=backend, max_excitations=6).solve()
                d = res.to_dict()
                E = float(res.energy)
                energies[backend] = E
                r["energy"] = E
                r["hf_energy"] = (float(d["hf_energy"]) if d.get("hf_energy") is not None else None)
                r["fci_energy"] = (float(d["fci_energy"]) if d.get("fci_energy") is not None else None)
                r["energy_le_hf"] = (r["energy"] <= r["hf_energy"] + 1e-6) if r["hf_energy"] else None
                if backend == "planck":
                    hist = d.get("energy_history")
                r["ok"] = True
            except Exception as e:
                r["ok"] = False
                r["error"] = f"{type(e).__name__}: {e}"
                r["trace"] = traceback.format_exc().splitlines()[-3:]
            r["t"] = round(time.perf_counter() - t, 2)
            cell["backends"][backend] = r
            print(f"  [{backend:11s}] ok={r.get('ok')} E={r.get('energy')} "
                  f"<=HF:{r.get('energy_le_hf')} ({r['t']}s)", flush=True)
        if "statevector" in energies and "planck" in energies:
            cell["parity_dE"] = abs(energies["planck"] - energies["statevector"])
            print(f"  -> energy parity |dE| = {cell['parity_dE']:.2e}", flush=True)

        cell["analysis"] = run_analysis_suite(sysm, hist)
        print(f"  -> analysis checks: {cell['analysis']}", flush=True)
        RESULTS["tierA"].append(cell)
        _save()


def run_analysis_suite(sysm, energy_history):
    """Run kanad's standalone analyzers on the (planck-backed) system and validate
    each returned value against its physical invariant. Each analyzer is independent
    (one failure doesn't sink the rest)."""
    from kanad.analysis import EnergyAnalyzer, BondingAnalyzer, PropertyCalculator
    ham = sysm.hamiltonian
    out = {}

    # --- bonding: classification + HOMO-LUMO gap (kanad's own output, faithfully recorded)
    try:
        b = BondingAnalyzer(ham).analyze_bonding_type()
        gap = b.get("homo_lumo_gap_ev", b.get("homo_lumo_gap"))
        out["bonding_type"] = b.get("bonding_type")
        out["bonding_gap_present"] = gap is not None      # finding: absent for ActiveHamiltonian systems
        if gap is not None:
            out["gap_nonneg"] = bool(float(gap) >= -1e-9 and np.isfinite(float(gap)))
    except Exception as e:
        out["bonding_error"] = f"{type(e).__name__}: {e}"

    # --- physical sanity: SCF HOMO-LUMO gap >= 0 (always available on the planck-backed system)
    try:
        mo_e = np.asarray(ham.mf.mo_energy, float)
        nocc = int(np.sum(np.asarray(ham.mf.mo_occ) > 0))
        if 0 < nocc < len(mo_e):
            out["mf_homo_lumo_gap_ev"] = float((mo_e[nocc] - mo_e[nocc - 1]) * 27.211386)
            out["mf_gap_nonneg"] = bool(out["mf_homo_lumo_gap_ev"] >= -1e-9)
    except Exception as e:
        out["mf_gap_error"] = f"{type(e).__name__}: {e}"

    # --- energy convergence on the planck energy_history
    try:
        if energy_history is not None and len(np.atleast_1d(energy_history)) > 1:
            conv = EnergyAnalyzer(ham).analyze_convergence(np.asarray(energy_history, float))
            fe = conv.get("final_energy")
            out["convergence_keys"] = sorted(conv.keys())
            out["convergence_final_finite"] = bool(fe is not None and np.isfinite(float(fe)))
    except Exception as e:
        out["convergence_error"] = f"{type(e).__name__}: {e}"

    # --- energy decomposition sums to the total (HF density)
    try:
        dm = ham.mf.make_rdm1()
        dec = EnergyAnalyzer(ham).decompose_energy(dm)
        tot = dec.get("total")
        # total = nuclear_repulsion + one_electron + two_electron
        # (coulomb/exchange are the SPLIT of two_electron — don't re-add them)
        parts = sum(dec.get(k, 0.0) for k in ("nuclear_repulsion", "one_electron", "two_electron"))
        out["decomp_sums_to_total"] = bool(tot is not None and abs(float(tot) - parts) < 1e-3)
    except Exception as e:
        out["decomp_error"] = f"{type(e).__name__}: {e}"

    # --- dipole magnitude >= 0 and finite
    try:
        dip = PropertyCalculator(ham).compute_dipole_moment()
        mag = dip.get("dipole_magnitude")
        out["dipole_nonneg_finite"] = bool(mag is not None and float(mag) >= -1e-12 and np.isfinite(float(mag)))
    except Exception as e:
        out["dipole_error"] = f"{type(e).__name__}: {e}"

    return out


# ================================================================ TIER B
def tier_b(qubit_list, full_expect_max=30, cpu_parity_max=26):
    from planck.statevector import StateVector
    from planck.pauli import PauliSum
    for nq in qubit_list:
        name, sysm = system_for_qubits(nq)
        dtype = "complex128" if nq <= 32 else "complex64"
        cell = {"qubits": nq, "system": name, "dtype": dtype}
        print(f"\n=== TIER B  {name}  ({nq}q, {dtype}) ===", flush=True)
        try:
            from planck.circuit import Circuit
            t = time.perf_counter()
            spo = sysm.hamiltonian.to_sparse_hamiltonian("jordan_wigner")
            cell["n_pauli_terms"] = int(len(spo))
            cell["hf_energy"] = float(sysm.hamiltonian.mf.e_tot)
            cell["t_sparse"] = round(time.perf_counter() - t, 2)

            qc = hea_circuit(nq, layers=2)
            pcirc = Circuit.from_qiskit(qc)

            vram0 = vram_used_bytes()
            t = time.perf_counter()
            sv = StateVector(nq, dtype=dtype)
            pcirc.run(sv)                              # in-place, one batched device call
            cell["t_build"] = round(time.perf_counter() - t, 2)
            vram1 = vram_used_bytes()

            theo_bytes = (2 ** nq) * (16 if dtype == "complex128" else 8)
            cell["state_GB_theoretical"] = round(theo_bytes / 1e9, 2)
            if vram0 is not None and vram1 is not None:
                cell["vram_delta_GB"] = round((vram1 - vram0) / 1e9, 2)
                cell["vram_used_GB"] = round(vram1 / 1e9, 2)

            # norm: device-side, no host transfer (critical at 137 GB)
            nrm2 = sv.vdot(sv)
            cell["norm"] = float(np.sqrt(abs(nrm2.real)))
            cell["norm_dev"] = abs(cell["norm"] - 1.0)
            cell["norm_ok"] = bool(cell["norm_dev"] < 1e-6)

            # cheap single-pass observable <Sum Z_i> (validates expectation kernel at scale)
            if dtype == "complex128":
                t = time.perf_counter()
                cell["sum_z"] = float(PauliSum.from_qiskit(sum_z_op(nq)).expectation(sv))
                cell["t_sumz"] = round(time.perf_counter() - t, 2)

            # full molecular <H> where it fits (c128) and time permits
            if dtype == "complex128" and nq <= full_expect_max:
                t = time.perf_counter()
                E = float(PauliSum.from_qiskit(spo).expectation(sv))
                cell["energy_H"] = E
                cell["t_expect_H"] = round(time.perf_counter() - t, 2)
                # trivial variational lower bound: <H> >= -sum|c_k|
                lb = -float(np.sum(np.abs(spo.coeffs)))
                cell["energy_ge_lower_bound"] = bool(E >= lb - 1e-6)
                cell["energy_finite_real"] = bool(np.isfinite(E))

            # CPU parity (statevector overlap + <H>) where it still fits
            if nq <= cpu_parity_max:
                from qiskit.quantum_info import Statevector
                ref = Statevector(qc)
                got = sv.to_numpy()
                cell["cpu_overlap"] = float(abs(np.vdot(ref.data, got)))
                cell["cpu_overlap_ok"] = bool(abs(cell["cpu_overlap"] - 1.0) < 1e-5)
                if "energy_H" in cell:
                    Eref = float(np.real_if_close(ref.expectation_value(spo)))
                    cell["cpu_energy_H"] = Eref
                    cell["energy_parity_dE"] = abs(cell["energy_H"] - Eref)

            cell["ok"] = True
        except Exception as e:
            cell["ok"] = False
            cell["error"] = f"{type(e).__name__}: {e}"
            cell["trace"] = traceback.format_exc().splitlines()[-4:]
        RESULTS["tierB"].append(cell)
        _save()
        print(f"  -> {json.dumps({k: v for k, v in cell.items() if k not in ('trace',)}, default=str)}",
              flush=True)


if __name__ == "__main__":
    tA = [int(x) for x in os.environ.get("SCALE_TIERA", "14,20").split(",") if x]
    tB = [int(x) for x in os.environ.get("SCALE_TIERB", "20,24,28,32").split(",") if x]
    full_exp = int(os.environ.get("SCALE_HH_FULL_EXPECT", "30"))
    a_parity_max = int(os.environ.get("SCALE_A_PARITY_MAX", "20"))
    if tA:
        tier_a(tA, cpu_parity_max=a_parity_max)
    if tB:
        tier_b(tB, full_expect_max=full_exp)
    _save()
    print("\nwrote", OUT)
    print("SCALE_AUDIT_DONE")
