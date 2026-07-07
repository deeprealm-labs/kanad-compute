"""Campaign C — API PROBE for the broad component sweep. Defensive: each block tries a
component and prints the working call signature or the exception, so the real diverse
batteries don't fail on guessed APIs. Touches VQE+ansätze, mappers, Hubbard/metallic,
periodic, and the property calculators.

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework \
        /root/miniconda3/bin/python -m benchmarks.tier_C_probe
"""
from __future__ import annotations
import traceback


def probe(name, fn):
    try:
        out = fn()
        print(f"[OK ] {name}: {out}", flush=True)
    except Exception as e:
        print(f"[ERR] {name}: {type(e).__name__}: {str(e)[:140]}", flush=True)


def main():
    from kanad import MolecularBuilder
    H2 = [('H', (0, 0, 0)), ('H', (0, 0, 0.74))]
    H2O = [('O', (0, 0, 0)), ('H', (0, 0.757, 0.587)), ('H', (0, -0.757, 0.587))]

    print("=== VQE + ansätze ===", flush=True)

    def vqe_givens():
        qs = (MolecularBuilder.from_atoms(H2).basis('sto-3g')
              .ansatz('givens_sd').solver('vqe', backend='statevector').build())
        return round(qs.solve()['energy'], 6)
    probe("VQE/givens_sd H2", vqe_givens)

    def vqe_hea():
        qs = (MolecularBuilder.from_atoms(H2).basis('sto-3g')
              .ansatz('hardware_efficient', n_layers=2).solver('vqe', backend='statevector').build())
        return round(qs.solve()['energy'], 6)
    probe("VQE/HEA H2", vqe_hea)

    print("=== mappers ===", flush=True)

    def mapper_bk():
        qs = (MolecularBuilder.from_atoms(H2).basis('sto-3g')
              .mapper('bravyi_kitaev').solver('ci').build())
        return round(qs.solve()['energy'], 6)
    probe("CI mapper=bravyi_kitaev H2", mapper_bk)

    print("=== Hubbard / metallic ===", flush=True)

    def hubbard_direct():
        from kanad.core.hamiltonians.metallic_hamiltonian import MetallicHamiltonian
        import inspect
        sig = str(inspect.signature(MetallicHamiltonian.__init__))
        return f"signature={sig}"
    probe("MetallicHamiltonian sig", hubbard_direct)

    def hubbard_model():
        from kanad.core.models import IonicHubbardModel
        import inspect
        return f"IonicHubbardModel sig={inspect.signature(IonicHubbardModel.__init__)}"
    probe("IonicHubbardModel sig", hubbard_model)

    print("=== periodic ===", flush=True)

    def periodic_sig():
        from kanad.core.hamiltonians.periodic_hamiltonian import PeriodicHamiltonian
        from kanad.core.lattice import Lattice
        import inspect
        return (f"PeriodicHamiltonian={inspect.signature(PeriodicHamiltonian.__init__)} ; "
                f"Lattice={inspect.signature(Lattice.__init__)}")
    probe("Periodic/Lattice sig", periodic_sig)

    print("=== property calculators ===", flush=True)

    def observables_builder():
        qs = MolecularBuilder.from_atoms(H2O).basis('sto-3g').solver('ci').build()
        qs.solve()
        o = qs.observables('core')
        return {k: o[k] for k in list(o)[:4]}
    probe("observables('core') H2O", observables_builder)

    def propcalc_sig():
        from kanad.analysis.property_calculator import PropertyCalculator
        import inspect
        return f"PropertyCalculator={inspect.signature(PropertyCalculator.__init__)}"
    probe("PropertyCalculator sig", propcalc_sig)

    def freq_sig():
        from kanad.analysis.vibrational_analysis import FrequencyCalculator
        import inspect
        return f"FrequencyCalculator={inspect.signature(FrequencyCalculator.__init__)}"
    probe("FrequencyCalculator sig", freq_sig)

    def thermo_sig():
        from kanad.analysis.thermochemistry import ThermochemistryCalculator
        import inspect
        return f"Thermo={inspect.signature(ThermochemistryCalculator.__init__)}"
    probe("ThermochemistryCalculator sig", thermo_sig)

    def bonding_sig():
        from kanad.analysis.energy_analysis import BondingAnalyzer
        import inspect
        return f"BondingAnalyzer={inspect.signature(BondingAnalyzer.__init__)}"
    probe("BondingAnalyzer sig", bonding_sig)

    print("\nPROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
