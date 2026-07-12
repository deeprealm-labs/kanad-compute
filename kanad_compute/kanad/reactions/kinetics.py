"""Transition-state-theory reaction kinetics from quantum thermochemistry.

The rate constant of an elementary reaction follows Eyring TST:

    k(T) = κ(T) · (k_B T / h) · exp(−ΔG‡ / k_B T)

where ΔG‡ = G(TS) − G(reactant) is the **Gibbs free energy of activation** and κ is the
transmission coefficient (here the Wigner tunneling correction from the TS imaginary
frequency). This module keeps the physics pure and testable: feed it a ΔG‡ (ideally from
the quantum thermochemistry of Phase 5 — correlated electronic energy + quantum-Hessian
frequencies at BOTH the reactant and the TS) and, optionally, the TS imaginary frequency.

This is the honest upgrade over a bare *potential-barrier* Eyring estimate: the activation
entropy and ZPE enter through ΔG‡, and quantum tunneling enters through κ.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# CODATA 2018 (SI)
_kB = 1.380649e-23        # J/K
_h = 6.62607015e-34       # J·s
_c = 2.99792458e10        # cm/s
_Ha_J = 4.3597447222071e-18


def wigner_tunneling(imaginary_freq_cm: float, temperature: float) -> float:
    """Wigner tunneling correction for a parabolic barrier of imaginary frequency ν‡:

        κ_W = 1 + (1/24) · (h |ν‡| / k_B T)²

    ``imaginary_freq_cm`` is the magnitude of the TS imaginary frequency in cm⁻¹ (pass a
    positive number; a negative one is treated by magnitude). κ_W ≥ 1 and → 1 as T → ∞.
    """
    u = _h * abs(float(imaginary_freq_cm)) * _c / (_kB * float(temperature))
    return 1.0 + (u * u) / 24.0


def eyring_rate_constant(
    dG_act_ha: float,
    temperature: float,
    *,
    imaginary_freq_cm: Optional[float] = None,
    kappa: float = 1.0,
    molecularity: int = 1,
) -> dict:
    """Eyring TST rate constant.

        k = κ · (k_B T / h) · exp(−ΔG‡ / k_B T)

    Args:
        dG_act_ha: Gibbs free energy of activation ΔG‡ in Hartree.
        temperature: Temperature in K.
        imaginary_freq_cm: TS imaginary frequency magnitude (cm⁻¹). If given, κ is
            multiplied by the Wigner tunneling correction.
        kappa: An additional transmission coefficient (e.g. Kramers friction); default 1.
        molecularity: 1 for a unimolecular rate (units 1/s). (>1 would need a standard-state
            concentration factor; not applied here — returned rate is the TST prefactor form.)

    Returns a dict with ``rate_constant_s`` (1/s for unimolecular), ``kappa`` (total
    transmission incl. tunneling), ``kappa_tunneling``, ``dG_act_ha`` and ``half_life_s``.
    """
    T = float(temperature)
    kappa_tun = 1.0
    if imaginary_freq_cm is not None:
        kappa_tun = wigner_tunneling(imaginary_freq_cm, T)
    kappa_total = float(kappa) * kappa_tun
    dG_J = float(dG_act_ha) * _Ha_J
    prefactor = _kB * T / _h                       # 1/s
    k = kappa_total * prefactor * np.exp(-dG_J / (_kB * T))
    half_life = np.log(2.0) / k if k > 0 else np.inf
    return {
        'rate_constant_s': float(k),
        'prefactor_s': float(prefactor),
        'kappa': float(kappa_total),
        'kappa_tunneling': float(kappa_tun),
        'dG_act_ha': float(dG_act_ha),
        'dG_act_kcal': float(dG_act_ha) * 627.509,
        'half_life_s': float(half_life),
        'temperature': T,
        'molecularity': int(molecularity),
    }


def quantum_rate_constant(
    reactant_thermo: dict,
    ts_thermo: dict,
    temperature: float,
    *,
    ts_imaginary_freq_cm: Optional[float] = None,
) -> dict:
    """Rate constant from quantum thermochemistry at the reactant and the TS.

    ``reactant_thermo`` / ``ts_thermo`` are the dicts returned by
    ``ThermochemistryCalculator.compute_thermochemistry(e_elec=<correlated E>, ...)`` with
    quantum-Hessian frequencies — so ΔG‡ = G(TS) − G(reactant) carries the correlated
    electronic barrier, the ZPE difference and the activation entropy. If the TS imaginary
    frequency is supplied, Wigner tunneling is included.
    """
    dG = float(ts_thermo['g']) - float(reactant_thermo['g'])
    out = eyring_rate_constant(dG, temperature, imaginary_freq_cm=ts_imaginary_freq_cm)
    out['g_reactant_ha'] = float(reactant_thermo['g'])
    out['g_ts_ha'] = float(ts_thermo['g'])
    return out
