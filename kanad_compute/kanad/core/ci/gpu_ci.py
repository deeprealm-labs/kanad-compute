"""GPU eigensolver hook for selected-CI diagonalization.

Routes the subspace CI diagonalization (selected_ci.diagonalize_custom /
diagonalize_pyscf) to rocm-planck's ``det_ci`` when it will actually accelerate
— i.e. an explicit GPU device was requested, or device='auto' AND planck has
on-device det_ci kernels. Returns None otherwise (planck absent, CPU-only build,
or device='cpu'), so callers transparently use the native scipy path (which is
faster than planck's scalar CPU fallback for small subspaces).

planck.det_ci.to_dict() is byte-compatible with the selected_ci result dict
(energy / eigenvector / determinants / n_determinants), so this is a drop-in.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_GPU_DEVICES = ("amd", "nvidia", "gpu")

# Serialize GPU det_ci across worker threads. A compute node runs several jobs
# concurrently (ThreadPoolExecutor), and planck.det_ci releases the GIL during
# the (long) on-device eigensolve — so two SQD jobs would otherwise launch
# kernels on the SAME physical GPU at once and contend, turning an 8.9s C2 solve
# into ~80s for both. One GPU = one solve at a time: each job then gets the full
# device and finishes fast; the CPU sampling / QPU-recovery phases still overlap.
# No-op on the app server (planck absent → we return before acquiring).
_GPU_DET_CI_LOCK = threading.Lock()


def try_planck_det_ci(h1, h2, nuc, determinants, n_orb, n_elec,
                      *, sz: float = 0.0, n_roots: int = 1, device: str = "auto"):
    """Return a selected_ci-compatible result dict from planck.det_ci, or None to
    fall back to the native CPU diagonalizer."""
    if not device or device == "cpu":
        return None
    try:
        import planck
    except Exception:
        return None  # planck not installed (e.g. the app server) — use native CPU
    if getattr(planck, "det_ci", None) is None:
        return None
    # On 'auto', only route when planck can actually run det_ci on a GPU; else the
    # native vectorized scipy path beats planck's scalar CPU fallback. An explicit
    # GPU device routes regardless (planck handles its own CPU fallback + warns).
    if device == "auto":
        try:
            if not planck.has_gpu_det_ci():
                return None
        except Exception:
            return None
    elif device not in _GPU_DEVICES:
        return None
    try:
        dets = list(determinants)
        # Hold the GPU for the eigensolve only — see _GPU_DET_CI_LOCK above.
        with _GPU_DET_CI_LOCK:
            res = planck.det_ci(h1, h2, nuc, dets, n_orb, n_elec,
                                sz=(sz or 0.0), n_roots=n_roots, device=device)
            out = res.to_dict()
        logger.info("selected-CI diagonalized via planck.det_ci on %s (n_det=%d)",
                    out.get("device_used"), out.get("n_determinants"))
        return out
    except Exception as e:
        logger.warning("planck.det_ci failed (%s); falling back to native CPU diagonalizer", e)
        return None
