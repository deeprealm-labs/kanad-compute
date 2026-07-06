"""System information detection — CPU, RAM, GPU, installed packages."""

import platform
import psutil


def get_system_info(check_gpu: bool = False) -> dict:
    """Gather system hardware and software info."""
    info = {
        "platform": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 1),
        "gpu_available": False,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "cuda_available": False,
    }

    info["gpu_vendor"] = None

    # Check GPU
    if check_gpu:
        info.update(_detect_gpu())

    # Statevector/det_ci GPU engines available to the worker
    info["planck_available"] = _planck_gpu_available()
    info["cudaq_available"] = _module_present("cudaq") or _module_present("cuda_quantum")
    info["gpu_engine"] = (
        "planck" if info["planck_available"]
        else "cudaq" if info["cudaq_available"]
        else "aer-gpu" if info.get("cuda_available")
        else "cpu"
    )

    # Check installed quantum packages
    info["packages"] = _detect_packages()

    return info


def _planck_gpu_available() -> bool:
    """rocm-planck GPU statevector core present + loadable (PLANCK_GPU_PLATFORM build).

    Current planck exposes the compiled core as planck.statevector.StateVector (there is no
    _GPU_CORE_AVAILABLE flag) — a successful import of that is the availability signal the
    framework's planck_adapter actually relies on.
    """
    try:
        from planck.statevector import StateVector  # noqa: F401
        return True
    except Exception:
        return False


def _module_present(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _detect_gpu() -> dict:
    """Detect the GPU + vendor (AMD via rocm-smi, NVIDIA via nvidia-smi)."""
    result = {"gpu_available": False, "gpu_name": None, "gpu_memory_gb": None,
              "cuda_available": False, "gpu_vendor": None}

    # AMD (ROCm) via rocm-smi
    try:
        import subprocess
        out = subprocess.run(["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                if "Card Series" in line:
                    result["gpu_available"] = True
                    result["gpu_vendor"] = "amd"
                    result["gpu_name"] = line.split(":")[-1].strip()
                    break
    except Exception:
        pass

    # NVIDIA via nvidia-smi (only if AMD not found)
    if not result["gpu_available"]:
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                parts = out.stdout.strip().split(",")
                result["gpu_available"] = True
                result["gpu_vendor"] = "nvidia"
                result["gpu_name"] = parts[0].strip()
                result["gpu_memory_gb"] = round(int(parts[1].strip()) / 1024, 1) if len(parts) > 1 else None
        except Exception:
            pass

    # Method 2: Check CUDA via qiskit-aer
    try:
        from qiskit_aer import AerSimulator
        sim = AerSimulator(method="statevector", device="GPU")
        result["cuda_available"] = True
    except Exception:
        pass

    return result


def resource_usage() -> dict:
    """Live CPU / RAM / GPU utilization snapshot for progress messages.

    CPU/RAM come from psutil (always present). GPU utilization + VRAM come from
    rocm-smi (AMD) or nvidia-smi (NVIDIA), best-effort — fields are None when the
    tool isn't available or the parse fails. Cheap enough to call once per phase.
    """
    usage = {"cpu_percent": None, "ram_percent": None, "gpu_percent": None,
             "vram_used_gb": None, "vram_total_gb": None, "gpu_vendor": None}
    try:
        usage["cpu_percent"] = round(psutil.cpu_percent(interval=None), 1)
        usage["ram_percent"] = round(psutil.virtual_memory().percent, 1)
    except Exception:
        pass

    # AMD ROCm (rocm-planck node) via rocm-smi --json
    try:
        import subprocess, json as _json
        out = subprocess.run(["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"],
                             capture_output=True, text=True, timeout=4)
        if out.returncode == 0 and out.stdout.strip():
            data = _json.loads(out.stdout)
            uses, used, total = [], 0, 0
            for _card, vals in data.items():
                if not isinstance(vals, dict):
                    continue
                for k, v in vals.items():
                    kl = str(k).lower()
                    try:
                        if "gpu use" in kl:
                            uses.append(float(str(v).replace("%", "").strip()))
                        elif "vram total used memory" in kl:
                            used += int(v)
                        elif "vram total memory" in kl:
                            total += int(v)
                    except Exception:
                        pass
            if uses:
                usage["gpu_percent"] = round(sum(uses) / len(uses), 1)
            if total:
                usage["vram_total_gb"] = round(total / (1024 ** 3), 1)
                usage["vram_used_gb"] = round(used / (1024 ** 3), 1)
            if uses or total:
                usage["gpu_vendor"] = "amd"
                return usage
    except Exception:
        pass

    # NVIDIA via nvidia-smi
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4)
        if out.returncode == 0 and out.stdout.strip():
            parts = out.stdout.strip().splitlines()[0].split(",")
            usage["gpu_percent"] = float(parts[0].strip())
            usage["vram_used_gb"] = round(int(parts[1].strip()) / 1024, 1)
            usage["vram_total_gb"] = round(int(parts[2].strip()) / 1024, 1)
            usage["gpu_vendor"] = "nvidia"
    except Exception:
        pass
    return usage


def resource_summary() -> str:
    """One-line human string for progress logs.

    e.g. 'CPU 45% · RAM 30% · GPU 78% · VRAM 32/192 GB'.
    """
    u = resource_usage()
    parts = []
    if u.get("cpu_percent") is not None:
        parts.append(f"CPU {u['cpu_percent']:.0f}%")
    if u.get("ram_percent") is not None:
        parts.append(f"RAM {u['ram_percent']:.0f}%")
    if u.get("gpu_percent") is not None:
        parts.append(f"GPU {u['gpu_percent']:.0f}%")
    if u.get("vram_used_gb") is not None and u.get("vram_total_gb"):
        parts.append(f"VRAM {u['vram_used_gb']:.0f}/{u['vram_total_gb']:.0f} GB")
    return " · ".join(parts) if parts else "resource stats unavailable"


def _detect_packages() -> dict:
    """Check which quantum packages are installed."""
    packages = {}
    for pkg in [
        "qiskit", "qiskit_aer", "pyscf", "numpy", "scipy",
        "qiskit_ibm_runtime", "qiskit_ionq", "bluequbit",
        "cuda_quantum", "planck",
    ]:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            packages[pkg] = None
    return packages
