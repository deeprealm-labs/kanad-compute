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

    # Check GPU
    if check_gpu:
        info.update(_detect_gpu())

    # Check installed quantum packages
    info["packages"] = _detect_packages()

    return info


def _detect_gpu() -> dict:
    """Try to detect NVIDIA GPU via multiple methods."""
    result = {"gpu_available": False, "gpu_name": None, "gpu_memory_gb": None, "cuda_available": False}

    # Method 1: nvidia-smi via subprocess
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            parts = out.stdout.strip().split(",")
            result["gpu_available"] = True
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


def _detect_packages() -> dict:
    """Check which quantum packages are installed."""
    packages = {}
    for pkg in [
        "qiskit", "qiskit_aer", "pyscf", "numpy", "scipy",
        "qiskit_ibm_runtime", "qiskit_ionq", "bluequbit",
        "cuda_quantum",
    ]:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            packages[pkg] = None
    return packages
