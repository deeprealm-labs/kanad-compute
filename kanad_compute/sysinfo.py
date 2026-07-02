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
    """rocm-planck GPU statevector core present (PLANCK_GPU_PLATFORM build loaded)."""
    try:
        import planck
        return bool(getattr(planck, "_GPU_CORE_AVAILABLE", False))
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
