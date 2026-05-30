"""Pin a single, consistent CUDA cuDNN/cuBLAS stack on Windows.

ctranslate2 (faster-whisper) needs cuDNN 9.10+ symbols such as
``cudnnGetLibConfig``, while torch 2.6 bundles an older cuDNN 9.1 in
``torch/lib``. Windows can only hold one ``cudnn64_9.dll`` per process, so
whichever loads first wins for *every* consumer. When torch (Chatterbox TTS)
loads its 9.1 cuDNN first, ctranslate2 then fails with:

    Could not load symbol cudnnGetLibConfig. Error code 127

This module preloads the full cuDNN 9.10 set (and its matching cuBLAS) shipped
by the ``nvidia-cudnn-cu12`` / ``nvidia-cublas-cu12`` wheels *by absolute path*
before anything else touches CUDA. cuDNN/cuBLAS keep backward compatibility
within their major/CUDA version, so torch (built against 9.1 / CUDA 12.4) runs
fine against the 9.10 / 12.9 runtime. The result: one consistent stack that
satisfies both torch and ctranslate2 regardless of import order or stray DLLs on
PATH.

The whole thing is best-effort and never raises: on non-Windows, when the
wheels are absent, or when there is no GPU, it simply does nothing.
"""

from __future__ import annotations

import os

_DONE = False


def _package_bin(module_name: str) -> str | None:
    """Return the ``bin`` directory of an installed ``nvidia.*`` wheel, if any."""
    import importlib.util

    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    bin_dir = os.path.join(os.path.dirname(spec.origin), "bin")
    return bin_dir if os.path.isdir(bin_dir) else None


def preload_cuda_dlls() -> None:
    """Preload the cuDNN 9.10 / cuBLAS stack so ctranslate2 + torch agree.

    Safe to call multiple times; only the first call does any work. Failures are
    swallowed so the app keeps running (CUDA features will just fall back).
    """
    global _DONE
    if _DONE or os.name != "nt":
        _DONE = True
        return
    _DONE = True

    import ctypes

    cublas_bin = _package_bin("nvidia.cublas")
    cudnn_bin = _package_bin("nvidia.cudnn")

    # Register the directories so the cuDNN dispatcher can find its sub-libraries.
    for directory in (cublas_bin, cudnn_bin):
        if directory and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(directory)
            except (OSError, AttributeError):
                pass

    # Preload by absolute path, dependencies first. cuBLAS before cuDNN; within
    # cuDNN, sub-libraries before the cudnn64_9 dispatcher that loads them.
    ordered = [
        (cublas_bin, "cublas64_12.dll"),
        (cublas_bin, "cublasLt64_12.dll"),
        (cudnn_bin, "cudnn_graph64_9.dll"),
        (cudnn_bin, "cudnn_engines_precompiled64_9.dll"),
        (cudnn_bin, "cudnn_engines_runtime_compiled64_9.dll"),
        (cudnn_bin, "cudnn_heuristic64_9.dll"),
        (cudnn_bin, "cudnn_ops64_9.dll"),
        (cudnn_bin, "cudnn_cnn64_9.dll"),
        (cudnn_bin, "cudnn_adv64_9.dll"),
        (cudnn_bin, "cudnn64_9.dll"),
    ]
    for directory, name in ordered:
        if not directory:
            continue
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            try:
                ctypes.WinDLL(path)
            except OSError:
                # A sub-library may fail to load in isolation; that's fine, the
                # dispatcher will resolve what it needs from the registered dirs.
                pass
