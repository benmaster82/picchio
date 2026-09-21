#!/usr/bin/env python3
"""Regenerate the embedded CUDA Driver PTX using the Toolkit's NVRTC DLL.

This is a build-time helper only. Normal Picchio builds consume the checked-in
gpu_native_ptx.inc and do not require Python, NVRTC, CUDA Runtime, or a Toolkit.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path


def find_nvrtc(cuda_root):
    root = Path(cuda_root or os.environ.get("CUDA_PATH", ""))
    if not root:
        root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6")
    candidates = sorted((root / "bin").glob("nvrtc64_*.dll"))
    candidates = [path for path in candidates if ".alt." not in path.name]
    if not candidates:
        raise FileNotFoundError(f"NVRTC not found below {root}")
    return root, candidates[-1]


def compile_ptx(source, cuda_root, nvrtc_path):
    lib = ctypes.WinDLL(str(nvrtc_path))
    program = ctypes.c_void_p()
    lib.nvrtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_char_p)]
    lib.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p,
                                           ctypes.POINTER(ctypes.c_size_t)]
    lib.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    data = source.read_bytes()
    rc = lib.nvrtcCreateProgram(ctypes.byref(program), data, source.name.encode(),
                                0, None, None)
    if rc:
        raise RuntimeError(f"nvrtcCreateProgram failed: {rc}")
    options = [b"--gpu-architecture=compute_75", b"--fmad=true",
               f"--include-path={cuda_root / 'include'}".encode()]
    array = (ctypes.c_char_p * len(options))(*options)
    rc = lib.nvrtcCompileProgram(program, len(options), array)
    log_size = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(program, ctypes.byref(log_size))
    if log_size.value > 1:
        log = ctypes.create_string_buffer(log_size.value)
        lib.nvrtcGetProgramLog(program, log)
        print(log.value.decode(errors="replace"))
    if rc:
        lib.nvrtcDestroyProgram(ctypes.byref(program))
        raise RuntimeError(f"nvrtcCompileProgram failed: {rc}")
    ptx_size = ctypes.c_size_t()
    lib.nvrtcGetPTXSize(program, ctypes.byref(ptx_size))
    ptx = ctypes.create_string_buffer(ptx_size.value)
    lib.nvrtcGetPTX(program, ptx)
    lib.nvrtcDestroyProgram(ctypes.byref(program))
    return ptx.raw[:-1].decode("ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-root")
    parser.add_argument("--keep-ptx", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    cuda_root, nvrtc = find_nvrtc(args.cuda_root)
    ptx = compile_ptx(here / "gpu_router_kernel.cu", cuda_root, nvrtc)
    if args.keep_ptx:
        (here / "gpu_router_kernel.generated.ptx").write_text(ptx, encoding="ascii")
    lines = [json.dumps(line + "\n") for line in ptx.splitlines()]
    (here / "gpu_native_ptx.inc").write_text("\n".join(lines) + "\n",
                                               encoding="ascii")
    print(f"generated {len(ptx.encode('ascii'))} PTX bytes via {nvrtc.name}")


if __name__ == "__main__":
    main()
