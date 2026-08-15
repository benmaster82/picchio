#!/usr/bin/env python3
"""flat_bench.py — S1 harness: verify byte-identity and measure I/O (model-agnostic).

1. Byte-verify: every expert read from the flat store must equal the same expert
   read from the safetensors container (the L0 gate).
2. Microbench (I/O isolated from compute): read a fixed expert set and report
   wall time, MB/s, IOPS, and per-read latency (p50/p99), in three modes:
     - buffered pass 1  (first touch, cold-ish)
     - buffered pass 2  (warm: OS page cache serving)
     - unbuffered       (FILE_FLAG_NO_BUFFERING: true disk, no page cache)
   buffered-warm vs unbuffered is the page-cache effect that phase S2 removes.

Env: FLAT_MODEL, FLAT_OUT, FLAT_SAMPLE (experts to bench, default 384).
"""
import os
import random
import struct
import sys
import time

import flat_common as fc

MODEL = os.environ.get("FLAT_MODEL", "D:/qwen3_30b_i4")
FLAT = os.environ.get("FLAT_OUT", "D:/qwen3_flat.picchioflat")
BS = fc.BS

with open(FLAT, "rb") as f:
    sb = f.read(BS)
    (magic, ver, bs, NL, NE, n, idx_off, idx_len, data_off) = struct.unpack_from(
        "<8sIIIIIQQQ", sb)
    assert magic == b"PCHIOFL1", magic
    f.seek(idx_off)
    raw = f.read(idx_len)
index = [struct.unpack_from("<QII", raw, i * 16) for i in range(n)]
avg = idx_off / n / 1e6
print(f"flat: {n} experts, {avg:.2f} MB/expert, payload {idx_off/1e9:.2f} GB, "
      f"disk {FLAT[0]}:")

cfg, cNL, cNE, name2shard, scheme = fc.load(MODEL)


def st_payload(L, E):
    return b"".join(fc.tbytes(name2shard, nm) for nm in scheme(L, E))


# ── 1) byte-verify ──
bad = 0
with open(FLAT, "rb") as f:
    for i in range(n):
        L, E = divmod(i, NE)
        off, ln, pd = index[i]
        f.seek(off)
        if f.read(ln) != st_payload(L, E):
            bad += 1
print(f"byte-verify: {n-bad}/{n} identical" + (" OK" if bad == 0 else f"  FAIL ({bad})"))
if bad:
    sys.exit(1)

# ── 2) microbench ──
random.seed(0)
K = min(int(os.environ.get("FLAT_SAMPLE", "384")), n)
sample = random.sample(index, K)
per_pass = sum(pd for _, _, pd in sample) / 1e6


def report(name, times, total):
    wall = sum(times)
    ts = sorted(times)
    p50 = ts[len(ts) // 2] * 1e6
    p99 = ts[min(len(ts) - 1, int(len(ts) * 0.99))] * 1e6
    print(f"  {name:<16} {total/1e6:6.0f} MB  {wall:6.2f}s  "
          f"{total/1e6/wall:6.0f} MB/s  {len(times)/wall:6.0f} IOPS  "
          f"us/read p50={p50:5.0f} p99={p99:6.0f}")


def buffered(entries):
    fd = os.open(FLAT, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    times, total = [], 0
    for off, ln, pd in entries:
        t = time.perf_counter()
        os.lseek(fd, off, 0)
        b = os.read(fd, pd)
        times.append(time.perf_counter() - t)
        total += len(b)
    os.close(fd)
    return times, total


def unbuffered_win(entries):
    import ctypes
    import ctypes.wintypes as w
    k = ctypes.windll.kernel32
    k.CreateFileW.restype = ctypes.c_void_p
    k.VirtualAlloc.restype = ctypes.c_void_p
    h = k.CreateFileW(FLAT, 0x80000000, 1, None, 3, 0x20000000, None)  # NO_BUFFERING
    if h == ctypes.c_void_p(-1).value:
        raise OSError("CreateFileW failed")
    maxlen = max(pd for _, _, pd in entries)
    buf = k.VirtualAlloc(None, ctypes.c_size_t(maxlen), 0x3000, 4)
    read = w.DWORD()
    times, total = [], 0
    try:
        for off, ln, pd in entries:
            k.SetFilePointerEx(ctypes.c_void_p(h), ctypes.c_longlong(off), None, 0)
            t = time.perf_counter()
            ok = k.ReadFile(ctypes.c_void_p(h), ctypes.c_void_p(buf), pd,
                            ctypes.byref(read), None)
            times.append(time.perf_counter() - t)
            if not ok:
                raise OSError("ReadFile failed")
            total += read.value
    finally:
        k.CloseHandle(ctypes.c_void_p(h))
    return times, total


print(f"microbench: {K} experts, {per_pass:.0f} MB per pass\n"
      f"  {'mode':<16} {'bytes':>6}     {'wall':>5}   {'MB/s':>6}    {'IOPS':>4}   latency")
report("buffered pass1", *buffered(sample))
report("buffered pass2", *buffered(sample))
try:
    report("unbuffered", *unbuffered_win(sample))
except Exception as e:
    print(f"  unbuffered: skipped ({type(e).__name__}: {e})")

print("\nreading: buffered-warm ~ page cache; unbuffered ~ true disk. "
      "their gap is what S2 (O_DIRECT) removes; S3 (async QD) then raises MB/s/IOPS.")
