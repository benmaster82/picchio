#!/usr/bin/env python3
"""flat_bench_qd.py — S3 measurement: async reads at high queue depth (Windows).

Reads a fixed expert set from a .picchioflat store using overlapped I/O plus an
IOCP completion port, unbuffered (FILE_FLAG_NO_BUFFERING), keeping QD reads in
flight. Sweeps QD to show how much throughput rises above the QD=1 baseline.

io_uring is the Linux equivalent; this is the Windows path. The point of S3 is
aggregate throughput at high QD, so we report MB/s and IOPS per QD (per-read
latency is meaningless once reads overlap).

Env: FLAT_OUT (the .picchioflat), FLAT_SAMPLE (experts, default 384),
     FLAT_QD (comma list, default "1,2,4,8,16,32,64").
"""
import ctypes
import ctypes.wintypes as w
import os
import random
import struct
import sys
import time

import flat_common as fc

if sys.platform != "win32":
    sys.exit("this S3 prototype is the Windows (IOCP) path; use io_uring on Linux")

FLAT = os.environ.get("FLAT_OUT", "D:/qwen3_flat.picchioflat")
BS = fc.BS
k = ctypes.windll.kernel32
for fn in ("CreateFileW", "CreateIoCompletionPort", "VirtualAlloc"):
    getattr(k, fn).restype = ctypes.c_void_p
INVALID = ctypes.c_void_p(-1).value
IO_PENDING = 997


class OVERLAPPED(ctypes.Structure):
    _fields_ = [("Internal", ctypes.c_void_p), ("InternalHigh", ctypes.c_void_p),
                ("Offset", w.DWORD), ("OffsetHigh", w.DWORD),
                ("hEvent", ctypes.c_void_p)]


# ── read superblock + index ──
with open(FLAT, "rb") as f:
    sb = f.read(BS)
    magic, ver, bs, NL, NE, n, idx_off, idx_len, data_off = struct.unpack_from(
        "<8sIIIIIQQQ", sb)
    assert magic == b"PCHIOFL1", magic
    f.seek(idx_off)
    raw = f.read(idx_len)
index = [struct.unpack_from("<QII", raw, i * 16) for i in range(n)]
random.seed(0)
K = min(int(os.environ.get("FLAT_SAMPLE", "384")), n)
sample = random.sample(index, K)
per_pass = sum(pd for _, _, pd in sample)
print(f"flat: {n} experts, {idx_off/n/1e6:.2f} MB/expert, disk {FLAT[0]}:  "
      f"set={K} experts, {per_pass/1e6:.0f} MB/pass")


def qd_read(entries, qd):
    h = k.CreateFileW(FLAT, 0x80000000, 1, None, 3, 0x40000000 | 0x20000000, None)
    if h == INVALID:
        raise OSError("CreateFileW failed")
    port = k.CreateIoCompletionPort(ctypes.c_void_p(h), None, 0, 0)
    if not port:
        raise OSError("CreateIoCompletionPort failed")
    maxlen = max(pd for _, _, pd in entries)
    bufs = [k.VirtualAlloc(None, ctypes.c_size_t(maxlen), 0x3000, 4) for _ in range(qd)]
    ovs = [OVERLAPPED() for _ in range(qd)]
    addr2slot = {ctypes.addressof(ovs[j]): j for j in range(qd)}

    def submit(slot, off, pd):
        ov = ovs[slot]
        ctypes.memset(ctypes.byref(ov), 0, ctypes.sizeof(ov))
        ov.Offset = off & 0xFFFFFFFF
        ov.OffsetHigh = (off >> 32) & 0xFFFFFFFF
        ok = k.ReadFile(ctypes.c_void_p(h), ctypes.c_void_p(bufs[slot]), pd,
                        None, ctypes.byref(ov))
        if not ok and k.GetLastError() != IO_PENDING:
            raise OSError(f"ReadFile QD err={k.GetLastError()}")

    N = len(entries)
    nbytes = w.DWORD()
    key = ctypes.c_size_t()
    ovp = ctypes.c_void_p()
    total = 0
    t0 = time.perf_counter()
    i = 0
    free = list(range(qd))
    while free and i < N:                       # prime the pipe
        s = free.pop()
        submit(s, entries[i][0], entries[i][2])
        i += 1
    reaped = 0
    while reaped < N:
        if not k.GetQueuedCompletionStatus(ctypes.c_void_p(port), ctypes.byref(nbytes),
                                            ctypes.byref(key), ctypes.byref(ovp),
                                            0xFFFFFFFF):
            raise OSError(f"GQCS err={k.GetLastError()}")
        slot = addr2slot[ovp.value]
        total += nbytes.value
        reaped += 1
        if i < N:
            submit(slot, entries[i][0], entries[i][2])
            i += 1
    wall = time.perf_counter() - t0
    k.CloseHandle(ctypes.c_void_p(port))
    k.CloseHandle(ctypes.c_void_p(h))
    return wall, total


QDS = [int(x) for x in os.environ.get("FLAT_QD", "1,2,4,8,16,32,64").split(",")]
print("\nS3 async QD sweep (overlapped + IOCP, unbuffered = true disk):")
print(f"  {'QD':>3}  {'MB/s':>7}  {'IOPS':>6}  {'wall':>7}  vs QD1")
base = None
for qd in QDS:
    best = min(qd_read(sample, qd) for _ in range(2))   # 2 runs, keep the fast one
    wall, total = best
    mbs = total / 1e6 / wall
    if base is None:
        base = mbs
    print(f"  {qd:>3}  {mbs:7.0f}  {len(sample)/wall:6.0f}  {wall:6.2f}s  {mbs/base:4.2f}x")

print("\nreading: QD1 ~ the synchronous unbuffered baseline; the climb is the S3 win "
      "from overlapping reads. Small experts (Qwen) should gain most; big experts "
      "(GPT-OSS) already saturate near QD1.")
