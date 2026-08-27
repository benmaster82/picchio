#!/usr/bin/env python3
"""pipe_node.py — Step 1 skeleton: 2-stage pipeline with byte-identity verification.

Goal of this spike: prove that splitting the forward pass at a layer boundary and
shipping the residual stream over TCP reassembles a BYTE-IDENTICAL result, and
measure the per-token round-trip latency. Nothing model-specific yet: the "run
these layers" step is a placeholder deterministic bit-exact transform, so any
mismatch is a transport/protocol bug, not floating-point nondeterminism.

Topology (the only LAN-viable split on a shared/jittery link):
    coordinator (stage A, layers [0, CUT))  --activation-->  worker (stage B, [CUT, L))
                                            <---result------

Modes (stdlib only, zero dependencies):
    python pipe_node.py --selftest                 # both stages in one process (ground truth gate)
    python pipe_node.py --worker                   # run stage B (machine B)
    python pipe_node.py --coordinator 192.168.1.50 # run stage A against worker (machine A)

The wire protocol, framing, and the byte-identity harness are the real, reusable
parts. The SEAM marked below is where picchio's partial forward replaces run_layers:
  - worker holds picchio with only layers [CUT, L) resident, resumes from the residual
  - coordinator holds layers [0, CUT), emits the residual after layer CUT
  - the payload becomes the raw float32 residual stream (same framing)
  - byte-identity is then checked once at bring-up by comparing distributed logits
    to the single-node logits, exactly as --selftest does here.
"""
import argparse
import hashlib
import socket
import struct
import sys
import threading
import time

# ── wire protocol ──
MAGIC = b"PCPL"
VERSION = 1
T_FWD = 1        # coordinator -> worker: run layers [lo, hi) on this activation
T_RES = 2        # worker -> coordinator: the resulting activation
T_ERR = 0xFF
# magic, version, type, reserved, seq, layer_lo, layer_hi, payload_bytes
SUBHDR = struct.Struct("!4sBBHIIII")


def set_fast(sock):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if r == 0:
            return None
        got += r
    return buf


def send_msg(sock, mtype, seq, lo, hi, payload):
    sock.sendall(SUBHDR.pack(MAGIC, VERSION, mtype, 0, seq, lo, hi, len(payload)))
    sock.sendall(payload)


def recv_msg(sock):
    head = recv_exact(sock, SUBHDR.size)
    if head is None:
        return None
    magic, ver, mtype, _rsv, seq, lo, hi, nb = SUBHDR.unpack(head)
    if magic != MAGIC or ver != VERSION:
        raise OSError(f"bad header: magic={magic!r} ver={ver}")
    payload = recv_exact(sock, nb)
    if payload is None:
        return None
    return mtype, seq, lo, hi, bytes(payload)


# ── the placeholder "forward" (SEAM: replace with picchio partial forward) ──
def run_layers(buf, lo, hi):
    """Deterministic, bit-exact stand-in for running transformer layers [lo, hi).
    Sequential and order-sensitive, so [0,CUT) then [CUT,L) equals [0,L) exactly,
    and any transport corruption cascades and is caught by the byte-identity gate.
    Operates on the buffer as an array of uint32 words (residual stream = f32)."""
    a = bytearray(buf)
    words = memoryview(a).cast("I")
    for L in range(lo, hi):
        c = (L * 2654435761 + 0x9E3779B1) & 0xFFFFFFFF
        prev = 0
        for i in range(len(words)):
            x = (words[i] + c) & 0xFFFFFFFF
            x ^= x >> 15
            x = (x * 0x2545F491 + prev) & 0xFFFFFFFF
            words[i] = x
            prev = x
    return bytes(a)


def make_input(seq, nbytes):
    """Deterministic per-token input activation, so both machines agree on ground
    truth (stands in for the embedding + prompt state at the cut layer)."""
    a = bytearray(nbytes)
    w = memoryview(a).cast("I")
    x = (seq * 2246822519 + 3266489917) & 0xFFFFFFFF
    for i in range(len(w)):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        w[i] = x
    return bytes(a)


def short_digest(b):
    return hashlib.sha256(b).hexdigest()[:12]


# ── stage B: worker ──
def serve_conn(conn):
    set_fast(conn)
    try:
        while True:
            msg = recv_msg(conn)
            if msg is None:
                break
            mtype, seq, lo, hi, payload = msg
            if mtype != T_FWD:
                send_msg(conn, T_ERR, seq, lo, hi, b"expected T_FWD")
                continue
            out = run_layers(payload, lo, hi)     # SEAM: picchio resumes [lo, hi)
            send_msg(conn, T_RES, seq, lo, hi, out)
    except (ConnectionError, OSError):
        pass
    finally:
        conn.close()


def run_worker(host, port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(16)
    print(f"pipe worker (stage B) listening on {host}:{port}  (Ctrl-C to stop)")
    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=serve_conn, args=(conn,), daemon=True).start()
    except KeyboardInterrupt:
        print("\nworker stopped")
    finally:
        srv.close()


# ── stage A: coordinator ──
def pctl(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))]


def run_coordinator(host, port, hidden, layers, cut, tokens, verify):
    nbytes = hidden * 4
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    set_fast(sock)
    print(f"pipe coordinator (stage A) -> {host}:{port}")
    print(f"  hidden={hidden} ({nbytes} B)  layers={layers}  cut@{cut}  "
          f"stageA=[0,{cut})  stageB=[{cut},{layers})")
    lat, mism = [], 0
    for seq in range(tokens):
        x = make_input(seq, nbytes)
        midA = run_layers(x, 0, cut)              # SEAM: picchio emits residual @cut
        t = time.perf_counter()
        send_msg(sock, T_FWD, seq, cut, layers, midA)
        msg = recv_msg(sock)
        lat.append((time.perf_counter() - t) * 1e3)
        if msg is None:
            sys.exit("worker closed the connection")
        mtype, rseq, lo, hi, final_dist = msg
        if mtype != T_RES or rseq != seq:
            sys.exit(f"unexpected reply: type={mtype} seq={rseq}")
        if verify:
            ground = run_layers(x, 0, layers)     # bring-up only: A has all layer logic
            if final_dist != ground:
                mism += 1
                if mism <= 3:
                    print(f"  MISMATCH seq={seq}: dist={short_digest(final_dist)} "
                          f"ground={short_digest(ground)}")
    sock.close()

    p50, p99 = pctl(lat, 50), pctl(lat, 99)
    print(f"\n{tokens} tokens  round-trip ms: min {min(lat):.1f}  p50 {p50:.1f}  "
          f"p99 {p99:.1f}  max {max(lat):.1f}")
    if verify:
        if mism == 0:
            print(f"byte-identity: {tokens}/{tokens} IDENTICAL  "
                  f"== pipeline split PASSED ==")
        else:
            print(f"byte-identity: {tokens-mism}/{tokens} identical  FAIL ({mism})")
            sys.exit(1)
    else:
        print("byte-identity: skipped (--no-verify); worker holds the far layers")


# ── selftest: worker on a background thread, coordinator against it ──
def run_selftest(hidden, layers, cut, tokens):
    port = 52123
    t = threading.Thread(target=run_worker, args=("127.0.0.1", port), daemon=True)
    t.start()
    time.sleep(0.3)
    print("pipe self-test: stage B on a local thread, splitting the forward\n")
    run_coordinator("127.0.0.1", port, hidden, layers, cut, tokens, verify=True)


def main():
    ap = argparse.ArgumentParser(description="Step 1 pipeline skeleton with "
                                             "byte-identity verification")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true",
                   help="run both stages in one process (correctness gate)")
    g.add_argument("--worker", action="store_true", help="run stage B (far layers)")
    g.add_argument("--coordinator", metavar="HOST", help="run stage A against HOST")
    ap.add_argument("--port", type=int, default=52100)
    ap.add_argument("--host", default="0.0.0.0", help="worker bind address")
    ap.add_argument("--hidden", type=int, default=2880,
                    help="residual width (GPT-OSS 2880, Qwen3 2048)")
    ap.add_argument("--layers", type=int, default=24, help="total layers")
    ap.add_argument("--cut", type=int, default=12, help="stage boundary layer")
    ap.add_argument("--tokens", type=int, default=64, help="tokens to stream")
    ap.add_argument("--no-verify", action="store_true",
                    help="coordinator does not compute ground truth (real topology)")
    a = ap.parse_args()

    if not 0 < a.cut < a.layers:
        ap.error("--cut must be between 1 and layers-1")
    if a.selftest:
        run_selftest(a.hidden, a.layers, a.cut, a.tokens)
    elif a.worker:
        run_worker(a.host, a.port)
    else:
        run_coordinator(a.coordinator, a.port, a.hidden, a.layers, a.cut,
                        a.tokens, verify=not a.no_verify)


if __name__ == "__main__":
    main()
