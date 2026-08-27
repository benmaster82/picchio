#!/usr/bin/env python3
"""net_bench.py — Step 0 harness for distributed MoE inference: measure the LAN.

The cost of splitting the forward across machines is not bandwidth (an activation
is ~11.5 KB) but round-trip latency times how many hops a token needs. This tool
measures exactly that on the real network, so the split granularity is chosen from
data, not intuition.

Two modes (stdlib only, zero dependencies):
  - server:  echoes framed payloads back (run on machine B)
  - client:  measures RTT (p50/p90/p99, jitter) for one activation-sized payload,
             then sustained fan-out throughput with several connections in flight

TCP_NODELAY is set on every socket: without it Nagle's algorithm batches small
sends and destroys the latency measurement.

Run it twice: once over Ethernet (the strong case) and once over WiFi, both idle
and under load. Read the go/no-go thresholds printed at the end.

  machine B:  python net_bench.py --server
  machine A:  python net_bench.py --client 192.168.1.50

Payload defaults to hidden*4 bytes (Qwen3 2048 -> 8192, GPT-OSS 2880 -> 11520);
pass --bytes to match your model's residual stream.
"""
import argparse
import socket
import struct
import sys
import threading
import time

HDR = struct.Struct("!I")  # 4-byte big-endian length prefix


def set_fast(sock):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def recvall(sock, n):
    """Read exactly n bytes or return None on clean close."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if r == 0:
            return None
        got += r
    return buf


def send_frame(sock, payload):
    sock.sendall(HDR.pack(len(payload)))
    sock.sendall(payload)


def recv_frame(sock):
    head = recvall(sock, HDR.size)
    if head is None:
        return None
    (n,) = HDR.unpack(head)
    return recvall(sock, n)


# ── server ──
def serve_conn(conn, addr):
    set_fast(conn)
    try:
        while True:
            payload = recv_frame(conn)
            if payload is None:
                break
            send_frame(conn, payload)          # echo it straight back
    except (ConnectionError, OSError):
        pass
    finally:
        conn.close()


def run_server(host, port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(64)
    print(f"net_bench server listening on {host}:{port}  (Ctrl-C to stop)")
    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=serve_conn, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        print("\nserver stopped")
    finally:
        srv.close()


# ── client ──
def pctl(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[i]


def rtt_test(host, port, nbytes, iters, warmup):
    payload = b"\xa5" * nbytes
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    set_fast(sock)
    lat = []
    for i in range(iters + warmup):
        t = time.perf_counter()
        send_frame(sock, payload)
        echo = recv_frame(sock)
        dt = time.perf_counter() - t
        if echo is None or len(echo) != nbytes:
            sys.exit("RTT: bad echo from server (framing/size mismatch)")
        if i >= warmup:
            lat.append(dt)
    sock.close()
    us = [x * 1e6 for x in lat]
    p50, p90, p99 = pctl(us, 50), pctl(us, 90), pctl(us, 99)
    print(f"\nRTT ping-pong  ({nbytes} B payload, {iters} samples, {warmup} warmup)")
    print(f"  min {min(us):7.0f}   p50 {p50:7.0f}   p90 {p90:7.0f}   "
          f"p99 {p99:7.0f}   max {max(us):7.0f}   us")
    print(f"  mean {sum(us)/len(us):6.0f} us   jitter (p99/p50) {p99/max(p50,1e-9):4.1f}x")
    return p50, p99


def throughput_worker(host, port, nbytes, stop, counter):
    payload = b"\xa5" * nbytes
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    set_fast(sock)
    n = 0
    while not stop.is_set():
        send_frame(sock, payload)
        if recv_frame(sock) is None:
            break
        n += 1
    counter[0] = n
    sock.close()


def throughput_test(host, port, nbytes, conns, secs):
    stop = threading.Event()
    counters = [[0] for _ in range(conns)]
    threads = [threading.Thread(target=throughput_worker,
                                args=(host, port, nbytes, stop, counters[i]))
               for i in range(conns)]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    time.sleep(secs)
    stop.set()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    msgs = sum(c[0] for c in counters)
    # each round-trip moves the payload twice (there and back)
    mbs = msgs * nbytes * 2 / 1e6 / wall
    print(f"\nfan-out throughput  ({conns} connections in flight, {secs}s)")
    print(f"  {msgs} round-trips   {msgs/wall:7.0f} req/s   {mbs:7.1f} MB/s aggregate")
    return msgs / wall


def verdict(p50_us, p99_us):
    p50, p99 = p50_us / 1000.0, p99_us / 1000.0   # to ms
    print("\ngo/no-go (per-token network budget for a disk-bound 120B ~ 200-400 ms):")
    print("  pipeline (1 hop/token):        always fine on this link")
    hops = 36                                        # ~ one dispatch round per layer
    print(f"  expert-parallel (~{hops} hops/token): "
          f"~{hops*p50:5.0f} ms typical, ~{hops*p99:5.0f} ms p99")
    if p50 <= 1.0 and p99 <= 3.0:
        print("  -> RTT low and stable: fine-grained expert-parallel is viable.")
    elif p50 <= 5.0:
        print("  -> moderate/jittery: pipeline is safe; expert-parallel only if you "
              "coalesce experts per message (fewer rounds).")
    else:
        print("  -> high RTT or heavy jitter (WiFi?): stick to pipeline for now.")


def run_client(host, port, nbytes, iters, warmup, conns, secs):
    print(f"net_bench client -> {host}:{port}")
    p50, p99 = rtt_test(host, port, nbytes, iters, warmup)
    throughput_test(host, port, nbytes, conns, secs)
    verdict(p50, p99)


def main():
    ap = argparse.ArgumentParser(description="LAN latency/throughput harness for "
                                             "distributed MoE inference")
    ap.add_argument("--server", action="store_true", help="run the echo server")
    ap.add_argument("--client", metavar="HOST", help="run the client against HOST")
    ap.add_argument("--port", type=int, default=51515)
    ap.add_argument("--host", default="0.0.0.0", help="server bind address")
    ap.add_argument("--bytes", type=int, default=11520,
                    help="payload size (default 11520 = GPT-OSS hidden 2880 x 4)")
    ap.add_argument("--iters", type=int, default=2000, help="RTT samples")
    ap.add_argument("--warmup", type=int, default=200, help="RTT warmup (discarded)")
    ap.add_argument("--conns", type=int, default=8, help="throughput connections")
    ap.add_argument("--secs", type=float, default=5.0, help="throughput duration")
    a = ap.parse_args()

    if a.server == bool(a.client):
        ap.error("pick exactly one of --server or --client HOST")
    if a.server:
        run_server(a.host, a.port)
    else:
        run_client(a.client, a.port, a.bytes, a.iters, a.warmup, a.conns, a.secs)


if __name__ == "__main__":
    main()
