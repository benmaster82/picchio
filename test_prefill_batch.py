#!/usr/bin/env python3
"""Il prefill batched deve produrre gli stessi token del percorso sequenziale."""
import os
import subprocess
import sys
from pathlib import Path

MODEL = sys.argv[1] if len(sys.argv) > 1 else "test_tiny_picchio"
EXE = Path("picchio.exe").resolve()
PROMPT = [1 + (i % 97) for i in range(70)]
MAX_NEW = 6


def run(batch):
    env = os.environ.copy()
    for name in ("INPUT", "PROMPT", "INPUT_FILE", "OUTPUT", "MODEL_AUX",
                 "TRACE_NUMERIC", "ORACLE_DIR", "PILOT"):
        env.pop(name, None)
    env.update({"SERVICE": "1", "TEMPERATURE": "0", "REP": "1", "PIN_GB": "1",
                "CTX": "256", "PREFILL_BATCH": str(batch)})
    proc = subprocess.Popen([str(EXE), MODEL], env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, encoding="ascii",
                            errors="replace", bufsize=1)
    assert proc.stdout.readline().startswith("READY ")
    proc.stdin.write(f"TURN {MAX_NEW} 0 0 1 0 {len(PROMPT)} " +
                     " ".join(map(str, PROMPT)) + "\n")
    proc.stdin.flush()
    tokens = []
    while True:
        line = proc.stdout.readline().strip()
        if line.startswith("TOKEN "):
            tokens.append(int(line.split()[1]))
        elif line.startswith("DONE "):
            pos = int(line.split()[3])
            break
        else:
            raise RuntimeError(line)
    proc.stdin.write("SHUTDOWN\n")
    proc.stdin.flush()
    proc.wait(timeout=120)
    return tokens, pos


sequential, pos_seq = run(1)
batched, pos_batch = run(64)
print(f"sequenziale batch=1  tokens={sequential} pos={pos_seq}")
print(f"batched     batch=64 tokens={batched} pos={pos_batch}")
assert sequential == batched, (sequential, batched)
assert pos_seq == pos_batch, (pos_seq, pos_batch)
print("OK: prefill batched identico al sequenziale")
