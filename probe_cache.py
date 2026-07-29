#!/usr/bin/env python3
"""Misura quanti slot expert per layer alloca Picchio per diversi valori di PIN_GB."""
import os
import re
import subprocess
import sys
from pathlib import Path

EXE = Path("picchio.exe").resolve()
MODEL = sys.argv[1] if len(sys.argv) > 1 else r"D:\gptoss_i4"
AUX = ";".join(str(Path(p).resolve()) for p in
               ("expert_biases.safetensors", "model-00012.safetensors")
               if Path(p).is_file())


def probe(pin_gb):
    env = os.environ.copy()
    for name in ("INPUT", "PROMPT", "INPUT_FILE", "OUTPUT", "TRACE_NUMERIC"):
        env.pop(name, None)
    env.update({"SERVICE": "1", "CTX": "256", "PIN_GB": str(pin_gb),
                "TEMPERATURE": "0", "REP": "1", "MODEL_AUX": AUX})
    proc = subprocess.Popen([str(EXE), MODEL], env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="ascii", errors="replace", bufsize=1)
    line = ""
    try:
        while True:
            line = proc.stderr.readline()
            if not line:
                break
            match = re.search(r"expert cache: (\d+) slot/layer .*?(\d+) expert totali, ~([\d.]+) GB",
                              line)
            if match:
                return int(match.group(1)), int(match.group(2)), float(match.group(3))
    finally:
        try:
            proc.stdin.write("SHUTDOWN\n")
            proc.stdin.flush()
        except OSError:
            pass
        proc.kill()
    raise RuntimeError(f"riga cache non trovata (ultima: {line!r})")


def main():
    print(f"{'PIN_GB':>7} {'slot/layer':>11} {'expert':>7} {'cache_GB':>9}")
    for pin_gb in (1, 2, 3, 4, 6, 8):
        slots, experts, gb = probe(pin_gb)
        print(f"{pin_gb:>7} {slots:>11} {experts:>7} {gb:>9.1f}")
    print("Sommare la parte densa riportata dal runtime per il totale residente.")


if __name__ == "__main__":
    main()
