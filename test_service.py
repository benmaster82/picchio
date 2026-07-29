#!/usr/bin/env python3
"""Valida la sessione persistente: riuso del prefisso identico al prefill completo."""
import os
import subprocess
import sys
from pathlib import Path

MODEL = sys.argv[1] if len(sys.argv) > 1 else "test_tiny_picchio"
EXE = Path("picchio.exe").resolve()


class Service:
    def __init__(self, model, ctx=256):
        env = os.environ.copy()
        for name in ("INPUT", "PROMPT", "INPUT_FILE", "OUTPUT", "MODEL_AUX",
                     "TRACE_NUMERIC", "ORACLE_DIR"):
            env.pop(name, None)
        env.update({"SERVICE": "1", "TEMPERATURE": "0", "REP": "1",
                    "PIN_GB": "1", "CTX": str(ctx)})
        self.proc = subprocess.Popen([str(EXE), str(model)], env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, encoding="ascii", errors="replace",
                                     bufsize=1)
        ready = self._line()
        assert ready.startswith("READY "), ready
        parts = ready.split()
        self.ctx, self.vocab = int(parts[1]), int(parts[2])
        self.stop = [int(x) for x in parts[3:]]

    def _line(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("servizio terminato inaspettatamente")
        return line.strip()

    def turn(self, ids, max_new, keep=0):
        payload = f"TURN {max_new} {keep} {len(ids)} " + " ".join(map(str, ids)) + "\n"
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()
        tokens = []
        while True:
            line = self._line()
            if line.startswith("TOKEN "):
                tokens.append(int(line.split()[1]))
            elif line.startswith("DONE "):
                _, reason, produced, pos = line.split()
                assert int(produced) == len(tokens), (produced, tokens)
                return tokens, reason, int(pos)
            elif line.startswith("ERROR "):
                return None, line, None
            else:
                raise RuntimeError(f"frame inatteso: {line}")

    def close(self):
        try:
            self.proc.stdin.write("SHUTDOWN\n")
            self.proc.stdin.flush()
        except OSError:
            pass
        self.proc.wait(timeout=60)


def main():
    prompt = [1]
    extra = [7]

    a = Service(MODEL)
    print(f"READY ctx={a.ctx} vocab={a.vocab} stop={a.stop}")
    first, reason1, pos1 = a.turn(prompt, max_new=3)
    print(f"turno1 tokens={first} reason={reason1} pos={pos1}")
    assert pos1 == len(prompt) + len(first), (pos1, first)
    second, reason2, pos2 = a.turn(extra, max_new=2, keep=pos1)
    print(f"turno2 tokens={second} reason={reason2} pos={pos2}")
    a.close()

    b = Service(MODEL)
    full = prompt + first + extra
    fresh, reason3, pos3 = b.turn(full, max_new=2)
    print(f"fresh   tokens={fresh} reason={reason3} pos={pos3}")
    b.close()

    assert second == fresh, (second, fresh)
    assert pos2 == pos3, (pos2, pos3)
    print("OK: riuso del prefisso equivalente al prefill completo")

    c = Service(MODEL)
    bad, msg, _ = c.turn([1], max_new=1, keep=999)
    assert bad is None and "BAD_RANGE" in msg, msg
    ok, reason, pos = c.turn([1], max_new=1)
    assert ok and pos == 2, (ok, pos)
    reset = c.turn([1], max_new=1, keep=0)
    assert reset[2] == 2, reset
    c.close()
    print("OK: keep invalido rifiutato senza corrompere la sessione")


if __name__ == "__main__":
    main()
