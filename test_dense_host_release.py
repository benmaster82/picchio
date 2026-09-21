"""Real SERVICE regression on a tiny generated model (requires NVIDIA GPU).

Checks identical retained/released GPU output, KV prefix truncation, RESET,
zero-output turns, native phase counters, and fatal released-host GPU failure.
"""
from pathlib import Path
import subprocess
import tempfile

from chat import PicchioSession
from make_test_model import make_test_model


def main():
    root = Path(__file__).resolve().parents[2]
    exe = Path(__file__).with_name("picchio.exe")
    with tempfile.TemporaryDirectory(prefix="dense-release-", dir=root / "build") as directory:
        make_test_model(directory)
        baseline = None
        for release in (False, True):
            session = PicchioSession(exe, directory, 64, 0.0015, 2, None,
                {"TEMPERATURE": 0, "TOPP": 1, "TOPK": 1, "SEED": 1234},
                gpu_dense=True, gpu_dense_release_host=release, flat="0")
            try:
                initial = session.stats()
                assert bool(initial["gpu_dense_host_released_bytes"]) == release
                first, _, pos = session.turn([1, 5, 3], 4, 0, lambda _: None)
                assert pos == 7 and len(first) == 4
                assert session.last_timing["prefill_tokens"] == 3
                assert session.last_timing["decode_tokens"] == 4
                assert session.last_timing["decode_seconds"] > 0
                second, _, pos = session.turn([7], 2, 3, lambda _: None)
                assert pos == 6 and session.last_timing["prefill_tokens"] == 1
                session.reset()
                assert session.stats()["position"] == 0
                repeated, _, _ = session.turn([1, 5, 3], 4, 0, lambda _: None)
                assert repeated == first
                session.reset()
                empty, _, pos = session.turn([1], 0, 0, lambda _: None)
                assert empty == [] and pos == 1
                assert session.last_timing["decode_tokens"] == 0
                assert session.last_timing["decode_tokens_per_s"] is None
                final = session.stats()
                assert final["gpu_dense_fallbacks"] == 0
                assert final["gpu_dense_calls"] > 0
                assert final["prefill_tokens"] == 8 and final["decode_tokens"] == 10
                if baseline is None:
                    baseline = (first, second, final["resident_gb"])
                else:
                    assert (first, second) == baseline[:2]
                    assert final["resident_gb"] < baseline[2]
            finally:
                session.close()
    failure = subprocess.run([str(exe), "--gpu-dense-test", "fail-after-release"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert failure.returncode == 1, (failure.returncode, failure.stderr)
    assert "fatal: GPU failure after host weights were released" in failure.stderr
    print("PASS: retained/released output, KV reuse/reset, phase timings, fatal GPU failure")


if __name__ == "__main__":
    main()
