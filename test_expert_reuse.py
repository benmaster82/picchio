"""Regression for packed cache storage, eviction, indexing and short reads.

Uses only generated INT3/INT4 fixtures; never modifies the user's checkpoint.
"""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np
from safetensors.numpy import load_file, save_file
from chat import PicchioSession
from make_test_model import make_test_model


def packed_fixture(directory, bits):
    directory = Path(directory)
    make_test_model(str(directory))
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text())
    config.update(picchio_expert_bits=bits, num_local_experts=8)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    path = directory / "model.safetensors"
    tensors = {k: v for k, v in load_file(str(path)).items() if ".mlp.experts." not in k}
    rng = np.random.default_rng(42)
    for layer in range(2):
        prefix = f"model.layers.{layer}.mlp"
        tensors[prefix + ".router.weight"] = (rng.standard_normal((8, 64)) * 0.1).astype("float32")
        tensors[prefix + ".router.bias"] = np.zeros(8, dtype="float32")
        for eid in range(8):
            for part, output, input_dim in (("gate_up_proj", 256, 64), ("down_proj", 64, 128)):
                key = f"{prefix}.experts.{part}.{eid}"
                groups = input_dim // 64
                tensors[key] = rng.integers(0, 256, (output, groups * (24 if bits == 3 else 32)), dtype="uint8")
                tensors[key + ".qs"] = np.full((output, groups), 0.01, dtype="float32")
                tensors[f"{prefix}.experts.{eid}.{part}_bias"] = np.zeros(output, dtype="float32")
    save_file(tensors, str(path))
    return path


def main():
    root = Path(__file__).resolve().parents[2]
    exe = Path(__file__).with_name("picchio.exe")
    with tempfile.TemporaryDirectory(prefix="expert-reuse-", dir=root / "build") as directory:
        for bits in (3, 4):
            path = packed_fixture(directory, bits)
            reference = None
            for reuse, index in ((False, False), (True, False), (True, True)):
                session = PicchioSession(exe, directory, 64, 0.0001, 2, None,
                    {"TEMPERATURE": 0, "TOPP": 1, "TOPK": 1, "SEED": 1234,
                     "PREFETCH": 1, "ECAP": 4}, async_moe=True, direct=True,
                    io_threads=2, flat="0", expert_reuse=reuse, tensor_index=index)
                try:
                    first, _, _ = session.turn([1, 5, 3], 12, 0, lambda _: None)
                    second, _, _ = session.turn([7, 8], 5, 5, lambda _: None)
                    session.reset()
                    repeated, _, _ = session.turn([1, 5, 3], 12, 0, lambda _: None)
                    assert repeated == first
                    if reference is None:
                        reference = (first, second)
                    else:
                        assert (first, second) == reference, (bits, reuse, index)
                    stats = session.stats()
                    assert bool(stats["expert_buffer_reuses"]) == reuse, stats
                finally:
                    session.close()
            checked = subprocess.run([str(exe), "--expert-io-test", directory],
                                     capture_output=True, text=True, encoding="utf-8", errors="replace")
            assert checked.returncode == 0, checked.stderr

        # SafeTensors stores F32 dense/scales before U8 packed experts. Remove
        # only the generated packed payload so startup succeeds and an I/O worker
        # must reject the short read. It must exit promptly, not deadlock in atexit.
        with path.open("r+b") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_length))
            packed_start = min(t["data_offsets"][0] for t in header.values()
                               if isinstance(t, dict) and t.get("dtype") == "U8")
            stream.truncate(8 + header_length + packed_start)
        env = os.environ.copy()
        env.update(SERVICE="1", CTX="64", PIN_GB="0.0001", OMP_NUM_THREADS="2",
                   EXPERT_REUSE="1", TENSOR_INDEX="1", ASYNC_MOE="1", DIRECT="1",
                   IO_THREADS="2", FLAT="0", PREFETCH="0", GPU="0", GPU_DENSE="0",
                   GPU_ROUTER="0", GPU_PREFETCH="0", GPU_EXPERTS="0", GPU_LMHEAD="0",
                   GPU_DENSE_RELEASE_HOST="0", ECAP="4")
        env.pop("MODEL_AUX", None)
        failed = subprocess.run([str(exe), directory], env=env,
            input="TURN 1 0 0 1 1 1 1\n", capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=15)
        assert failed.returncode == 1, (failed.returncode, failed.stderr)
        assert "fatal: incomplete expert read" in failed.stderr, failed.stderr
    print("PASS: INT3/INT4 bytes, eviction, prefetch/async, KV/reset, index, short-read termination")


if __name__ == "__main__":
    main()
