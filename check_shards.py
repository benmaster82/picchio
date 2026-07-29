#!/usr/bin/env python3
import json
import struct
from pathlib import Path
from safetensors import safe_open

for path in sorted(Path("D:/gptoss_i4").glob("model-*.safetensors")):
    try:
        with safe_open(str(path), framework="pt") as handle:
            count = len(list(handle.keys()))
        status = f"OK tensors={count}"
    except Exception as error:
        status = f"FAIL {error}"
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    ends = [
        value["data_offsets"][1]
        for name, value in header.items()
        if name != "__metadata__"
    ]
    expected = 8 + header_len + max(ends, default=0)
    print(
        path.name, status, f"size={path.stat().st_size}",
        f"expected={expected}", f"delta={path.stat().st_size - expected}",
        f"header={header_len}",
    )
