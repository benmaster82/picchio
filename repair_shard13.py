#!/usr/bin/env python3
"""Ripara in-place la coda troncata dello shard Picchio 13, in modo resumable."""
import argparse
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import hf_hub_url

from convert import FP4_LUT, quantize_int4

REPO = "openai/gpt-oss-120b"
RAW_NAME = "model-00013-of-00014.safetensors"
TARGET = Path("D:/gptoss_i4/model-00013.safetensors")
GROUP_SIZE = 64


def read_header_local(path):
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    return length, header


def request_range(session, url, start, end, retries=6):
    for attempt in range(retries):
        try:
            response = session.get(
                url, headers={"Range": f"bytes={start}-{end}"}, timeout=120
            )
            response.raise_for_status()
            expected = end - start + 1
            if response.status_code != 206 or len(response.content) != expected:
                raise IOError(
                    f"range {start}-{end}: status={response.status_code}, "
                    f"bytes={len(response.content)}/{expected}"
                )
            return response.content
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)


def read_header_remote(session, url):
    length = struct.unpack("<Q", request_range(session, url, 0, 7))[0]
    raw = request_range(session, url, 8, 7 + length)
    return length, json.loads(raw)

def raw_expert(session, url, raw_header_len, raw_header, name, expert):
    meta = raw_header[name]
    shape = meta["shape"]
    if not shape or expert < 0 or expert >= shape[0]:
        raise ValueError(f"expert {expert} non valido per {name}: {shape}")
    start, end = meta["data_offsets"]
    stride = (end - start) // shape[0]
    expert_start = start + expert * stride
    raw = request_range(
        session, url, 8 + raw_header_len + expert_start,
        7 + raw_header_len + expert_start + stride,
    )
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape[1:])


def convert_expert(session, url, raw_header_len, raw_header, base, expert):
    blocks = raw_expert(
        session, url, raw_header_len, raw_header, base + "_blocks", expert
    )
    scales = raw_expert(
        session, url, raw_header_len, raw_header, base + "_scales", expert
    )
    rows, n_blocks = blocks.shape[0], blocks.shape[1]
    flat = blocks.reshape(rows, n_blocks * 16)
    nibbles = np.empty((rows, n_blocks * 32), dtype=np.uint8)
    nibbles[:, 0::2] = flat & 0x0F
    nibbles[:, 1::2] = (flat >> 4) & 0x0F
    values = FP4_LUT[nibbles]
    scale_f32 = np.ldexp(
        np.ones_like(scales, dtype=np.float32), scales.astype(np.int32) - 127
    )
    values *= np.repeat(scale_f32, 32, axis=1)
    packed, converted_scales = quantize_int4(values, group_size=GROUP_SIZE)
    return packed.tobytes(), converted_scales.astype("<f4", copy=False).tobytes()


def parse_target(name):
    parts = name.split(".")
    expert = int(parts[-1])
    projection = parts[-2]
    layer = int(parts[2])
    base = f"model.layers.{layer}.mlp.experts.{projection}"
    return base, expert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=TARGET)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    target = args.target
    header_len, header = read_header_local(target)
    data_base = 8 + header_len
    entries = sorted(
        (meta["data_offsets"][0], meta["data_offsets"][1], name, meta)
        for name, meta in header.items() if name != "__metadata__"
    )
    final_size = data_base + entries[-1][1]
    if target.stat().st_size == final_size:
        print("Shard gia completo")
        return
    if target.stat().st_size > final_size:
        raise RuntimeError("Il file supera la dimensione dichiarata")

    session = requests.Session()
    url = hf_hub_url(REPO, RAW_NAME)
    raw_header_len, raw_header = read_header_remote(session, url)
    repaired = 0

    with target.open("r+b") as handle:
        while handle.seek(0, 2) < final_size:
            size = handle.tell()
            pending = next((entry for entry in entries if data_base + entry[1] > size), None)
            if pending is None:
                break
            start, end, name, meta = pending
            absolute_start = data_base + start
            absolute_end = data_base + end
            if size != absolute_start:
                raise RuntimeError(
                    f"confine non allineato: size={size}, prossimo={absolute_start} {name}"
                )
            if name.endswith(".qs"):
                raise RuntimeError(f"scala inattesa nella coda troncata: {name}")

            base, expert = parse_target(name)
            payload, converted_scales = convert_expert(
                session, url, raw_header_len, raw_header, base, expert
            )

            # Le scale sono già nella parte integra: devono coincidere bit-per-bit.
            scale_name = name + ".qs"
            scale_meta = header.get(scale_name)
            if scale_meta is None:
                raise RuntimeError(f"scale mancanti nell'header: {scale_name}")
            scale_start, scale_end = scale_meta["data_offsets"]
            handle.seek(data_base + scale_start)
            stored_scales = handle.read(scale_end - scale_start)
            if stored_scales != converted_scales:
                raise RuntimeError(f"scale non coincidenti: {scale_name}")

            expected = absolute_end - absolute_start
            if len(payload) != expected:
                raise RuntimeError(f"{name}: bytes={len(payload)}, attesi={expected}")
            if args.check_only:
                print(f"CHECK {name}: {expected} byte, scale verificate")
                return
            handle.seek(0, 2)
            handle.write(payload)
            handle.flush()
            repaired += 1
            if repaired % 10 == 0 or handle.tell() == final_size:
                print(
                    f"{repaired} tensori aggiunti; {handle.tell()/1e9:.3f}/"
                    f"{final_size/1e9:.3f} GB", flush=True
                )
    print(f"Riparazione completata: {target} ({target.stat().st_size} byte)")


if __name__ == "__main__":
    main()
