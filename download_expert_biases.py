#!/usr/bin/env python3
"""Scarica solo i bias expert originali GPT-OSS e crea un sidecar F32."""
import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import HfApi, hf_hub_url
from safetensors.numpy import save_file

REPO = "openai/gpt-oss-120b"


def request_range(session, url, start, end, retries=6):
    for attempt in range(retries):
        try:
            response = session.get(
                url, headers={"Range": f"bytes={start}-{end}"}, timeout=120
            )
            response.raise_for_status()
            expected = end - start + 1
            if response.status_code != 206 or len(response.content) != expected:
                raise IOError(f"range incompleto: {len(response.content)}/{expected}")
            return response.content
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)


def remote_header(session, filename):
    url = hf_hub_url(REPO, filename)
    length = struct.unpack("<Q", request_range(session, url, 0, 7))[0]
    header = json.loads(request_range(session, url, 8, 7 + length))
    return url, length, header

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="C:/picchio/expert_biases.safetensors")
    args = parser.parse_args()
    output = Path(args.output)
    session = requests.Session()
    repo_files = HfApi().model_info(REPO).siblings
    shards = sorted(s.rfilename for s in repo_files if s.rfilename.endswith(".safetensors"))
    tensors = {}

    for index, filename in enumerate(shards, 1):
        url, header_len, header = remote_header(session, filename)
        names = [name for name in header if name.endswith(
            ("experts.gate_up_proj_bias", "experts.down_proj_bias")
        )]
        for name in names:
            meta = header[name]
            start, end = meta["data_offsets"]
            raw = request_range(
                session, url, 8 + header_len + start, 7 + header_len + end
            )
            dtype = meta["dtype"]
            if dtype == "BF16":
                u16 = np.frombuffer(raw, dtype="<u2")
                array = (u16.astype(np.uint32) << 16).view(np.float32)
            elif dtype == "F32":
                array = np.frombuffer(raw, dtype="<f4")
            elif dtype == "F16":
                array = np.frombuffer(raw, dtype="<f2").astype(np.float32)
            else:
                raise ValueError(f"dtype non supportato: {dtype} per {name}")
            tensors[name] = array.reshape(meta["shape"]).copy()
        print(f"[{index}/{len(shards)}] {filename}: {len(names)} bias", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata={
        "source": REPO,
        "purpose": "Picchio expert biases F32 sidecar",
    })
    print(f"Sidecar: {output} — {len(tensors)} tensori — {output.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
