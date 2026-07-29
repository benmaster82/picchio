#!/usr/bin/env python3
"""Confronta due directory di dump Picchio/Transformers e trova il primo mismatch."""
import argparse
import json
from pathlib import Path

import numpy as np

ORDER = [
    "embedding", "rope_cos", "rope_sin",
    "input", "pre_attn_norm", "q_nobias", "k_nobias", "v_nobias",
    "q_bias", "k_bias", "v_bias", "q_rope", "k_rope", "v_heads",
    "attn_scores", "sink_logits", "attn_probs_with_sink", "attn_concat",
    "attn_out", "post_attn_residual", "pre_moe_norm", "router_logits",
    "top_indices", "top_weights", "gate_up", "activated", "output",
    "contribution", "moe_out", "final_norm", "logits",
]


def sort_key(name):
    for index, marker in enumerate(ORDER):
        if name == marker or name.endswith("." + marker):
            return index, name
    return len(ORDER), name


def compare(reference, candidate, atol, rtol):
    ref = Path(reference)
    got = Path(candidate)
    ref_names = {p.stem for p in ref.glob("*.npy")}
    got_names = {p.stem for p in got.glob("*.npy")}
    common = sorted(ref_names & got_names, key=sort_key)
    if not common:
        raise SystemExit("Nessun dump .npy con nome comune")
    failed = False
    for name in common:
        a = np.load(ref / f"{name}.npy", allow_pickle=False)
        b = np.load(got / f"{name}.npy", allow_pickle=False)
        if a.shape != b.shape:
            print(f"FAIL {name}: shape {b.shape}, attesa {a.shape}")
            failed = True
            break
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        max_abs = float(diff.max(initial=0))
        denom = np.maximum(np.abs(a.astype(np.float64)), 1e-12)
        max_rel = float((diff / denom).max(initial=0))
        ok = bool(np.allclose(a, b, atol=atol, rtol=rtol))
        print(f"{'OK  ' if ok else 'FAIL'} {name}: abs={max_abs:.6g} rel={max_rel:.6g}")
        if not ok:
            failed = True
            break
    missing = sorted(ref_names - got_names)
    if missing:
        print(f"Non ancora prodotti dal candidato: {len(missing)} dump")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    raise SystemExit(compare(args.reference, args.candidate, args.atol, args.rtol))


if __name__ == "__main__":
    main()
