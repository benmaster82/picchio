"""Ispeziona il formato reale dei tensori nel modello GPT-OSS-120B."""
import torch
from safetensors import safe_open

# Apri il primo shard
path = "D:/gptoss_raw/model-00000-of-00014.safetensors"
print(f"Ispezionando: {path}\n")

with safe_open(path, framework="pt") as f:
    keys = list(f.keys())
    print(f"{len(keys)} tensori nel file\n")
    
    # Mostra tutti i tensori con dtype e shape
    for k in keys[:30]:
        t = f.get_tensor(k)
        print(f"  {k}")
        print(f"    dtype={t.dtype} shape={list(t.shape)}")
        if t.dtype == torch.uint8:
            print(f"    (packed) min={t.min().item()} max={t.max().item()}")
        elif t.dtype in (torch.float32, torch.bfloat16, torch.float16):
            tf = t.float()
            print(f"    range=[{tf.min().item():.4f}, {tf.max().item():.4f}]")
        print()

    # Cerca expert specifico
    print("\n=== Expert layer 0, expert 0 ===")
    expert_keys = [k for k in keys if "layers.0.mlp.experts" in k and ".0." in k.split("experts")[1][:3]]
    for k in expert_keys:
        t = f.get_tensor(k)
        print(f"  {k}: dtype={t.dtype} shape={list(t.shape)}")
