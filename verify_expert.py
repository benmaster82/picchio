"""Verifica: carica expert 0 layer 0 dal convertito, matmul, confronta con originale."""
import numpy as np
from safetensors import safe_open
import sys
sys.path.insert(0, '.')
from convert import FP4_LUT, quantize_int4

# Carica expert convertito (INT4 gs64)
with safe_open('D:/gptoss_i4/model-00009.safetensors', framework='numpy') as f:
    gu_packed = f.get_tensor('model.layers.0.mlp.experts.gate_up_proj.0')
    gu_scales = f.get_tensor('model.layers.0.mlp.experts.gate_up_proj.0.qs')
    print(f"Convertito gate_up: packed={gu_packed.shape} scales={gu_scales.shape}")

# Carica expert originale (MXFP4) e dequantizza
with safe_open('D:/gptoss_raw/model-00009-of-00014.safetensors', framework='numpy') as f:
    keys = [k for k in f.keys() if 'layer' in k and '0.mlp' in k]
    blocks_key = 'model.layers.0.mlp.experts.gate_up_proj_blocks'
    scales_key = 'model.layers.0.mlp.experts.gate_up_proj_scales'
    blocks = f.get_tensor(blocks_key)  # [128, 5760, 90, 16]
    scales = f.get_tensor(scales_key)  # [128, 5760, 90]
    print(f"Originale: blocks={blocks.shape} scales={scales.shape}")

# Dequantizza expert 0
e = 0
blk = blocks[e]  # [5760, 90, 16]
sc = scales[e]   # [5760, 90]
rows, n_blk = 5760, 90
cols = n_blk * 32  # = 2880

blk_flat = blk.reshape(rows, n_blk * 16)
lo = blk_flat & 0x0F
hi = (blk_flat >> 4) & 0x0F
nibbles = np.empty((rows, n_blk * 32), dtype=np.uint8)
nibbles[:, 0::2] = lo
nibbles[:, 1::2] = hi
nibbles = nibbles[:, :cols]
values = FP4_LUT[nibbles]
sc_float = np.ldexp(np.ones_like(sc, dtype=np.float32), sc.astype(np.int32) - 127)
sc_expanded = np.repeat(sc_float, 32, axis=1)[:, :cols]
original_f32 = values * sc_expanded

print(f"Dequantizzato: shape={original_f32.shape} range=[{original_f32.min():.2f}, {original_f32.max():.2f}]")

# Requantizza con gs64 (come fa il convertitore)
packed_test, scales_test = quantize_int4(original_f32, group_size=64)
print(f"Requantizzato: packed={packed_test.shape} scales={scales_test.shape}")

# Confronta con quello salvato
print(f"\nScale dal file: shape={gu_scales.shape} range=[{gu_scales.min():.4f}, {gu_scales.max():.4f}]")
print(f"Scale ricalcolate: shape={scales_test.shape} range=[{scales_test.min():.4f}, {scales_test.max():.4f}]")

# Matmul test: x = [0.01, 0.01, ...] 
x = np.full(cols, 0.01, dtype=np.float32)

# Reference: F32 matmul
y_ref = original_f32 @ x
print(f"\ny_ref (F32): first 5 = {y_ref[:5]}")

# INT4 gs64 matmul (simulata)
O, I = rows, cols
gs = 64
n_groups = (I + gs - 1) // gs
scales_2d = scales_test.reshape(O, n_groups)
packed_2d = packed_test.reshape(O, (I+1)//2)

y_q = np.zeros(O, dtype=np.float32)
for o in range(min(O, 5)):
    val = 0.0
    for g in range(n_groups):
        gs_start = g * gs
        gs_end = min(gs_start + gs, I)
        group_acc = 0.0
        for i in range(gs_start, gs_end, 2):
            byte = packed_2d[o, i // 2]
            lo_v = int(byte & 0xF) - 8
            hi_v = int(byte >> 4) - 8
            group_acc += x[i] * lo_v
            if i + 1 < gs_end:
                group_acc += x[i+1] * hi_v
        val += group_acc * scales_2d[o, g]
    y_q[o] = val
    
print(f"y_q (INT4 gs64): first 5 = {y_q[:5]}")
print(f"Errore max (prime 5 righe): {np.max(np.abs(y_ref[:5] - y_q[:5])):.4f}")
