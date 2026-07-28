"""Riconverte shard 0 con attention in F32."""
import sys, time, json
sys.path.insert(0, 'C:/picchio')
from convert import *
from pathlib import Path

cfg = json.load(open('D:/gptoss_raw/config.json'))
stats = {'total_tensors':0,'norms':0,'bias':0,'router':0,'dense_i4':0,'expert_i4':0,'expert_blocks':0,'expert_scales':0,'other':0}
output_tensors = {}
t0 = time.time()
print('Converting shard 0 (attention=F32)...')
convert_shard('D:/gptoss_raw/model-00000-of-00014.safetensors', output_tensors, cfg, stats)

# Verifica
attn_keys = [k for k in output_tensors if 'self_attn' in k and 'weight' in k and '.qs' not in k]
print(f'Attention tensors: {len(attn_keys)}')
if attn_keys:
    print(f'  {attn_keys[0]}: dtype={output_tensors[attn_keys[0]].dtype} shape={output_tensors[attn_keys[0]].shape}')

# MXFP4 post-process
blocks_keys = [k for k in list(output_tensors.keys()) if k.endswith('_blocks')]
for bk in blocks_keys:
    base = bk[:-7]
    sk = base + '_scales'
    if sk not in output_tensors: continue
    blocks = output_tensors[bk]
    scales = output_tensors[sk]
    n_exp, rows, n_blk = blocks.shape[0], blocks.shape[1], blocks.shape[2]
    cols = n_blk * 32
    print(f'  MXFP4: {base} [{n_exp}x{rows}x{cols}]...', end='', flush=True)
    for e in range(n_exp):
        blk_flat = blocks[e].reshape(rows, n_blk * 16)
        lo = blk_flat & 0x0F
        hi = (blk_flat >> 4) & 0x0F
        nibbles = np.empty((rows, n_blk * 32), dtype=np.uint8)
        nibbles[:, 0::2] = lo
        nibbles[:, 1::2] = hi
        nibbles = nibbles[:, :cols]
        values = FP4_LUT[nibbles]
        sc = scales[e]
        sc_float = np.ldexp(np.ones_like(sc, dtype=np.float32), sc.astype(np.int32) - 127)
        sc_expanded = np.repeat(sc_float, 32, axis=1)[:, :cols]
        values *= sc_expanded
        packed, row_scales = quantize_int4(values)
        output_tensors[f'{base}.{e}'] = packed
        output_tensors[f'{base}.{e}.qs'] = row_scales
    del output_tensors[bk]
    del output_tensors[sk]
    print(f' done')

for bk in [k for k in list(output_tensors.keys()) if 'experts' in k and 'bias' in k]:
    t = output_tensors[bk]
    if t.dtype != np.float32: output_tensors[bk] = t.astype(np.float32)

out_path = 'D:/gptoss_i4/model-00000.safetensors'
save_file(output_tensors, out_path)
sz = Path(out_path).stat().st_size
print(f'Done in {time.time()-t0:.0f}s — {sz/1e9:.2f} GB')
