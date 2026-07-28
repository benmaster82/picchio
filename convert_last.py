"""Riconverte solo l'ultimo shard."""
import sys, time
sys.path.insert(0, 'C:/picchio')
from convert import *
import json
from pathlib import Path

cfg = json.load(open('D:/gptoss_raw/config.json'))
stats = {'total_tensors':0,'norms':0,'bias':0,'router':0,'dense_i4':0,'expert_i4':0,'expert_blocks':0,'expert_scales':0,'other':0}
output_tensors = {}

shard = 'D:/gptoss_raw/model-00014-of-00014.safetensors'
print(f'Convertendo: {shard}')
t0 = time.time()
convert_shard(shard, output_tensors, cfg, stats)

blocks_keys = [k for k in list(output_tensors.keys()) if k.endswith('_blocks')]
for bk in blocks_keys:
    base = bk[:-7]
    sk = base + '_scales'
    if sk not in output_tensors: continue
    blocks = output_tensors[bk]
    scales = output_tensors[sk]
    n_exp, rows, n_blk = blocks.shape[0], blocks.shape[1], blocks.shape[2]
    block_size = 32
    cols = n_blk * block_size
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
        sc_expanded = np.repeat(sc_float, block_size, axis=1)[:, :cols]
        values *= sc_expanded
        packed, row_scales = quantize_int4(values)
        output_tensors[f'{base}.{e}'] = packed
        output_tensors[f'{base}.{e}.qs'] = row_scales
    del output_tensors[bk]
    del output_tensors[sk]
    print(f' done ({n_exp} expert)')

for bk in [k for k in list(output_tensors.keys()) if 'experts' in k and 'bias' in k]:
    t = output_tensors[bk]
    if t.dtype != np.float32: output_tensors[bk] = t.astype(np.float32)

save_file(output_tensors, 'D:/gptoss_i4/model-00014.safetensors')
print(f'Done in {time.time()-t0:.0f}s')
