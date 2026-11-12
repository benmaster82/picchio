"""Compare grouped decode to the legacy scheduler on identical generated weights.

python test_moe_grouped.py --exe ./picchio.exe
Requires numpy and safetensors. Exercises full oracle dumps, not just text.
"""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from safetensors.numpy import load_file, save_file
from make_test_model import make_test_model
from convert import quantize_int3, quantize_int4


def fixture(directory, bits, family):
    make_test_model(str(directory))
    config_path = directory / 'config.json'
    cfg = json.loads(config_path.read_text())
    cfg['picchio_expert_bits'] = 3 if bits == 3 else 4
    cfg['model_type'] = {'gpt': 'gpt_oss', 'qwen': 'qwen3_moe', 'minimax': 'minimax_m2'}[family]
    config_path.write_text(json.dumps(cfg))
    path = directory / 'model.safetensors'
    tensors = load_file(str(path))
    rng = np.random.default_rng(17)
    for layer in range(2):
        prefix = f'model.layers.{layer}'
        if family != 'gpt':
            tensors[prefix + '.self_attn.q_norm.weight'] = np.ones(64 if family == 'minimax' else 16, dtype='float32')
            tensors[prefix + '.self_attn.k_norm.weight'] = np.ones(32 if family == 'minimax' else 16, dtype='float32')
        for expert in range(4):
            base = f'{prefix}.mlp.experts.{expert}'
            for part in ('gate_up_proj', 'down_proj'):
                # Nonzero biases exercise addition before activation/scaling.
                bias_key = base + '.' + part + '_bias'
                tensors[bias_key] = (rng.standard_normal(tensors[bias_key].shape) * .01).astype('float32')
                if part == 'gate_up_proj':
                    # Exercise positive gate/up and negative up clipping too.
                    tensors[bias_key][0:4] = [8, 9, -8, -9]
                key = base + '.' + part
                if bits:
                    weights = tensors.pop(key)
                    if bits == -4:
                        packed, scales = quantize_int4(weights, group_size=0)
                    else:
                        packed, scales = (quantize_int3(weights) if bits == 3 else quantize_int4(weights))
                    # Packed loader supports this projection-first naming too.
                    new_key = f'{prefix}.mlp.experts.{part}.{expert}'
                    tensors[new_key], tensors[new_key + '.qs'] = packed, scales
    save_file(tensors, str(path))


def run(exe, model, grouped, threads, root, *, oracle=True, async_moe=False, idot=False):
    dump = Path(tempfile.mkdtemp(dir=root, prefix='oracle-'))
    env = os.environ.copy()
    for key in ('MODEL', 'MODEL_AUX', 'PIPE_ROLE', 'INPUT', 'PROMPT', 'OUTPUT', 'INPUT_FILE', 'TRACE_NUMERIC', 'ORACLE_DIR'):
        env.pop(key, None)
    env.update(SERVICE='1', CTX='64', PIN_GB='.001', ECAP='2',
        OMP_NUM_THREADS=str(threads), IO_THREADS='2', MOE_GROUPED=str(grouped),
        IDOT=str(int(idot)), PREFETCH='1', FLAT='0', ASYNC_MOE=str(int(async_moe)),
        DIRECT='1', REP='1', GPU='0', GPU_DENSE='0', GPU_ROUTER='0',
        GPU_PREFETCH='0', GPU_EXPERTS='0', GPU_LMHEAD='0', GPU_DENSE_RELEASE_HOST='0',
        HOTSTORE_PRELOAD='0')
    if oracle:
        env['ORACLE_DIR'] = str(dump)
    commands = ('STATS\nTURN 8 0 0 1 1 3 1 5 3\nSTATS\n'
                'TURN 3 3 0 1 1 1 7\nRESET\nTURN 8 0 0 1 1 3 1 5 3\nSTATS\nSHUTDOWN\n')
    result = subprocess.run([str(exe), str(model)], env=env, input=commands,
        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=40)
    assert result.returncode == 0, result.stderr
    frames = [line for line in result.stdout.splitlines() if line.startswith(('TOKEN ', 'DONE '))]
    stats = [json.loads(line[6:]) for line in result.stdout.splitlines() if line.startswith('STATS ')]
    if grouped:
        assert stats[-1]['moe_scratch_allocs'] == 1, stats
        assert stats[-1]['moe_scratch_bytes'] == stats[1]['moe_scratch_bytes']
        if not async_moe and not idot:
            assert stats[-1]['moe_grouped_calls'] > 0, stats
        if async_moe and not oracle:
            assert stats[-1]['async_batches'] > 0, stats
            assert stats[-1]['moe_grouped_calls'] == 0, stats
    dumps = {p.name: p.read_bytes() for p in dump.glob('*.npy')}
    if oracle:
        assert any('logits' in name for name in dumps)
        assert all(np.isfinite(np.load(io.BytesIO(value))).all() for value in dumps.values())
    return frames, dumps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', default='./picchio.exe' if os.name == 'nt' else './picchio')
    args = parser.parse_args()
    exe = Path(args.exe).resolve()
    with tempfile.TemporaryDirectory(prefix='moe-grouped-') as tmp:
        root = Path(tmp)
        for family in ('gpt', 'qwen', 'minimax'):
            for bits in (0, 3, 4, -4):
                model = root / f'{family}-{bits}'
                fixture(model, bits, family)
                reference = run(exe, model, 0, 1, root)
                for threads in (1, 2, 4):
                    candidate = run(exe, model, 1, threads, root)
                    assert candidate[0] == reference[0], (family, bits, threads, 'tokens')
                    assert candidate[1] == reference[1], (family, bits, threads, 'oracle tensors')
                # Exercise the production batched-prefill path and the async fallback.
                for async_moe, idot in ((False, False), (True, False), (False, bits == 4)):
                    before = run(exe, model, 0, 2, root, oracle=False, async_moe=async_moe, idot=idot)
                    after = run(exe, model, 1, 2, root, oracle=False, async_moe=async_moe, idot=idot)
                    assert before == after, (family, bits, async_moe, idot)
                print(f'PASS {family} expert format {bits}: identical tokens and all oracle tensors', flush=True)
    print('PASS grouped decode: scratch reuse, threads, formats, architectures, eviction, KV/reset, async/IDOT fallback')


if __name__ == '__main__':
    main()
