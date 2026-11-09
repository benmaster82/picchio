"""Real-process warm-start regressions using generated weights only.

Usage: python test_hotstore.py [--exe ./picchio.exe]
Requires numpy and safetensors; no tokenizer, GPU or downloaded model.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

from make_test_model import make_test_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', default='./picchio.exe' if os.name == 'nt' else './picchio')
    parser.add_argument('--bits', type=int, choices=(0, 3, 4), default=0,
                        help='expert format: 0=F32, 3/4=synthetic packed weights')
    args = parser.parse_args()
    exe = str(Path(args.exe).resolve())
    with tempfile.TemporaryDirectory(prefix='picchio-hotstore-') as tmp:
        model = Path(tmp) / 'model'
        make_test_model(str(model))
        if args.bits:
            import numpy as np
            from safetensors.numpy import load_file, save_file
            config = json.loads((model / 'config.json').read_text())
            config['picchio_expert_bits'] = args.bits
            (model / 'config.json').write_text(json.dumps(config))
            path = model / 'model.safetensors'
            tensors = {k: v for k, v in load_file(str(path)).items()
                       if '.mlp.experts.' not in k}
            rng = np.random.default_rng(42)
            for layer in range(2):
                prefix = f'model.layers.{layer}.mlp'
                for eid in range(4):
                    for part, output, dim in (('gate_up_proj', 256, 64), ('down_proj', 64, 128)):
                        key = f'{prefix}.experts.{part}.{eid}'
                        groups = dim // 64
                        tensors[key] = rng.integers(0, 256,
                            (output, groups * (24 if args.bits == 3 else 32)), dtype='uint8')
                        tensors[key + '.qs'] = np.full((output, groups), 0.01, dtype='float32')
                        tensors[f'{prefix}.experts.{eid}.{part}_bias'] = np.zeros(output, dtype='float32')
            save_file(tensors, str(path))
        usage = model / '.picchio_usage'
        valid = struct.pack('<12I', 0x50494343, 1, 2, 4, 1, 100, 0, 50, 1, 100, 0, 50)
        baseline_tokens = None
        baseline_logits = None

        def run(preload, payload, *, async_moe=False, generate=True, commands_override=None):
            if payload is None:
                usage.unlink(missing_ok=True)
            else:
                usage.write_bytes(payload)
            oracle = Path(tempfile.mkdtemp(prefix='oracle-', dir=tmp))
            env = os.environ.copy()
            for key in ('MODEL', 'MODEL_AUX', 'PIPE_ROLE', 'INPUT', 'INPUT_FILE',
                        'PROMPT', 'OUTPUT', 'TRACE_NUMERIC'):
                env.pop(key, None)
            env.update(SERVICE='1', CTX='64', PIN_GB='0.001', ECAP='2',
                       OMP_NUM_THREADS='2', IO_THREADS='2', PREFETCH='1', PILOT='0',
                       FLAT='0', DIRECT='1', ASYNC_MOE=str(int(async_moe)), REP='1',
                       GPU='0', GPU_DENSE='0', GPU_ROUTER='0', GPU_PREFETCH='0',
                       GPU_EXPERTS='0', GPU_LMHEAD='0', GPU_DENSE_RELEASE_HOST='0',
                       HOTSTORE_PRELOAD=str(preload), ORACLE_DIR=str(oracle))
            commands = 'STATS\n'
            if generate:
                commands += 'TURN 8 0 0 1 1 3 1 5 3\nRESET\nTURN 8 0 0 1 1 3 1 5 3\n'
            commands += 'STATS\nSHUTDOWN\n'
            if commands_override is not None:
                commands = commands_override
            result = subprocess.run([exe, str(model)], input=commands, env=env,
                                    capture_output=True, text=True, encoding='utf-8',
                                    errors='replace', timeout=30)
            return result, oracle

        for preload, payload, expected in (
                (0, valid, 0), (1, valid, 2), (2, valid, 4), (999, valid, 4),
                (2, None, 0), (2, valid[:20], 0),
                (2, valid[:4] + struct.pack('<I', 2) + valid[8:], 0),
                (2, valid[:8] + struct.pack('<I', 3) + valid[12:], 0),
                (2, b'bad', 0), (2, valid[:16] + bytes(32), 0)):
            for async_moe in (False, True):
                result, oracle = run(preload, payload, async_moe=async_moe)
                assert result.returncode == 0, result.stderr
                stats = [json.loads(line[6:]) for line in result.stdout.splitlines()
                         if line.startswith('STATS ')]
                assert stats[0]['expert_loads'] == expected, (preload, result.stderr)
                assert stats[0]['expert_slots_per_layer'] == 2
                tokens = [line for line in result.stdout.splitlines() if line.startswith('TOKEN ')]
                assert tokens and len(tokens) % 2 == 0, result.stdout
                half = len(tokens) // 2
                assert tokens[:half] == tokens[half:], result.stdout
                logits = {p.name: p.read_bytes() for p in oracle.glob('*logits*.npy')}
                assert logits, 'expected oracle logit dumps'
                if baseline_tokens is None:
                    baseline_tokens, baseline_logits = tokens, logits
                assert tokens == baseline_tokens, result.stdout
                assert logits == baseline_logits, 'warm start changed logits'

        # Empty or malformed values must fail, not silently enable preloading.
        for value in ('', '-1', 'nan', '1.5', 'garbage', '2147483648'):
            result, _ = run(value, valid, generate=False)
            assert result.returncode == 1, (value, result.stderr)
            assert 'HOTSTORE_PRELOAD must be' in result.stderr

        # Force top-1 routing with a frequency tie at IDs 1 and 3. The first
        # preloaded entry must be ID 1 (not a zero-count or the other tied ID).
        import numpy as np
        from safetensors.numpy import load_file, save_file
        config_path = model / 'config.json'
        config = json.loads(config_path.read_text())
        config['num_experts_per_tok'] = 1
        config_path.write_text(json.dumps(config))
        path = model / 'model.safetensors'
        tensors = load_file(str(path))
        for layer in range(2):
            prefix = f'model.layers.{layer}.mlp.router'
            tensors[prefix + '.weight'] = np.zeros((4, 64), dtype='float32')
            tensors[prefix + '.bias'] = np.array([0, 100, 0, 50], dtype='float32')
        save_file(tensors, str(path))
        tied = struct.pack('<12I', 0x50494343, 1, 2, 4, 0, 100, 0, 100, 0, 100, 0, 100)
        result, _ = run(1, tied, commands_override=
            'STATS\nTURN 0 0 0 1 1 1 1\nSTATS\nSHUTDOWN\n')
        assert result.returncode == 0, result.stderr
        stats = [json.loads(line[6:]) for line in result.stdout.splitlines()
                 if line.startswith('STATS ')]
        assert stats[1]['cache_hits'] - stats[0]['cache_hits'] == 2, result.stdout
        assert stats[1]['expert_loads'] == stats[0]['expert_loads'], result.stdout

    print('PASS: warm-start bounds, invalid/missing usage, sync/async eviction, RESET, identical tokens/logits')


if __name__ == '__main__':
    main()
