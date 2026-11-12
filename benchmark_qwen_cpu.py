"""Validate exact grouped Qwen decode across thread/cache settings and KV reuse.

Uses an existing converted model and local tokenizer. Records raw generated IDs,
phase timings and memory counters; never enables IDOT or GPU execution.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import queue
import subprocess
import threading
import time

from transformers import AutoTokenizer


def child_cpu_seconds(proc):
    """Optional Windows CPU time, to distinguish wall-time stalls from work."""
    if os.name != 'nt':
        return None
    handle = getattr(proc, '_handle', None)
    if handle is None:
        return None
    import ctypes
    from ctypes import wintypes
    get_times = ctypes.WinDLL('kernel32', use_last_error=True).GetProcessTimes
    get_times.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    get_times.restype = wintypes.BOOL
    creation, exit_time, kernel, user = [wintypes.FILETIME() for _ in range(4)]
    if not get_times(int(handle), ctypes.byref(creation), ctypes.byref(exit_time),
                     ctypes.byref(kernel), ctypes.byref(user)):
        return None
    return (((kernel.dwHighDateTime << 32) | kernel.dwLowDateTime) +
            ((user.dwHighDateTime << 32) | user.dwLowDateTime)) / 10_000_000


class Engine:
    def __init__(self, exe, model, threads, cache, grouped, log, timeout):
        self.timeout = timeout
        env = os.environ.copy()
        for key in ('MODEL', 'MODEL_AUX', 'INPUT', 'INPUT_FILE', 'OUTPUT', 'PROMPT',
                    'ORACLE_DIR', 'TRACE_NUMERIC', 'PIPE_ROLE', 'ECAP', 'PIPE', 'PREFILL'):
            env.pop(key, None)
        env.update(SERVICE='1', CTX='1024', PIN_GB=str(cache), OMP_NUM_THREADS=str(threads),
            IO_THREADS='4', MOE_GROUPED=str(grouped), MOE_SHARED_IDOT='0', IDOT='0',
            HOTSTORE_PRELOAD='0', GPU='0', GPU_DENSE='0', GPU_ROUTER='0', GPU_PREFETCH='0',
            GPU_EXPERTS='0', GPU_LMHEAD='0', GPU_DENSE_RELEASE_HOST='0',
            FLAT='0', PREFETCH='0', PILOT='0', ASYNC_MOE='0', DIRECT='0', REP='1')
        self.log = log.open('w', encoding='utf-8')
        self.lines = queue.Queue()
        self.proc = subprocess.Popen([str(exe), str(model)], env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, encoding='utf-8', bufsize=1)
        def reader():
            try:
                for line in self.proc.stdout:
                    self.lines.put(line.strip())
            finally:
                self.lines.put(None)
        threading.Thread(target=reader, daemon=True).start()
        start = time.monotonic()
        try:
            assert self.line().startswith('READY ')
        except BaseException:
            self.close()
            raise
        self.startup_seconds = time.monotonic() - start

    def line(self):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                value = self.lines.get(timeout=.5)
            except queue.Empty:
                continue
            if value is None:
                raise RuntimeError(f'engine exited, status {self.proc.poll()}; see stderr log')
            return value
        raise TimeoutError('engine protocol timeout; partial stderr log retained')

    def send(self, value):
        self.proc.stdin.write(value + '\n')
        self.proc.stdin.flush()

    def stats(self):
        self.send('STATS')
        value = self.line()
        assert value.startswith('STATS '), value
        return json.loads(value[6:])

    def reset(self):
        self.send('RESET')
        assert self.line().startswith('DONE RESET ')

    def turn(self, full, committed, maximum):
        keep = 0
        for a, b in zip(full, committed):
            if a != b:
                break
            keep += 1
        if keep == len(full):
            keep -= 1
        delta = full[keep:]
        before = self.stats()
        cpu_before = child_cpu_seconds(self.proc)
        start = time.monotonic()
        self.send(f'TURN {maximum} {keep} 0 1 1 {len(delta)} ' + ' '.join(map(str, delta)))
        tokens = []
        while True:
            frame = self.line()
            if frame.startswith('TOKEN '):
                tokens.append(int(frame.split()[1]))
            elif frame.startswith('DONE '):
                _, reason, count, position = frame.split()
                assert int(count) == len(tokens)
                break
            else:
                raise RuntimeError(frame)
        elapsed = time.monotonic() - start
        cpu_after = child_cpu_seconds(self.proc)
        after = self.stats()
        timers = ('prefill_seconds', 'decode_seconds', 'attention_seconds', 'moe_seconds',
                  'disk_seconds', 'head_seconds', 'cache_hits', 'cache_misses', 'expert_loads',
                  'moe_grouped_calls')
        return dict(tokens=tokens, reason=reason, position=int(position), prompt_tokens=len(full),
            reused_tokens=keep, new_prompt_tokens=len(delta), elapsed_seconds=elapsed,
            process_cpu_seconds=None if cpu_before is None or cpu_after is None else cpu_after - cpu_before,
            phase={key: after[key] - before[key] for key in timers},
            rss_gb=after['rss_gb'], peak_rss_gib=after.get('peak_rss_gib') or None,
            scratch_allocations=after['moe_scratch_allocs'],
            scratch_bytes=after['moe_scratch_bytes'], engine_before=before, engine_after=after)

    def close(self):
        if self.proc.poll() is None:
            try:
                self.send('SHUTDOWN')
                self.proc.wait(timeout=15)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait()
        self.log.close()


def render(tokenizer, messages):
    value = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
        tokenize=True, enable_thinking=False)
    if hasattr(value, 'keys'):
        value = value['input_ids']
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--threads', nargs='+', type=int, default=[4, 6])
    parser.add_argument('--cache-gib', nargs='+', type=float, default=[4, 2])
    parser.add_argument('--short-only', action='store_true', help='omit long-prompt and conversation cases')
    args = parser.parse_args()
    if any(n < 1 for n in args.threads) or any(n <= 0 for n in args.cache_gib):
        parser.error('threads and cache budgets must be positive')
    exe, model, output = Path(args.exe).resolve(), Path(args.model).resolve(), Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    short = 'Spiega in italiano come un modello Mixture of Experts sceglie gli esperti. Usa quattro frasi.'
    facts = '\n'.join(f'Stazione {i}: il progetto ORIONE-{i} raccoglie misure di temperatura e pressione ogni venti minuti.'
                      for i in range(12))
    long_prompt = facts + '\nRiassumi in italiano lo scopo delle stazioni e le due grandezze misurate.'
    conversation = ['Memorizza: progetto Picchio, codice ORIONE-42, colore verde. Rispondi soltanto OK.',
                    'Qual e il codice del progetto? Rispondi soltanto con il codice.',
                    'Quale colore era associato al progetto? Rispondi con una sola parola.']
    report = dict(executable=str(exe), executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                  model=str(model), platform=platform.platform(), python=platform.python_version(),
                  packages={name: importlib.metadata.version(name) for name in ('transformers', 'tokenizers')},
                  settings='IDOT/GPU/prefetch/async off, ctx 1024',
                  status='running', results=[])
    def save():
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    short_reference = None
    try:
        configurations = [(threads, cache) for cache in args.cache_gib for threads in args.threads]
        for config, (threads, cache) in enumerate(configurations):
            modes = (0, 1) if config % 2 == 0 else (1, 0)
            paired = {}
            for grouped in modes:
                tag = f't{threads}-c{cache}-g{grouped}'
                engine = Engine(exe, model, threads, cache, grouped,
                    output.with_name(output.stem + '-' + tag + '.log'), args.timeout)
                row = dict(threads=threads, cache_gib=cache, grouped=grouped,
                           startup_seconds=engine.startup_seconds, turns=[])
                report['results'].append(row)
                save()
                try:
                    workloads = [('short', [short], 32)]
                    if config == 0 and not args.short_only:
                        workloads += [('long', [long_prompt], 24), ('conversation', conversation, 16)]
                    for label, prompts, maximum in workloads:
                        engine.reset()
                        messages, committed = [], []
                        for number, prompt in enumerate(prompts):
                            messages.append(dict(role='user', content=prompt))
                            full = render(tokenizer, messages)
                            assert len(full) + maximum <= 1024
                            print(f'RUN {tag} {label} turn {number + 1}, {len(full)} prompt tokens', flush=True)
                            turn = engine.turn(full, committed, maximum)
                            turn.update(workload=label, turn=number + 1,
                                        text=tokenizer.decode(turn['tokens'], skip_special_tokens=True).strip())
                            messages.append(dict(role='assistant', content=turn['text']))
                            committed = full + turn['tokens']
                            if label == 'conversation' and number > 0:
                                assert turn['reused_tokens'] > 0, 'KV reuse was not exercised'
                            if grouped:
                                assert turn['phase']['moe_grouped_calls'] > 0
                                assert turn['scratch_allocations'] == 1
                            row['turns'].append(turn)
                            save()
                            print(json.dumps(dict(config=tag, workload=label, turn=number + 1,
                                prompt_tokens=len(full), generated=len(turn['tokens']),
                                elapsed=round(turn['elapsed_seconds'], 3), reused=turn['reused_tokens'])), flush=True)
                    paired[grouped] = row
                    save()
                finally:
                    engine.close()
            for before, after in zip(paired[0]['turns'], paired[1]['turns']):
                for key in ('tokens', 'reason', 'position', 'prompt_tokens', 'reused_tokens'):
                    assert before[key] == after[key], (threads, cache, before['workload'], key)
            ids = paired[0]['turns'][0]['tokens']
            if short_reference is None:
                short_reference = ids
            assert ids == short_reference, 'thread/cache setting changed the exact response'
            print(f'PASS identical token IDs: threads={threads}, cache={cache} GiB', flush=True)
        report.update(status='passed', identical_token_ids=True)
        save()
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
