# Porting Picchio to Qwen3-MoE

Status: **implemented, pending validation on the real model.** The runtime and
converter changes compile and the GPT-OSS self-test still passes (no regression).
Token-exactness against `transformers` has not yet been run (needs the Qwen3
weights + a converted container); see [Validation](#validation).

Picchio was written for GPT-OSS. Qwen3-MoE (e.g. `Qwen/Qwen3-30B-A3B`) is the
closest non-GPT-OSS family: GQA, RMSNorm, RoPE, SwiGLU experts, softmax router.
The port makes the GPT-OSS-specific behaviours **config-gated** so the GPT-OSS
path stays byte-identical, and flips them for Qwen3.

## Architecture deltas handled

| Aspect | GPT-OSS | Qwen3-MoE | Where |
|---|---|---|---|
| QK-Norm | none | RMSNorm on Q/K per head, pre-RoPE | `gqa_attention`, `rmsnorm_heads` |
| Activation | clipped SwiGLU `(up+1)·gate·σ(1.702·gate)` | plain SiLU `gate·σ(gate)·up` | `moe_forward`, `expert_apply` |
| Router norm | softmax over top-k logits | softmax over all experts, renorm top-k | `router_weights` |
| Attention sinks | mandatory | none | loader gated on `use_sinks` |
| Sliding window | alternated with full | all full | `cfg_load` default |
| Attention bias | q/k/v/o bias | none | already `attention_bias` in config |
| Expert layout | fused MXFP4 `gate_up_proj` + `down_proj` | separate BF16 gate/up/down | `convert.py` |
| Expert intermediate | `intermediate_size` | `moe_intermediate_size` | `cfg_load` |
| Router tensor | `mlp.router.weight` | `mlp.gate.weight` | loader fallback |
| Stop token | Harmony `<|return|>`/`<|call|>` | ChatML `<|im_end|>` (eos) | `cfg_load` reads `eos_token_id` |
| Chat template | Harmony (`openai-harmony`) | ChatML (`transformers`) | `chat_qwen.py` |

Detection is by `model_type` in `config.json` (`"qwen*"` → Qwen mode). All flags
live in `Cfg` (`qk_norm`, `swiglu_clipped`, `router_norm`, `use_sinks`).

## What changed

**`picchio.c`** (runtime, all config-gated):
- `Cfg`: `qk_norm`, `swiglu_clipped`, `router_norm`, `use_sinks`. `Layer`:
  `q_norm`, `k_norm`.
- `cfg_load`: architecture detection; `moe_intermediate_size`; full-attention
  default and ChatML stop id for Qwen.
- `rmsnorm_heads` helper; QK-Norm applied in `gqa_attention` before RoPE.
- `router_weights` helper; used in both `moe_forward` (decode) and
  `forward_prefill` (batched prefill).
- SwiGLU branch (clipped vs plain SiLU) in `moe_forward` and `expert_apply`.
- Loader: loads `q_norm`/`k_norm`; sinks optional; `mlp.gate.weight` router name.
- `self_test` sets the GPT-OSS flags explicitly (keeps the synthetic model a
  faithful GPT-OSS check).

**`convert.py`**:
- Detects Qwen from `model_type`; keeps Qwen expert gate/up/down raw F32 in
  `convert_shard`.
- `interleave_qwen_experts`: fuses gate+up into the interleaved `gate_up_proj`
  layout the runtime expects (`gu[2i]=gate_i`, `gu[2i+1]=up_i`), quantizes
  gate_up and down to group-scaled INT4, buffers half-seen experts across shards.
- Aborts if any expert is left incomplete.

**`chat_qwen.py`** (new): reuses the exact SERVICE pipe transport; swaps Harmony
for `AutoTokenizer.apply_chat_template` (ChatML, `enable_thinking` ↔
`--no-reasoning`). Single-shot and interactive, with KV-prefix reuse.

## Build, convert, run

```powershell
.\build.bat                              # unchanged; GPT-OSS still works
pip install transformers torch safetensors numpy huggingface_hub

python convert.py --model Qwen/Qwen3-30B-A3B --output C:\models\qwen3_30b_i4 --download
python chat_qwen.py --model C:\models\qwen3_30b_i4 --no-reasoning `
    --ctx 2048 --pin-gb 8 --max-tokens 256 --temperature 0.7
```

## Validation

Verified so far:
- Runtime compiles clean (only the pre-existing `tok.h` strncpy warning).
- `picchio --self-test` still PASSES (GPT-OSS path unchanged).
- `interleave_qwen_experts` round-trip: dequantized even rows ≈ gate, odd ≈ up,
  down ≈ down, within INT4 gs64 noise; tensor names match the runtime reader.

Still to do (needs the real weights + `transformers`):
1. Convert a Qwen3-30B-A3B checkpoint and confirm the container loads
   (`picchio` prints `config: ... E=128 top8`, dense weights, no missing tensor).
2. L1 oracle: extend `make_test_model.py`/`test_forward.py` with a Qwen3 fixture
   (`tiny-random/qwen3-moe` style) and check per-checkpoint QK-Norm, router
   normalization, and plain-SiLU against `transformers`.
3. L2: greedy token match vs `transformers` on the real checkpoint (allowing for
   INT4 lossiness, per DESIGN §0.2).
4. Confirm `chat_qwen.py` prefix reuse (`reused k/N` should climb across turns).

## Known caveats
- Tied embeddings: if a Qwen variant sets `tie_word_embeddings=true`, `lm_head`
  is absent and the loader will fail — needs a tie fallback (reuse `embed_tokens`).
- Transient RAM in conversion: a shard's Qwen experts are held as F32 before
  quantization (~2× the BF16 shard). Fine for 30B on a roomy machine; watch it on
  16 GB.
- `chat_qwen.py` is written but not yet exercised end-to-end (no converted model
  available at implementation time).
