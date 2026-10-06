# Vendored third-party reference code

This directory contains **unmodified** third-party source, kept here only so the
MiniMax-M2 validation fixture can be built from the real upstream architecture
rather than from a reimplementation that might share a bug with it.

It is **not** part of the Picchio engine and is never compiled or imported by
`picchio.c`. Only `make_minimax_test_model.py` and `make_minimax_test_model_deep.py`
import it.

## `minimax_m2/`

| | |
|---|---|
| Origin | [`ModelCloud/MiniMax-M2-GPTQMODEL-W4A16`](https://huggingface.co/ModelCloud/MiniMax-M2-GPTQMODEL-W4A16) (the `modeling_minimax_m2.py` / `configuration_minimax_m2.py` shipped with the checkpoint) |
| Copyright | 2024-2025 ModelCloud.ai, qubitium@modelcloud.ai |
| License | Apache-2.0 (`SPDX-License-Identifier: Apache-2.0`, retained in each file header) |
| Changes | **None.** All three files (`__init__.py`, `configuration_minimax_m2.py`, `modeling_minimax_m2.py`) are byte-identical to the upstream copies, verified by `diff`. The only addition is the empty `reference/__init__.py` one level up, which exists solely to make this an importable package path and contains no licensed content. |

Picchio itself is MIT-licensed (see [`../LICENSE`](../LICENSE)). Apache-2.0 permits
this redistribution; the original copyright and license identifiers are preserved
in the file headers above, and no modifications were made to the licensed files.

For the full Apache License 2.0 text, see <https://www.apache.org/licenses/LICENSE-2.0>.

## Compatibility note

This is a pinned snapshot that targets **transformers < 5.0**. On transformers 5.x
the RoPE it builds via `LlamaRotaryEmbedding` ignores `partial_rotary_factor` and
rotates the full head, which is wrong for MiniMax-M2 and would yield a silently
incorrect oracle — so `make_minimax_test_model.py` refuses to run there. See
[huggingface/transformers#48241](https://github.com/huggingface/transformers/issues/48241).
