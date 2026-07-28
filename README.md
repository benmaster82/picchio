# Picchio 🪶

Motore MoE streaming per **GPT-OSS-120B** (117B parametri, 5.1B attivi/token) su hardware consumer.

Ispirato a [Colibri](https://github.com/JustVugg/colibri) (GLM-5.2), adattato per GPT-OSS.

## Quick Start

### 1. Test senza compilatore (Python oracle)

```bash
pip install safetensors numpy
python make_test_model.py        # genera mini-modello (~900 KB)
python test_forward.py test_model  # valida il forward pass
```

### 2. Compila il motore C

Su Linux/macOS:
```bash
make
./picchio --self-test
```

Su Windows (MinGW):
```bash
gcc -O2 -Wall -o picchio.exe picchio.c -lm
picchio.exe --self-test
```

### 3. Testa con il mini-modello

```bash
./picchio test_model
```

### 4. Modello reale (57 GB)

```bash
python convert.py --model openai/gpt-oss-120b --output /nvme/gptoss_i4
MODEL=/nvme/gptoss_i4 ./picchio
```

## Architettura GPT-OSS-120B

| Proprietà | Valore |
|---|---|
| Parametri totali | 116.8B |
| Parametri attivi/token | 5.1B |
| Hidden size | 2880 |
| Layer | 36 (tutti MoE) |
| Attention | GQA: 64 Q heads, 8 KV heads, head_dim=64 |
| Expert/layer | 128 |
| Expert attivi/token | 4 (top-4) |
| Expert size (INT4) | ~12.4 MB |
| Modello totale (INT4) | ~57 GB |
| Context | 131K (sliding window 128 + full, alternati) |

## Requisiti hardware (16 GB RAM)

- Parte densa residente: ~1.5 GB
- KV-cache (4K token): ~75 MB
- Cache expert LRU: ~10 GB (~800 expert)
- Performance stimata: 2-4 tok/s (warm)

## File

```
picchio.c        — motore principale
quant.h          — kernel matmul (F32, INT8, INT4) + SIMD
json.h           — parser config.json
st.h             — reader safetensors
convert.py       — conversione MXFP4 → INT4
make_test_model.py — genera mini-modello per test
test_forward.py  — oracle Python per validazione
```

## Stato

- [x] Forward pass completo (attention + MoE + routing)
- [x] GQA con sliding window + full attention alternati
- [x] Attention bias
- [x] Cache LRU per expert con eviction
- [x] Reader safetensors + config loader
- [x] Self-test con mini-modello sintetico
- [x] Oracle Python per validazione
- [ ] Tokenizer o200k_harmony
- [ ] PILOT prefetch (thread separato)
- [ ] Sampling (temperature, top-p)
- [ ] Hot-store appreso persistente
- [ ] Server API OpenAI-compatible

## Licenza

Apache 2.0
