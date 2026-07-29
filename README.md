# Picchio 🪶

Motore MoE streaming per **GPT-OSS** (120B e 20B) su hardware consumer.

Il 20B è la configurazione consigliata con 16 GB di RAM: modello convertito ~14 GB,
cache hit ~57% e circa 4× più veloce del 120B sullo stesso hardware, senza modifiche
al codice (layer, expert e attention sono letti da `config.json`).

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

### 4. Modello reale

GPT-OSS-20B, consigliato con 16 GB di RAM (~14 GB convertiti):

```powershell
python convert.py --model openai/gpt-oss-20b --output D:\gptoss20b_i4 --download
python export_vocab.py D:\gptoss20b_i4\tokenizer.json D:\gptoss20b_i4\picchio_vocab.bin
python chat.py "Ciao" --model D:\gptoss20b_i4 --pin-gb 4 --ctx 512
```

GPT-OSS-120B (~66,65 GB convertiti, richiede il sidecar dei bias):

```bash
python convert.py --model openai/gpt-oss-120b --output /nvme/gptoss_i4
MODEL=/nvme/gptoss_i4 ./picchio
```

Nota: il download da Hugging Face include anche la cartella `original/`, che è lo stesso
modello in formato alternativo e non serve. Conviene interromperla o rimuoverla per non
occupare spazio inutilmente.

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
| Modello convertito corrente (INT4 gs64 + parte densa F32) | ~66,65 GB |
| Context | 131K (sliding window 128 + full, alternati) |

## Requisiti hardware (16 GB RAM)

- Parte densa residente misurata: ~4,46 GB
- KV-cache iniziale configurabile con `CTX` (default 512 posizioni)
- Cache expert configurabile con `PIN_GB` (default 6 GB; usare 1 GB con poco margine)
- La build **deve** usare `-fopenmp` e `-mavx2 -mfma`: senza questi flag i kernel matmul
  restano su un solo core e in versione scalare
- Prestazione misurata sul 20B (6 core, `PIN_GB=4`, modello su NVMe): ~0,6 s per token
- Tenere il modello su un disco interno: da un box USB 2.0 il tempo di I/O raddoppia
- `PILOT=1` è sconsigliato con poca RAM libera: misurato ~11% più lento
- Il modello convertito occupa circa 66,65 GB

Con uno shard su C: e gli altri su D:, usare PowerShell:

```powershell
$env:MODEL_AUX='C:\picchio\expert_biases.safetensors;C:\picchio\model-00012.safetensors'
$env:OMP_NUM_THREADS='6'
$env:PIN_GB='1'
.\picchio.exe D:\gptoss_i4
```

`expert_biases.safetensors` contiene i bias F32 originali. Si rigenera senza scaricare
il modello completo con `python download_expert_biases.py`.

## Chat token-exact

Il bridge Python usa Harmony ufficiale e passa al runtime soltanto ID raw; non usa
l'encode approssimato di `tok.h`:

```powershell
python -m pip install -r requirements-chat.txt
python chat.py "Scrivi un saluto breve in italiano" --max-tokens 64
```

Chat multi-turn nello stesso processo, con modello e KV-cache mantenuti in memoria:

```powershell
python chat.py --model D:\gptoss20b_i4 --pin-gb 4 --ctx 1024 --max-tokens 200 --temperature 0.7
```

Usare `--temperature 0.7` per l'uso conversazionale: in modalità greedy (`0`, il default,
utile per i confronti riproducibili) il modello può ciclare nel canale `analysis` senza
emettere la risposta finale. Altre opzioni: `--top-p`, `--top-k`, `--seed`,
`--show-analysis`, `--json`.

Importante: la conversione mantiene `embed_tokens` e `lm_head` a **INT8**, non INT4, come
richiede la lista di esclusione ufficiale del modello. A INT4 la testa di uscita ha un
errore dell'11% e cambia il token più probabile in un caso su quattro, con generazioni
lunghe che degenerano. I container prodotti prima di questa correzione vanno riconvertiti.

Tra turni viene riusato il prefisso comune della KV-cache (tipicamente circa l'88%),
perché il re-render Harmony scarta l'analysis e trasforma `<|return|>` in `<|end|>`.

Per controllare prompt e ID senza caricare il modello da 66,65 GB:

```powershell
python chat.py "Ciao" --dry-run
```

Il bridge rileva automaticamente lo shard 12 e il sidecar bias nelle posizioni correnti.
`--model-aux` sovrascrive esplicitamente `MODEL_AUX`. `tok.h` resta un fallback
interattivo, ma non è dichiarato token-exact.

## File

```
picchio.c        — motore principale (include la modalità SERVICE persistente)
chat.py           — chat Harmony ufficiale ↔ ID raw, multi-turn
test_service.py   — equivalenza riuso prefisso vs prefill completo
check_harmony_delta.py — verifica prefix-preserving del render Harmony
requirements-chat.txt — dipendenza Harmony fissata
quant.h          — kernel matmul (F32, INT8, INT4) + SIMD
json.h           — parser config.json
st.h             — reader safetensors
convert.py       — conversione MXFP4 → INT4
make_test_model.py — genera mini-modello per test
test_forward.py  — oracle Python per validazione
```

## Stato

- [x] Forward pass completo (attention + MoE + routing)
- [x] GQA con sliding/full attention, attention sinks e YaRN
- [x] Clipped SwiGLU GPT-OSS e gate/up interleaved
- [x] Bias attention ed expert aggregati
- [x] INT4 group-scaled, embedding e lm-head inclusi
- [x] Cache LRU expert, hot-store e sampling
- [x] Reader safetensors multi-shard e `MODEL_AUX` multi-disco
- [x] Oracle Transformers sul checkpoint `tiny-random/gpt-oss`
- [x] Equivalenza F32 e INT4 layer-by-layer, 32 token greedy e posizione 130
- [x] Forward e generazione autoregressiva del GPT-OSS-120B senza NaN/Inf
- [x] Bridge chat con rendering/parsing Harmony ufficiale e trasporto ID raw
- [x] Sessione persistente multi-turn con riuso del prefisso KV
- [ ] Tokenizer o200k_harmony nativo C token-exact (`tok.h` resta approssimato)
- [ ] Prestazioni: prefill e I/O expert sono il collo di bottiglia (~0,09 token/s)
- [x] Kernel matmul realmente paralleli (OpenMP abilitato): ~1,7× sul turno completo
- [ ] PILOT prefetch utile: va riscritto per popolare direttamente la cache LRU
- [ ] Server API OpenAI-compatible

## Licenza

Apache 2.0
