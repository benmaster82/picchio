# Picchio — Motore MoE Streaming per GPT-OSS-120B

> *Il picchio tamburella cento volte al secondo su un tronco enorme —
> noi tamburelliamo 128 expert su un disco enorme.*

**Obiettivo:** eseguire GPT-OSS-120B (117B parametri, MoE) su hardware consumer
(16 GB RAM, SSD NVMe, GPU opzionale) in puro C, con expert streamati da disco.

Licenza: Apache 2.0.

## 0. Stato e contratto normativo (revisione luglio 2026)

Le sezioni storiche successive descrivono l'idea iniziale e possono contenere stime
superate. In caso di conflitto, questa sezione ha precedenza. L'architettura reale
usata da Picchio è: hidden size 2880, 36 layer tutti MoE, 64 query head, 8 KV head,
head dimension 64, 128 expert per layer, top-4, intermediate size 2880, gate/up fuso
5760, attention sliding-window 128 alternata con full attention, `rope_theta=150000`
e vocabolario circa 201K. GPT-OSS usa inoltre attention sinks, YaRN e una variante
clipped SwiGLU: nessuno di questi dettagli può essere sostituito da un'approssimazione
senza una prova oracle.

Lo stato corrente è **runtime end-to-end e forward numericamente validati** contro
Transformers sul tiny model, sia in F32 sia in INT4 group-scaled. Il GPT-OSS-120B reale
ha inoltre completato più passi autoregressivi con KV-cache e logits finiti. Questa
validazione riguarda runtime e forward su ID token raw; tokenizer o200k e Harmony non
sono ancora token-exact e restano una suite separata.

### 0.1 Fixture ufficiale

- Modello: `tiny-random/gpt-oss`
- Revisione bloccata: `02ba5c61f879b5a38a8b1f7a8e0409b8e1bb8f38`
- Scopo: debug architetturale; pesi casuali, non valutazione della qualità linguistica.
- Input primario: ID token raw fissi. Tokenizer e Harmony sono una suite separata.
- Riferimento: `transformers==4.57.1`, `torch==2.6.0`, `safetensors==0.6.2`.
- Esecuzione: batch 1, `eval()`, nessun gradiente, nessun sampling, PILOT disattivato,
  repetition penalty 1.0 e argmax deterministico.

### 0.2 Tre livelli di validazione

**L0 — Container-exact.** Il convertitore deve produrre un manifest con nome, dtype,
shape, byte count, schema di quantizzazione, group size e hash. Ogni peso obbligatorio,
inclusi bias expert e attention sinks, deve esistere. Packed INT4, scale e righe
dequantizzate campione devono coincidere tra Python e C entro la tolleranza dichiarata.
Short read, shape inattesa o fallback silenzioso sono errori fatali.

**L1 — Implementation-exact.** Picchio C viene confrontato con un oracle Python che
legge gli stessi file convertiti Picchio e riproduce la stessa dequantizzazione. Gli
indici top-k e l'argmax devono essere identici; per tensori F32 si registrano max-abs,
max-rel e cosine error. Questo livello separa i bug del runtime dagli effetti della
quantizzazione lossy.

**L2 — Transformers token-exact.** Con gli stessi ID di input, Picchio viene confrontato
con Transformers sul checkpoint originale. Gli ID greedy devono coincidere. Poiché
MXFP4/BF16 → INT4 gs64 è lossy, un mismatch L2 dopo il superamento di L0 e L1 deve
riportare anche top-1/top-2 margin e differenza logits: non dimostra da solo un bug C.

### 0.3 Checkpoint e ordine obbligatorio

Per un token a posizione 0 e layer 0 confrontare, nell'ordine:

1. embedding e RMSNorm pre-attention;
2. Q/K/V prima e dopo bias;
3. Q/K dopo RoPE/YaRN;
4. KV scritto, intervallo della mask e attention scores;
5. sink logit, massa softmax del sink, concat head e output projection;
6. residual e RMSNorm pre-MoE;
7. router logits, top-k ordinato e relativi pesi;
8. gate/up con bias, split, clipped SwiGLU, down con bias e contributi pesati;
9. residual di fine layer;
10. final norm, logits, top-10, margine top-1/top-2 e argmax.

Dopo il singolo token: sequenza corta sul layer 0; posizioni 127/128/129 per la sliding
window; un token attraverso tutti i layer; prefill raw fisso; infine 8–32 token greedy.
Si corregge esclusivamente il primo checkpoint divergente, poi si ripete dall'inizio.

I dump sono F32 little-endian o `.npy`, accompagnati da JSON con versione, revisione,
input IDs, posizione, layer, shape, dtype, hash e statistiche. Sul tiny si possono
salvare tensori completi; sul 120B si usano hash e probe. OpenMP/SIMD, cache eviction,
hot-store, PILOT, tokenizer e Harmony vengono riabilitati e verificati solo dopo L1.

### 0.4 Criteri di accettazione della milestone “token corretto”

- Fixture riproducibile dalla revisione bloccata e senza dipendere dal modello 120B.
- Nessun peso obbligatorio mancante o sostituito con valori di default.
- Test isolati superati per INT4 gs64, embedding, RMSNorm, RoPE/YaRN, softmax con sink,
  routing e clipped SwiGLU.
- Tutti i checkpoint L1 entro le tolleranze registrate e top-k/argmax identici.
- Sequenza greedy L1 identica per almeno 32 token raw.
- Risultato L2 documentato separatamente; solo dopo si applicano le correzioni validate
  alla conversione e al runtime del GPT-OSS-120B.

### 0.5 Risultati della validazione

Validazione completata il 29 luglio 2026:

- checkpoint F32 layer-by-layer entro `atol=rtol=1e-5`;
- top-k, pesi router e argmax identici;
- 32 token greedy identici tra Picchio e Transformers;
- KV-cache e confine sliding verificati alla posizione 130;
- container INT4 group-scaled confrontato con un riferimento dequantizzato identico;
- tutti i 15 shard reali validati con il parser safetensors ufficiale;
- shard 13, trovato troncato, ricostruito expert-per-expert con verifica bit-per-bit
  delle scale già presenti;
- GPT-OSS-120B eseguito per più passi autoregressivi senza NaN/Inf.

La vecchia conversione aveva quantizzato per errore i bias expert. Il runtime supporta
quel formato per retrocompatibilità, ma la modalità raccomandata usa il sidecar F32
prodotto da `download_expert_biases.py`. Le conversioni future preservano direttamente
i bias expert in F32.

### 0.6 Milestone chat single-turn token-exact

Il primo percorso chat non duplica o200k/Harmony in C. Un bridge Python, basato sulla
libreria ufficiale `openai-harmony` a versione fissata, possiede rendering del prompt,
tokenizzazione, parsing dei canali e decode. Picchio riceve e produce esclusivamente
ID token raw; `tok.h` resta un fallback interattivo approssimato e non fa parte del
contratto token-exact.

Il protocollo runtime deve offrire:

- input da file di ID decimali, per non dipendere dal limite delle variabili d'ambiente
  Windows; `RAW=1` e `INPUT` restano compatibili;
- output machine-readable contenente ogni ID generato, inclusi token Harmony e stop,
  senza filtro o decode C;
- stdout riservato agli ID in tale modalità e diagnostica su stderr;
- stop esplicito su `<|return|>` e `<|call|>`, riportando anche il terminatore;
  `<|end|>` chiude un messaggio ma non l'intera risposta assistant;
- campionamento greedy riproducibile per la prima validazione.

Criteri di accettazione: gli ID renderizzati dal bridge coincidono con Harmony
ufficiale; il trasporto file→C è esatto; la sequenza di output completa è parsabile da
Harmony; la modalità raw esistente e le suite tiny F32/INT4 non regrediscono.

Risultato: milestone raggiunta il 29 luglio 2026. Il GPT-OSS-120B reale ha risposto
"Ciao!" con prompt Harmony di 77 token, prefill 930,34 s, 23 token in 251,34 s, circa
0,09 token/s e 20,3% di cache hit expert.

### 0.7 Milestone chat multi-turn persistente

Obiettivo: non ricaricare il modello e non ricalcolare il prefisso già elaborato.

Vincolo misurato: il rendering Harmony **non** è prefix-preserving tra turni. Nel
re-render canonico il canale `analysis` viene scartato e il `<|return|>` finale diventa
`<|end|>`. Conservando l'analysis la divergenza si riduce al solo terminatore, con circa
l'88% delle posizioni riusabili. Il protocollo non può quindi limitarsi ad accodare un
delta: il bridge calcola il prefisso comune più lungo e il runtime riparte da quella
posizione, sovrascrivendo la KV successiva.

Invariante obbligatorio: ogni token trasmesso deve essere anche consumato dal forward,
inclusi `<|return|>` e `<|call|>`, così che `pos` coincida sempre con il numero di
posizioni valide in KV. Un token emesso ma non consumato renderebbe il turno successivo
numericamente errato.

Protocollo di servizio, righe di testo su pipe, stdout riservato al protocollo e stderr
alla diagnostica:

- `READY <ctx_capacity> <vocab> <stop_ids...>` all'avvio;
- `TURN <max_new> <keep> <n_ids> <ids...>`: riusa `keep` posizioni e consuma i nuovi ID;
- `TOKEN <id>` per ogni token generato, terminatore incluso;
- `DONE <RETURN|CALL|MAX_TOKENS|CONTEXT_FULL> <n_output> <pos>`;
- `ERROR <code> <fatal> <messaggio>` con validazione prima di mutare la KV;
- `RESET` riporta `pos` a 0 conservando modello e cache expert; `SHUTDOWN` esce pulito.

La capacità KV reale è `CTX`: nessun prompt può superarla, perché oltre quel limite le
scritture verrebbero ignorate silenziosamente producendo risultati errati.

Risultati del 29 luglio 2026, verificati sul tiny model:

- il riuso del prefisso produce token identici al prefill completo, con `pos` coerente;
- `keep` fuori range viene rifiutato senza corrompere la sessione;
- due turni consecutivi hanno riusato 68 posizioni su 77;
- le suite tiny F32 e INT4 non sono regredite.

Sul GPT-OSS-120B reale la sessione persistente ha risposto correttamente, con canale
`analysis` e `final` separati dal parser ufficiale.

### 0.8 Prestazioni: misure e vincoli hardware

Profilo di un turno reale da 77 token di prompt e 24 generati:

- `t_moe` 983,70 s, di cui **565,73 s di attesa disco** su 11.569 letture;
- `t_attn` 134,20 s, `t_head` 71,29 s;
- cache hit expert 20,5% su 14.544 richieste;
- circa 0,09 token/s.

Vincolo di memoria misurato: 15,83 GB totali con circa 6,86 GB liberi. La parte densa
occupa 4,46 GB, quindi la cache expert non può superare circa 2,4 GB. `PIN_GB` 1 e 2
producono entrambi il minimo di 4 slot per layer; `PIN_GB=3` porterebbe il totale a
7,6 GB, oltre la memoria libera, causando paging. L'aumento della cache richiede
quindi più RAM, non solo un parametro diverso.

Vincolo di I/O: il modello risiede su un SSD esterno **USB** con bridge JMicron, non
sull'NVMe interno, coerente con i circa 253 MB/s osservati.

Prefetch su Windows: il thread PILOT non modifica la cache LRU e non condivide gli
handle del thread principale. Apre handle propri, copia l'input di routing, calcola il
top-k del layer successivo e legge gli intervalli degli expert previsti per portarli
nella cache del sistema operativo. Le strutture condivise vengono solo lette, quindi
non esiste corsa critica che possa alterare il risultato numerico; il controllo di
presenza in cache è volutamente senza lock e un esito impreciso costa al massimo una
lettura inutile. La correttezza con prefetch attivo è verificata dalle suite tiny.

**OpenMP.** Il binario veniva compilato senza `-fopenmp`, quindi i `#pragma omp` dei
kernel matmul in `quant.h` erano ignorati e tutto il calcolo restava su un core, su una
CPU con 6 core fisici. L'opzione `-Wno-unknown-pragmas` nascondeva l'avviso. Entrambe le
cose sono state corrette: `-fopenmp` è obbligatorio e l'avviso deve restare visibile.

Misure comparate sullo stesso turno, 77 token di prompt e 24 generati:

| Configurazione | `t_attn` | `t_moe` | `t_head` | disco | totale fasi |
|---|---|---|---|---|---|
| Base, un core | 134,20 s | 983,70 s | 71,29 s | 565,73 s | ~1.189 s |
| OpenMP | 30,98 s | 649,91 s | 19,93 s | 534,69 s | ~701 s |
| OpenMP + PILOT | 30,64 s | 730,05 s | 19,30 s | 614,21 s | ~780 s |

Conclusioni: OpenMP vale circa 1,7× e il MoE al netto del disco scende da circa 418 s a
circa 115 s. Il PILOT così concepito **peggiora di circa l'11%**, perché scalda la cache
del sistema operativo che qui non ha spazio: le pagine vengono espulse e il disco viene
letto due volte. Per questo il prefetch resta opzionale e disattivato per default.

Riprogettazione necessaria del prefetch: inserire l'expert direttamente nella cache LRU
con sincronizzazione, eliminando la seconda lettura, e misurare quanti expert previsti
vengono realmente usati, dato che il routing del layer successivo è stimato dallo stato
nascosto del layer corrente ed è quindi approssimato.

**Prefill batched (batch-union).** Con cache da 4 slot per layer e top-4, il percorso
token-per-token sfratta l'intera cache a ogni posizione, quindi il prompt rilegge gli
stessi expert molte volte. Il prefill elabora ora le posizioni a blocchi: per ogni layer
si esegue l'attention in ordine di posizione, poi si calcola il routing di tutte le
posizioni e si legge **una sola volta** ogni expert dell'unione, riusandolo per tutte le
posizioni che lo hanno selezionato. La matematica non cambia: cambia solo l'ordine delle
letture. Il percorso sequenziale resta attivo con dump oracle, tracing o repetition
penalty, così le suite di validazione non sono influenzate. `test_prefill_batch.py`
verifica che le due strade producano token identici.

### 0.9 GPT-OSS-20B

Stesso codice, nessuna modifica: dimensioni, layer, expert e pattern di attention sono
letti da `config.json`. Il 20B ha 24 layer e 32 expert per layer, top-4, hidden 2880.
La conversione ha prodotto 14,0 GB in 856 s, con 3,20 GB densi, 10,75 GB di expert e i
bias expert già in F32, quindi senza sidecar.

Confronto sullo stesso prompt Harmony e sullo stesso hardware:

| Metrica | 120B | 20B |
|---|---|---|
| Cache hit | 20,5% | 56,7% |
| Attesa disco | 534,69 s | 97,83 s |
| `t_moe` | 649,91 s | 181,88 s |
| `t_attn` | 30,98 s | 19,63 s |
| `t_head` | 19,93 s | 6,94 s |
| Totale fasi | ~701 s | ~208 s |

Circa 1,7 s per token contro circa 7 s, quindi un fattore 4. La ragione strutturale è la
copertura della cache: 4 slot per layer valgono il 3% di 128 expert nel 120B ma una quota
molto maggiore dei 32 expert del 20B.

Effetto della dimensione della cache sul 20B, con parte densa 3,14 GB e 8,37 GB di RAM
libera misurata:

| `PIN_GB` | slot/layer | cache | hit | disco | totale fasi |
|---|---|---|---|---|---|
| 3 | 10 | 3,0 GB | 56,7% | 97,83 s | ~208 s |
| 4 | 14 | 4,2 GB | 67,1% | 76,75 s | ~190 s |

Oltre `PIN_GB=4` il totale residente supererebbe la RAM libera e causerebbe paging.

**SIMD.** Come per OpenMP, i percorsi AVX2 di `quant.h` sono protetti da `#ifdef __AVX2__`
e la build non passava flag di architettura, quindi i kernel usavano solo lo scalare.
Aggiunti `-mavx2 -mfma`; il binario richiede una CPU con AVX2. FMA cambia l'ordine di
arrotondamento, ma le suite tiny F32 e INT4 restano entro tolleranza con errore massimo
`3,73e-08`.

Effetto sul 20B a `PIN_GB=4`, stesso prompt e stesso output di 43 token:

| Metrica | Senza SIMD | Con SIMD |
|---|---|---|
| `t_moe` | 162,70 s | 88,49 s |
| `t_attn` | 20,55 s | 13,93 s |
| `t_head` | 6,91 s | 1,33 s |
| disco | 76,75 s | 71,14 s |
| totale fasi | ~190 s | ~104 s |

Il calcolo al netto del disco scende da circa 113 s a circa 32,6 s, e il MoE puro da
circa 85,9 s a circa 17,4 s. Il collo di bottiglia torna quindi l'I/O, che pesa 71 s su
104, cioè il 68%: i prossimi interventi utili sono più RAM per la cache expert, il
modello su NVMe interno e un prefetch che popoli direttamente la cache LRU.

**Lezione di metodo.** Due dei guadagni maggiori non sono venuti da nuovo codice ma da
flag di compilazione mancanti, `-fopenmp` e `-mavx2 -mfma`, con gli avvisi silenziati.
Prima di ottimizzare, verificare che il codice esistente sia realmente compilato.

**Campionamento e uso conversazionale.** La decodifica greedy, usata per il determinismo
delle validazioni, può entrare in cicli: su una domanda aperta il 20B ha consumato 200
token nel canale `analysis` senza raggiungere `final`. Con `--temperature 0.7` la stessa
domanda ha prodotto una risposta corretta. `chat.py` espone quindi `--temperature`,
`--top-p`, `--top-k` e `--seed`, mantenendo il default greedy per non alterare i test.

Conversazione reale a due turni sul 20B: il secondo turno ha riusato 279 posizioni su
294, cioè il 95%, elaborandone solo 15, e la risposta era corretta. La cache hit sale con
l'uso grazie all'hot-store, dal 67,1% al 79,2% nella stessa sessione.

**Posizione del modello.** La copia del container dall'SSD esterno ha misurato 52,7 MB/s
sequenziali, valore compatibile con un collegamento USB 2.0 e non con le prestazioni del
supporto. Spostando il 20B sull'NVMe interno, a parità di tutto il resto (stessi 120
forward, 43 token e 67,1% di cache hit):

| Metrica | SSD USB | NVMe interno |
|---|---|---|
| disco | 71,14 s | 37,22 s |
| `t_moe` | 88,49 s | 55,35 s |
| totale fasi | ~104 s | ~73,5 s |
| caricamento | 8,4 s | 4,3 s |

Il MoE al netto del disco resta invariato, circa 18 s, quindi il guadagno è interamente
di I/O. Il disco scende dal 68% al 51% del tempo. Riepilogo del percorso sul 20B: da
~208 s (senza SIMD, su USB) a ~73,5 s, quindi 2,8×. Con 768 expert totali, circa 9,5 GB, la residenza
completa è raggiungibile con più RAM, condizione in cui il disco esce dal percorso
critico. La risposta del 20B termina con `RETURN`, quindi il ciclo Harmony completo,
terminatore incluso, è verificato sul modello reale.

---

## 1. Analisi dell'architettura GPT-OSS-120B

### 1.1 Numeri fondamentali

| Proprietà                    | Valore                          |
|------------------------------|---------------------------------|
| Parametri totali             | 116.8B                          |
| Parametri attivi per token   | 5.1B                            |
| Layer                        | 36                              |
| Expert totali per layer MoE  | 128                             |
| Expert attivi per token      | 4                               |
| Hidden dimension (D)         | 6144 (stimato da param count)   |
| Intermediate MoE (I_moe)     | ~12288 (stimato, 2×D)           |
| Intermediate Dense (I_dense) | ~24576 (stimato, 4×D)           |
| Attention heads              | 48 (stimato)                    |
| GQA groups                   | 8                               |
| KV heads                     | 6 (48/8)                        |
| Head dim                     | 128 (stimato, D/heads)          |
| Context length               | 128K                            |
| Vocabolario                  | ~200K (o200k_harmony)           |
| Quantizzazione nativa        | MXFP4 (MoE), BF16 (resto)      |

> **Nota:** le dimensioni esatte (hidden, heads, intermediate) verranno lette
> dal `config.json` del modello a runtime, come fa Colibri. I valori sopra
> sono stime ragionevoli basate sul parameter count e la documentazione.

### 1.2 Struttura dei layer

GPT-OSS-120B usa un'architettura con **alternanza di layer densi e MoE**.
I primi N layer sono densi (FFN classica), i restanti sono MoE.
Dalla documentazione, il pattern di attention alterna tra:
- **Dense attention** (full causal)
- **Locally banded sparse attention** (finestra locale)

Questo è simile a GPT-3 e diverso da GLM-5.2 (che usa MLA + DSA).

### 1.3 Attenzione: GQA (non MLA)

A differenza di GLM-5.2 che usa Multi-head Latent Attention (MLA) con compressione
KV a 576 float/token, GPT-OSS usa **Grouped Query Attention** standard:
- 48 query heads
- 6 KV heads (group size 8)
- Head dim 128
- RoPE standard (non interleaved)

**KV-cache per token:** 2 × 6 × 128 = 1536 float × 36 layer = 55.296 float
A BF16: 55.296 × 2 byte = ~110 KB/token. Per 4K token: ~430 MB.
Per 128K: ~13.5 GB (significativo — serve gestione oculata).

### 1.4 Formato pesi MXFP4

I pesi MoE usano **MXFP4** (Microscaling FP4, standard OCP):
- Ogni valore è un FP4 (4 bit: 1 sign + 2 exp + 1 mantissa), range ±6.0
- 2 valori impacchettati per byte (`tensor.blocks`, uint8)
- Scala condivisa per blocco di 32 elementi (`tensor.scales`, E8M0 o FP8)
- La scala è lungo l'ultima dimensione

Questo è diverso dall'INT4 simmetrico di Colibri (che usa range [-8,7] con offset).
Servono kernel di dequantizzazione MXFP4 dedicati.

**Alternativa:** convertire MXFP4 → INT4 simmetrico a tempo di conversione,
perdendo ~0.1-0.3% di qualità ma riusando i kernel veloci di Colibri.
Questa è la scelta raccomandata per la prima versione.

### 1.5 Dimensione degli expert

Un expert MoE ha 3 matrici (gate_proj, up_proj, down_proj):
- gate: [I_moe, D] = [12288, 6144]
- up:   [I_moe, D] = [12288, 6144]
- down: [D, I_moe] = [6144, 12288]

Parametri per expert: 3 × 12288 × 6144 = ~226M parametri.
A INT4 (0.5 byte/param): ~113 MB per expert.
A MXFP4 (0.5 byte/param + scale): ~113 MB + scale.

**Expert totali su disco:** 128 expert × ~30 layer MoE × 113 MB ≈ **430 GB** a MXFP4.

> Se la dimensione dell'intermediate MoE è più piccola (es. 8192), gli expert
> scendono a ~75 MB ciascuno e il totale a ~290 GB. Il config.json lo dirà.

### 1.6 Parte densa residente

La parte densa include:
- Embedding + lm_head: ~200K × 6144 × 2 ≈ 2.4B param
- Attention per 36 layer: Q/K/V/O projections
- Layer densi (FFN classica nei primi layer)
- Shared experts (se presenti — da verificare nel config)
- LayerNorm / RMSNorm weights

Stima a INT4: **~3-5 GB** residenti in RAM.
A BF16: **~8-10 GB**.

---

## 2. Architettura del motore

### 2.1 Gerarchia di memoria (come Colibri)

```
┌─────────────┐
│   GPU VRAM   │  Tier 0: expert "caldissimi" + densa (opzionale)
├─────────────┤
│   RAM        │  Tier 1: densa residente + cache LRU expert
├─────────────┤
│   NVMe/SSD   │  Tier 2: tutti gli expert (cold storage, streaming)
└─────────────┘
```

**Politica fondamentale (da Colibri):** il placement decide SOLO la velocità,
mai la precisione o la semantica del routing. L'output è identico
indipendentemente da dove risiedono gli expert.

### 2.2 Pipeline per-token

```
Per ogni layer l in [0, N_layers):
  1. RMSNorm(input)
  2. GQA Attention
     a. Q = x @ Wq               (48 heads × 128 dim)
     b. K = x @ Wk               (6 KV heads × 128 dim)
     c. V = x @ Wv               (6 KV heads × 128 dim)
     d. RoPE(Q, K, pos)
     e. Aggiorna KV-cache[l]
     f. scores = Q @ K^T / sqrt(128)
     g. Se layer sparse: applica banded mask
     h. attn = softmax(scores) @ V
     i. out = attn @ Wo
  3. Residual: h = h + out
  4. RMSNorm(h)
  5. Se layer DENSO:
     a. FFN: SiLU(gate(x)) * up(x) → down → out
  6. Se layer MoE:
     a. ROUTE: router(x) → top-4 expert con pesi
     b. UNION: raccogli set unico di expert (per batch)
     c. PLACE: cerca in VRAM → RAM cache → disco
     d. LOAD: carica expert mancanti (pread coalescente)
     e. COMPUTE: SiLU(gate_e(x)) * up_e(x) → down_e → pesato
     f. SHARED: shared expert (se presente) sempre residente
     g. out = somma pesata expert + shared
  7. Residual: h = h + out

Testa finale:
  RMSNorm(h) → lm_head → logits → sampling
```

### 2.3 Prefetch pilotato (PILOT)

Come Colibri, un thread separato esegue il routing del layer L+1
mentre il layer L sta calcolando, e lancia `pread`/`posix_fadvise`
sugli expert predetti. Con 4 expert attivi per token (vs 8 di GLM),
la prevedibilità potrebbe essere diversa — da misurare.

### 2.4 Cache LRU per-layer

Ogni layer MoE ha un pool di `ESlot` (slot expert) riusabili.
La politica è LRU: l'expert meno recentemente usato viene evictato.
In aggiunta, un hot-store "appreso" (.picchio_usage) tiene traccia
della frequenza per expert e pinna automaticamente i più caldi.

### 2.5 Dual-SSD

Stesso concetto di Colibri: se c'è un secondo SSD, gli expert vengono
distribuiti tra i due drive con hash deterministico pesato per bandwidth.

---

## 3. Strutture dati principali (C)

```c
/* ── Configurazione (letta da config.json) ── */
typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, n_experts, topk;
    int moe_inter, dense_inter, head_dim;
    int first_dense;          /* primi N layer densi (senza MoE) */
    int vocab;
    int ctx_len;              /* max context (128K) */
    int stop_ids[8], n_stop;
    float eps, theta;         /* RMSNorm epsilon, RoPE theta */
    float routed_scale;       /* fattore di scala per expert routing */
    int8_t attn_type[128];    /* per layer: 0=dense, 1=banded sparse */
} Cfg;

/* ── Tensore quantizzato [O, I] ── */
/* fmt: 0=F32, 1=INT8, 2=INT4, 3=MXFP4, 4=BF16 */
typedef struct {
    int fmt;
    float *qf;               /* F32 data */
    int8_t *q8;              /* INT8 data */
    uint8_t *q4;             /* INT4/MXFP4 packed data */
    float *s;                /* scale per riga (INT8/INT4) o per blocco (MXFP4) */
    int O, I;                /* dimensioni output × input */
    int block_size;          /* MXFP4: dimensione blocco scale (32) */
} QT;

/* ── Layer ── */
typedef struct {
    float *in_ln, *post_ln;  /* RMSNorm weights */

    /* GQA Attention */
    QT wq, wk, wv, wo;      /* proiezioni Q/K/V/O */

    int sparse;              /* 0=layer denso, 1=layer MoE */

    /* FFN densa (sparse==0) */
    QT gate_proj, up_proj, down_proj;

    /* MoE (sparse==1) */
    float *router;           /* router weights [n_experts, D] */
    float *router_bias;      /* bias di correzione (se presente) */
    QT sh_gate, sh_up, sh_down; /* shared expert (se presente) */
} Layer;

/* ── Slot Expert (riusabile, cache LRU) ── */
typedef struct {
    int eid;                 /* ID expert (-1 = vuoto) */
    QT g, u, d;             /* gate/up/down projections */
    uint8_t *slab;           /* buffer coalescente per pread */
    float *fslab;            /* buffer scale */
    int64_t slab_cap;
    uint64_t last_used;      /* timestamp per LRU */
} ESlot;

/* ── KV-Cache ── */
typedef struct {
    /* GQA: K e V per ogni KV head, per layer */
    /* K[layer][pos][kv_head * head_dim] */
    /* V[layer][pos][kv_head * head_dim] */
    float **K, **V;          /* [n_layers][max_pos * n_kv_heads * head_dim] */
    int max_pos;             /* posizioni allocate */
    int cur_pos;             /* posizione corrente */
} KVCache;

/* ── Modello ── */
typedef struct {
    Cfg c;
    /* shards S; */          /* reader safetensors */

    QT embed, lm_head;
    float *final_norm;
    Layer *L;                /* [n_layers] */

    KVCache kv;

    /* Cache expert per-layer */
    ESlot **ecache;          /* [n_layers][ecap] */
    int *ecn;                /* expert cached per layer */
    int ecap;                /* capacità cache per layer */

    /* Hot-store appreso */
    ESlot **pin;             /* expert pinnati per layer */
    int *npin;
    uint32_t **eusage;       /* contatori persistenti */
    uint32_t **eheat;        /* calore recente */

    /* Working set corrente */
    ESlot ws[32];            /* max topk * batch expert in flight */

    /* Statistiche */
    uint64_t eclock, hits, miss, ereq;
    uint64_t n_fw, n_emit;
    double t_edisk, t_emm, t_attn, t_head;
    int64_t resident_bytes;
} Model;
```

---

## 4. Layout dei file del modello convertito

```
/path/to/gptoss_i4/
├── config.json              ← copiato dall'originale
├── tokenizer.json           ← o200k_harmony
├── params.json              ← metadati di conversione
│
├── dense.safetensors        ← embed + lm_head + attention + FFN densa
│                               (tutti a INT4 o BF16, ~4 GB)
│
├── experts-00.safetensors   ← expert del layer 0..5 (shardati per parallelismo)
├── experts-01.safetensors   ← expert del layer 6..11
├── ...                      ← ~6 shard da ~50-70 GB
│
├── .picchio_usage           ← contatori di routing (aggiornato ogni turno)
└── .picchio_kv              ← KV-cache persistente (opzionale)
```

Ogni expert è memorizzato come 3 tensori contigui:
```
model.layers.{L}.mlp.experts.{E}.gate_proj.weight     → uint8 packed
model.layers.{L}.mlp.experts.{E}.gate_proj.weight.qs   → float32 scale
model.layers.{L}.mlp.experts.{E}.up_proj.weight        → uint8 packed
model.layers.{L}.mlp.experts.{E}.up_proj.weight.qs     → float32 scale
model.layers.{L}.mlp.experts.{E}.down_proj.weight      → uint8 packed
model.layers.{L}.mlp.experts.{E}.down_proj.weight.qs   → float32 scale
```

La contiguità nel file è cruciale: una singola `pread` carica tutto l'expert.

---

## 5. Moduli del progetto

```
picchio/
├── DESIGN.md                ← questo documento
├── Makefile                 ← build + check + clean
├── c/
│   ├── picchio.c            ← motore principale (forward pass, MoE loop, decode)
│   ├── st.h                 ← reader safetensors (come Colibri)
│   ├── tok.h                ← tokenizer o200k_harmony
│   ├── json.h               ← parser JSON minimale
│   ├── tier.h               ← gerarchia VRAM/RAM/disco, LRU, hot-store
│   ├── quant.h              ← kernel quantizzazione: INT4, INT8, MXFP4, IDOT
│   ├── attn.h               ← GQA attention + RoPE + KV-cache
│   ├── simd.h               ← primitive SIMD: AVX2, AVX-512, NEON
│   ├── pilot.h              ← prefetch pilotato (thread separato)
│   ├── backend_cuda.h/.cu   ← tier VRAM opzionale
│   ├── convert.py           ← conversione HF MXFP4 → INT4 container
│   ├── openai_server.py     ← gateway API OpenAI-compatible
│   ├── setup.sh             ← build + self-test
│   └── tests/
│       ├── test_matmul.c    ← validazione kernel vs reference float
│       ├── test_attn.c      ← validazione GQA vs torch
│       └── oracle.py        ← genera reference tokens da transformers
├── web/                     ← dashboard browser (opzionale, fase 2)
└── docs/
    ├── benchmarks.md
    └── tuning.md
```

---

## 6. Piano di sviluppo incrementale

### Fase 1: "Token corretto" (settimane 1-3)
- [ ] Reader safetensors (`st.h`) — riusabile da Colibri con adattamenti
- [ ] Parser config.json → struct `Cfg`
- [ ] Tokenizer o200k_harmony (wrapper del .json con BPE)
- [ ] Caricamento parte densa (embed, attention, FFN) a BF16/INT4
- [ ] Forward pass densa: RMSNorm → GQA → FFN → residual
- [ ] KV-cache GQA
- [ ] RoPE standard
- [ ] Validazione: token-exact vs `transformers` su un prompt di test
- [ ] **Milestone:** genera il primo token corretto

### Fase 2: "MoE funzionante" (settimane 3-5)
- [ ] Router: top-4 expert selection (sigmoid/softmax + top-k)
- [ ] Expert loading da disco (`pread` coalescente)
- [ ] Cache LRU per-layer
- [ ] Shared expert residente (se il modello ne ha)
- [ ] Validazione MoE: expert routing identico a transformers
- [ ] **Milestone:** genera testo coerente (anche se lento)

### Fase 3: "Velocità" (settimane 5-8)
- [ ] Kernel SIMD: AVX2/NEON per matmul INT4/INT8
- [ ] IDOT (dot-product intero: quantizza attivazioni → int8)
- [ ] Prefetch pilotato (PILOT thread)
- [ ] I/O asincrono (PIPE: pool di thread per pread parallele)
- [ ] Batch-union (ogni expert unico letto una sola volta per batch)
- [ ] O_DIRECT opzionale
- [ ] Hot-store appreso (.picchio_usage)
- [ ] **Milestone:** >0.5 tok/s su NVMe consumer

### Fase 4: "Produzione" (settimane 8-12)
- [ ] Convertitore MXFP4 → INT4 (Python, shard-by-shard)
- [ ] Server API OpenAI-compatible
- [ ] Sampling: temperature, top-p, top-k, repetition penalty
- [ ] Chat template harmony
- [ ] KV-cache persistente su disco
- [ ] Dual-SSD
- [ ] Profiling e tuning dashboard
- [ ] **Milestone:** demo end-to-end, chat interattiva

### Fase 5: "GPU e oltre" (opzionale)
- [ ] Tier VRAM con backend CUDA
- [ ] Expert residenti in GPU
- [ ] Metal backend per Apple Silicon
- [ ] Speculative decoding (se il modello ha una draft head)

---

## 7. Stime di performance

### 7.1 Scenario: Desktop 32 GB RAM, NVMe 3.5 GB/s

**Parte densa residente:** ~4 GB (INT4), lascia ~25 GB per cache expert.
**Expert per token:** 4 expert × ~113 MB = ~452 MB per layer MoE.
**Layer MoE:** ~30. Expert unici per token: fino a 4×30 = 120 (ma con overlap).

Caso freddo (tutto da disco):
- 120 expert × 113 MB = ~13.5 GB di letture per token
- A 3.5 GB/s: ~3.9 secondi per token → **~0.26 tok/s**

Caso tiepido (50% cache hit):
- ~6.8 GB di letture → ~1.9 s/tok → **~0.5 tok/s**

Caso caldo (90% cache hit, dopo warm-up):
- ~1.35 GB di letture → ~0.4 s/tok → **~2.5 tok/s**

Con prefetch PILOT e I/O asincrono: +30-50% → **~1-3.5 tok/s**

### 7.2 Scenario: 64 GB RAM, tutto pinned

Expert totali per 30 layer MoE × 128 = 3840 expert × 113 MB = ~430 GB.
Non ci stanno tutti, ma con 60 GB disponibili si pinnano ~530 expert (i più caldi).
Se il routing è concentrato (come misurato per GLM: pochi expert dominanti),
il cache hit rate può superare il 95% → **3-5 tok/s CPU-only**.

### 7.3 Scenario: GPU RTX 4090 (24 GB VRAM) + 32 GB RAM

~210 expert in VRAM + densa su GPU.
Per la maggior parte dei token: 0 accessi a disco → **5-10 tok/s** (stimato).

### 7.4 Confronto con GLM-5.2 (Colibri)

GPT-OSS-120B è molto più favorevole per lo streaming:
- Expert ~7× più piccoli (113 MB vs 19 MB — da verificare con dim reali)
- Solo 4 expert attivi vs ~8
- Meno layer MoE (~30 vs 75)
- **Cache miss ~15× più economico** in termini di I/O

> Se le dimensioni reali dell'intermediate MoE sono ~8192 invece di 12288,
> gli expert scendono a ~75 MB e tutto migliora ulteriormente.

---

## 8. Differenze chiave rispetto a Colibri

| Aspetto | Colibri (GLM-5.2) | Picchio (GPT-OSS-120B) |
|---|---|---|
| Attenzione | MLA (KV compressa, 576 float/tok) | GQA standard (1536 float/tok) |
| KV-cache | Ultra-compressa, ricostruzione al volo | Standard GQA, più grande ma più semplice |
| RoPE | Interleaved parziale | Standard |
| Router | Sigmoid + noaux_tc + bias | Da determinare (probabilmente softmax top-k) |
| Expert/token | ~8 | 4 |
| Expert/layer | 256 | 128 |
| Formato pesi | INT4 simmetrico per-riga | MXFP4 nativo (convertiremo a INT4) |
| Speculative | MTP head nativa | Da verificare (potrebbe non averne) |
| Sparse attention | DSA lightning indexer | Banded (finestra locale, più semplice) |
| Dimensione modello | ~370 GB (int4) | ~60-430 GB (dipende da intermediate dim) |

---

## 9. Decisioni di design aperte

1. **Formato di conversione:** MXFP4 nativo o INT4 simmetrico?
   - Raccomandazione: INT4 per la v1 (riusa kernel Colibri), MXFP4 nativo per v2

2. **Dimensioni reali del modello:** servono i numeri esatti dal config.json.
   Scaricare e ispezionare `openai/gpt-oss-120b` su HuggingFace.

3. **Shared expert:** GPT-OSS potrebbe non avere shared expert come GLM-5.2.
   Da verificare nel model code.

4. **Router type:** sigmoid con bias (come GLM) o softmax classico?
   Da leggere nel codice `gpt_oss/torch/model.py`.

5. **Banded attention pattern:** qual è la window size? Alternanza esatta?
   Impatta la KV-cache (possiamo scartare token oltre la finestra).

6. **Chat format:** il modello richiede harmony encoding, servono
   i template corretti per il prompt.

---

## 10. Dipendenze e requisiti di build

**Runtime (zero dipendenze):**
- Compilatore C: gcc ≥ 9 o clang ≥ 12, con OpenMP
- POSIX: pread, posix_fadvise, mmap, pthread
- Opzionale: CUDA toolkit (per tier VRAM)

**Build-time:**
- make
- Python 3.10+ (solo per il convertitore e il server API)

**Piattaforme target:**
- Linux x86_64 (primaria)
- macOS arm64 (Apple Silicon, NEON)
- Windows x86_64 (MinGW/MSVC, secondaria)

---

*Documento fondativo v0.1 — Picchio Project, luglio 2026*
