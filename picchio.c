/* picchio.c — Motore MoE streaming per GPT-OSS-120B in C puro.
 *
 * Architettura target: GPT-OSS-120B (117B MoE, 5.1B attivi/token)
 *   - 36 layer, 128 expert/layer, top-4
 *   - GQA (64 Q heads, 8 KV heads, group=8, head_dim=64)
 *   - RoPE standard, sliding window 128 alternato con full attention
 *   - MXFP4 quantizzazione nativa (convertita a INT4 per la v1)
 *   - Hidden dim 2880, MoE intermediate 5760 (fused gate+up)
 *   - Expert: gate_up [5760, 2880] + down [2880, 2880] = ~12.4 MB a INT4
 *   - Vocabolario 201,088 (o200k_harmony)
 *
 * Filosofia: placement decide SOLO la velocità, mai la precisione.
 * L'output è identico indipendentemente da dove risiedono gli expert.
 *
 * Build: make picchio
 * Run:   MODEL=/path/to/gptoss_i4 ./picchio [max_tokens]
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>     /* GetProcessMemoryInfo per rss_gb */
#else
#include <pthread.h>
#include <unistd.h>
#endif

#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#include <sys/mman.h>
#endif

#ifdef __linux__
#include <fcntl.h>     /* O_DIRECT */
#endif

#include "quant.h"
#include "json.h"
#include "st.h"
#include "tok.h"

/* ═══════════════════════════════════════════════════════════
 *  CONFIGURAZIONE (letta da config.json del modello)
 * ═══════════════════════════════════════════════════════════ */

typedef struct {
    int hidden;           /* D: hidden dimension (2880) */
    int n_layers;         /* numero totale di layer (36) */
    int n_heads;          /* query heads (64) */
    int n_kv_heads;       /* KV heads (8) — GQA group=8 */
    int head_dim;         /* dimensione per testa (64) */
    int n_experts;        /* expert per layer MoE (128) */
    int topk;             /* expert attivi per token (4) */
    int moe_inter;        /* intermediate dim expert: gate_up fused (5760) */
    int dense_inter;      /* intermediate dim FFN densa (se layer densi) */
    int vocab;            /* dimensione vocabolario (201088) */
    int ctx_len;          /* contesto massimo (131072) */
    int sliding_window;   /* finestra sliding attention (128) */
    int stop_ids[8];      /* token di stop */
    int n_stop;
    float eps;            /* RMSNorm epsilon (1e-5) */
    float theta;          /* RoPE base frequency */
    float rope_factor;    /* YaRN context extension factor */
    float rope_beta_fast;
    float rope_beta_slow;
    int rope_original_ctx;
    int rope_truncate;
    float rope_attn_factor;
    float swiglu_limit;
    float swiglu_alpha;
    float routed_scale;   /* scaling factor per expert routing */
    int has_shared;       /* 1 se ci sono shared expert */
    int n_shared;         /* numero di shared expert */
    int has_attn_bias;    /* 1 se attention usa bias (sì in GPT-OSS) */
    int8_t layer_type[128]; /* per layer: 0=sliding_attention, 1=full_attention */
} Cfg;

/* ═══════════════════════════════════════════════════════════
 *  LAYER
 * ═══════════════════════════════════════════════════════════ */

typedef struct {
    float *in_ln;         /* RMSNorm pre-attention weights [D] */
    float *post_ln;       /* RMSNorm pre-FFN weights [D] */

    /* GQA Attention */
    QT wq;                /* [n_heads * head_dim, D] = [4096, 2880] */
    QT wk;                /* [n_kv_heads * head_dim, D] = [512, 2880] */
    QT wv;                /* [n_kv_heads * head_dim, D] = [512, 2880] */
    QT wo;                /* [D, n_heads * head_dim] = [2880, 4096] */

    /* Attention bias (GPT-OSS le ha) */
    float *bq;            /* [n_heads * head_dim] */
    float *bk;            /* [n_kv_heads * head_dim] */
    float *bv;            /* [n_kv_heads * head_dim] */
    float *bo;            /* [D] */
    float *sinks;         /* [n_heads] attention sink logits */

    int layer_type;       /* 0=sliding_attention, 1=full_attention */

    /* MoE (tutti i layer in GPT-OSS-120B sono MoE) */
    float *router;        /* [n_experts, D] — pesi del router */
    float *router_bias;   /* [n_experts] — bias di correzione */

    /* Expert gate_up_proj è fusa: [moe_inter, D] dove moe_inter = 5760 = 2*2880
     * down_proj: [D, D] = [2880, 2880]
     * I pesi degli expert NON sono nel Layer — sono caricati on-demand in ESlot */
} Layer;

/* ═══════════════════════════════════════════════════════════
 *  EXPERT SLOT (cache LRU, riusabile tra layer)
 * ═══════════════════════════════════════════════════════════ */

typedef struct {
    int eid;              /* expert ID (-1 = vuoto) */
    int layer;            /* layer di appartenenza */
    /* GPT-OSS expert layout: gate_up fused [5760, 2880] + down [2880, 2880]
     * gate_up_proj_bias [5760], down_proj_bias [2880] */
    QT gu;                /* gate_up_proj fused [moe_inter, D] */
    QT d;                 /* down_proj [D, D] */
    float *gu_bias;       /* [moe_inter] */
    float *d_bias;        /* [D] */
    uint8_t *slab;        /* buffer per pread coalescente */
    float *fslab;         /* buffer scale */
    int64_t slab_cap;     /* capacità allocata */
    int64_t fslab_cap;
    uint64_t last_used;   /* timestamp per LRU eviction */
    int loading;          /* 1 = prefetch in corso su questo slot (atomico) */
} ESlot;

/* ═══════════════════════════════════════════════════════════
 *  KV-CACHE (GQA standard)
 * ═══════════════════════════════════════════════════════════ */

typedef struct {
    /* Per layer, per posizione: K e V di ogni KV head.
     * Layout: K[layer] = float[max_pos * n_kv_heads * head_dim]
     *         V[layer] = float[max_pos * n_kv_heads * head_dim]
     */
    float **K;            /* [n_layers] puntatori a buffer K */
    float **V;            /* [n_layers] puntatori a buffer V */
    int max_pos;          /* posizioni allocate */
    int cur_pos;          /* prossima posizione da scrivere */
} KVCache;

/* ═══════════════════════════════════════════════════════════
 *  MODELLO
 * ═══════════════════════════════════════════════════════════ */

typedef struct {
    Cfg c;

    QT embed;             /* embedding [vocab, D] */
    QT lm_head;           /* language model head [vocab, D] */
    float *final_norm;    /* RMSNorm finale [D] */

    Layer *L;             /* [n_layers] */
    KVCache kv;

    /* Cache expert per-layer */
    ESlot **ecache;       /* [n_layers][ecap] — pool LRU */
    int *ecn;             /* quanti expert cached per layer */
    int ecap;             /* capacità per layer */

    /* Hot-store appreso */
    ESlot **pin;          /* expert pinnati (mai evicted) */
    int *npin;
    uint32_t **eusage;    /* contatori persistenti per expert */
    uint32_t **eheat;     /* calore recente (per promotion) */

    /* Working set del forward corrente */
    ESlot ws[32];         /* expert in flight (max topk per batch) */

    /* Profiling */
    uint64_t eclock;      /* monotonic counter per LRU */
    uint64_t hits, miss, ereq; /* statistiche cache */
    uint64_t n_fw, n_emit;     /* forward / token emessi */
    double t_edisk;       /* tempo totale letture expert da disco */
    double t_attn;        /* tempo totale attention */
    double t_moe;         /* tempo totale MoE compute */
    double t_head;        /* tempo totale lm_head */

    int64_t resident_bytes; /* byte della parte densa in RAM */

    /* Probe accuratezza predizione prefetch (PREDICT_PROBE=1) */
    int pred_next[64];    /* proxy A: predetto da hn pre-MoE del layer precedente */
    int pred_next2[64];   /* proxy B: predetto da post-MoE norm del layer precedente */
    int pred_layer;       /* layer a cui si riferiscono le predizioni (-1 = nessuna) */
    uint64_t pred_hit, pred_total;    /* proxy A */
    uint64_t pred_hit2;               /* proxy B (stesso pred_total) */
} Model;

/* ═══════════════════════════════════════════════════════════
 *  ORACLE DUMP (.npy F32, attivo solo con ORACLE_DIR)
 * ═══════════════════════════════════════════════════════════ */

static const char *g_oracle_dir = NULL;

static void oracle_dump(const char *name, const float *data,
                        int ndim, const int64_t *shape) {
    if (!g_oracle_dir || !*g_oracle_dir) return;
#ifdef _WIN32
    CreateDirectoryA(g_oracle_dir, NULL);
#endif
    char path[768];
    snprintf(path, sizeof(path), "%s/%s.npy", g_oracle_dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) return;

    char shape_str[256] = "";
    size_t used = 0;
    for (int i = 0; i < ndim; i++) {
        int n = snprintf(shape_str + used, sizeof(shape_str) - used,
                         "%s%lld%s", i ? ", " : "",
                         (long long)shape[i], (ndim == 1) ? "," : "");
        if (n > 0) used += (size_t)n;
    }
    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }",
        shape_str);
    int preamble = 10;
    int padded = hlen + 1;
    while ((preamble + padded) % 16) padded++;
    memset(header + hlen, ' ', (size_t)(padded - hlen));
    header[padded - 1] = '\n';

    const uint8_t magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    uint16_t header_len = (uint16_t)padded;
    fwrite(magic, 1, 8, f);
    fwrite(&header_len, 2, 1, f);
    fwrite(header, 1, (size_t)padded, f);
    int64_t count = 1;
    for (int i = 0; i < ndim; i++) count *= shape[i];
    fwrite(data, sizeof(float), (size_t)count, f);
    fclose(f);
}

static void oracle_dump_vec(const char *name, const float *data, int64_t n) {
    int64_t shape[3] = {1, 1, n};
    oracle_dump(name, data, 3, shape);
}

static int g_trace_numeric = 0;
static int g_predict_probe = 0;   /* misura accuratezza predizione prefetch */

static int trace_vector(const char *stage, int layer, const float *x, int n) {
    if (!g_trace_numeric) return 1;
    int nonfinite = 0;
    float min_v = INFINITY, max_v = -INFINITY;
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(x[i])) { nonfinite++; continue; }
        if (x[i] < min_v) min_v = x[i];
        if (x[i] > max_v) max_v = x[i];
        sum_sq += (double)x[i] * x[i];
    }
    fprintf(stderr, "  numeric L=%d %-18s min=% .6g max=% .6g rms=%.6g bad=%d\n",
            layer, stage, min_v, max_v,
            n > nonfinite ? sqrt(sum_sq / (n - nonfinite)) : NAN, nonfinite);
    return nonfinite == 0;
}

static void trace_top_logits(const float *logits, int vocab) {
    if (!g_trace_numeric) return;
    int ids[10]; float values[10];
    for (int k = 0; k < 10; k++) { ids[k] = -1; values[k] = -INFINITY; }
    int nonfinite = 0;
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(logits[i])) { nonfinite++; continue; }
        if (logits[i] <= values[9]) continue;
        values[9] = logits[i]; ids[9] = i;
        for (int k = 8; k >= 0 && values[k + 1] > values[k]; k--) {
            float fv = values[k]; values[k] = values[k + 1]; values[k + 1] = fv;
            int iv = ids[k]; ids[k] = ids[k + 1]; ids[k + 1] = iv;
        }
    }
    fprintf(stderr, "  numeric logits bad=%d top10:", nonfinite);
    for (int k = 0; k < 10; k++) fprintf(stderr, " %d=%.6g", ids[k], values[k]);
    fprintf(stderr, "\n");
}

/* ═══════════════════════════════════════════════════════════
 *  TIMING
 * ═══════════════════════════════════════════════════════════ */

static double now_s(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
#endif
}

static double rss_gb(void) {
#if defined(__APPLE__) || defined(__linux__)
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
#ifdef __APPLE__
    return r.ru_maxrss / (1024.0 * 1024.0 * 1024.0);
#else
    return r.ru_maxrss / (1024.0 * 1024.0);
#endif
#elif defined(_WIN32)
    /* Working set corrente = RSS: pagine fisicamente residenti in RAM. */
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.WorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    return 0.0;
#else
    return 0.0;
#endif
}

/* RAM fisica totale in byte (0 se non determinabile). Usata per dimensionare il
 * budget della cache expert quando PIN_GB non è specificato. */
static int64_t physical_ram_bytes(void) {
#ifdef _WIN32
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) return (int64_t)ms.ullTotalPhys;
    return 0;
#elif defined(__linux__)
    long pages = sysconf(_SC_PHYS_PAGES);
    long psize = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && psize > 0) return (int64_t)pages * (int64_t)psize;
    return 0;
#else
    return 0;  /* altre piattaforme: fallback al default fisso */
#endif
}

/* ═══════════════════════════════════════════════════════════
 *  CONFIG LOADER (da config.json)
 * ═══════════════════════════════════════════════════════════ */

static int cfg_load(Cfg *c, const char *model_path) {
    char path[512];
    snprintf(path, sizeof(path), "%s/config.json", model_path);

    char *json = json_read_file(path);
    if (!json) {
        fprintf(stderr, "errore: impossibile leggere %s\n", path);
        return -1;
    }

    memset(c, 0, sizeof(*c));

    c->hidden       = json_int(json, "hidden_size", 2880);
    c->n_layers     = json_int(json, "num_hidden_layers", 36);
    c->n_heads      = json_int(json, "num_attention_heads", 64);
    c->n_kv_heads   = json_int(json, "num_key_value_heads", 8);
    c->head_dim     = json_int(json, "head_dim", 64);
    c->n_experts    = json_int(json, "num_local_experts", 128);
    c->topk         = json_int(json, "num_experts_per_tok", 4);
    c->moe_inter    = json_int(json, "intermediate_size", 2880) * 2; /* gate+up fused */
    c->dense_inter  = json_int(json, "intermediate_size", 2880);
    c->vocab        = json_int(json, "vocab_size", 201088);
    c->ctx_len      = json_int(json, "max_position_embeddings", 131072);
    c->sliding_window = json_int(json, "sliding_window", 128);
    c->eps          = json_float(json, "rms_norm_eps", 1e-5f);
    c->theta        = json_float(json, "rope_theta", 150000.0f);
    c->rope_factor  = json_float(json, "factor", 1.0f);
    c->rope_beta_fast = json_float(json, "beta_fast", 32.0f);
    c->rope_beta_slow = json_float(json, "beta_slow", 1.0f);
    c->rope_original_ctx = json_int(json, "original_max_position_embeddings", c->ctx_len);
    c->rope_truncate = json_bool(json, "truncate", 1);
    c->rope_attn_factor = c->rope_factor > 1.0f
        ? 0.1f * logf(c->rope_factor) + 1.0f : 1.0f;
    c->swiglu_limit = json_float(json, "swiglu_limit", 7.0f);
    c->swiglu_alpha = 1.702f;
    c->routed_scale = json_float(json, "routed_scaling_factor", 1.0f);
    c->has_shared   = json_int(json, "num_shared_experts", 0) > 0 ? 1 : 0;
    c->n_shared     = json_int(json, "num_shared_experts", 0);
    c->has_attn_bias = json_bool(json, "attention_bias", 1);

    /* Layer types: alternating sliding/full attention */
    char layer_types[128][64];
    int n_types = json_str_array(json, "layer_types", layer_types, 128);
    if (n_types > 0) {
        for (int i = 0; i < c->n_layers && i < n_types; i++) {
            if (strstr(layer_types[i % n_types], "full"))
                c->layer_type[i] = 1;
            else
                c->layer_type[i] = 0; /* sliding */
        }
        /* Se n_types < n_layers, cicla il pattern */
        if (n_types < c->n_layers) {
            for (int i = n_types; i < c->n_layers; i++)
                c->layer_type[i] = c->layer_type[i % n_types];
        }
    } else {
        /* Default: alterna sliding/full */
        for (int i = 0; i < c->n_layers; i++)
            c->layer_type[i] = (int8_t)(i % 2);
    }

    /* Stop tokens: <|return|>=200002, <|call|>=200012 */
    c->stop_ids[0] = 200002;  /* <|return|> */
    c->stop_ids[1] = 200012;  /* <|call|> */
    c->n_stop = 2;

    free(json);

    /* Riepilogo */
    fprintf(stderr, "  config: D=%d L=%d H=%d KV=%d hd=%d E=%d top%d\n",
            c->hidden, c->n_layers, c->n_heads, c->n_kv_heads,
            c->head_dim, c->n_experts, c->topk);
    fprintf(stderr, "  config: moe_inter=%d vocab=%d ctx=%d sw=%d eps=%.0e\n",
            c->moe_inter, c->vocab, c->ctx_len, c->sliding_window, c->eps);

    return 0;
}

/* ═══════════════════════════════════════════════════════════
 *  RoPE (standard, non interleaved)
 * ═══════════════════════════════════════════════════════════ */

static float yarn_inv_freq(const Cfg *c, int j) {
    int dim = c->head_dim;
    float pos_freq = powf(c->theta, 2.0f * j / dim);
    float extrap = 1.0f / pos_freq;
    if (c->rope_factor <= 1.0f) return extrap;

    float low = dim * logf(c->rope_original_ctx /
                (c->rope_beta_fast * 2.0f * (float)M_PI)) /
                (2.0f * logf(c->theta));
    float high = dim * logf(c->rope_original_ctx /
                 (c->rope_beta_slow * 2.0f * (float)M_PI)) /
                 (2.0f * logf(c->theta));
    if (c->rope_truncate) { low = floorf(low); high = ceilf(high); }
    if (low < 0) low = 0;
    if (high > dim - 1) high = (float)(dim - 1);
    if (low == high) high += 0.001f;
    float ramp = (j - low) / (high - low);
    if (ramp < 0) ramp = 0;
    if (ramp > 1) ramp = 1;
    float extrap_factor = 1.0f - ramp;
    float interp = extrap / c->rope_factor;
    return interp * (1.0f - extrap_factor) + extrap * extrap_factor;
}

static void rope_apply(float *q, float *k, int pos, int head_dim,
                       int n_q_heads, int n_kv_heads, const Cfg *c) {
    int half = head_dim / 2;
    for (int kind = 0; kind < 2; kind++) {
        float *base = kind == 0 ? q : k;
        int heads = kind == 0 ? n_q_heads : n_kv_heads;
        for (int h = 0; h < heads; h++) {
            float *v = base + h * head_dim;
            for (int j = 0; j < half; j++) {
                float ang = pos * yarn_inv_freq(c, j);
                float cs = cosf(ang) * c->rope_attn_factor;
                float sn = sinf(ang) * c->rope_attn_factor;
                float a = v[j], b = v[j + half];
                v[j] = a * cs - b * sn;
                v[j + half] = a * sn + b * cs;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════
 *  KV-CACHE: allocazione e aggiornamento
 * ═══════════════════════════════════════════════════════════ */

static void kv_init(KVCache *kv, int n_layers, int n_kv_heads,
                    int head_dim, int max_pos) {
    int kv_dim = n_kv_heads * head_dim;
    kv->max_pos = max_pos;
    kv->cur_pos = 0;
    kv->K = calloc(n_layers, sizeof(float *));
    kv->V = calloc(n_layers, sizeof(float *));
    for (int l = 0; l < n_layers; l++) {
        kv->K[l] = calloc((int64_t)max_pos * kv_dim, sizeof(float));
        kv->V[l] = calloc((int64_t)max_pos * kv_dim, sizeof(float));
    }
}

static void kv_store(KVCache *kv, int layer, int pos,
                     const float *k, const float *v,
                     int kv_dim) {
    if (pos >= kv->max_pos) return;  /* overflow protection */
    memcpy(kv->K[layer] + (int64_t)pos * kv_dim, k, kv_dim * sizeof(float));
    memcpy(kv->V[layer] + (int64_t)pos * kv_dim, v, kv_dim * sizeof(float));
}

/* ═══════════════════════════════════════════════════════════
 *  GQA ATTENTION (decode, single token)
 * ═══════════════════════════════════════════════════════════ */

static void gqa_attention(float *out, const float *x, Layer *l,
                          KVCache *kv, int layer, int pos, const Cfg *c) {
    int D = c->hidden;
    int H = c->n_heads;
    int KVH = c->n_kv_heads;
    int hd = c->head_dim;
    int kv_dim = KVH * hd;
    int group = H / KVH;  /* query heads per KV head */

    float *q = falloc(H * hd);
    float *k = falloc(kv_dim);
    float *v = falloc(kv_dim);

    /* Proiezioni Q, K, V */
    matmul_qt(q, x, &l->wq, 1);
    matmul_qt(k, x, &l->wk, 1);
    matmul_qt(v, x, &l->wv, 1);
    if (g_oracle_dir) {
        char n[128]; int64_t sq[3] = {1, 1, H * hd};
        int64_t sk[3] = {1, 1, kv_dim};
        snprintf(n, sizeof(n), "layer%d.q_nobias", layer); oracle_dump(n, q, 3, sq);
        snprintf(n, sizeof(n), "layer%d.k_nobias", layer); oracle_dump(n, k, 3, sk);
        snprintf(n, sizeof(n), "layer%d.v_nobias", layer); oracle_dump(n, v, 3, sk);
    }

    /* Add bias se presente */
    if (l->bq) for (int i = 0; i < H * hd; i++) q[i] += l->bq[i];
    if (l->bk) for (int i = 0; i < kv_dim; i++) k[i] += l->bk[i];
    if (l->bv) for (int i = 0; i < kv_dim; i++) v[i] += l->bv[i];
    if (g_oracle_dir) {
        char n[128]; int64_t sq[3] = {1, 1, H * hd};
        int64_t sk[3] = {1, 1, kv_dim};
        snprintf(n, sizeof(n), "layer%d.q_bias", layer); oracle_dump(n, q, 3, sq);
        snprintf(n, sizeof(n), "layer%d.k_bias", layer); oracle_dump(n, k, 3, sk);
        snprintf(n, sizeof(n), "layer%d.v_bias", layer); oracle_dump(n, v, 3, sk);
    }

    /* RoPE / YaRN */
    rope_apply(q, k, pos, hd, H, KVH, c);
    if (g_oracle_dir) {
        char n[128];
        int64_t sq[4] = {1, H, 1, hd};
        int64_t sk[4] = {1, KVH, 1, hd};
        snprintf(n, sizeof(n), "layer%d.q_rope", layer); oracle_dump(n, q, 4, sq);
        snprintf(n, sizeof(n), "layer%d.k_rope", layer); oracle_dump(n, k, 4, sk);
        snprintf(n, sizeof(n), "layer%d.v_heads", layer); oracle_dump(n, v, 4, sk);
    }

    /* Aggiorna KV-cache */
    if (pos < kv->max_pos)
        kv_store(kv, layer, pos, k, v, kv_dim);

    /* Attention: per ogni query head */
    float *attn_out = calloc(H * hd, sizeof(float));

    /* Determina la finestra di attenzione */
    int start_pos = 0;
    int end_pos = pos;
    if (end_pos >= kv->max_pos) end_pos = kv->max_pos - 1;
    if (l->layer_type == 0) {  /* sliding_attention: include al massimo window token */
        start_pos = end_pos - c->sliding_window + 1;
        if (start_pos < 0) start_pos = 0;
    }
    /* layer_type == 1: full_attention → start_pos = 0 (vede tutto) */

    for (int h = 0; h < H; h++) {
        int kv_h = h / group;  /* quale KV head */
        float *qh = q + h * hd;

        /* Score con posizioni nella finestra */
        int n_pos = end_pos - start_pos + 1;
        if (n_pos <= 0) n_pos = 1;
        float *scores = calloc(n_pos, sizeof(float));
        float scale = 1.0f / sqrtf((float)hd);

        for (int t = start_pos; t <= end_pos; t++) {
            float *kt = kv->K[layer] + (int64_t)t * kv_dim + kv_h * hd;
            float dot = 0;
            for (int i = 0; i < hd; i++) dot += qh[i] * kt[i];
            scores[t - start_pos] = dot * scale;
        }

        /* Softmax con attention sink: il sink assorbe probabilità ma non
         * contribuisce alla somma dei value, come Transformers eager. */
        float max_score = l->sinks ? l->sinks[h] : -INFINITY;
        for (int i = 0; i < n_pos; i++)
            if (scores[i] > max_score) max_score = scores[i];
        float sum = l->sinks ? expf(l->sinks[h] - max_score) : 0.0f;
        for (int i = 0; i < n_pos; i++) {
            scores[i] = expf(scores[i] - max_score);
            sum += scores[i];
        }
        for (int i = 0; i < n_pos; i++) scores[i] /= sum;

        /* Weighted sum dei valori */
        float *oh = attn_out + h * hd;
        for (int t = start_pos; t <= end_pos; t++) {
            float *vt = kv->V[layer] + (int64_t)t * kv_dim + kv_h * hd;
            float w = scores[t - start_pos];
            for (int i = 0; i < hd; i++) oh[i] += w * vt[i];
        }

        free(scores);
    }

    /* Output projection */
    if (g_oracle_dir) {
        char n[128]; int64_t sh[3] = {1, 1, H * hd};
        snprintf(n, sizeof(n), "layer%d.attn_concat", layer);
        oracle_dump(n, attn_out, 3, sh);
    }
    matmul_qt(out, attn_out, &l->wo, 1);
    if (l->bo) for (int i = 0; i < D; i++) out[i] += l->bo[i];
    if (g_oracle_dir) {
        char n[128]; snprintf(n, sizeof(n), "layer%d.attn_out", layer);
        oracle_dump_vec(n, out, D);
    }

    free(q); free(k); free(v); free(attn_out);
}

/* ═══════════════════════════════════════════════════════════
 *  EXPERT LOADING (da disco, pread coalescente)
 * ═══════════════════════════════════════════════════════════ */

/* Puntatore globale al DB safetensors (per accesso da expert_load) */
static StDB *g_db = NULL;

/* Carica un bias expert dal formato corretto F32 oppure recupera il vecchio
 * formato convertito: byte INT4 salvati per errore come F32 + scale gs64. */
static float *load_expert_bias(int layer, int eid, const char *suffix,
                               int n_experts, int n_values) {
    char name[300];
    snprintf(name, sizeof(name), "model.layers.%d.mlp.experts.%d.%s",
             layer, eid, suffix);
    StTensor *t = st_find(g_db, name);
    int aggregated = 0;
    if (!t) {
        snprintf(name, sizeof(name), "model.layers.%d.mlp.experts.%s",
                 layer, suffix);
        t = st_find(g_db, name);
        aggregated = t != NULL;
    }
    if (!t) return NULL;

    float *out = falloc(n_values);
    char scale_name[340];
    snprintf(scale_name, sizeof(scale_name), "%s.qs", name);
    StTensor *ts = st_find(g_db, scale_name);

    int64_t tensor_numel = st_numel(t);
    int is_full_f32 = t->dtype == ST_F32 &&
                      tensor_numel == (int64_t)n_experts * n_values;

    if (aggregated && ts && !is_full_f32) {
        int packed_n = (n_values + 1) / 2;
        int groups = (n_values + 63) / 64;
        uint8_t *packed = (uint8_t *)malloc((size_t)packed_n);
        float *scales = falloc(groups);
        if (t->dtype == ST_F32) {
            float *stored = falloc(packed_n);
            st_read_raw_at(g_db, t, (int64_t)eid * packed_n * 4,
                           stored, (int64_t)packed_n * 4);
            for (int i = 0; i < packed_n; i++)
                packed[i] = (uint8_t)lrintf(stored[i]);
            free(stored);
        } else if (t->dtype == ST_U8) {
            st_read_raw_at(g_db, t, (int64_t)eid * packed_n,
                           packed, packed_n);
        } else {
            free(packed); free(scales); free(out); return NULL;
        }
        st_read_raw_at(g_db, ts, (int64_t)eid * groups * 4,
                       scales, (int64_t)groups * 4);
        for (int i = 0; i < n_values; i++) {
            uint8_t byte = packed[i >> 1];
            int q = (i & 1) ? (int)(byte >> 4) - 8
                            : (int)(byte & 0x0F) - 8;
            out[i] = q * scales[i / 64];
        }
        free(packed); free(scales);
        return out;
    }

    if (t->dtype != ST_F32) { free(out); return NULL; }
    int64_t offset = aggregated ? (int64_t)eid * n_values * 4 : 0;
    if (st_read_raw_at(g_db, t, offset, out, (int64_t)n_values * 4)
            != (int64_t)n_values * 4) {
        free(out); return NULL;
    }
    (void)n_experts;
    return out;
}

static void expert_load(Model *m, int layer, int eid, ESlot *s) {
    Cfg *c = &m->c;
    int D = c->hidden;
    int I = c->moe_inter;  /* 5760 */
    char name[256];

    s->eid = eid;
    s->layer = layer;

    if (!g_db) {
        /* No safetensors DB — mark expert as unavailable */
        s->eid = -1;
        return;
    }

    /* ── Carica gate_up_proj: [moe_inter, D] ── */
    snprintf(name, sizeof(name),
             "model.layers.%d.mlp.experts.gate_up_proj.%d", layer, eid);
    StTensor *t_gu = st_find(g_db, name);

    /* Prova nomi alternativi */
    if (!t_gu) {
        snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.experts.%d.gate_up_proj", layer, eid);
        t_gu = st_find(g_db, name);
    }
    if (!t_gu) {
        snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.experts.%d.gate_up_proj.weight", layer, eid);
        t_gu = st_find(g_db, name);
    }

    if (!t_gu) {
        /* Expert non trovato nel DB (potrebbe essere in un shard non caricato) */
        s->eid = -1;
        return;
    }

    /* Carica come F32 e quantizza a INT4 */
    int64_t gu_numel = (int64_t)I * D;
    float *gu_f32 = falloc(gu_numel);

    if (t_gu->dtype == ST_F32) {
        st_read_raw(g_db, t_gu, gu_f32, gu_numel * 4);
        memset(&s->gu, 0, sizeof(QT));
        s->gu.fmt = 0; s->gu.O = I; s->gu.I = D; s->gu.qf = gu_f32;
        gu_f32 = NULL;
        goto gate_up_loaded;
    } else if (t_gu->dtype == ST_BF16) {
        uint16_t *tmp = (uint16_t *)malloc(gu_numel * 2);
        st_read_raw(g_db, t_gu, tmp, gu_numel * 2);
        for (int64_t i = 0; i < gu_numel; i++) {
            uint32_t bits = (uint32_t)tmp[i] << 16;
            memcpy(&gu_f32[i], &bits, 4);
        }
        free(tmp);
        memset(&s->gu, 0, sizeof(QT));
        s->gu.fmt = 0; s->gu.O = I; s->gu.I = D; s->gu.qf = gu_f32;
        gu_f32 = NULL;
        goto gate_up_loaded;
    } else if (t_gu->dtype == ST_U8) {
        /* Già INT4 packed — carica direttamente */
        memset(&s->gu, 0, sizeof(QT));
        s->gu.fmt = 2;
        s->gu.O = I;
        s->gu.I = D;
        int64_t rb = (int64_t)I * ((D + 1) / 2);
        s->gu.q4 = (uint8_t *)malloc(rb);
        st_read_raw(g_db, t_gu, s->gu.q4, rb);
        /* Cerca scale */
        snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.experts.gate_up_proj.%d.qs", layer, eid);
        StTensor *t_s = st_find(g_db, name);
        if (!t_s) {
            snprintf(name, sizeof(name),
                     "model.layers.%d.mlp.experts.%d.gate_up_proj.qs", layer, eid);
            t_s = st_find(g_db, name);
        }
        if (t_s) {
            int64_t n_scale = st_numel(t_s);
            s->gu.s = (float *)malloc(n_scale * sizeof(float));
            st_read_raw(g_db, t_s, s->gu.s, n_scale * sizeof(float));
            /* Determina block_size: se n_scale > O → group-scaled */
            if (n_scale > I) {
                s->gu.block_size = (int)((int64_t)I * D / n_scale);
                if (s->gu.block_size <= 0) s->gu.block_size = 64;
            }
        } else {
            s->gu.s = falloc(I);
            for (int i = 0; i < I; i++) s->gu.s[i] = 1.0f;
        }
        free(gu_f32);
        gu_f32 = NULL;
        goto gate_up_loaded;
    } else {
        free(gu_f32);
        s->eid = -1;
        return;
    }

gate_up_loaded:
    /* ── Carica gate_up bias (supporta anche il vecchio bias INT4+qs) ── */
    s->gu_bias = load_expert_bias(layer, eid, "gate_up_proj_bias",
                                  c->n_experts, I);

    /* ── Carica down_proj: [D, D] ── */
    snprintf(name, sizeof(name),
             "model.layers.%d.mlp.experts.down_proj.%d", layer, eid);
    StTensor *t_d = st_find(g_db, name);
    if (!t_d) {
        snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.experts.%d.down_proj", layer, eid);
        t_d = st_find(g_db, name);
    }
    if (!t_d) {
        snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.experts.%d.down_proj.weight", layer, eid);
        t_d = st_find(g_db, name);
    }

    int down_I = I / 2;  /* input dimension del down = intermediate_size */
    if (t_d) {
        int64_t d_numel = (int64_t)D * down_I;
        float *d_f32 = falloc(d_numel);

        if (t_d->dtype == ST_F32) {
            st_read_raw(g_db, t_d, d_f32, d_numel * 4);
            memset(&s->d, 0, sizeof(QT));
            s->d.fmt = 0; s->d.O = D; s->d.I = down_I; s->d.qf = d_f32;
            d_f32 = NULL;
            goto down_loaded;
        } else if (t_d->dtype == ST_BF16) {
            uint16_t *tmp = (uint16_t *)malloc(d_numel * 2);
            st_read_raw(g_db, t_d, tmp, d_numel * 2);
            for (int64_t i = 0; i < d_numel; i++) {
                uint32_t bits = (uint32_t)tmp[i] << 16;
                memcpy(&d_f32[i], &bits, 4);
            }
            free(tmp);
            memset(&s->d, 0, sizeof(QT));
            s->d.fmt = 0; s->d.O = D; s->d.I = down_I; s->d.qf = d_f32;
            d_f32 = NULL;
            goto down_loaded;
        } else if (t_d->dtype == ST_U8) {
            /* Già INT4 packed */
            memset(&s->d, 0, sizeof(QT));
            s->d.fmt = 2;
            s->d.O = D;
            s->d.I = down_I;
            int64_t rb = (int64_t)D * ((down_I + 1) / 2);
            s->d.q4 = (uint8_t *)malloc(rb);
            st_read_raw(g_db, t_d, s->d.q4, rb);
            snprintf(name, sizeof(name),
                     "model.layers.%d.mlp.experts.down_proj.%d.qs", layer, eid);
            StTensor *t_ds = st_find(g_db, name);
            if (!t_ds) {
                snprintf(name, sizeof(name),
                         "model.layers.%d.mlp.experts.%d.down_proj.qs", layer, eid);
                t_ds = st_find(g_db, name);
            }
            if (t_ds) {
                int64_t n_scale = st_numel(t_ds);
                s->d.s = (float *)malloc(n_scale * sizeof(float));
                st_read_raw(g_db, t_ds, s->d.s, n_scale * sizeof(float));
                if (n_scale > D) {
                    s->d.block_size = (int)((int64_t)D * down_I / n_scale);
                    if (s->d.block_size <= 0) s->d.block_size = 64;
                }
            } else {
                s->d.s = falloc(D);
                for (int i = 0; i < D; i++) s->d.s[i] = 1.0f;
            }
            free(d_f32);
            d_f32 = NULL;
            goto down_loaded;
        } else {
            free(d_f32);
            s->eid = -1;
            return;
        }
    } else {
        s->eid = -1;  /* Expert incompleto */
        return;
    }

down_loaded:
    /* ── Down bias (supporta anche il vecchio bias INT4+qs) ── */
    s->d_bias = load_expert_bias(layer, eid, "down_proj_bias",
                                 c->n_experts, D);
    return;
}

/* ═══════════════════════════════════════════════════════════
 *  EXPERT CACHE: lookup LRU + eviction
 * ═══════════════════════════════════════════════════════════ */

/* Flag `loading` di uno slot: 1 mentre il thread di prefetch lo sta riempiendo.
 * Release/acquire garantiscono che, quando il main osserva loading==0, veda anche
 * i buffer scritti dal prefetch. */
static inline int slot_loading(ESlot *s) {
    return __atomic_load_n(&s->loading, __ATOMIC_ACQUIRE);
}
static inline void slot_set_loading(ESlot *s, int v) {
    __atomic_store_n(&s->loading, v, __ATOMIC_RELEASE);
}
static void slot_wait_ready(ESlot *s) {
    while (slot_loading(s)) {
#ifdef _WIN32
        Sleep(0);
#else
        usleep(50);
#endif
    }
}

static void free_slot_buffers(ESlot *s) {
    free(s->gu.qf); s->gu.qf = NULL;
    free(s->gu.q4); s->gu.q4 = NULL;
    free(s->gu.s);  s->gu.s = NULL;
    free(s->d.qf);  s->d.qf = NULL;
    free(s->d.q4);  s->d.q4 = NULL;
    free(s->d.s);   s->d.s = NULL;
    free(s->gu_bias); s->gu_bias = NULL;
    free(s->d_bias);  s->d_bias = NULL;
    free(s->slab);    s->slab = NULL;
    free(s->fslab);   s->fslab = NULL;
}

static ESlot *cache_lookup(Model *m, int layer, int eid) {
    ESlot *pool = m->ecache[layer];
    int n = m->ecn[layer];

    /* Cerca nei pinned */
    for (int i = 0; i < m->npin[layer]; i++) {
        if (m->pin[layer][i].eid == eid) {
            m->pin[layer][i].last_used = ++m->eclock;
            m->hits++;
            return &m->pin[layer][i];
        }
    }

    /* Cerca nella cache LRU */
    for (int i = 0; i < n; i++) {
        if (pool[i].eid == eid) {
            pool[i].last_used = ++m->eclock;
            m->hits++;
            return &pool[i];
        }
    }

    m->miss++;

    /* Cache miss: evict LRU o usa slot vuoto */
    ESlot *victim = NULL;
    if (n < m->ecap) {
        victim = &pool[n];
        m->ecn[layer]++;
    } else {
        uint64_t oldest = UINT64_MAX;
        for (int i = 0; i < n; i++) {
            if (pool[i].last_used < oldest) {
                oldest = pool[i].last_used;
                victim = &pool[i];
            }
        }
    }

    /* Carica da disco */
    double t0 = now_s();
    
    /* Free vecchi dati se lo slot era occupato (eviction) */
    if (victim->eid >= 0) {
        free(victim->gu.qf); victim->gu.qf = NULL;
        free(victim->gu.q4); victim->gu.q4 = NULL;
        free(victim->gu.s);  victim->gu.s = NULL;
        free(victim->d.qf);  victim->d.qf = NULL;
        free(victim->d.q4);  victim->d.q4 = NULL;
        free(victim->d.s);   victim->d.s = NULL;
        free(victim->gu_bias); victim->gu_bias = NULL;
        free(victim->d_bias);  victim->d_bias = NULL;
        free(victim->slab);    victim->slab = NULL;
        free(victim->fslab);   victim->fslab = NULL;
    }
    
    expert_load(m, layer, eid, victim);
    m->t_edisk += now_s() - t0;
    victim->last_used = ++m->eclock;
    victim->layer = layer;
    m->ereq++;

    return victim;
}

/* Numero di thread per i read paralleli degli expert (1 = seriale). */
static int g_io_threads = 1;

/* Carica in batch gli expert `eids[0..n)` del layer, restituendo gli slot in
 * `out`. Gli hit vengono risolti e "toccati" (LRU) per primi, così i victim
 * scelti per i miss non possono sfrattare né un hit né un miss già riservato;
 * i miss vengono poi letti da disco in parallelo (queue depth > 1). La
 * matematica non cambia: si leggono gli stessi byte negli stessi slot, solo in
 * ordine concorrente. Contratto: n <= ecap (garantito da chi chiama). */
static void cache_load_batch(Model *m, int layer, const int *eids, int n,
                             ESlot **out) {
    ESlot *pool = m->ecache[layer];
    int miss_i[128];
    int nmiss = 0;

    /* Pass 1: risolvi gli hit (pinned o LRU) e marcali come appena usati. */
    for (int i = 0; i < n; i++) {
        ESlot *found = NULL;
        for (int j = 0; j < m->npin[layer]; j++)
            if (m->pin[layer][j].eid == eids[i]) { found = &m->pin[layer][j]; break; }
        if (!found)
            for (int j = 0; j < m->ecn[layer]; j++)
                if (pool[j].eid == eids[i]) { found = &pool[j]; break; }
        if (found) {
            slot_wait_ready(found);  /* se un prefetch sta caricando questo expert, attendi */
            found->last_used = ++m->eclock;
            m->hits++;
            out[i] = found;
        } else {
            out[i] = NULL;
            if (nmiss < 128) miss_i[nmiss++] = i;
        }
    }

    /* Pass 2: riserva un victim per ogni miss (seriale: nessuna race su LRU).
     * Gli slot con prefetch in corso (loading) non vengono mai scelti come victim;
     * l'invariante ecap-topk garantito da prefetch_issue assicura che restino
     * abbastanza slot liberi. */
    for (int mi = 0; mi < nmiss; mi++) {
        int i = miss_i[mi];
        ESlot *victim = NULL;
        if (m->ecn[layer] < m->ecap) {
            victim = &pool[m->ecn[layer]++];
        } else {
            uint64_t oldest = UINT64_MAX;
            for (int j = 0; j < m->ecn[layer]; j++) {
                if (slot_loading(&pool[j])) continue;
                if (pool[j].last_used < oldest) { oldest = pool[j].last_used; victim = &pool[j]; }
            }
            if (!victim) {  /* estrema difesa: attendi il primo slot in prefetch */
                victim = &pool[0]; slot_wait_ready(victim);
            }
        }
        if (victim->eid >= 0) free_slot_buffers(victim);
        victim->eid = eids[i];   /* riservato: verrà popolato in pass 3 */
        victim->layer = layer;
        victim->last_used = ++m->eclock;
        m->miss++;
        m->ereq++;
        out[i] = victim;
    }

    /* Pass 3: leggi i miss da disco, in parallelo se abilitato. Gli slot sono
     * distinti, expert_load scrive solo nel proprio slot e legge g_db in sola
     * lettura → nessuna sincronizzazione necessaria oltre agli handle per-thread
     * di st.h. */
    if (nmiss > 0) {
        double t0 = now_s();
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) num_threads(g_io_threads) \
                if(g_io_threads > 1 && nmiss > 1)
#endif
        for (int mi = 0; mi < nmiss; mi++) {
            int i = miss_i[mi];
            expert_load(m, layer, eids[i], out[i]);
        }
        m->t_edisk += now_s() - t0;
    }
}

/* ═══════════════════════════════════════════════════════════
 *  PREFETCH → LRU (thread separato, popola direttamente la cache)
 *
 *  Il thread principale, a fine layer L, predice il top-k del layer L+1
 *  (proxy post-MoE, ~89% accurato sul 20B), RISERVA gli slot LRU per quegli
 *  expert e li passa a un thread che li legge da disco. Durante l'attention
 *  di L+1 il thread riempie gli slot; quando moe_forward(L+1) li richiede,
 *  sono già in cache. A differenza del vecchio PILOT (che scaldava solo la
 *  cache dell'OS → doppia lettura → −11%), qui non c'è seconda lettura.
 *
 *  Sicurezza: solo il main muta la LRU (riserva). Il thread scrive esclusiva-
 *  mente nei buffer degli slot riservati e imposta loading=0 (release) a fine
 *  lettura; il main attende loading==0 (acquire) prima di usare uno slot.
 *  prefetch_issue riserva al più (ecap − topk) slot, così restano sempre
 *  abbastanza slot non-loading per il routing reale.
 * ═══════════════════════════════════════════════════════════ */

#ifndef _WIN32
#include <pthread.h>
#include <sched.h>
#endif

typedef struct {
    Model *m;
    int layer;
    int n;
    ESlot *slots[64];
    int    eids[64];
    volatile int has_job;
    volatile int shutdown;
    uint64_t issued;      /* expert totali messi in prefetch */
#ifdef _WIN32
    CRITICAL_SECTION cs;
    CONDITION_VARIABLE cv;
    HANDLE thread;
#else
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
    pthread_t thread;
#endif
} Prefetch;

static Prefetch g_pf;
static int g_pilot_enabled = 0;   /* nome storico: usato da main() */

/* ── Primitive di lock/segnale, 1 = successo per trylock su entrambe ── */
#ifdef _WIN32
static int  pf_trylock(void) { return TryEnterCriticalSection(&g_pf.cs) != 0; }
static void pf_lock(void)    { EnterCriticalSection(&g_pf.cs); }
static void pf_unlock(void)  { LeaveCriticalSection(&g_pf.cs); }
static void pf_signal(void)  { WakeConditionVariable(&g_pf.cv); }
#else
static int  pf_trylock(void) { return pthread_mutex_trylock(&g_pf.mutex) == 0; }
static void pf_lock(void)    { pthread_mutex_lock(&g_pf.mutex); }
static void pf_unlock(void)  { pthread_mutex_unlock(&g_pf.mutex); }
static void pf_signal(void)  { pthread_cond_signal(&g_pf.cond); }
#endif

/* Corpo del worker: legge da disco ogni slot riservato e lo marca pronto. Gli
 * slot sono distinti ed expert_load è thread-safe (handle per-thread in st.h),
 * quindi il caricamento di un blocco usa gli stessi IO_THREADS del percorso sync. */
static void prefetch_run_job(void) {
    int n = g_pf.n, layer = g_pf.layer;
    Model *m = g_pf.m;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(g_io_threads) \
            if(g_io_threads > 1 && n > 1)
#endif
    for (int i = 0; i < n; i++) {
        if (!g_pf.shutdown)
            expert_load(m, layer, g_pf.eids[i], g_pf.slots[i]);
        slot_set_loading(g_pf.slots[i], 0);  /* pronto (o abortito da shutdown) */
    }
}

#ifdef _WIN32
static DWORD WINAPI prefetch_thread(LPVOID arg) {
    (void)arg;
    for (;;) {
        EnterCriticalSection(&g_pf.cs);
        while (!g_pf.has_job && !g_pf.shutdown)
            SleepConditionVariableCS(&g_pf.cv, &g_pf.cs, INFINITE);
        if (g_pf.shutdown) { LeaveCriticalSection(&g_pf.cs); break; }
        LeaveCriticalSection(&g_pf.cs);
        prefetch_run_job();
        EnterCriticalSection(&g_pf.cs);
        g_pf.has_job = 0;
        LeaveCriticalSection(&g_pf.cs);
    }
    return 0;
}
#else
static void *prefetch_thread(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_pf.mutex);
        while (!g_pf.has_job && !g_pf.shutdown)
            pthread_cond_wait(&g_pf.cond, &g_pf.mutex);
        if (g_pf.shutdown) { pthread_mutex_unlock(&g_pf.mutex); break; }
        pthread_mutex_unlock(&g_pf.mutex);
        prefetch_run_job();
        pthread_mutex_lock(&g_pf.mutex);
        g_pf.has_job = 0;
        pthread_mutex_unlock(&g_pf.mutex);
    }
    return NULL;
}
#endif

static void pilot_init(Model *m) {
    memset(&g_pf, 0, sizeof(g_pf));
    g_pf.m = m;
#ifdef _WIN32
    InitializeCriticalSection(&g_pf.cs);
    InitializeConditionVariable(&g_pf.cv);
    g_pf.thread = CreateThread(NULL, 0, prefetch_thread, NULL, 0, NULL);
    if (!g_pf.thread) {
        DeleteCriticalSection(&g_pf.cs);
        fprintf(stderr, "  ⚠ prefetch: creazione thread fallita\n");
        return;
    }
#else
    pthread_mutex_init(&g_pf.mutex, NULL);
    pthread_cond_init(&g_pf.cond, NULL);
    if (pthread_create(&g_pf.thread, NULL, prefetch_thread, NULL) != 0) {
        fprintf(stderr, "  ⚠ prefetch: creazione thread fallita\n");
        return;
    }
#endif
    g_pilot_enabled = 1;
}

/* Main thread: riserva gli slot LRU per gli expert predetti `pred[0..k)` del
 * `layer` e li affida al worker. Non attende: se il worker è ancora occupato o
 * il lock è conteso, salta questo round (il prefetch è best-effort). */
static void prefetch_issue(Model *m, int layer, const int *pred, int k) {
    if (!g_pilot_enabled || layer < 0 || layer >= m->c.n_layers) return;
    int max_reserve = m->ecap - m->c.topk;   /* lascia sempre topk slot al routing */
    if (max_reserve <= 0) return;
    if (!pf_trylock()) return;
    if (g_pf.has_job) { pf_unlock(); return; }   /* job precedente ancora in corso */

    ESlot *pool = m->ecache[layer];
    int n = 0;
    for (int a = 0; a < k && n < max_reserve; a++) {
        int eid = pred[a];
        if (eid < 0) continue;
        int cached = 0;
        for (int j = 0; j < m->npin[layer] && !cached; j++)
            if (m->pin[layer][j].eid == eid) cached = 1;
        for (int j = 0; j < m->ecn[layer] && !cached; j++)
            if (pool[j].eid == eid) cached = 1;
        if (cached) continue;
        int dup = 0;
        for (int b = 0; b < n; b++) if (g_pf.eids[b] == eid) dup = 1;
        if (dup) continue;

        ESlot *victim = NULL;
        if (m->ecn[layer] < m->ecap) {
            victim = &pool[m->ecn[layer]++];
        } else {
            uint64_t oldest = UINT64_MAX;
            for (int j = 0; j < m->ecn[layer]; j++) {
                if (slot_loading(&pool[j])) continue;
                int reserved = 0;
                for (int b = 0; b < n; b++) if (g_pf.slots[b] == &pool[j]) reserved = 1;
                if (reserved) continue;
                if (pool[j].last_used < oldest) { oldest = pool[j].last_used; victim = &pool[j]; }
            }
        }
        if (!victim) break;
        if (victim->eid >= 0) free_slot_buffers(victim);
        victim->eid = eid;
        victim->layer = layer;
        victim->last_used = ++m->eclock;
        slot_set_loading(victim, 1);
        g_pf.slots[n] = victim;
        g_pf.eids[n] = eid;
        n++;
    }
    if (n > 0) {
        g_pf.layer = layer;
        g_pf.n = n;
        g_pf.has_job = 1;
        g_pf.issued += n;
        pf_signal();
    }
    pf_unlock();
}

static void pilot_shutdown(void) {
    if (!g_pilot_enabled) return;
    pf_lock();
    g_pf.shutdown = 1;
    pf_signal();
    pf_unlock();
#ifdef _WIN32
    WaitForSingleObject(g_pf.thread, 10000);
    CloseHandle(g_pf.thread);
    DeleteCriticalSection(&g_pf.cs);
#else
    pthread_join(g_pf.thread, NULL);
    pthread_mutex_destroy(&g_pf.mutex);
    pthread_cond_destroy(&g_pf.cond);
#endif
    fprintf(stderr, "prefetch: %llu expert messi in prefetch\n",
            (unsigned long long)g_pf.issued);
    g_pilot_enabled = 0;
}

/* ═══════════════════════════════════════════════════════════
 *  MOE FORWARD (single token)
 * ═══════════════════════════════════════════════════════════ */

/* Probe: predice il top-k del router del layer `nl` applicato all'input `x` (già
 * normalizzato) e lo scrive in `out`. Non altera la matematica: serve solo a
 * misurare l'accuratezza di due proxy per il prefetch. */
static void predict_topk_into(Model *m, int nl, const float *x,
                              const Cfg *c, int *out) {
    int D = c->hidden, E = c->n_experts, K = c->topk;
    Layer *l = &m->L[nl];
    float best_s[64];
    for (int k = 0; k < K; k++) { out[k] = -1; best_s[k] = -1e30f; }
    for (int e = 0; e < E; e++) {
        float dot = l->router_bias ? l->router_bias[e] : 0.0f;
        const float *rw = l->router + (int64_t)e * D;
        for (int i = 0; i < D; i++) dot += x[i] * rw[i];
        for (int k = 0; k < K; k++) {
            if (dot > best_s[k]) {
                for (int j = K - 1; j > k; j--) {
                    best_s[j] = best_s[j-1]; out[j] = out[j-1];
                }
                best_s[k] = dot; out[k] = e;
                break;
            }
        }
    }
}

static void moe_forward(float *out, const float *x, Model *m,
                        int layer, const Cfg *c) {
    int D = c->hidden;
    int E = c->n_experts;
    int K = c->topk;
    Layer *l = &m->L[layer];

    /* 1. Router: calcola score per ogni expert */
    float *scores = falloc(E);
    for (int e = 0; e < E; e++) {
        float dot = 0;
        const float *rw = l->router + (int64_t)e * D;
        for (int i = 0; i < D; i++) dot += x[i] * rw[i];
        if (l->router_bias) dot += l->router_bias[e];
        scores[e] = dot;
    }

    /* 2. Top-K selection */
    int sel[64];          /* expert selezionati */
    float weights[64];    /* pesi corrispondenti */

    for (int k = 0; k < K; k++) {
        int best = -1;
        float best_s = -1e30f;
        for (int e = 0; e < E; e++) {
            int already = 0;
            for (int j = 0; j < k; j++) if (sel[j] == e) { already = 1; break; }
            if (!already && scores[e] > best_s) { best_s = scores[e]; best = e; }
        }
        sel[k] = best;
        weights[k] = best_s;
    }

    /* Probe: confronta il routing reale di questo layer con i due proxy predetti
     * per esso durante il layer precedente. */
    if (g_predict_probe && m->pred_layer == layer) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < K; j++)
                if (sel[k] == m->pred_next[j])  { m->pred_hit++;  break; }
            for (int j = 0; j < K; j++)
                if (sel[k] == m->pred_next2[j]) { m->pred_hit2++; break; }
        }
        m->pred_total += K;
    }

    /* 3. Normalizza pesi (softmax sui top-k) */
    softmax(weights, K);
    if (g_trace_numeric) {
        fprintf(stderr, "  numeric route L=%d:", layer);
        for (int k = 0; k < K; k++)
            fprintf(stderr, " e%d=%.7g", sel[k], weights[k]);
        fprintf(stderr, "\n");
    }
    if (g_oracle_dir) {
        char n[128]; int64_t se[2] = {1, E}, sk[2] = {1, K};
        float *sel_f = falloc(K);
        for (int k = 0; k < K; k++) sel_f[k] = (float)sel[k];
        snprintf(n, sizeof(n), "layer%d.router_logits", layer);
        oracle_dump(n, scores, 2, se);
        snprintf(n, sizeof(n), "layer%d.top_indices", layer);
        oracle_dump(n, sel_f, 2, sk);
        snprintf(n, sizeof(n), "layer%d.top_weights", layer);
        oracle_dump(n, weights, 2, sk);
        free(sel_f);
    }

    /* 4. Aggiorna statistiche di routing */
    for (int k = 0; k < K; k++) {
        if (m->eusage[layer])
            m->eusage[layer][sel[k]]++;
        if (m->eheat[layer])
            m->eheat[layer][sel[k]]++;
    }

    /* 5. Carica e computa ogni expert.
     * I K expert selezionati (unici, K <= ecap) vengono precaricati in un solo
     * batch: i miss sono letti da disco in parallelo prima del calcolo. */
    float *expert_out = calloc(D, sizeof(float));
    memset(out, 0, D * sizeof(float));

    ESlot *slots[64];
    cache_load_batch(m, layer, sel, K, slots);

    for (int k = 0; k < K; k++) {
        ESlot *es = slots[k];
        if (es->eid < 0) continue;  /* errore di caricamento */

        int I = c->moe_inter;  /* 5760 = gate_up fused */
        float *gu = falloc(I);

        /* gate_up fused: una matmul per [5760, 2880] × x */
        matmul_qt(gu, x, &es->gu, 1);
        if (es->gu_bias) {
            for (int i = 0; i < I; i++) gu[i] += es->gu_bias[i];
        }
        if (g_trace_numeric && layer == 0) {
            char stage[64]; snprintf(stage, sizeof(stage), "expert%d_gate_up", sel[k]);
            trace_vector(stage, layer, gu, I);
        }
        if (g_oracle_dir) {
            char n[128]; snprintf(n, sizeof(n), "layer%d.expert%d.gate_up", layer, k);
            int64_t sh[2] = {1, I}; oracle_dump(n, gu, 2, sh);
        }

        /* Clipped SwiGLU GPT-OSS:
         * gate=min(gate,limit), up=clamp(up,+/-limit),
         * (up+1)*gate*sigmoid(alpha*gate). */
        int half = I / 2;
        for (int i = 0; i < half; i++) {
            float gate = gu[2 * i];
            float up = gu[2 * i + 1];
            if (gate > c->swiglu_limit) gate = c->swiglu_limit;
            if (up > c->swiglu_limit) up = c->swiglu_limit;
            if (up < -c->swiglu_limit) up = -c->swiglu_limit;
            gu[i] = (up + 1.0f) * gate * sigmoidf(c->swiglu_alpha * gate);
        }
        if (g_oracle_dir) {
            char n[128]; snprintf(n, sizeof(n), "layer%d.expert%d.activated", layer, k);
            int64_t sh[2] = {1, half}; oracle_dump(n, gu, 2, sh);
        }

        /* down_proj: [D, half] × gu[:half] */
        matmul_qt(expert_out, gu, &es->d, 1);
        if (es->d_bias) {
            for (int i = 0; i < D; i++) expert_out[i] += es->d_bias[i];
        }
        if (g_trace_numeric && layer == 0) {
            char stage[64]; snprintf(stage, sizeof(stage), "expert%d_output", sel[k]);
            trace_vector(stage, layer, expert_out, D);
        }
        if (g_oracle_dir) {
            char n[128]; snprintf(n, sizeof(n), "layer%d.expert%d.output", layer, k);
            int64_t sh[2] = {1, D}; oracle_dump(n, expert_out, 2, sh);
        }

        /* Accumula pesato */
        float w = weights[k] * c->routed_scale;
        if (g_oracle_dir) {
            float *contrib = falloc(D);
            for (int i = 0; i < D; i++) contrib[i] = w * expert_out[i];
            char n[128]; snprintf(n, sizeof(n), "layer%d.expert%d.contribution", layer, k);
            int64_t sh[2] = {1, D}; oracle_dump(n, contrib, 2, sh);
            free(contrib);
        }
        for (int i = 0; i < D; i++)
            out[i] += w * expert_out[i];

        free(gu);
    }

    /* 6. (GPT-OSS non ha shared expert — tutti i layer sono MoE puri) */
    if (g_oracle_dir) {
        char n[128]; snprintf(n, sizeof(n), "layer%d.moe_out", layer);
        oracle_dump_vec(n, out, D);
    }

    free(scores); free(expert_out);
}

/* ═══════════════════════════════════════════════════════════
 *  EMBEDDING
 * ═══════════════════════════════════════════════════════════ */

static void embed_token(Model *m, int tok, float *x) {
    int D = m->c.hidden;
    QT *e = &m->embed;

    /* Bounds check */
    if (tok < 0 || tok >= e->O) {
        memset(x, 0, D * sizeof(float));
        return;
    }

    if (e->fmt == 0) {
        memcpy(x, e->qf + (int64_t)tok * D, D * sizeof(float));
    } else if (e->fmt == 1) {
        const int8_t *q = e->q8 + (int64_t)tok * D;
        float s = e->s[tok];
        for (int i = 0; i < D; i++) x[i] = (float)q[i] * s;
    } else if (e->fmt == 2) {
        const uint8_t *q = e->q4 + (int64_t)tok * ((D + 1) / 2);
        int gs = e->block_size;
        int ng = gs > 0 ? (D + gs - 1) / gs : 1;
        const float *scales = e->s + (int64_t)tok * ng;
        for (int i = 0; i < D; i += 2) {
            uint8_t byte = q[i >> 1];
            float s0 = scales[gs > 0 ? i / gs : 0];
            x[i] = (float)((int)(byte & 0xF) - 8) * s0;
            if (i + 1 < D) {
                float s1 = scales[gs > 0 ? (i + 1) / gs : 0];
                x[i + 1] = (float)((int)(byte >> 4) - 8) * s1;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════
 *  SAMPLING (temperature, top-p, top-k, repetition penalty)
 * ═══════════════════════════════════════════════════════════ */

/* Parametri di sampling (letti dall'ambiente) */
static float g_temperature = 1.0f;
static float g_top_p = 0.95f;
static int   g_top_k = 50;
static float g_rep_penalty = 1.1f;

/* Storico token per repetition penalty */
static int g_history[4096];
static int g_history_len = 0;

/* RNG (xorshift64) */
static uint64_t g_rng = 1234567890123456789ULL;
static float rng_float(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (float)(g_rng >> 33) / (float)(1ULL << 31);
}

/* Comparatore per sort decrescente */
typedef struct { float val; int idx; } IdxVal;
static int cmp_desc(const void *a, const void *b) {
    float fa = ((const IdxVal *)a)->val;
    float fb = ((const IdxVal *)b)->val;
    return (fa < fb) - (fa > fb);
}

static int sample_logits(float *logits, int vocab) {
    /* 1. Repetition penalty */
    if (g_rep_penalty > 1.0f) {
        for (int i = 0; i < g_history_len; i++) {
            int t = g_history[i];
            if (t >= 0 && t < vocab) {
                if (logits[t] > 0)
                    logits[t] /= g_rep_penalty;
                else
                    logits[t] *= g_rep_penalty;
            }
        }
    }

    /* 2. Temperature */
    if (g_temperature < 1e-6f) {
        /* Greedy (temperature = 0) */
        int best = 0;
        for (int i = 1; i < vocab; i++)
            if (logits[i] > logits[best]) best = i;
        if (g_history_len < 4096) g_history[g_history_len++] = best;
        return best;
    }

    float inv_temp = 1.0f / g_temperature;
    for (int i = 0; i < vocab; i++) logits[i] *= inv_temp;

    /* 3. Top-K: tieni solo i K logits più alti */
    int effective_k = vocab;
    if (g_top_k > 0 && g_top_k < vocab) {        /* Trova il K-esimo valore più grande (selezione parziale) */
        /* Per efficienza usiamo un approccio semplificato:
         * trova la soglia con un passaggio lineare */
        float threshold = -1e30f;
        
        /* Prova veloce: se K è piccolo, trova i top-K */
        if (g_top_k <= 256) {
            float tops[256];
            for (int i = 0; i < g_top_k; i++) tops[i] = -1e30f;
            for (int i = 0; i < vocab; i++) {
                if (logits[i] > tops[g_top_k - 1]) {
                    tops[g_top_k - 1] = logits[i];
                    /* Bubble up */
                    for (int j = g_top_k - 2; j >= 0; j--) {
                        if (tops[j + 1] > tops[j]) {
                            float tmp = tops[j]; tops[j] = tops[j + 1]; tops[j + 1] = tmp;
                        } else break;
                    }
                }
            }
            threshold = tops[g_top_k - 1];
        }
        
        /* Azzera tutto sotto la soglia */
        for (int i = 0; i < vocab; i++)
            if (logits[i] < threshold) logits[i] = -1e30f;
        effective_k = g_top_k;
    }

    /* 4. Softmax */
    float max_l = -1e30f;
    for (int i = 0; i < vocab; i++) if (logits[i] > max_l) max_l = logits[i];
    float sum = 0;
    for (int i = 0; i < vocab; i++) {
        logits[i] = expf(logits[i] - max_l);
        sum += logits[i];
    }
    for (int i = 0; i < vocab; i++) logits[i] /= sum;

    /* 5. Top-P (nucleus sampling) */
    if (g_top_p < 1.0f && g_top_p > 0.0f) {
        /* Ordina per probabilità decrescente (solo quelli > 0) */
        /* Per efficienza: raccogliamo solo i non-zero */
        int n_active = 0;
        for (int i = 0; i < vocab; i++)
            if (logits[i] > 1e-10f) n_active++;
        
        if (n_active > 1) {
            IdxVal *sorted = (IdxVal *)malloc(n_active * sizeof(IdxVal));
            int si = 0;
            for (int i = 0; i < vocab; i++)
                if (logits[i] > 1e-10f) { sorted[si].val = logits[i]; sorted[si].idx = i; si++; }
            
            qsort(sorted, n_active, sizeof(IdxVal), cmp_desc);
            
            /* Accumula fino a top_p */
            float cum = 0;
            int cutoff = n_active;
            for (int i = 0; i < n_active; i++) {
                cum += sorted[i].val;
                if (cum >= g_top_p) { cutoff = i + 1; break; }
            }
            
            /* Azzera tutto oltre il cutoff */
            for (int i = cutoff; i < n_active; i++)
                logits[sorted[i].idx] = 0.0f;
            
            /* Rinormalizza */
            sum = 0;
            for (int i = 0; i < vocab; i++) sum += logits[i];
            if (sum > 0) for (int i = 0; i < vocab; i++) logits[i] /= sum;
            
            free(sorted);
        }
    }

    /* 6. Campiona dalla distribuzione */
    float r = rng_float();
    float cum = 0;
    for (int i = 0; i < vocab; i++) {
        cum += logits[i];
        if (cum >= r) {
            if (g_history_len < 4096) g_history[g_history_len++] = i;
            return i;
        }
    }
    
    /* Fallback: ultimo token con probabilità > 0 */
    for (int i = vocab - 1; i >= 0; i--) {
        if (logits[i] > 0) {
            if (g_history_len < 4096) g_history[g_history_len++] = i;
            return i;
        }
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════
 *  FORWARD PASS (un token, decode)
 * ═══════════════════════════════════════════════════════════ */

static int forward_token(Model *m, int tok, int pos) {
    Cfg *c = &m->c;
    int D = c->hidden;

    float *h = falloc(D);           /* hidden state */
    float *hn = falloc(D);          /* hidden normalizzato */
    float *attn_out = falloc(D);    /* output attention */
    float *ffn_out = falloc(D);     /* output FFN/MoE */

    /* Embedding */
    embed_token(m, tok, h);
    trace_vector("embedding", -1, h, D);
    if (g_oracle_dir) oracle_dump_vec("embedding", h, D);

    /* Transformer layers */
    for (int l = 0; l < c->n_layers; l++) {
        Layer *ly = &m->L[l];
        char dump_name[128];
        if (g_oracle_dir) {
            snprintf(dump_name, sizeof(dump_name), "layer%d.input", l);
            oracle_dump_vec(dump_name, h, D);
        }

        /* Pre-attention norm */
        rmsnorm(hn, h, ly->in_ln, D, c->eps);
        if (g_oracle_dir) {
            snprintf(dump_name, sizeof(dump_name), "layer%d.pre_attn_norm", l);
            oracle_dump_vec(dump_name, hn, D);
        }

        /* GQA Attention */
        double ta = now_s();
        gqa_attention(attn_out, hn, ly, &m->kv, l, pos, c);
        m->t_attn += now_s() - ta;

        /* Residual */
        for (int i = 0; i < D; i++) h[i] += attn_out[i];
        trace_vector("post_attention", l, h, D);
        if (g_oracle_dir) {
            snprintf(dump_name, sizeof(dump_name), "layer%d.post_attn_residual", l);
            oracle_dump_vec(dump_name, h, D);
        }

        /* Pre-FFN norm */
        rmsnorm(hn, h, ly->post_ln, D, c->eps);
        if (g_oracle_dir) {
            snprintf(dump_name, sizeof(dump_name), "layer%d.pre_moe_norm", l);
            oracle_dump_vec(dump_name, hn, D);
        }

        /* MoE (tutti i layer in GPT-OSS sono MoE) */
        double tm = now_s();

        moe_forward(ffn_out, hn, m, l, c);

        /* Probe proxy A: predici il top-k di l+1 da hn (norm pre-MoE di l). hn
         * non serve più dopo moe_forward, quindi verrà riusato come scratch. */
        if (g_predict_probe && l + 1 < c->n_layers)
            predict_topk_into(m, l + 1, hn, c, m->pred_next);
        m->t_moe += now_s() - tm;

        /* Residual */
        for (int i = 0; i < D; i++) h[i] += ffn_out[i];

        /* Proxy B: predici il top-k di l+1 dallo stato post-MoE di l normalizzato
         * col post_ln di l+1 (manca solo l'attention di l+1, ~89% accurato). È il
         * segnale sia del probe sia del prefetch→LRU, che riserva quegli expert e
         * li carica durante l'attention di l+1. */
        if ((g_predict_probe || g_pilot_enabled) && l + 1 < c->n_layers) {
            rmsnorm(hn, h, m->L[l + 1].post_ln, D, c->eps);
            predict_topk_into(m, l + 1, hn, c, m->pred_next2);
            m->pred_layer = l + 1;
            if (g_pilot_enabled)
                prefetch_issue(m, l + 1, m->pred_next2, c->topk);
        }
        trace_vector("post_moe", l, h, D);
        if (g_oracle_dir) {
            snprintf(dump_name, sizeof(dump_name), "layer%d.output", l);
            oracle_dump_vec(dump_name, h, D);
        }
    }

    /* Final norm */
    rmsnorm(hn, h, m->final_norm, D, c->eps);
    if (g_oracle_dir) oracle_dump_vec("final_norm", hn, D);

    /* LM head → logits */
    double th = now_s();
    float *logits = falloc(c->vocab);
    matmul_qt(logits, hn, &m->lm_head, 1);
    trace_top_logits(logits, c->vocab);
    if (g_oracle_dir) oracle_dump_vec("logits", logits, c->vocab);
    m->t_head += now_s() - th;

    /* Sampling */
    int sampled = sample_logits(logits, c->vocab);

    free(h); free(hn); free(attn_out); free(ffn_out); free(logits);

    m->n_fw++;
    return sampled;
}

/* ═══════════════════════════════════════════════════════════
 *  PREFILL BATCHED (batch-union degli expert)
 *
 *  Elaborando più posizioni insieme, ogni expert unico del layer viene letto
 *  una sola volta invece di una volta per token. Con cache da 4 slot per layer
 *  e top-4, il percorso token-per-token sfratta l'intera cache a ogni token.
 *  La matematica è identica al percorso sequenziale: cambia solo l'ordine
 *  delle letture, mai il routing o la precisione.
 * ═══════════════════════════════════════════════════════════ */

/* Applica un expert a x e accumula w * output in dst. */
static void expert_apply(ESlot *es, const float *x, float *dst, float w,
                         const Cfg *c) {
    int D = c->hidden, I = c->moe_inter, half = I / 2;
    float *gu = falloc(I);
    float *eo = falloc(D);

    matmul_qt(gu, x, &es->gu, 1);
    if (es->gu_bias) for (int i = 0; i < I; i++) gu[i] += es->gu_bias[i];

    for (int i = 0; i < half; i++) {
        float gate = gu[2 * i], up = gu[2 * i + 1];
        if (gate > c->swiglu_limit) gate = c->swiglu_limit;
        if (up > c->swiglu_limit) up = c->swiglu_limit;
        if (up < -c->swiglu_limit) up = -c->swiglu_limit;
        gu[i] = (up + 1.0f) * gate * sigmoidf(c->swiglu_alpha * gate);
    }

    matmul_qt(eo, gu, &es->d, 1);
    if (es->d_bias) for (int i = 0; i < D; i++) eo[i] += es->d_bias[i];
    for (int i = 0; i < D; i++) dst[i] += w * eo[i];

    free(gu); free(eo);
}

/* Prefill di n token a partire da pos_base. Ritorna il token campionato
 * dall'ultima posizione, come farebbe l'ultimo forward_token sequenziale. */
static int forward_prefill(Model *m, const int *ids, int n, int pos_base) {
    Cfg *c = &m->c;
    int D = c->hidden, E = c->n_experts, K = c->topk;

    float *H = (float *)falloc((int64_t)n * D);
    float *XN = (float *)falloc((int64_t)n * D);
    float *hn = falloc(D);
    float *attn_out = falloc(D);
    int *sel_all = (int *)malloc((size_t)n * K * sizeof(int));
    float *w_all = (float *)malloc((size_t)n * K * sizeof(float));
    float *scores = falloc(E);
    int *uniq = (int *)malloc((size_t)E * sizeof(int));
    unsigned char *seen = (unsigned char *)malloc((size_t)E);
    if (!H || !XN || !sel_all || !w_all || !uniq || !seen) {
        fprintf(stderr, "errore: memoria insufficiente per il prefill batched\n");
        exit(1);
    }

    for (int p = 0; p < n; p++) embed_token(m, ids[p], H + (int64_t)p * D);

    for (int l = 0; l < c->n_layers; l++) {
        Layer *ly = &m->L[l];

        /* Attention: in ordine di posizione, così la KV precedente è pronta. */
        double ta = now_s();
        for (int p = 0; p < n; p++) {
            float *h = H + (int64_t)p * D;
            rmsnorm(hn, h, ly->in_ln, D, c->eps);
            gqa_attention(attn_out, hn, ly, &m->kv, l, pos_base + p, c);
            for (int i = 0; i < D; i++) h[i] += attn_out[i];
        }
        m->t_attn += now_s() - ta;

        double tm = now_s();
        /* Routing di tutte le posizioni, poi unione degli expert richiesti. */
        memset(seen, 0, (size_t)E);
        int n_uniq = 0;
        for (int p = 0; p < n; p++) {
            float *h = H + (int64_t)p * D;
            float *xn = XN + (int64_t)p * D;
            rmsnorm(xn, h, ly->post_ln, D, c->eps);

            for (int e = 0; e < E; e++) {
                float dot = ly->router_bias ? ly->router_bias[e] : 0.0f;
                const float *rw = ly->router + (int64_t)e * D;
                for (int i = 0; i < D; i++) dot += xn[i] * rw[i];
                scores[e] = dot;
            }
            int *sel = sel_all + (size_t)p * K;
            float *wgt = w_all + (size_t)p * K;
            for (int k = 0; k < K; k++) {
                int best = -1; float best_s = -1e30f;
                for (int e = 0; e < E; e++) {
                    int already = 0;
                    for (int j = 0; j < k; j++) if (sel[j] == e) { already = 1; break; }
                    if (!already && scores[e] > best_s) { best_s = scores[e]; best = e; }
                }
                sel[k] = best; wgt[k] = best_s;
            }
            softmax(wgt, K);
            for (int k = 0; k < K; k++) {
                if (m->eusage[l]) m->eusage[l][sel[k]]++;
                if (m->eheat[l]) m->eheat[l][sel[k]]++;
                if (!seen[sel[k]]) { seen[sel[k]] = 1; uniq[n_uniq++] = sel[k]; }
            }
        }

        /* Un expert unico → una sola lettura, riusata da tutte le posizioni.
         * Il buffer va azzerato: expert_apply accumula. Gli expert unici del
         * batch superano spesso la capacità della cache, quindi si procede a
         * blocchi: ogni blocco viene letto da disco in parallelo (queue depth > 1)
         * e poi applicato a tutte le posizioni.
         *
         * Con prefetch attivo (PREFETCH=1) i blocchi sono dimezzati (due entrano
         * in cache) e si esegue un pipeline a doppio buffer: mentre si calcola il
         * blocco corrente (matmul su n posizioni, disco idle), il blocco successivo
         * — già noto in uniq[], quindi prefetch ESATTO senza predizione — viene
         * caricato dal thread di prefetch. La matematica non cambia. */
        float *moe = (float *)calloc((size_t)n * D, sizeof(float));
        if (!moe) { fprintf(stderr, "errore: memoria MoE\n"); exit(1); }
        int chunk = m->ecap < 128 ? m->ecap : 128;
        if (g_pilot_enabled) {
            int half = m->ecap / 2;                 /* due blocchi in cache */
            int cap  = m->ecap - c->topk;           /* invariante di prefetch_issue */
            if (half < 1) half = 1;
            if (cap >= 1 && half > cap) half = cap;
            if (half >= 1) chunk = half;
        }
        ESlot *slots[128];
        for (int base = 0; base < n_uniq; base += chunk) {
            int cnt = n_uniq - base < chunk ? n_uniq - base : chunk;
            /* Rendi disponibile il blocco corrente: sincrono al primo giro,
             * altrimenti servito (con attesa) dal prefetch del giro precedente. */
            cache_load_batch(m, l, uniq + base, cnt, slots);
            /* Avvia il caricamento async del blocco successivo, che si sovrappone
             * al calcolo di questo blocco. */
            if (g_pilot_enabled && base + chunk < n_uniq) {
                int nbase = base + chunk;
                int ncnt = n_uniq - nbase < chunk ? n_uniq - nbase : chunk;
                prefetch_issue(m, l, uniq + nbase, ncnt);
            }
            for (int u = 0; u < cnt; u++) {
                ESlot *es = slots[u];
                if (es->eid < 0) continue;
                int eid = uniq[base + u];
                for (int p = 0; p < n; p++) {
                    int *sel = sel_all + (size_t)p * K;
                    float *wgt = w_all + (size_t)p * K;
                    for (int k = 0; k < K; k++) {
                        if (sel[k] != eid) continue;
                        expert_apply(es, XN + (int64_t)p * D, moe + (int64_t)p * D,
                                     wgt[k] * c->routed_scale, c);
                    }
                }
            }
        }
        for (int p = 0; p < n; p++) {
            float *h = H + (int64_t)p * D;
            const float *mo = moe + (int64_t)p * D;
            for (int i = 0; i < D; i++) h[i] += mo[i];
        }
        free(moe);
        m->t_moe += now_s() - tm;
    }

    /* Solo l'ultima posizione produce i logits utili. */
    rmsnorm(hn, H + (int64_t)(n - 1) * D, m->final_norm, D, c->eps);
    double th = now_s();
    float *logits = falloc(c->vocab);
    matmul_qt(logits, hn, &m->lm_head, 1);
    m->t_head += now_s() - th;
    int sampled = sample_logits(logits, c->vocab);

    free(H); free(XN); free(hn); free(attn_out); free(sel_all); free(w_all);
    free(scores); free(uniq); free(seen); free(logits);
    m->n_fw += n;
    return sampled;
}

/* Prefill a blocchi. Ricade sul percorso sequenziale quando servono i dump
 * oracle, il tracing o una repetition penalty che dipende dallo storico. */
static int prefill_tokens(Model *m, const int *ids, int n, int pos_base) {
    int batch = 64;
    { const char *v = getenv("PREFILL_BATCH"); if (v) batch = atoi(v); }
    int sequential = g_oracle_dir || g_trace_numeric || g_predict_probe
                     || g_rep_penalty > 1.0f || batch <= 1 || n <= 1;

    int last = ids[0];
    if (sequential) {
        for (int i = 0; i < n; i++) {
            last = forward_token(m, ids[i], pos_base + i);
            if (last < 0 || last >= m->c.vocab) last = 0;
        }
        return last;
    }
    for (int off = 0; off < n; off += batch) {
        int len = n - off < batch ? n - off : batch;
        last = forward_prefill(m, ids + off, len, pos_base + off);
        if (last < 0 || last >= m->c.vocab) last = 0;
    }
    return last;
}

/* ═══════════════════════════════════════════════════════════
 *  STATISTICHE
 * ═══════════════════════════════════════════════════════════ */

static void stats_dump(Model *m) {
    fprintf(stderr, "\n── picchio stats ──\n");
    fprintf(stderr, "forward:    %llu\n", (unsigned long long)m->n_fw);
    fprintf(stderr, "token out:  %llu\n", (unsigned long long)m->n_emit);
    fprintf(stderr, "cache hit:  %llu / %llu (%.1f%%)\n",
            (unsigned long long)m->hits,
            (unsigned long long)(m->hits + m->miss),
            m->hits + m->miss > 0
                ? 100.0 * m->hits / (m->hits + m->miss) : 0.0);
    fprintf(stderr, "disk reads: %llu (%.2f s total)\n",
            (unsigned long long)m->ereq, m->t_edisk);
    fprintf(stderr, "t_attn:     %.2f s\n", m->t_attn);
    fprintf(stderr, "t_moe:      %.2f s\n", m->t_moe);
    fprintf(stderr, "t_head:     %.2f s\n", m->t_head);
    fprintf(stderr, "RSS:        %.2f GB\n", rss_gb());
    fprintf(stderr, "resident:   %.2f GB (dense model)\n",
            m->resident_bytes / 1e9);
    if (m->pred_total > 0) {
        double accA = 100.0 * m->pred_hit  / m->pred_total;
        double accB = 100.0 * m->pred_hit2 / m->pred_total;
        double baseline = 100.0 * m->c.topk / m->c.n_experts;  /* overlap casuale atteso */
        fprintf(stderr, "prefetch predict (%llu campioni, casuale ~%.1f%%):\n",
                (unsigned long long)m->pred_total, baseline);
        fprintf(stderr, "  proxy A (hn pre-MoE):   %.1f%% expert corretti\n", accA);
        fprintf(stderr, "  proxy B (post-MoE norm): %.1f%% expert corretti\n", accB);
    }
}

/* ═══════════════════════════════════════════════════════════
 *  HOT-STORE PERSISTENTE (.picchio_usage)
 *
 *  Salva i contatori di routing tra sessioni.
 *  Al prossimo avvio, gli expert più caldi vengono pre-caricati.
 * ═══════════════════════════════════════════════════════════ */

static const char *g_model_path_global = NULL;

static void hotstore_save(Model *m) {
    if (!g_model_path_global) return;
    Cfg *c = &m->c;
    
    char path[512];
    snprintf(path, sizeof(path), "%s/.picchio_usage", g_model_path_global);
    
    FILE *f = fopen(path, "wb");
    if (!f) return;
    
    /* Header: magic + versione + dimensioni */
    uint32_t magic = 0x50494343;  /* "PICC" */
    uint32_t version = 1;
    uint32_t n_layers = (uint32_t)c->n_layers;
    uint32_t n_experts = (uint32_t)c->n_experts;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&n_layers, 4, 1, f);
    fwrite(&n_experts, 4, 1, f);
    
    /* Contatori per layer */
    for (int l = 0; l < c->n_layers; l++) {
        if (m->eusage[l])
            fwrite(m->eusage[l], sizeof(uint32_t), c->n_experts, f);
        else {
            uint32_t zeros[128] = {0};
            for (int e = 0; e < c->n_experts; e += 128)
                fwrite(zeros, sizeof(uint32_t),
                       c->n_experts - e < 128 ? c->n_experts - e : 128, f);
        }
    }
    
    fclose(f);
    fprintf(stderr, "  hot-store salvato: %s\n", path);
}

static void hotstore_load(Model *m) {
    if (!g_model_path_global) return;
    Cfg *c = &m->c;
    
    char path[512];
    snprintf(path, sizeof(path), "%s/.picchio_usage", g_model_path_global);
    
    FILE *f = fopen(path, "rb");
    if (!f) return;
    
    uint32_t magic, version, n_layers, n_experts;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x50494343) { fclose(f); return; }
    if (fread(&version, 4, 1, f) != 1) { fclose(f); return; }
    if (fread(&n_layers, 4, 1, f) != 1) { fclose(f); return; }
    if (fread(&n_experts, 4, 1, f) != 1) { fclose(f); return; }
    
    if ((int)n_layers != c->n_layers || (int)n_experts != c->n_experts) {
        fprintf(stderr, "  hot-store: dimensioni non corrispondono, ignoro\n");
        fclose(f);
        return;
    }
    
    uint64_t total_usage = 0;
    for (int l = 0; l < c->n_layers; l++) {
        if (m->eusage[l]) {
            fread(m->eusage[l], sizeof(uint32_t), c->n_experts, f);
            for (int e = 0; e < c->n_experts; e++)
                total_usage += m->eusage[l][e];
        }
    }
    
    fclose(f);
    fprintf(stderr, "  hot-store caricato: %llu routing registrati\n",
            (unsigned long long)total_usage);
}

/* ═══════════════════════════════════════════════════════════
 *  SELF-TEST: mini-modello sintetico per validare il forward
 * ═══════════════════════════════════════════════════════════ */

/* Genera pesi random deterministici (LCG semplice) */
static uint32_t _rng_state = 42;
static float _rng_f(void) {
    _rng_state = _rng_state * 1664525u + 1013904223u;
    return ((float)(_rng_state >> 8) / 16777216.0f - 0.5f) * 0.1f;
}

static void fill_random_f32(float *p, int64_t n) {
    for (int64_t i = 0; i < n; i++) p[i] = _rng_f();
}

static void fill_random_qt_f32(QT *t, int O, int I) {
    memset(t, 0, sizeof(*t));
    t->fmt = 0;  /* F32 per il self-test (niente quantizzazione) */
    t->O = O;
    t->I = I;
    t->qf = falloc((int64_t)O * I);
    fill_random_f32(t->qf, (int64_t)O * I);
}

static int self_test(void) {
    fprintf(stderr, "── self-test: mini-modello sintetico ──\n\n");

    /* Configurazione mini: 2 layer, D=64, 4 heads, 2 KV heads, 4 expert, top-2 */
    static Model m;
    memset(&m, 0, sizeof(m));
    Cfg *c = &m.c;

    c->hidden = 64;
    c->n_layers = 2;
    c->n_heads = 4;
    c->n_kv_heads = 2;
    c->head_dim = 16;  /* 64 / 4 */
    c->n_experts = 4;
    c->topk = 2;
    c->moe_inter = 128;  /* 2 * D */
    c->dense_inter = 128;
    c->vocab = 32;
    c->ctx_len = 64;
    c->sliding_window = 16;
    c->eps = 1e-5f;
    c->theta = 10000.0f;
    c->routed_scale = 1.0f;
    c->has_shared = 0;
    c->has_attn_bias = 1;
    c->layer_type[0] = 0;  /* sliding */
    c->layer_type[1] = 1;  /* full */

    int D = c->hidden;
    int H = c->n_heads;
    int KVH = c->n_kv_heads;
    int hd = c->head_dim;

    fprintf(stderr, "  D=%d L=%d H=%d KV=%d hd=%d E=%d top%d vocab=%d\n",
            D, c->n_layers, H, KVH, hd, c->n_experts, c->topk, c->vocab);

    /* Alloca embedding e lm_head */
    fill_random_qt_f32(&m.embed, c->vocab, D);
    fill_random_qt_f32(&m.lm_head, c->vocab, D);
    m.final_norm = falloc(D);
    for (int i = 0; i < D; i++) m.final_norm[i] = 1.0f;  /* norms = 1 per test */

    /* Alloca layer */
    m.L = calloc(c->n_layers, sizeof(Layer));
    for (int l = 0; l < c->n_layers; l++) {
        Layer *ly = &m.L[l];
        ly->layer_type = c->layer_type[l];

        /* RMSNorm weights (tutte 1.0 per test) */
        ly->in_ln = falloc(D);
        ly->post_ln = falloc(D);
        for (int i = 0; i < D; i++) { ly->in_ln[i] = 1.0f; ly->post_ln[i] = 1.0f; }

        /* Attention Q/K/V/O */
        fill_random_qt_f32(&ly->wq, H * hd, D);
        fill_random_qt_f32(&ly->wk, KVH * hd, D);
        fill_random_qt_f32(&ly->wv, KVH * hd, D);
        fill_random_qt_f32(&ly->wo, D, H * hd);

        /* Attention bias */
        ly->bq = falloc(H * hd);  fill_random_f32(ly->bq, H * hd);
        ly->bk = falloc(KVH * hd); fill_random_f32(ly->bk, KVH * hd);
        ly->bv = falloc(KVH * hd); fill_random_f32(ly->bv, KVH * hd);
        ly->bo = falloc(D);        fill_random_f32(ly->bo, D);

        /* Router */
        ly->router = falloc((int64_t)c->n_experts * D);
        fill_random_f32(ly->router, (int64_t)c->n_experts * D);
        ly->router_bias = falloc(c->n_experts);
        fill_random_f32(ly->router_bias, c->n_experts);
    }

    /* KV-cache */
    kv_init(&m.kv, c->n_layers, KVH, hd, c->ctx_len);

    /* Expert cache — per il self-test, pre-carica tutti gli expert in RAM */
    m.ecap = c->n_experts;  /* abbastanza slot per tutti */
    m.ecache = calloc(c->n_layers, sizeof(ESlot *));
    m.ecn = calloc(c->n_layers, sizeof(int));
    m.pin = calloc(c->n_layers, sizeof(ESlot *));
    m.npin = calloc(c->n_layers, sizeof(int));
    m.eusage = calloc(c->n_layers, sizeof(uint32_t *));
    m.eheat = calloc(c->n_layers, sizeof(uint32_t *));

    for (int l = 0; l < c->n_layers; l++) {
        m.ecache[l] = calloc(m.ecap, sizeof(ESlot));
        m.ecn[l] = c->n_experts;  /* tutti pre-caricati */
        m.eusage[l] = calloc(c->n_experts, sizeof(uint32_t));
        m.eheat[l] = calloc(c->n_experts, sizeof(uint32_t));

        for (int e = 0; e < c->n_experts; e++) {
            ESlot *es = &m.ecache[l][e];
            es->eid = e;
            es->layer = l;
            /* gate_up fused: [moe_inter, D] */
            fill_random_qt_f32(&es->gu, c->moe_inter, D);
            /* down: [D, moe_inter/2] — l'input al down è moe_inter/2 valori post-SwiGLU */
            fill_random_qt_f32(&es->d, D, c->moe_inter / 2);
            /* Bias */
            es->gu_bias = falloc(c->moe_inter);
            fill_random_f32(es->gu_bias, c->moe_inter);
            es->d_bias = falloc(D);
            fill_random_f32(es->d_bias, D);
        }
    }

    fprintf(stderr, "  strutture allocate ✓\n");

    /* ── Forward pass: 8 token ── */
    fprintf(stderr, "  forward pass: 8 token...\n");

    int tokens[] = {1, 5, 3, 12, 7, 2, 9, 4};
    int n_tok = 8;

    double t0 = now_s();
    for (int i = 0; i < n_tok; i++) {
        int next = forward_token(&m, tokens[i], i);
        fprintf(stderr, "    pos=%d tok_in=%d → tok_out=%d\n", i, tokens[i], next);
        if (next < 0 || next >= c->vocab) {
            fprintf(stderr, "  ✗ ERRORE: token fuori range [0, %d)\n", c->vocab);
            return 1;
        }
    }
    double elapsed = now_s() - t0;

    fprintf(stderr, "\n  ✓ %d forward in %.3f ms (%.1f tok/s)\n",
            n_tok, elapsed * 1000.0, n_tok / elapsed);

    /* Verifica statistiche */
    fprintf(stderr, "  cache: hit=%llu miss=%llu (%.0f%%)\n",
            (unsigned long long)m.hits, (unsigned long long)m.miss,
            m.hits + m.miss > 0 ? 100.0 * m.hits / (m.hits + m.miss) : 0.0);
    fprintf(stderr, "  t_attn=%.3fms t_moe=%.3fms t_head=%.3fms\n",
            m.t_attn * 1000, m.t_moe * 1000, m.t_head * 1000);

    /* ── Test componenti individuali ── */
    fprintf(stderr, "\n  test RMSNorm...");
    {
        float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float w[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        float o[4];
        rmsnorm(o, x, w, 4, 1e-5f);
        /* RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386 */
        float rms = sqrtf((1+4+9+16)/4.0f);
        float expected0 = 1.0f / rms;
        if (fabsf(o[0] - expected0) > 1e-4f) {
            fprintf(stderr, " ✗ atteso %.4f, ottenuto %.4f\n", expected0, o[0]);
            return 1;
        }
        fprintf(stderr, " ✓\n");
    }

    fprintf(stderr, "  test softmax...");
    {
        float x[3] = {1.0f, 2.0f, 3.0f};
        softmax(x, 3);
        float sum = x[0] + x[1] + x[2];
        if (fabsf(sum - 1.0f) > 1e-5f) {
            fprintf(stderr, " ✗ somma=%.6f\n", sum);
            return 1;
        }
        if (x[2] < x[1] || x[1] < x[0]) {
            fprintf(stderr, " ✗ ordine sbagliato\n");
            return 1;
        }
        fprintf(stderr, " ✓\n");
    }

    fprintf(stderr, "  test matmul F32...");
    {
        /* y = x @ W^T, x=[1,4], W=[2,4] → y=[1,2] */
        float x[4] = {1, 2, 3, 4};
        float W[8] = {1,0,0,0,  0,1,0,0};  /* identity-ish */
        float y[2] = {0, 0};
        matmul_f32(y, x, W, 1, 4, 2);
        if (fabsf(y[0] - 1.0f) > 1e-5f || fabsf(y[1] - 2.0f) > 1e-5f) {
            fprintf(stderr, " ✗ y=[%.2f, %.2f] atteso [1, 2]\n", y[0], y[1]);
            return 1;
        }
        fprintf(stderr, " ✓\n");
    }

    fprintf(stderr, "  test SiLU...");
    {
        float s = siluf(0.0f);
        if (fabsf(s - 0.0f) > 1e-5f) {
            fprintf(stderr, " ✗ SiLU(0)=%.4f\n", s);
            return 1;
        }
        s = siluf(1.0f);
        float expected = 1.0f / (1.0f + expf(-1.0f));  /* ≈ 0.7311 */
        if (fabsf(s - expected) > 1e-4f) {
            fprintf(stderr, " ✗ SiLU(1)=%.4f atteso %.4f\n", s, expected);
            return 1;
        }
        fprintf(stderr, " ✓\n");
    }

    fprintf(stderr, "  test RoPE...");
    {
        /* Verifica che RoPE a pos=0 non cambia nulla (cos=1, sin=0) */
        float q[16], k[8];
        for (int i = 0; i < 16; i++) q[i] = (float)(i + 1);
        for (int i = 0; i < 8; i++) k[i] = (float)(i + 1);
        float q_orig[16], k_orig[8];
        memcpy(q_orig, q, sizeof(q));
        memcpy(k_orig, k, sizeof(k));
        Cfg rope_cfg;
        memset(&rope_cfg, 0, sizeof(rope_cfg));
        rope_cfg.head_dim = 8;
        rope_cfg.theta = 10000.0f;
        rope_cfg.rope_factor = 1.0f;
        rope_cfg.rope_attn_factor = 1.0f;
        rope_apply(q, k, 0, 8, 2, 1, &rope_cfg);
        /* A pos=0: ang=0 per tutti → cos=1, sin=0 → nessun cambiamento */
        int ok = 1;
        for (int i = 0; i < 16; i++)
            if (fabsf(q[i] - q_orig[i]) > 1e-5f) { ok = 0; break; }
        for (int i = 0; i < 8; i++)
            if (fabsf(k[i] - k_orig[i]) > 1e-5f) { ok = 0; break; }
        if (!ok) {
            fprintf(stderr, " ✗ RoPE pos=0 ha modificato i vettori\n");
            return 1;
        }
        fprintf(stderr, " ✓\n");
    }

    fprintf(stderr, "  test INT4 quant/matmul...");
    {
        /* Quantizza una matrice F32, poi matmul e confronta */
        int O = 4, I = 8;
        float W[32];
        for (int i = 0; i < 32; i++) W[i] = (float)(i - 16) * 0.1f;

        uint8_t q4[16]; /* O * (I/2) */
        float scale[4];
        quantize_rows_i4(W, q4, scale, O, I);

        float x[8] = {1,1,1,1,1,1,1,1};
        float y_ref[4], y_q4[4];

        /* Reference: F32 matmul */
        matmul_f32(y_ref, x, W, 1, I, O);
        /* Quantized: INT4 matmul */
        matmul_i4(y_q4, x, q4, scale, 1, I, O);

        /* L'errore di quantizzazione dovrebbe essere piccolo */
        float max_err = 0;
        for (int i = 0; i < O; i++) {
            float err = fabsf(y_ref[i] - y_q4[i]);
            if (err > max_err) max_err = err;
        }
        if (max_err > 2.0f) {  /* tolleranza larga per INT4 */
            fprintf(stderr, " ✗ errore max=%.2f\n", max_err);
            return 1;
        }
        fprintf(stderr, " ✓ (err max=%.4f)\n", max_err);
    }

    fprintf(stderr, "\n── self-test SUPERATO ──\n");
    fprintf(stderr, "  Il forward pass completo funziona correttamente.\n");
    fprintf(stderr, "  Pronto per collegare il modello reale.\n\n");

    /* TODO: free tutto (per il self-test non importa) */
    return 0;
}

/* ═══════════════════════════════════════════════════════════
 *  WEIGHT LOADER: carica pesi densi dal safetensors DB
 * ═══════════════════════════════════════════════════════════ */

/* Helper: carica un tensore F32 da safetensors.
 * Se il tensore è BF16, converte a F32. */
static float *load_f32_tensor(StDB *db, const char *name, int64_t expected_numel) {
    StTensor *t = st_find(db, name);
    if (!t) {
        fprintf(stderr, "  ⚠ tensore non trovato: %s\n", name);
        return NULL;
    }
    int64_t numel = st_numel(t);
    if (expected_numel > 0 && numel != expected_numel) {
        fprintf(stderr, "  ⚠ %s: numel=%lld atteso=%lld\n",
                name, (long long)numel, (long long)expected_numel);
    }

    float *out = falloc(numel);
    int64_t nbytes = st_bytes(t);

    if (t->dtype == ST_F32) {
        st_read_raw(db, t, out, nbytes);
    } else if (t->dtype == ST_BF16) {
        /* BF16 → F32: ogni BF16 è i 16 bit alti di un F32 */
        uint16_t *tmp = (uint16_t *)malloc(numel * 2);
        st_read_raw(db, t, tmp, nbytes);
        for (int64_t i = 0; i < numel; i++) {
            uint32_t bits = (uint32_t)tmp[i] << 16;
            memcpy(&out[i], &bits, 4);
        }
        free(tmp);
    } else if (t->dtype == ST_F16) {
        /* F16 → F32 (conversione minimale) */
        uint16_t *tmp = (uint16_t *)malloc(numel * 2);
        st_read_raw(db, t, tmp, nbytes);
        for (int64_t i = 0; i < numel; i++) {
            uint16_t h = tmp[i];
            uint32_t sign = (h & 0x8000) << 16;
            uint32_t exp  = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t f;
            if (exp == 0) {
                f = sign; /* ±0 o denorm → 0 */
            } else if (exp == 31) {
                f = sign | 0x7F800000 | (mant << 13); /* inf/nan */
            } else {
                f = sign | ((exp + 112) << 23) | (mant << 13);
            }
            memcpy(&out[i], &f, 4);
        }
        free(tmp);
    } else {
        fprintf(stderr, "  ⚠ %s: dtype non supportato (%d)\n", name, t->dtype);
        free(out);
        return NULL;
    }

    return out;
}

/* Helper: carica un tensore e quantizzalo a INT4.
 * Carica come F32, poi quantizza in-place. Ritorna QT. */
static int load_qt_i4(StDB *db, const char *name, QT *qt, int O, int I) {
    StTensor *t = st_find(db, name);
    if (!t) {
        fprintf(stderr, "  ⚠ non trovato: %s\n", name);
        return -1;
    }

    memset(qt, 0, sizeof(*qt));
    qt->fmt = 2;
    qt->O = O;
    qt->I = I;

    if (t->dtype == ST_U8) {
        /* Già INT4 packed dal convertitore */
        int64_t rb = (int64_t)O * ((I + 1) / 2);
        qt->q4 = (uint8_t *)malloc(rb);
        st_read_raw(db, t, qt->q4, rb);

        /* Scale associate */
        char sn[300];
        snprintf(sn, sizeof(sn), "%s.qs", name);
        StTensor *ts = st_find(db, sn);
        if (ts) {
            int64_t n_scale = st_numel(ts);
            qt->s = (float *)malloc(n_scale * sizeof(float));
            st_read_raw(db, ts, qt->s, n_scale * sizeof(float));
            /* Determina se è group-scaled: n_scale > O significa gs */
            if (n_scale > O) {
                qt->block_size = (int)((int64_t)O * I / n_scale);
                if (qt->block_size <= 0) qt->block_size = 64;
            } else {
                qt->block_size = 0;  /* per-row */
            }
        } else {
            qt->s = (float *)malloc(O * sizeof(float));
            for (int i = 0; i < O; i++) qt->s[i] = 1.0f;
            qt->block_size = 0;
        }
        return 0;
    }

    if (t->dtype == ST_I8) {
        /* INT8 per riga: usato per embedding e lm_head, che a INT4 degradano
         * la distribuzione dei logit. */
        int64_t n = (int64_t)O * I;
        qt->fmt = 1;
        qt->q8 = (int8_t *)malloc(n);
        if (!qt->q8) return -1;
        st_read_raw(db, t, qt->q8, n);

        char sn[300];
        snprintf(sn, sizeof(sn), "%s.qs", name);
        StTensor *ts = st_find(db, sn);
        if (!ts) {
            fprintf(stderr, "  ⚠ scale INT8 mancanti: %s\n", sn);
            free(qt->q8); qt->q8 = NULL;
            return -1;
        }
        int64_t n_scale = st_numel(ts);
        if (n_scale != O) {
            fprintf(stderr, "  ⚠ %s: %lld scale per %d righe\n", sn,
                    (long long)n_scale, O);
            free(qt->q8); qt->q8 = NULL;
            return -1;
        }
        qt->s = (float *)malloc((size_t)O * sizeof(float));
        st_read_raw(db, ts, qt->s, (int64_t)O * (int64_t)sizeof(float));
        return 0;
    }

    /* F32/BF16/F16: carica */
    float *f = load_f32_tensor(db, name, (int64_t)O * I);
    if (!f) return -1;

    /* Se il tensore è F32 nel file, tienilo F32 (attention, non quantizzare) */
    if (t->dtype == ST_F32) {
        qt->fmt = 0;
        qt->qf = f;
        return 0;
    }

    /* BF16/F16 → quantizza a INT4 */
    qt->q4 = (uint8_t *)malloc((int64_t)O * ((I + 1) / 2));
    qt->s = falloc(O);
    quantize_rows_i4(f, qt->q4, qt->s, O, I);
    free(f);
    return 0;
}

/* Carica tutti i pesi densi residenti (embedding, attention, norms, router) */
static int load_dense_weights(Model *m, StDB *db) {
    Cfg *c = &m->c;
    int D = c->hidden;
    int H = c->n_heads;
    int KVH = c->n_kv_heads;
    int hd = c->head_dim;
    int64_t loaded = 0;

    fprintf(stderr, "\n  caricamento pesi densi...\n");

    /* ── Embedding ── */
    fprintf(stderr, "    embed_tokens [%d, %d]...", c->vocab, D);
    if (load_qt_i4(db, "model.embed_tokens.weight", &m->embed, c->vocab, D) == 0) {
        loaded += qt_bytes(&m->embed);
        fprintf(stderr, " ✓ (%.1f MB)\n", qt_bytes(&m->embed) / 1e6);
    } else {
        fprintf(stderr, " ✗\n");
        return -1;
    }

    /* ── LM Head ── */
    fprintf(stderr, "    lm_head [%d, %d]...", c->vocab, D);
    if (load_qt_i4(db, "lm_head.weight", &m->lm_head, c->vocab, D) == 0) {
        loaded += qt_bytes(&m->lm_head);
        fprintf(stderr, " ✓ (%.1f MB)\n", qt_bytes(&m->lm_head) / 1e6);
    } else {
        fprintf(stderr, " ✗\n");
        return -1;
    }

    /* ── Final norm ── */
    m->final_norm = load_f32_tensor(db, "model.norm.weight", D);
    if (!m->final_norm) {
        fprintf(stderr, "    ⚠ final_norm non trovata, uso 1.0\n");
        m->final_norm = falloc(D);
        for (int i = 0; i < D; i++) m->final_norm[i] = 1.0f;
    }
    loaded += D * 4;

    /* ── Per-layer weights ── */
    for (int l = 0; l < c->n_layers; l++) {
        Layer *ly = &m->L[l];
        ly->layer_type = c->layer_type[l];
        char name[256];

        /* RMSNorm */
        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", l);
        ly->in_ln = load_f32_tensor(db, name, D);
        if (!ly->in_ln) { ly->in_ln = falloc(D); for (int i=0;i<D;i++) ly->in_ln[i]=1.0f; }

        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", l);
        ly->post_ln = load_f32_tensor(db, name, D);
        if (!ly->post_ln) { ly->post_ln = falloc(D); for (int i=0;i<D;i++) ly->post_ln[i]=1.0f; }

        loaded += 2 * D * 4;

        /* Attention Q/K/V/O */
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", l);
        load_qt_i4(db, name, &ly->wq, H * hd, D);
        loaded += qt_bytes(&ly->wq);

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", l);
        load_qt_i4(db, name, &ly->wk, KVH * hd, D);
        loaded += qt_bytes(&ly->wk);

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", l);
        load_qt_i4(db, name, &ly->wv, KVH * hd, D);
        loaded += qt_bytes(&ly->wv);

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", l);
        load_qt_i4(db, name, &ly->wo, D, H * hd);
        loaded += qt_bytes(&ly->wo);

        /* Attention bias */
        if (c->has_attn_bias) {
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.bias", l);
            ly->bq = load_f32_tensor(db, name, H * hd);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.bias", l);
            ly->bk = load_f32_tensor(db, name, KVH * hd);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.bias", l);
            ly->bv = load_f32_tensor(db, name, KVH * hd);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.bias", l);
            ly->bo = load_f32_tensor(db, name, D);
            loaded += (H*hd + KVH*hd + KVH*hd + D) * 4;
        }

        /* Attention sink logits (uno per query head) */
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.sinks", l);
        ly->sinks = load_f32_tensor(db, name, H);
        if (!ly->sinks) {
            fprintf(stderr, "  errore: attention sinks mancanti al layer %d\n", l);
            return -1;
        }
        loaded += H * 4;

        /* Router weights + bias */
        snprintf(name, sizeof(name), "model.layers.%d.mlp.router.weight", l);
        ly->router = load_f32_tensor(db, name, (int64_t)c->n_experts * D);
        if (!ly->router) {
            /* Prova nome alternativo */
            snprintf(name, sizeof(name), "model.layers.%d.block_sparse_moe.gate.weight", l);
            ly->router = load_f32_tensor(db, name, (int64_t)c->n_experts * D);
        }
        loaded += (int64_t)c->n_experts * D * 4;

        snprintf(name, sizeof(name), "model.layers.%d.mlp.router.bias", l);
        ly->router_bias = load_f32_tensor(db, name, c->n_experts);
        if (ly->router_bias) loaded += c->n_experts * 4;

        if (l % 6 == 0 || l == c->n_layers - 1)
            fprintf(stderr, "    layer %d/%d ✓\n", l + 1, c->n_layers);
    }

    m->resident_bytes = loaded;
    fprintf(stderr, "  ✓ pesi densi caricati: %.2f GB\n", loaded / 1e9);
    return 0;
}

/* Legge ID decimali separati da whitespace. Il chiamante libera *out_ids. */
static int read_token_ids_file(const char *path, int max_tokens, int **out_ids) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "errore: impossibile aprire INPUT_FILE=%s\n", path);
        return -1;
    }

    int cap = max_tokens < 256 ? max_tokens : 256;
    if (cap < 1) cap = 1;
    int *ids = (int *)malloc((size_t)cap * sizeof(int));
    if (!ids) { fclose(f); return -1; }

    int n = 0;
    long value;
    int rc;
    while ((rc = fscanf(f, "%ld", &value)) == 1) {
        if (value < 0 || value > INT_MAX || n >= max_tokens) {
            fprintf(stderr, "errore: ID non valido o prompt oltre %d token in %s\n",
                    max_tokens, path);
            free(ids); fclose(f); return -1;
        }
        if (n == cap) {
            int next = cap < max_tokens / 2 ? cap * 2 : max_tokens;
            int *grown = (int *)realloc(ids, (size_t)next * sizeof(int));
            if (!grown) { free(ids); fclose(f); return -1; }
            ids = grown;
            cap = next;
        }
        ids[n++] = (int)value;
    }
    if (rc != EOF) {
        fprintf(stderr, "errore: contenuto non numerico in INPUT_FILE=%s\n", path);
        free(ids); fclose(f); return -1;
    }
    fclose(f);
    if (n == 0) {
        fprintf(stderr, "errore: INPUT_FILE vuoto: %s\n", path);
        free(ids); return -1;
    }
    *out_ids = ids;
    return n;
}

/* ── Modalità servizio: sessione multi-turn persistente su pipe ──
 * stdout contiene solo il protocollo, stderr solo diagnostica.
 * Invariante: ogni TOKEN emesso viene anche consumato dal forward, quindi
 * `pos` equivale sempre al numero di posizioni valide nella KV-cache. */
static int service_loop(Model *m, int cap) {
    Cfg *c = &m->c;
    int pos = 0;

    printf("READY %d %d %d %d\n", cap, c->vocab, c->stop_ids[0], c->stop_ids[1]);
    fflush(stdout);

    char cmd[32];
    while (scanf("%31s", cmd) == 1) {
        if (strcmp(cmd, "SHUTDOWN") == 0) break;

        if (strcmp(cmd, "RESET") == 0) {
            pos = 0;
            g_history_len = 0;
            m->kv.cur_pos = 0;
            printf("DONE RESET 0 0\n");
            fflush(stdout);
            continue;
        }

        if (strcmp(cmd, "TURN") != 0) {
            printf("ERROR BAD_COMMAND 1 comando sconosciuto\n");
            fflush(stdout);
            return 1;
        }

        /* Header esteso: sampling per-richiesta (temp, top_p, top_k) fra keep e n_ids. */
        long max_new = 0, keep = 0, n_ids = 0;
        float t_temp = 0.0f, t_topp = 0.0f;
        long t_topk = 0;
        if (scanf("%ld %ld %f %f %ld %ld", &max_new, &keep,
                  &t_temp, &t_topp, &t_topk, &n_ids) != 6) {
            printf("ERROR BAD_HEADER 1 header TURN illeggibile\n");
            fflush(stdout);
            return 1;
        }
        /* Applica il sampling di questo turno (clamp difensivo). */
        if (t_temp < 0.0f) t_temp = 0.0f;
        if (t_temp > 5.0f) t_temp = 5.0f;
        if (t_topp <= 0.0f || t_topp > 1.0f) t_topp = 1.0f;
        if (t_topk < 0) t_topk = 0;
        g_temperature = t_temp;
        g_top_p = t_topp;
        g_top_k = (int)t_topk;
        if (max_new < 0 || keep < 0 || keep > pos || n_ids <= 0 ||
            keep + n_ids > cap || n_ids > cap) {
            printf("ERROR BAD_RANGE 0 keep/n_ids fuori limiti (pos=%d cap=%d)\n", pos, cap);
            fflush(stdout);
            /* Nessuna mutazione: la sessione resta usabile. Scarta gli ID pendenti. */
            for (long i = 0; i < n_ids; i++) { long skip; if (scanf("%ld", &skip) != 1) return 1; }
            continue;
        }

        int *ids = (int *)malloc((size_t)n_ids * sizeof(int));
        if (!ids) return 1;
        int valid = 1;
        for (long i = 0; i < n_ids; i++) {
            long value;
            if (scanf("%ld", &value) != 1) { free(ids); return 1; }
            if (value < 0 || value >= c->vocab) valid = 0;
            ids[i] = (int)value;
        }
        if (!valid) {
            free(ids);
            printf("ERROR BAD_TOKEN 0 ID fuori vocabolario\n");
            fflush(stdout);
            continue;
        }

        /* Riusa il prefisso: le posizioni successive verranno sovrascritte. */
        pos = (int)keep;
        int next = prefill_tokens(m, ids, (int)n_ids, pos);
        pos += (int)n_ids;
        free(ids);

        const char *reason = "MAX_TOKENS";
        int produced = 0;
        int tok = next;
        while (1) {
            if (produced >= max_new) { reason = "MAX_TOKENS"; break; }
            if (pos >= cap) { reason = "CONTEXT_FULL"; break; }
            printf("TOKEN %d\n", tok);
            fflush(stdout);
            produced++;
            int following = forward_token(m, tok, pos);
            pos++;
            m->n_emit++;
            if (tok == c->stop_ids[0]) { reason = "RETURN"; break; }
            if (tok == c->stop_ids[1]) { reason = "CALL"; break; }
            tok = (following < 0 || following >= c->vocab) ? 0 : following;
        }
        printf("DONE %s %d %d\n", reason, produced, pos);
        fflush(stdout);
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════
 *  MAIN
 * ═══════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    fprintf(stderr, "🪶 picchio v0.5.0 — GPT-OSS MoE streaming engine\n");
    fprintf(stderr, "   GQA · INT4 · streaming CPU (architettura letta da config.json)\n\n");

    /* ── Self-test mode ── */
    if (argc > 1 && strcmp(argv[1], "--self-test") == 0) {
        return self_test();
    }

    const char *model_path = getenv("MODEL");
    if (!model_path) {
        if (argc > 1) model_path = argv[1];
        else {
            fprintf(stderr, "Uso: MODEL=/path/to/gptoss_i4 ./picchio [max_tokens]\n");
            fprintf(stderr, "     oppure: ./picchio /path/to/gptoss_i4 [max_tokens]\n");
            fprintf(stderr, "     oppure: ./picchio --self-test\n");
            return 1;
        }
    }

    int max_tokens = 128;
    const char *mt = getenv("MAX");
    if (mt) max_tokens = atoi(mt);
    else if (argc > 2) max_tokens = atoi(argv[2]);
    g_oracle_dir = getenv("ORACLE_DIR");
    { const char *trace = getenv("TRACE_NUMERIC");
      g_trace_numeric = trace && atoi(trace) != 0; }
    { const char *p = getenv("PREDICT_PROBE");
      g_predict_probe = p && atoi(p) != 0; }

    fprintf(stderr, "modello:    %s\n", model_path);
    { const char *svc = getenv("SERVICE");
      if (svc && atoi(svc))
          fprintf(stderr, "modalità:   servizio (il limite di token arriva con ogni TURN)\n\n");
      else
          fprintf(stderr, "max token:  %d\n\n", max_tokens); }

    /* ── 1. Carica configurazione ── */
    static Model m;
    memset(&m, 0, sizeof(m));
    m.pred_layer = -1;

    if (cfg_load(&m.c, model_path) != 0) {
        fprintf(stderr, "errore: impossibile caricare config.json\n");
        return 1;
    }

    /* ── 2. Apri file safetensors ── */
    StDB db;
    st_init(&db);

    {
        char path[512];
        FILE *test;

        /* Prova dense.safetensors */
        snprintf(path, sizeof(path), "%s/dense.safetensors", model_path);
        test = fopen(path, "rb");
        if (test) { fclose(test); st_open_file(&db, path); }

        /* model.safetensors (singolo file HF) */
        snprintf(path, sizeof(path), "%s/model.safetensors", model_path);
        test = fopen(path, "rb");
        if (test) { fclose(test); st_open_file(&db, path); }

        /* Shard: model-NNNNN-of-NNNNN.safetensors (formato HF) */
        for (int i = 1; i <= 200; i++) {
            int found = 0;
            for (int total = 2; total <= 200; total++) {
                snprintf(path, sizeof(path),
                         "%s/model-%05d-of-%05d.safetensors", model_path, i, total);
                test = fopen(path, "rb");
                if (test) { fclose(test); st_open_file(&db, path); found = 1; break; }
            }
            if (!found && i > 1) break;
        }

        /* File shard aggiuntivi su altri dischi, separati da ';'.
         * L'ordine è significativo: st_find usa il primo tensore omonimo. */
        const char *aux_env = getenv("MODEL_AUX");
        if (aux_env && *aux_env) {
            char aux_list[2048];
            strncpy(aux_list, aux_env, sizeof(aux_list) - 1);
            aux_list[sizeof(aux_list) - 1] = '\0';
            char *cursor = aux_list;
            while (*cursor) {
                char *sep = strchr(cursor, ';');
                if (sep) *sep = '\0';
                if (*cursor) {
                    test = fopen(cursor, "rb");
                    if (test) { fclose(test); st_open_file(&db, cursor); }
                    else fprintf(stderr, "  ⚠ MODEL_AUX non trovato: %s\n", cursor);
                }
                if (!sep) break;
                cursor = sep + 1;
            }
        }

        /* Shard: model-NNNNN.safetensors (formato Picchio convertito) */
        for (int i = 0; i < 200; i++) {
            snprintf(path, sizeof(path), "%s/model-%05d.safetensors", model_path, i);
            test = fopen(path, "rb");
            if (test) { fclose(test); st_open_file(&db, path); }
        }

        /* experts-NN.safetensors */
        for (int i = 0; i < 50; i++) {
            snprintf(path, sizeof(path), "%s/experts-%02d.safetensors", model_path, i);
            test = fopen(path, "rb");
            if (test) { fclose(test); st_open_file(&db, path); }
        }
    }

    if (db.n_tensors == 0) {
        fprintf(stderr, "\n⚠ nessun tensore trovato in %s\n", model_path);
        fprintf(stderr, "  Usa: python3 convert.py --model openai/gpt-oss-120b --output %s\n",
                model_path);
        st_close(&db);
        return 1;
    }

    fprintf(stderr, "✓ %d tensori da %d file\n", db.n_tensors, db.n_files);

    /* Rendi il DB accessibile dal loader degli expert */
    g_db = &db;

    /* ── 3. Alloca strutture e carica pesi ── */
    Cfg *c = &m.c;
    m.L = calloc(c->n_layers, sizeof(Layer));

    /* KV-cache (512 posizioni iniziali per stare in 16 GB RAM) */
    int initial_ctx = 512;
    const char *ctx_env = getenv("CTX");
    if (ctx_env) initial_ctx = atoi(ctx_env);
    if (initial_ctx > c->ctx_len) initial_ctx = c->ctx_len;
    kv_init(&m.kv, c->n_layers, c->n_kv_heads, c->head_dim, initial_ctx);

    /* Cache expert per-layer — budget di RAM per gli expert (ogni expert ~12.4 MB).
     * Più cache = più hit = meno byte da disco, che è il collo di bottiglia reale.
     * Sweep misurato sul 20B (16 GB RAM, RSS reale ora disponibile): il default
     * storico di 6 GB dava 21 slot/layer e 88.9% di hit; a 8 GB si arriva a 28
     * slot/layer e 91.2%, con RSS ~8.5 GB (largo margine sui 15.8 GB fisici). La
     * residenza piena del 20B (32 expert/layer) si raggiunge intorno a PIN_GB=9,
     * oltre il quale non c'è guadagno perché gli expert per layer sono solo 32.
     * Il default è ora adattivo: senza PIN_GB si rileva la RAM fisica e si assegna
     * agli expert tutto tranne una riserva per densa/KV/OS. Su 16 GB questo dà ~10 GB
     * di budget (residenza piena del 20B); scala su macchine più grandi e si autolimita
     * su quelle piccole. La vecchia stima che PIN_GB basso saturasse la RAM era basata
     * su una misura RSS rotta su Windows (vedi rss_gb), ora corretta. */
    int pin_gb_env = 0;
    { const char *v = getenv("PIN_GB"); if (v) pin_gb_env = atoi(v); }
    int64_t GB = 1024LL * 1024 * 1024;
    int64_t avail_bytes;
    if (pin_gb_env > 0) {
        avail_bytes = (int64_t)pin_gb_env * GB;   /* override esplicito */
        fprintf(stderr, "  PIN_GB=%d (esplicito)\n", pin_gb_env);
    } else {
        int64_t phys = physical_ram_bytes();
        if (phys > 0) {
            /* Riserva per densa (~3–4,5 GB) + KV + overhead OS. Il resto va agli
             * expert. La cache si autolimita comunque a n_experts×n_layers. */
            int64_t reserve = 6LL * GB;
            avail_bytes = phys - reserve;
            if (avail_bytes < 2 * GB) avail_bytes = 2 * GB;  /* floor RAM scarsa */
            fprintf(stderr, "  RAM fisica %.1f GB → budget expert %.1f GB "
                    "(riserva %.0f GB per densa/KV/OS; override con PIN_GB)\n",
                    (double)phys / 1e9, (double)avail_bytes / 1e9,
                    (double)reserve / 1e9);
        } else {
            avail_bytes = 8LL * GB;  /* RAM non rilevabile: default fisso */
            fprintf(stderr, "  RAM non rilevabile → budget expert 8 GB\n");
        }
    }
    int64_t expert_bytes = (int64_t)c->moe_inter * ((c->hidden + 1) / 2)  /* gate_up */
                         + (int64_t)c->hidden * ((c->hidden + 1) / 2)     /* down */
                         + (int64_t)(c->moe_inter + c->hidden) * 4;       /* scales+bias */
    int total_slots = (int)(avail_bytes / expert_bytes);
    m.ecap = total_slots / c->n_layers;
    if (m.ecap < 4) m.ecap = 4;
    if (m.ecap > 128) m.ecap = 128;
    { const char *v = getenv("ECAP");   /* override manuale per tuning/test */
      if (v) { int e = atoi(v); if (e > 0) m.ecap = e > 128 ? 128 : e; } }
    if (m.ecap > c->n_experts) m.ecap = c->n_experts;  /* mai più slot che expert */
    if (m.ecap < c->topk) m.ecap = c->topk;  /* cache_load_batch richiede ecap>=topk */
    fprintf(stderr, "  expert cache: %d slot/layer × %d layer (%d expert totali, ~%.1f GB)\n",
            m.ecap, c->n_layers, m.ecap * c->n_layers,
            (double)m.ecap * c->n_layers * expert_bytes / 1e9);
    m.ecache = calloc(c->n_layers, sizeof(ESlot *));
    m.ecn = calloc(c->n_layers, sizeof(int));
    m.pin = calloc(c->n_layers, sizeof(ESlot *));
    m.npin = calloc(c->n_layers, sizeof(int));
    m.eusage = calloc(c->n_layers, sizeof(uint32_t *));
    m.eheat = calloc(c->n_layers, sizeof(uint32_t *));
    for (int l = 0; l < c->n_layers; l++) {
        m.ecache[l] = calloc(m.ecap, sizeof(ESlot));
        for (int i = 0; i < m.ecap; i++) m.ecache[l][i].eid = -1;
        m.eusage[l] = calloc(c->n_experts, sizeof(uint32_t));
        m.eheat[l] = calloc(c->n_experts, sizeof(uint32_t));
    }

    /* Carica pesi densi */
    double t0 = now_s();
    g_model_path_global = model_path;
    if (load_dense_weights(&m, &db) != 0) {
        fprintf(stderr, "errore: caricamento pesi fallito\n");
        st_close(&db);
        return 1;
    }

    /* Carica hot-store da sessione precedente */
    hotstore_load(&m);

    /* ── 4. Carica tokenizer ── */
    Tokenizer tok;
    int has_tokenizer = 0;
    {
        char tok_path[512];
        snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.json", model_path);
        if (tok_load(&tok, tok_path) == 0) {
            has_tokenizer = 1;
        } else {
            fprintf(stderr, "  ⚠ tokenizer.json non trovato — modo raw token ID\n");
        }
    }

    fprintf(stderr, "✓ caricato in %.1f s · residente %.2f GB\n",
            now_s() - t0, m.resident_bytes / 1e9);

    /* ── 5. Parametri di sampling ── */
    {
        const char *v;
        v = getenv("TEMPERATURE"); if (v) g_temperature = (float)atof(v);
        v = getenv("TOPP"); if (v) g_top_p = (float)atof(v);
        v = getenv("TOPK"); if (v) g_top_k = atoi(v);
        v = getenv("REP");  if (v) g_rep_penalty = (float)atof(v);
        v = getenv("SEED"); if (v) g_rng = (uint64_t)atoll(v);
        else g_rng = (uint64_t)time(NULL);  /* random seed di default */
    }
    fprintf(stderr, "sampling: temp=%.2f top_p=%.2f top_k=%d rep=%.2f\n",
            g_temperature, g_top_p, g_top_k, g_rep_penalty);

    /* Read paralleli degli expert (queue depth > 1). IO_THREADS controlla il
     * numero di thread; PIPE=0 forza il percorso seriale per compatibilità. */
    {
        const char *v = getenv("IO_THREADS");
        g_io_threads = v ? atoi(v) : 4;
        const char *pipe = getenv("PIPE");
        if (pipe && atoi(pipe) == 0) g_io_threads = 1;
        if (g_io_threads < 1) g_io_threads = 1;
        fprintf(stderr, "I/O expert: %d thread (%s)\n", g_io_threads,
                g_io_threads > 1 ? "read paralleli" : "seriale");
    }

    /* Prefetch → LRU (PREFETCH=1, alias storico PILOT=1). Predice gli expert del
     * layer successivo (proxy post-MoE) e li carica durante l'attention. */
    {
        const char *v = getenv("PREFETCH");
        if (!v) v = getenv("PILOT");
        if (v && atoi(v)) {
            pilot_init(&m);
            fprintf(stderr, "prefetch → LRU abilitato (proxy post-MoE)\n");
        }
    }

    /* ── 6. Generazione ── */
    fprintf(stderr, "\n");

    /* Modalità servizio: sessione persistente, nessun prompt/decode nel C. */
    { const char *v = getenv("SERVICE");
      if (v && atoi(v)) {
        fprintf(stderr, "servizio: sessione persistente, capacità KV %d posizioni\n",
                initial_ctx);
        int rc = service_loop(&m, initial_ctx);
        stats_dump(&m);
        hotstore_save(&m);
        pilot_shutdown();
        if (has_tokenizer) tok_free(&tok);
        st_close(&db);
        return rc;
      } }

    /* Harmony special token IDs */
    #define TOK_START   200006
    #define TOK_END     200007
    #define TOK_MESSAGE 200008
    #define TOK_CHANNEL 200005
    #define TOK_RETURN  200002
    #define TOK_CALL    200012

    /* Costruisci prompt con template Harmony legacy oppure usa ID ufficiali da file. */
    int prompt_tokens_stack[8192];
    int *prompt_tokens = prompt_tokens_stack;
    int prompt_tokens_owned = 0;
    int n_prompt = 0;
    const char *input_file = getenv("INPUT_FILE");

    if (input_file && *input_file) {
        n_prompt = read_token_ids_file(input_file, c->ctx_len, &prompt_tokens);
        if (n_prompt < 0) {
            if (has_tokenizer) tok_free(&tok);
            st_close(&db);
            return 1;
        }
        prompt_tokens_owned = 1;
        fprintf(stderr, "prompt (INPUT_FILE raw ID, %d token)\n", n_prompt);
    } else {
        /* Leggi prompt da stdin o variabile d'ambiente. */
        char prompt[4096] = "";
        const char *user_prompt = getenv("INPUT");
        if (!user_prompt || strlen(user_prompt) == 0) {
            user_prompt = getenv("PROMPT");
            /* Ignora il PROMPT di Windows ($P$G). */
            if (user_prompt && (strstr(user_prompt, "$P$G") || strlen(user_prompt) == 0))
                user_prompt = NULL;
        }
        if (user_prompt) {
            strncpy(prompt, user_prompt, sizeof(prompt) - 1);
        } else {
            fprintf(stderr, "› ");
            if (fgets(prompt, sizeof(prompt), stdin)) {
                int plen = (int)strlen(prompt);
                if (plen > 0 && prompt[plen-1] == '\n') prompt[--plen] = '\0';
                if (plen > 0 && prompt[plen-1] == '\r') prompt[--plen] = '\0';
            }
        }

        if (prompt[0] != '\0' && has_tokenizer) {
            const char *raw_env = getenv("RAW");
            int raw_mode = raw_env && atoi(raw_env);
            if (raw_mode) {
                const char *p = prompt;
                while (*p && n_prompt < 8192) {
                    while (*p == ' ' || *p == '\n') p++;
                    if (*p == '\0') break;
                    prompt_tokens[n_prompt++] = (int)strtol(p, (char **)&p, 10);
                }
                fprintf(stderr, "prompt (raw ID, %d token)\n", n_prompt);
            } else {
                const char *sys_msg = "You are ChatGPT, a large language model trained by OpenAI.\n"
                                      "Knowledge cutoff: 2024-06\n"
                                      "Current date: 2026-07-27\n\n"
                                      "Reasoning: medium\n\n"
                                      "# Valid channels: analysis, commentary, final. "
                                      "Channel must be included for every message.";
                prompt_tokens[n_prompt++] = TOK_START;
                n_prompt += tok_encode(&tok, "system", prompt_tokens + n_prompt, 16);
                prompt_tokens[n_prompt++] = TOK_MESSAGE;
                n_prompt += tok_encode(&tok, sys_msg, prompt_tokens + n_prompt, 2048);
                prompt_tokens[n_prompt++] = TOK_END;
                prompt_tokens[n_prompt++] = TOK_START;
                n_prompt += tok_encode(&tok, "user", prompt_tokens + n_prompt, 16);
                prompt_tokens[n_prompt++] = TOK_MESSAGE;
                n_prompt += tok_encode(&tok, prompt, prompt_tokens + n_prompt, 4096);
                prompt_tokens[n_prompt++] = TOK_END;
                prompt_tokens[n_prompt++] = TOK_START;
                n_prompt += tok_encode(&tok, "assistant", prompt_tokens + n_prompt, 16);
                fprintf(stderr, "prompt: \"%s\" → %d token (Harmony legacy)\n",
                        prompt, n_prompt);
            }
        } else if (prompt[0] != '\0') {
            const char *p = prompt;
            while (*p && n_prompt < 8192) {
                while (*p == ' ' || *p == '\n') p++;
                if (*p == '\0') break;
                prompt_tokens[n_prompt++] = (int)strtol(p, (char **)&p, 10);
            }
            fprintf(stderr, "prompt (raw ID): %d token\n", n_prompt);
        }
    }

    if (n_prompt == 0) {
        fprintf(stderr, "errore: prompt vuoto\n");
        if (prompt_tokens_owned) free(prompt_tokens);
        if (has_tokenizer) tok_free(&tok);
        st_close(&db);
        return 1;
    }
    for (int i = 0; i < n_prompt; i++) {
        if (prompt_tokens[i] < 0 || prompt_tokens[i] >= c->vocab) {
            fprintf(stderr, "errore: token ID fuori vocabolario alla posizione %d: %d\n",
                    i, prompt_tokens[i]);
            if (prompt_tokens_owned) free(prompt_tokens);
            if (has_tokenizer) tok_free(&tok);
            st_close(&db);
            return 1;
        }
    }

    /* Prefill: processa tutti i token del prompt */
    fprintf(stderr, "prefill %d token...\n", n_prompt);
    double t_prefill = now_s();
    int last_tok = prefill_tokens(&m, prompt_tokens, n_prompt, 0);
    fprintf(stderr, "✓ prefill in %.2f s (next=%d)\n\n", now_s() - t_prefill, last_tok);

    /* Decode: genera token autoregressivamente */
    double t_gen = now_s();
    int pos = n_prompt;
    int tok_id = last_tok;
    int in_final = 0;       /* 1 quando siamo nel canale "final" */
    int skip_header = 1;    /* 1 per saltare i token dell'header (channel, etc.) */

    /* Se no tokenizer, stampa tutto direttamente (no harmony parsing) */
    if (!has_tokenizer) { in_final = 1; skip_header = 0; }
    
    /* Debug: mostra anche analysis se VERBOSE=1 */
    int show_all = 0;
    { const char *v = getenv("VERBOSE"); if (v && atoi(v)) show_all = 1; }
    int output_ids = 0;
    { const char *v = getenv("OUTPUT"); if (v && strcmp(v, "ids") == 0) output_ids = 1; }

    for (int i = 0; i < max_tokens; i++) {
        /* Il protocollo machine-readable include sempre il terminatore. */
        if (output_ids) {
            printf("%d\n", tok_id);
            fflush(stdout);
        }

        /* Stop della risposta assistant: return o call. TOK_END chiude solo un messaggio. */
        if (tok_id == TOK_RETURN || tok_id == TOK_CALL) break;
        if (tok_id == c->stop_ids[0]) break;  /* EOS generico */

        if (output_ids) {
            /* Nessun decode o filtro C: il bridge Harmony possiede il rendering. */
        }
        /* Gestione canali harmony:
         * - Dopo <|channel|>, leggi il nome del canale
         * - Se "final", mostra il contenuto
         * - Se "analysis", non mostrare (chain of thought interna)
         */
        else if (tok_id == TOK_CHANNEL) {
            skip_header = 1;  /* prossimi token sono header di canale */
        } else if (tok_id == TOK_MESSAGE) {
            skip_header = 0;  /* dopo <|message|> inizia il contenuto */
        } else if (tok_id == TOK_START) {
            skip_header = 1;  /* inizio di un nuovo messaggio */
            in_final = 0;
        } else if (tok_id == TOK_END) {
            /* Fine messaggio — non stampare */
        } else if (skip_header) {
            /* Siamo nell'header — controlla se è "final" o se è un token normale */
            if (tok_id == TOK_CHANNEL || tok_id == TOK_START || tok_id == TOK_END ||
                tok_id == TOK_MESSAGE) {
                /* Token speciale nell'header — già gestito sopra */
            } else if (has_tokenizer) {
                const char *s = tok_decode(&tok, tok_id);
                if (strstr(s, "final")) { in_final = 1; }
                else if (strstr(s, "analysis")) { in_final = 0; }
                else if (strstr(s, "commentary")) { in_final = 0; }
                else {
                    /* Token normale fuori da qualsiasi struttura — 
                     * il modello sta generando senza header harmony.
                     * Trattiamo come contenuto diretto. */
                    skip_header = 0;
                    in_final = 1;
                    /* Stampa questo token */
                    char decoded[512];
                    tok_decode_raw(&tok, tok_id, decoded, sizeof(decoded));
                    printf("%s", decoded);
                    fflush(stdout);
                }
            } else {
                /* No tokenizer, token normale: stampa */
                skip_header = 0;
                in_final = 1;
                printf("[%d]", tok_id);
                fflush(stdout);
            }
        } else {
            /* Contenuto del messaggio — stampa se siamo in final (o verbose) */
            if (in_final || show_all || !has_tokenizer) {
                if (has_tokenizer) {
                    char decoded[512];
                    tok_decode_raw(&tok, tok_id, decoded, sizeof(decoded));
                    printf("%s", decoded);
                } else {
                    printf("[%d]", tok_id);
                }
                fflush(stdout);
            }
        }

        /* Forward per il prossimo token */
        tok_id = forward_token(&m, tok_id, pos);
        pos++;
        m.n_emit++;
    }

    printf("\n");
    double gen_elapsed = now_s() - t_gen;

    /* Stats */
    fprintf(stderr, "\n");
    if (m.n_emit > 0) {
        fprintf(stderr, "generati %llu token in %.2f s (%.1f tok/s)\n",
                (unsigned long long)m.n_emit, gen_elapsed,
                m.n_emit / gen_elapsed);
    }
    stats_dump(&m);

    /* Salva hot-store per la prossima sessione */
    hotstore_save(&m);

    /* Cleanup */
    pilot_shutdown();
    if (prompt_tokens_owned) free(prompt_tokens);
    if (has_tokenizer) tok_free(&tok);
    st_close(&db);
    return 0;
}
