/* picchio.c — Streaming MoE engine for GPT-OSS-120B in pure C.
 *
 * Target architecture: GPT-OSS-120B (117B MoE, 5.1B active/token)
 *   - 36 layers, 128 experts/layer, top-4
 *   - GQA (64 Q heads, 8 KV heads, group=8, head_dim=64)
 *   - Standard RoPE, sliding window 128 alternated with full attention
 *   - Native MXFP4 quantization (converted to INT4 for v1)
 *   - Hidden dim 2880, MoE intermediate 5760 (fused gate+up)
 *   - Expert: gate_up [5760, 2880] + down [2880, 2880] = ~12.4 MB at INT4
 *   - Vocabulary 201,088 (o200k_harmony)
 *
 * Philosophy: placement decides ONLY speed, never precision.
 * The output is identical regardless of where the experts reside.
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

#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_max_threads(void) { return 1; }
#endif

#ifdef _WIN32
#include <winsock2.h>  /* must precede windows.h */
#include <ws2tcpip.h>
#include <windows.h>
#include <psapi.h>     /* GetProcessMemoryInfo for rss_gb */
#else
#include <pthread.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
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
#include "flat.h"
#include "tok.h"
#include "picchio_cuda.h"   /* optional GPU backend: PgpuExpert + entry-point types */

/* Optional GPU backend entry-point types + state. The loader functions
 * (gpu_backend_load/shutdown) are defined further down; these are declared up
 * here so moe_forward and forward_head_logits can reference them. */
typedef int  (*pgpu_init_t)(void);
typedef void (*pgpu_shutdown_t)(void);
typedef int  (*pgpu_i8_t)(float *, const float *, const int8_t *, const float *,
                          int, int, const void *);
typedef int  (*pgpu_router_upload_t)(const float *, const float *, int, int,
                                    const void *);
typedef int  (*pgpu_router_scores_t)(float *, const float *, const float *,
                                    const float *, int, int, const void *);
typedef int  (*pgpu_moe_t)(float *, const float *, int, int, int,
                           const PgpuExpert *, const float *, int, int,
                           int, int, float, float);
typedef int  (*pgpu_moe_resident_t)(int, int);  /* 1 if (layer,eid) is in VRAM */
#include "gpu_router_native.h"
static pgpu_shutdown_t g_pgpu_shutdown = NULL;
static pgpu_i8_t       g_pgpu_i8 = NULL;    /* lm_head (opt-in: GPU_LMHEAD=1) */
static pgpu_router_upload_t g_pgpu_router_upload = NULL;
static pgpu_router_scores_t g_pgpu_router_scores = NULL;
static pgpu_moe_t      g_pgpu_moe = NULL;   /* experts (the G2 win) */
static pgpu_moe_resident_t g_pgpu_moe_resident = NULL; /* skip disk for VRAM-resident */
static int             g_gpu_backend = 0;
static int             g_gpu_router = 0;    /* resident routers + score kernel */
static int             g_gpu_dense = 0;     /* resident FP16 Q/K/V/O projections */
static uint64_t        g_gpu_dense_host_released = 0;
static uint64_t        g_service_prefill_tokens = 0, g_service_decode_tokens = 0;
static double          g_service_prefill_seconds = 0, g_service_decode_seconds = 0;
static int             g_expert_reuse = 1;
static uint64_t        g_expert_buffer_allocs = 0, g_expert_buffer_reuses = 0;
static double          g_expert_compute_seconds = 0;
/* SPEC_PROBE: measurement harness for n-gram speculative decoding. It does NOT
 * change generation — it records per-decode-token expert routing and the token
 * id stream, then reports n-gram acceptance and expert-union at exit. */
static int   g_spec_probe = 0;
static int  *g_probe_hist = NULL;
static int   g_probe_hist_n = 0, g_probe_hist_cap = 0, g_probe_prompt_n = 0;
static int  *g_probe_exp = NULL;
static int   g_probe_ntok = 0, g_probe_captok = 0, g_probe_L = 0, g_probe_K = 0;
/* SELF_DRAFT_PROBE: measures how often a top-1 routing (a free self-draft) picks
 * the same next token as the real top-4 routing. g_force_top1 collapses the MoE
 * mixture to its single top expert for the draft forward only. */
static int   g_self_draft_probe = 0;
static int   g_force_top1 = 0;
static int   g_sd_draft_k = 1;   /* draft routing width (SELF_DRAFT_K), default top-1 */
static long  g_sd_total = 0, g_sd_match = 0, g_sd_run = 0, g_sd_runhist[8] = {0};
/* Speculative decoding: a small draft model (DRAFT_MODEL) proposes tokens that the
 * target verifies in one batched forward. The draft is tiny and fully resident, so
 * it never touches the global streaming db after load (no single-model conflict).
 * The draft Model/StDB are declared near load_draft (after those types exist). */
static int   g_have_draft = 0;
static int   g_spec_k = 4;       /* draft tokens proposed per round (SPEC_K) */
static int             g_gpu_on = 0;        /* experts on GPU */
static int             g_gpu_lmhead = 0;    /* lm_head on GPU (off by default) */
static int             g_gpu_prefetch = 0;  /* early L+1 proxy during current MoE */
static uint64_t        g_gpu_router_calls = 0;
static double          g_gpu_router_time = 0.0;
static uint64_t        g_gpu_dense_calls = 0;
static uint64_t        g_gpu_dense_fallbacks = 0;
static double          g_gpu_dense_time = 0.0;

/* ═══════════════════════════════════════════════════════════
 *  CONFIGURATION (read from the model's config.json)
 * ═══════════════════════════════════════════════════════════ */

typedef struct {
    int hidden;           /* D: hidden dimension (2880) */
    int n_layers;         /* total number of layers (36) */
    int n_heads;          /* query heads (64) */
    int n_kv_heads;       /* KV heads (8) — GQA group=8 */
    int head_dim;         /* dimension per head (64) */
    int n_experts;        /* experts per MoE layer (128) */
    int topk;             /* active experts per token (4) */
    int moe_inter;        /* expert intermediate dim: gate_up fused (5760) */
    int dense_inter;      /* dense FFN intermediate dim (if dense layers) */
    int vocab;            /* vocabulary size (201088) */
    int ctx_len;          /* maximum context (131072) */
    int sliding_window;   /* sliding attention window (128) */
    int stop_ids[8];      /* stop tokens */
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
    float routed_scale;   /* scaling factor for expert routing */
    int has_shared;       /* 1 if there are shared experts */
    int n_shared;         /* number of shared experts */
    int has_attn_bias;    /* 1 if attention uses bias (yes in GPT-OSS) */
    /* Architecture family flags (derived from model_type). GPT-OSS keeps the
     * original behaviour; Qwen3-MoE flips these. Everything is config-gated so
     * the GPT-OSS path stays byte-identical. */
    int qk_norm;          /* 1: RMSNorm on Q/K per head before RoPE (Qwen3) */
    int swiglu_clipped;   /* 1: GPT-OSS clipped SwiGLU; 0: plain SiLU (Qwen3) */
    int router_norm;      /* 0: softmax over top-k logits (GPT-OSS);
                             1: softmax over all experts, renormalize top-k (Qwen3) */
    int use_sinks;        /* 1: attention sinks are mandatory (GPT-OSS) */
    int expert_bits;      /* expert quant width: 4 = INT4 gs64 (default), 3 = INT3 gs64 */
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

    /* Attention bias (GPT-OSS has it) */
    float *bq;            /* [n_heads * head_dim] */
    float *bk;            /* [n_kv_heads * head_dim] */
    float *bv;            /* [n_kv_heads * head_dim] */
    float *bo;            /* [D] */
    float *sinks;         /* [n_heads] attention sink logits */
    float *q_norm;        /* [head_dim] QK-Norm weight for Q (Qwen3); NULL otherwise */
    float *k_norm;        /* [head_dim] QK-Norm weight for K (Qwen3); NULL otherwise */

    int layer_type;       /* 0=sliding_attention, 1=full_attention */

    /* MoE (all layers in GPT-OSS-120B are MoE) */
    float *router;        /* [n_experts, D] — router weights */
    float *router_bias;   /* [n_experts] — correction bias */

    /* Expert gate_up_proj is fused: [moe_inter, D] where moe_inter = 5760 = 2*2880
     * down_proj: [D, D] = [2880, 2880]
     * The expert weights are NOT in the Layer — they are loaded on-demand into ESlot */
} Layer;

/* ═══════════════════════════════════════════════════════════
 *  EXPERT SLOT (LRU cache, reusable across layers)
 * ═══════════════════════════════════════════════════════════ */

typedef struct {
    int eid;              /* expert ID (-1 = empty) */
    int layer;            /* owning layer */
    /* GPT-OSS expert layout: gate_up fused [5760, 2880] + down [2880, 2880]
     * gate_up_proj_bias [5760], down_proj_bias [2880] */
    QT gu;                /* gate_up_proj fused [moe_inter, D] */
    QT d;                 /* down_proj [D, D] */
    float *gu_bias;       /* [moe_inter] */
    float *d_bias;        /* [D] */
    uint8_t *slab;        /* buffer for coalesced pread */
    float *fslab;         /* scale buffer */
    int64_t slab_cap;     /* allocated capacity */
    int64_t fslab_cap;
    int slab_reusable;    /* aligned SafeTensors cache storage, retained on eviction */
    uint64_t last_used;   /* timestamp for LRU eviction */
    int loading;          /* 1 = prefetch in progress on this slot (atomic) */
} ESlot;

/* ═══════════════════════════════════════════════════════════
 *  KV-CACHE (GQA standard)
 * ═══════════════════════════════════════════════════════════ */

typedef struct {
    /* Per layer, per position: K and V of each KV head.
     * Layout: K[layer] = float[max_pos * n_kv_heads * head_dim]
     *         V[layer] = float[max_pos * n_kv_heads * head_dim]
     */
    float **K;            /* [n_layers] pointers to K buffers */
    float **V;            /* [n_layers] pointers to V buffers */
    int max_pos;          /* allocated positions */
    int cur_pos;          /* next position to write */
} KVCache;

/* ═══════════════════════════════════════════════════════════
 *  MODEL
 * ═══════════════════════════════════════════════════════════ */

typedef struct {
    Cfg c;

    QT embed;             /* embedding [vocab, D] */
    QT lm_head;           /* language model head [vocab, D] */
    float *final_norm;    /* final RMSNorm [D] */

    Layer *L;             /* [n_layers] */
    KVCache kv;

    /* Per-layer expert cache */
    ESlot **ecache;       /* [n_layers][ecap] — LRU pool */
    int *ecn;             /* how many experts cached per layer */
    int ecap;             /* capacity per layer */

    /* Learned hot-store */
    ESlot **pin;          /* pinned experts (never evicted) */
    int *npin;
    uint32_t **eusage;    /* persistent per-expert counters */
    uint32_t **eheat;     /* recent heat (for promotion) */

    /* Working set of the current forward */
    ESlot ws[32];         /* experts in flight (max topk per batch) */

    /* Profiling */
    uint64_t eclock;      /* monotonic counter for LRU */
    uint64_t hits, miss, ereq; /* cache statistics */
    uint64_t n_fw, n_emit;     /* forwards / emitted tokens */
    double t_edisk;       /* total time reading experts from disk */
    uint64_t async_jobs;  /* current-layer asynchronous read batches */
    double t_async_wait;  /* time with no ready expert left to compute */
    double t_attn;        /* total attention time */
    double t_moe;         /* total MoE compute time */
    double t_head;        /* total lm_head time */

    int64_t resident_bytes; /* bytes of the dense part in RAM */

    /* Prefetch prediction accuracy probe (PREDICT_PROBE=1) */
    int pred_next[64];    /* proxy A: predicted from the previous layer's pre-MoE hn */
    int pred_next2[64];   /* proxy B: predicted from the previous layer's post-MoE norm */
    int pred_layer;       /* layer the predictions refer to (-1 = none) */
    uint64_t pred_hit, pred_total;    /* proxy A */
    uint64_t pred_hit2;               /* proxy B (same pred_total) */
} Model;

/* ═══════════════════════════════════════════════════════════
 *  ORACLE DUMP (.npy F32, active only with ORACLE_DIR)
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
static int g_predict_probe = 0;   /* measures prefetch prediction accuracy */

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
    /* Current working set = RSS: pages physically resident in RAM. */
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.WorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    return 0.0;
#else
    return 0.0;
#endif
}

/* Total physical RAM in bytes (0 if undeterminable). Used to size the expert
 * cache budget when PIN_GB is not specified. */
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
    return 0;  /* other platforms: fall back to the fixed default */
#endif
}

/* RAM that can be committed without immediately reclaiming or paging other
 * processes. The physical total alone is not enough on a 16 GB workstation:
 * browsers, IDEs and the OS can already consume most of it before Picchio
 * starts. */
static int64_t available_ram_bytes(void) {
#ifdef _WIN32
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) return (int64_t)ms.ullAvailPhys;
    return 0;
#elif defined(__linux__)
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long psize = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && psize > 0) return (int64_t)pages * (int64_t)psize;
    return 0;
#else
    return 0;
#endif
}

/* Conservative estimate of weights that load after cache sizing. Converted
 * Picchio models keep embed/head in INT8 and attention/router/bias tensors in
 * F32. This estimate is used only to protect the OS from an over-optimistic
 * automatic expert budget; PIN_GB remains an explicit override. */
static int64_t dense_weight_estimate_bytes(const Cfg *c) {
    int64_t D = c->hidden;
    int64_t qdim = (int64_t)c->n_heads * c->head_dim;
    int64_t kvdim = (int64_t)c->n_kv_heads * c->head_dim;
    int64_t embed_head = 2LL * c->vocab * (D + 4); /* INT8 rows + F32 scales */
    int64_t attention = (int64_t)c->n_layers *
        (2LL * qdim * D + 2LL * kvdim * D) * 4;
    int64_t routers = (int64_t)c->n_layers * c->n_experts * D * 4;
    int64_t expert_biases = (int64_t)c->n_layers * c->n_experts *
        ((int64_t)c->moe_inter + D) * 4;
    int64_t margin = 64LL * 1024 * 1024; /* norms, sinks and projection biases */
    return embed_head + attention + routers + expert_biases + margin;
}

static int64_t expert_slot_estimate_bytes(const Cfg *c) {
    int64_t D = c->hidden;
    int64_t I = c->moe_inter;
    int64_t H = I / 2;
    int64_t gu_groups = (D + 63) / 64;
    int64_t d_groups = (H + 63) / 64;
    int64_t gu_packed;
    int64_t d_packed;
    if (c->expert_bits == 3) {
        gu_packed = I * gu_groups * 24;
        d_packed = D * d_groups * 24;
    } else {
        gu_packed = I * ((D + 1) / 2);
        d_packed = D * ((H + 1) / 2);
    }
    return gu_packed + d_packed
         + (I * gu_groups + D * d_groups) * 4
         + (I + D) * 4; /* biases */
}

/* ═══════════════════════════════════════════════════════════
 *  CONFIG LOADER (from config.json)
 * ═══════════════════════════════════════════════════════════ */

static int cfg_load(Cfg *c, const char *model_path) {
    char path[512];
    snprintf(path, sizeof(path), "%s/config.json", model_path);

    char *json = json_read_file(path);
    if (!json) {
        fprintf(stderr, "error: cannot read %s\n", path);
        return -1;
    }

    memset(c, 0, sizeof(*c));

    /* ── Architecture family ────────────────────────────────────────────
     * GPT-OSS is the default; Qwen3-MoE differs in a handful of numeric
     * details (QK-Norm, plain SiLU, router normalization, no sinks, no
     * sliding window). Detect it from model_type and flip the flags. */
    char arch[64];
    json_str(json, "model_type", arch, sizeof(arch), "gpt_oss");
    int is_qwen = (strstr(arch, "qwen") != NULL) || (strstr(arch, "Qwen") != NULL);
    c->qk_norm        = is_qwen ? 1 : 0;
    c->swiglu_clipped = is_qwen ? 0 : 1;
    c->router_norm    = is_qwen ? 1 : 0;
    c->use_sinks      = is_qwen ? 0 : 1;
    c->expert_bits    = json_int(json, "picchio_expert_bits", 4);  /* 3 = INT3 gs64 */

    c->hidden       = json_int(json, "hidden_size", 2880);
    c->n_layers     = json_int(json, "num_hidden_layers", 36);
    c->n_heads      = json_int(json, "num_attention_heads", 64);
    c->n_kv_heads   = json_int(json, "num_key_value_heads", 8);
    c->head_dim     = json_int(json, "head_dim", 64);
    c->n_experts    = json_int(json, "num_local_experts",
                               json_int(json, "num_experts", 128));
    c->topk         = json_int(json, "num_experts_per_tok", 4);
    /* Expert intermediate dim (gate+up fused, so ×2). Qwen3 sizes its experts
     * with moe_intermediate_size; GPT-OSS reuses intermediate_size. */
    if (is_qwen) {
        int moe_i = json_int(json, "moe_intermediate_size",
                             json_int(json, "intermediate_size", 768));
        c->moe_inter = moe_i * 2;
    } else {
        c->moe_inter = json_int(json, "intermediate_size", 2880) * 2; /* gate+up fused */
    }
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
        /* If n_types < n_layers, cycle the pattern */
        if (n_types < c->n_layers) {
            for (int i = n_types; i < c->n_layers; i++)
                c->layer_type[i] = c->layer_type[i % n_types];
        }
    } else if (is_qwen) {
        /* Qwen3-MoE: every layer is full attention (no sliding window). */
        for (int i = 0; i < c->n_layers; i++) c->layer_type[i] = 1;
    } else {
        /* GPT-OSS default: alternate sliding/full */
        for (int i = 0; i < c->n_layers; i++)
            c->layer_type[i] = (int8_t)(i % 2);
    }

    /* Stop tokens. GPT-OSS uses the Harmony terminators; Qwen3 uses the ChatML
     * end-of-turn id from config (eos_token_id, typically 151645 <|im_end|>). */
    if (is_qwen) {
        c->stop_ids[0] = json_int(json, "eos_token_id", 151645);
        c->n_stop = 1;
    } else {
        c->stop_ids[0] = 200002;  /* <|return|> */
        c->stop_ids[1] = 200012;  /* <|call|> */
        c->n_stop = 2;
    }

    free(json);

    /* Summary */
    fprintf(stderr, "  model: %s (%s)\n", arch,
            is_qwen ? "Qwen3-MoE" : "GPT-OSS");
    fprintf(stderr, "  config: D=%d L=%d H=%d KV=%d hd=%d E=%d top%d\n",
            c->hidden, c->n_layers, c->n_heads, c->n_kv_heads,
            c->head_dim, c->n_experts, c->topk);
    fprintf(stderr, "  config: moe_inter=%d vocab=%d ctx=%d sw=%d eps=%.0e\n",
            c->moe_inter, c->vocab, c->ctx_len, c->sliding_window, c->eps);

    return 0;
}

/* ═══════════════════════════════════════════════════════════
 *  RoPE (standard, not interleaved)
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
 *  KV-CACHE: allocation and update
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

/* Per-head RMSNorm over head_dim, applied in place to a [nheads * hd] buffer.
 * Used for Qwen3's QK-Norm (weight has length head_dim, shared across heads). */
static void rmsnorm_heads(float *x, const float *w, int nheads, int hd, float eps) {
    for (int h = 0; h < nheads; h++) {
        float *v = x + (int64_t)h * hd;
        float ss = 0.0f;
        for (int i = 0; i < hd; i++) ss += v[i] * v[i];
        float inv = 1.0f / sqrtf(ss / (float)hd + eps);
        for (int i = 0; i < hd; i++) v[i] = v[i] * inv * w[i];
    }
}

/* Native FP16 projections with FP32 accumulation. CPU fallback is available
 * while host weights are retained; release-host mode must fail closed. */
static void attention_projection(float *y, const float *x, QT *w) {
    if (g_gpu_dense && w->fmt == 0) {
        double t0 = now_s();
        int rc = pgr_native_dense_matmul(y, x, w->qf, w->O, w->I, w);
        g_gpu_dense_time += now_s() - t0;
        g_gpu_dense_calls++;
        if (rc == 0) return;
        if (g_gpu_dense_host_released) {
            fprintf(stderr, "[gpu-dense] fatal: GPU failure after host weights were released; "
                    "restart without GPU_DENSE_RELEASE_HOST to enable CPU fallback\n");
            exit(1);
        }
        g_gpu_dense_fallbacks++;
        g_gpu_dense = 0;
        fprintf(stderr, "[gpu-dense] disabled after backend failure; using CPU\n");
    }
    matmul_qt(y, x, w, 1);
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

    /* Q, K, V projections */
    attention_projection(q, x, &l->wq);
    attention_projection(k, x, &l->wk);
    attention_projection(v, x, &l->wv);
    if (g_oracle_dir) {
        char n[128]; int64_t sq[3] = {1, 1, H * hd};
        int64_t sk[3] = {1, 1, kv_dim};
        snprintf(n, sizeof(n), "layer%d.q_nobias", layer); oracle_dump(n, q, 3, sq);
        snprintf(n, sizeof(n), "layer%d.k_nobias", layer); oracle_dump(n, k, 3, sk);
        snprintf(n, sizeof(n), "layer%d.v_nobias", layer); oracle_dump(n, v, 3, sk);
    }

    /* Add bias if present */
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

    /* QK-Norm (Qwen3): RMSNorm over head_dim on every Q and K head, before RoPE. */
    if (c->qk_norm && l->q_norm && l->k_norm) {
        rmsnorm_heads(q, l->q_norm, H, hd, c->eps);
        rmsnorm_heads(k, l->k_norm, KVH, hd, c->eps);
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

    /* Update KV-cache */
    if (pos < kv->max_pos)
        kv_store(kv, layer, pos, k, v, kv_dim);

    /* Attention: for each query head */
    float *attn_out = calloc(H * hd, sizeof(float));

    /* Determine the attention window */
    int start_pos = 0;
    int end_pos = pos;
    if (end_pos >= kv->max_pos) end_pos = kv->max_pos - 1;
    if (l->layer_type == 0) {  /* sliding_attention: include at most `window` tokens */
        start_pos = end_pos - c->sliding_window + 1;
        if (start_pos < 0) start_pos = 0;
    }
    /* layer_type == 1: full_attention → start_pos = 0 (sees everything) */

    for (int h = 0; h < H; h++) {
        int kv_h = h / group;  /* which KV head */
        float *qh = q + h * hd;

        /* Score against positions in the window */
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

        /* Softmax with attention sink: the sink absorbs probability but does not
         * contribute to the value sum, like Transformers eager. */
        float max_score = l->sinks ? l->sinks[h] : -INFINITY;
        for (int i = 0; i < n_pos; i++)
            if (scores[i] > max_score) max_score = scores[i];
        float sum = l->sinks ? expf(l->sinks[h] - max_score) : 0.0f;
        for (int i = 0; i < n_pos; i++) {
            scores[i] = expf(scores[i] - max_score);
            sum += scores[i];
        }
        for (int i = 0; i < n_pos; i++) scores[i] /= sum;

        /* Weighted sum of the values */
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
    attention_projection(out, attn_out, &l->wo);
    if (l->bo) for (int i = 0; i < D; i++) out[i] += l->bo[i];
    if (g_oracle_dir) {
        char n[128]; snprintf(n, sizeof(n), "layer%d.attn_out", layer);
        oracle_dump_vec(n, out, D);
    }

    free(q); free(k); free(v); free(attn_out);
}

/* ═══════════════════════════════════════════════════════════
 *  EXPERT LOADING (from disk, coalesced pread)
 * ═══════════════════════════════════════════════════════════ */

/* Global pointer to the safetensors DB (for access from expert_load) */
static StDB *g_db = NULL;
static FlatStore g_flat;

static void flat_global_shutdown(void) { flat_close(&g_flat); }

/* Load an expert bias from the correct F32 format, or recover the old
 * converted format: INT4 bytes mistakenly saved as F32 + gs64 scale. */
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

/* Fast path for converted group-scaled INT4/INT3 experts. The flat payload is
 * gate_up packed | gate_up scales | down packed | down scales. Biases remain in
 * safetensors (small and architecture-dependent). Weight pointers slice one
 * aligned allocation, so DIRECT reads need no bounce buffer or memcpy. */
static int expert_load_flat(Model *m, int layer, int eid, ESlot *s) {
    FlatLoc *loc = flat_loc(&g_flat, layer, eid);
    if (!loc) return 0;
    Cfg *c = &m->c;
    int D = c->hidden, I = c->moe_inter, down_I = I / 2;
    int64_t gu_qb, d_qb;
    if (c->expert_bits == 3) {
        gu_qb = (int64_t)I * (((int64_t)D + 63) / 64) * 24;
        d_qb = (int64_t)D * (((int64_t)down_I + 63) / 64) * 24;
    } else if (c->expert_bits == 4) {
        gu_qb = (int64_t)I * ((D + 1) / 2);
        d_qb = (int64_t)D * ((down_I + 1) / 2);
    } else {
        return 0;
    }
    int64_t gu_sb = (int64_t)I * ((D + 63) / 64) * 4;
    int64_t d_sb = (int64_t)D * ((down_I + 63) / 64) * 4;
    int64_t expected = gu_qb + gu_sb + d_qb + d_sb;
    if (expected != loc->len) {
        static int warned = 0;
        if (!__atomic_exchange_n(&warned, 1, __ATOMIC_RELAXED))
            fprintf(stderr, "flat: expert payload size mismatch; using safetensors fallback\n");
        return 0;
    }

    uint8_t *blob = (uint8_t *)flat_aligned_alloc(loc->padded_len);
    if (!blob || !flat_read(&g_flat, loc, blob, st_direct)) {
        flat_aligned_free(blob);
        return 0;
    }
    memset(&s->gu, 0, sizeof(s->gu));
    memset(&s->d, 0, sizeof(s->d));
    s->slab = blob;
    s->slab_cap = loc->padded_len;
    s->gu.fmt = c->expert_bits == 3 ? 5 : 2;
    s->gu.O = I; s->gu.I = D; s->gu.block_size = 64;
    s->gu.q4 = blob;
    s->gu.s = (float *)(blob + gu_qb);
    s->d.fmt = c->expert_bits == 3 ? 5 : 2;
    s->d.O = D; s->d.I = down_I; s->d.block_size = 64;
    s->d.q4 = blob + gu_qb + gu_sb;
    s->d.s = (float *)(blob + gu_qb + gu_sb + d_qb);
    s->gu_bias = load_expert_bias(layer, eid, "gate_up_proj_bias",
                                  c->n_experts, I);
    s->d_bias = load_expert_bias(layer, eid, "down_proj_bias",
                                 c->n_experts, D);
    return 1;
}

static void free_slot_buffers(ESlot *s);

/* Converted gs64 experts can be read straight into reusable cache storage.
 * Four independently aligned regions preserve the original tensor bytes, even
 * when the shard offsets are not sector aligned. No disk re-layout required.
 * Return 0 for unsupported layouts, 1 for success; read errors are fatal rather
 * than silently dropping a selected expert or using stale slot contents. */
static int expert_load_reusable(Model *m, int layer, int eid, ESlot *s) {
    if (!g_expert_reuse || !g_db ||
        (m->c.expert_bits != 3 && m->c.expert_bits != 4)) return 0;
    int D = m->c.hidden, I = m->c.moe_inter, down_I = I / 2;
    if (D % 64 || down_I % 64) return 0;
    const char *parts[4] = {"gate_up_proj", "gate_up_proj", "down_proj", "down_proj"};
    StTensor *tensor[4];
    size_t region[4], total = 0;
    int64_t group_bytes = m->c.expert_bits == 3 ? 24 : 32;
    int64_t expected[4] = {(int64_t)I * (D / 64) * group_bytes,
                           (int64_t)I * (D / 64) * 4,
                           (int64_t)D * (down_I / 64) * group_bytes,
                           (int64_t)D * (down_I / 64) * 4};
    for (int j = 0; j < 4; j++) {
        char name[256];
        snprintf(name, sizeof(name), "model.layers.%d.mlp.experts.%s.%d%s",
                 layer, parts[j], eid, (j & 1) ? ".qs" : "");
        tensor[j] = st_find(g_db, name);
        if (!tensor[j] || tensor[j]->dtype != ((j & 1) ? ST_F32 : ST_U8) ||
            st_bytes(tensor[j]) != expected[j]) return 0;
        int64_t absolute = g_db->files[tensor[j]->file_idx].data_offset + tensor[j]->offset_start;
        if (absolute < 0 || ((j & 1) && (absolute & 3))) return 0;
        region[j] = (size_t)((expected[j] + (absolute & (ST_ALIGN - 1)) + ST_ALIGN - 1)
                            & ~(int64_t)(ST_ALIGN - 1));
        if (region[j] > SIZE_MAX - total) return 0;
        total += region[j];
    }
    if (!s->slab_reusable || (uint64_t)s->slab_cap < total) {
        free_slot_buffers(s);
        s->slab = flat_aligned_alloc(total);
        if (!s->slab) {
            fprintf(stderr, "OOM reusable expert buffer\n"); fflush(stderr);
            _Exit(1); /* may be an I/O worker: atexit joins would deadlock */
        }
        s->slab_cap = (int64_t)total;
        s->slab_reusable = 1;
        __atomic_fetch_add(&g_expert_buffer_allocs, 1, __ATOMIC_RELAXED);
    } else {
        __atomic_fetch_add(&g_expert_buffer_reuses, 1, __ATOMIC_RELAXED);
    }
    uint8_t *ptr[4];
    size_t offset = 0;
    for (int j = 0; j < 4; j++) {
        if (st_read_raw_aligned(g_db, tensor[j], s->slab + offset, region[j], &ptr[j]) != expected[j]) {
            fprintf(stderr, "fatal: incomplete expert read L=%d E=%d tensor=%s\n",
                    layer, eid, tensor[j]->name);
            fflush(stderr);
            _Exit(1); /* never wait for the worker pool from inside that pool */
        }
        offset += region[j];
    }
    memset(&s->gu, 0, sizeof(s->gu));
    memset(&s->d, 0, sizeof(s->d));
    s->gu.fmt = s->d.fmt = m->c.expert_bits == 3 ? 5 : 2;
    /* Legacy reader represents a single group as per-row scaling (block_size=0).
     * Preserve that dispatch as well as the bytes, including for tiny fixtures. */
    s->gu.block_size = D > 64 ? 64 : 0;
    s->d.block_size = down_I > 64 ? 64 : 0;
    s->gu.O = I; s->gu.I = D; s->gu.q4 = ptr[0]; s->gu.s = (float *)ptr[1];
    s->d.O = D; s->d.I = down_I; s->d.q4 = ptr[2]; s->d.s = (float *)ptr[3];
    s->gu_bias = load_expert_bias(layer, eid, "gate_up_proj_bias", m->c.n_experts, I);
    s->d_bias = load_expert_bias(layer, eid, "down_proj_bias", m->c.n_experts, D);
    return 1;
}

static void expert_load(Model *m, int layer, int eid, ESlot *s) {
    Cfg *c = &m->c;
    int D = c->hidden;
    int I = c->moe_inter;  /* 5760 */
    char name[256];

    s->eid = eid;
    s->layer = layer;

    if (s->slab_reusable && flat_loc(&g_flat, layer, eid)) free_slot_buffers(s);
    if (expert_load_flat(m, layer, eid, s)) return;
    if (expert_load_reusable(m, layer, eid, s)) return;
    if (s->slab_reusable) free_slot_buffers(s); /* fallback to another storage layout */

    if (!g_db) {
        /* No safetensors DB — mark expert as unavailable */
        s->eid = -1;
        return;
    }

    /* ── Load gate_up_proj: [moe_inter, D] ── */
    snprintf(name, sizeof(name),
             "model.layers.%d.mlp.experts.gate_up_proj.%d", layer, eid);
    StTensor *t_gu = st_find(g_db, name);

    /* Try alternative names */
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
        /* Expert not found in the DB (it may be in a shard that was not loaded) */
        s->eid = -1;
        return;
    }

    /* Load as F32 and quantize to INT4 */
    int64_t gu_numel = (int64_t)I * D;
    float *gu_f32 = t_gu->dtype == ST_U8 ? NULL : falloc(gu_numel);

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
        /* Already packed (INT4 gs64, or INT3 gs64 when expert_bits==3) */
        memset(&s->gu, 0, sizeof(QT));
        s->gu.O = I;
        s->gu.I = D;
        int64_t rb;
        if (c->expert_bits == 3) {
            s->gu.fmt = 5;
            rb = (int64_t)I * (((int64_t)D + 63) / 64) * 24;   /* planes: 24 B / 64 vals */
        } else {
            s->gu.fmt = 2;
            rb = (int64_t)I * ((D + 1) / 2);
        }
        s->gu.q4 = (uint8_t *)malloc(rb);
        if (!s->gu.q4 || st_read_raw(g_db, t_gu, s->gu.q4, rb) != rb) {
            /* Short/failed read (I/O pressure) would leave the tail garbage:
             * drop the expert rather than compute with corrupt weights. */
            free(gu_f32); free(s->gu.q4); s->gu.q4 = NULL;
            s->eid = -1; return;
        }
        /* Look for scales */
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
            /* Determine block_size: if n_scale > O → group-scaled */
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
    /* ── Load gate_up bias (also supports the old INT4+qs bias) ── */
    s->gu_bias = load_expert_bias(layer, eid, "gate_up_proj_bias",
                                  c->n_experts, I);

    /* ── Load down_proj: [D, D] ── */
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

    int down_I = I / 2;  /* input dimension of down = intermediate_size */
    if (t_d) {
        int64_t d_numel = (int64_t)D * down_I;
        float *d_f32 = t_d->dtype == ST_U8 ? NULL : falloc(d_numel);

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
            /* Already packed (INT4 gs64, or INT3 gs64 when expert_bits==3) */
            memset(&s->d, 0, sizeof(QT));
            s->d.O = D;
            s->d.I = down_I;
            int64_t rb;
            if (c->expert_bits == 3) {
                s->d.fmt = 5;
                rb = (int64_t)D * (((int64_t)down_I + 63) / 64) * 24;
            } else {
                s->d.fmt = 2;
                rb = (int64_t)D * ((down_I + 1) / 2);
            }
            s->d.q4 = (uint8_t *)malloc(rb);
            if (!s->d.q4 || st_read_raw(g_db, t_d, s->d.q4, rb) != rb) {
                /* gate_up already loaded: free it too, since an eid=-1 slot is
                 * skipped by the cache's guarded free and would otherwise leak. */
                free(d_f32); free(s->d.q4); s->d.q4 = NULL;
                free(s->gu.qf); s->gu.qf = NULL;
                free(s->gu.q4); s->gu.q4 = NULL;
                free(s->gu.s);  s->gu.s = NULL;
                free(s->gu_bias); s->gu_bias = NULL;
                s->eid = -1; return;
            }
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
        s->eid = -1;  /* Incomplete expert */
        return;
    }

down_loaded:
    /* ── Down bias (also supports the old INT4+qs bias) ── */
    s->d_bias = load_expert_bias(layer, eid, "down_proj_bias",
                                 c->n_experts, D);
    return;
}

/* ═══════════════════════════════════════════════════════════
 *  EXPERT CACHE: LRU lookup + eviction
 * ═══════════════════════════════════════════════════════════ */

/* A slot's `loading` flag: 1 while the prefetch thread is filling it.
 * Release/acquire guarantee that, when the main thread observes loading==0, it
 * also sees the buffers written by the prefetch. */
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
    if (s->slab) {
        flat_aligned_free(s->slab); s->slab = NULL;
        s->gu.qf = NULL; s->gu.q4 = NULL; s->gu.s = NULL;
        s->d.qf = NULL; s->d.q4 = NULL; s->d.s = NULL;
    } else {
        free(s->gu.qf); s->gu.qf = NULL;
        free(s->gu.q4); s->gu.q4 = NULL;
        free(s->gu.s);  s->gu.s = NULL;
        free(s->d.qf);  s->d.qf = NULL;
        free(s->d.q4);  s->d.q4 = NULL;
        free(s->d.s);   s->d.s = NULL;
    }
    free(s->gu_bias); s->gu_bias = NULL;
    free(s->d_bias);  s->d_bias = NULL;
    free(s->fslab);   s->fslab = NULL;
    s->slab_cap = s->fslab_cap = 0;
    s->slab_reusable = 0;
}

static void recycle_slot_buffers(ESlot *s) {
    if (!g_expert_reuse || !s->slab_reusable) { free_slot_buffers(s); return; }
    /* Workers have completed before eviction; preserve only the owning slab. */
    free(s->gu_bias); s->gu_bias = NULL;
    free(s->d_bias); s->d_bias = NULL;
    memset(&s->gu, 0, sizeof(s->gu));
    memset(&s->d, 0, sizeof(s->d));
}

static ESlot *cache_lookup(Model *m, int layer, int eid) {
    ESlot *pool = m->ecache[layer];
    int n = m->ecn[layer];

    /* Look in the pinned set */
    for (int i = 0; i < m->npin[layer]; i++) {
        if (m->pin[layer][i].eid == eid) {
            m->pin[layer][i].last_used = ++m->eclock;
            m->hits++;
            return &m->pin[layer][i];
        }
    }

    /* Look in the LRU cache */
    for (int i = 0; i < n; i++) {
        if (pool[i].eid == eid) {
            pool[i].last_used = ++m->eclock;
            m->hits++;
            return &pool[i];
        }
    }

    m->miss++;

    /* Cache miss: evict LRU or use an empty slot */
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

    /* Load from disk */
    double t0 = now_s();

    /* Free old data if the slot was occupied (eviction) */
    if (victim->eid >= 0) {
        recycle_slot_buffers(victim);
    }
    
    expert_load(m, layer, eid, victim);
    m->t_edisk += now_s() - t0;
    victim->last_used = ++m->eclock;
    victim->layer = layer;
    m->ereq++;

    return victim;
}

/* Number of threads for parallel expert reads (1 = serial). */
static int g_io_threads = 1;

/* Batch-load the experts `eids[0..n)` of the layer, returning the slots in
 * `out`. Hits are resolved and "touched" (LRU) first, so the victims chosen for
 * the misses cannot evict either a hit or a miss already reserved; the misses
 * are then read from disk in parallel (queue depth > 1). The math does not
 * change: the same bytes are read into the same slots, only in concurrent
 * order. Contract: n <= ecap (guaranteed by the caller). */
static void cache_load_batch(Model *m, int layer, const int *eids, int n,
                             ESlot **out) {
    ESlot *pool = m->ecache[layer];
    int miss_i[128];
    int nmiss = 0;

    /* Pass 1: resolve the hits (pinned or LRU) and mark them as just used. */
    for (int i = 0; i < n; i++) {
        ESlot *found = NULL;
        for (int j = 0; j < m->npin[layer]; j++)
            if (m->pin[layer][j].eid == eids[i]) { found = &m->pin[layer][j]; break; }
        if (!found)
            for (int j = 0; j < m->ecn[layer]; j++)
                if (pool[j].eid == eids[i]) { found = &pool[j]; break; }
        if (found) {
            slot_wait_ready(found);  /* if a prefetch is loading this expert, wait */
            found->last_used = ++m->eclock;
            m->hits++;
            out[i] = found;
        } else {
            out[i] = NULL;
            if (nmiss < 128) miss_i[nmiss++] = i;
        }
    }

    /* Pass 2: reserve a victim for each miss (serial: no race on the LRU).
     * Slots with a prefetch in progress (loading) are never chosen as victims;
     * the ecap-topk invariant guaranteed by prefetch_issue ensures that enough
     * free slots remain. */
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
            if (!victim) {  /* last-resort defense: wait for the first prefetching slot */
                victim = &pool[0]; slot_wait_ready(victim);
            }
        }
        if (victim->eid >= 0) recycle_slot_buffers(victim);
        victim->eid = eids[i];   /* reserved: it will be populated in pass 3 */
        victim->layer = layer;
        victim->last_used = ++m->eclock;
        m->miss++;
        m->ereq++;
        out[i] = victim;
    }

    /* Pass 3: read the misses from disk, in parallel if enabled. The slots are
     * distinct, expert_load writes only into its own slot and reads g_db
     * read-only → no synchronization needed beyond st.h's per-thread handles. */
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

/* Completion-driven current-layer I/O. Unlike the predictive prefetch below,
 * this worker owns only misses already selected by the router. Each slot is
 * published independently (loading=0, release), allowing the main thread to
 * compute ready experts while the remaining reads are still in flight. */
typedef struct {
    Model *m;
    int layer;
    int n;
    ESlot *slots[128];
    int eids[128];
    int has_job;
    int shutdown;
    int initialized;
#ifdef _WIN32
    CRITICAL_SECTION cs;
    CONDITION_VARIABLE cv;
    HANDLE thread;
#else
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    pthread_t thread;
#endif
} AsyncLoader;

static AsyncLoader g_async_loader;
static int g_async_moe_enabled = 0;

static void async_loader_run_job(void) {
    int n = g_async_loader.n, layer = g_async_loader.layer;
    Model *m = g_async_loader.m;
    double t0 = now_s();
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(g_io_threads) \
            if(g_io_threads > 1 && n > 1)
#endif
    for (int i = 0; i < n; i++) {
        if (!g_async_loader.shutdown)
            expert_load(m, layer, g_async_loader.eids[i],
                        g_async_loader.slots[i]);
        slot_set_loading(g_async_loader.slots[i], 0);
    }
    m->t_edisk += now_s() - t0;
}

#ifdef _WIN32
static DWORD WINAPI async_loader_thread(LPVOID arg) {
    (void)arg;
    for (;;) {
        EnterCriticalSection(&g_async_loader.cs);
        while (!g_async_loader.has_job && !g_async_loader.shutdown)
            SleepConditionVariableCS(&g_async_loader.cv, &g_async_loader.cs,
                                     INFINITE);
        if (g_async_loader.shutdown) {
            LeaveCriticalSection(&g_async_loader.cs);
            break;
        }
        LeaveCriticalSection(&g_async_loader.cs);
        async_loader_run_job();
        EnterCriticalSection(&g_async_loader.cs);
        g_async_loader.has_job = 0;
        WakeAllConditionVariable(&g_async_loader.cv);
        LeaveCriticalSection(&g_async_loader.cs);
    }
    return 0;
}
#else
static void *async_loader_thread(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_async_loader.mutex);
        while (!g_async_loader.has_job && !g_async_loader.shutdown)
            pthread_cond_wait(&g_async_loader.cond, &g_async_loader.mutex);
        if (g_async_loader.shutdown) {
            pthread_mutex_unlock(&g_async_loader.mutex);
            break;
        }
        pthread_mutex_unlock(&g_async_loader.mutex);
        async_loader_run_job();
        pthread_mutex_lock(&g_async_loader.mutex);
        g_async_loader.has_job = 0;
        pthread_cond_broadcast(&g_async_loader.cond);
        pthread_mutex_unlock(&g_async_loader.mutex);
    }
    return NULL;
}
#endif

static void async_loader_shutdown(void) {
    if (!g_async_loader.initialized) return;
#ifdef _WIN32
    EnterCriticalSection(&g_async_loader.cs);
    while (g_async_loader.has_job)
        SleepConditionVariableCS(&g_async_loader.cv, &g_async_loader.cs,
                                 INFINITE);
    g_async_loader.shutdown = 1;
    WakeConditionVariable(&g_async_loader.cv);
    LeaveCriticalSection(&g_async_loader.cs);
    WaitForSingleObject(g_async_loader.thread, INFINITE);
    CloseHandle(g_async_loader.thread);
    DeleteCriticalSection(&g_async_loader.cs);
#else
    pthread_mutex_lock(&g_async_loader.mutex);
    while (g_async_loader.has_job)
        pthread_cond_wait(&g_async_loader.cond, &g_async_loader.mutex);
    g_async_loader.shutdown = 1;
    pthread_cond_signal(&g_async_loader.cond);
    pthread_mutex_unlock(&g_async_loader.mutex);
    pthread_join(g_async_loader.thread, NULL);
    pthread_mutex_destroy(&g_async_loader.mutex);
    pthread_cond_destroy(&g_async_loader.cond);
#endif
    g_async_loader.initialized = 0;
    g_async_moe_enabled = 0;
}

static int async_loader_init(Model *m) {
    memset(&g_async_loader, 0, sizeof(g_async_loader));
    g_async_loader.m = m;
#ifdef _WIN32
    InitializeCriticalSection(&g_async_loader.cs);
    InitializeConditionVariable(&g_async_loader.cv);
    g_async_loader.thread = CreateThread(NULL, 0, async_loader_thread,
                                         NULL, 0, NULL);
    if (!g_async_loader.thread) {
        DeleteCriticalSection(&g_async_loader.cs);
        return 0;
    }
#else
    pthread_mutex_init(&g_async_loader.mutex, NULL);
    pthread_cond_init(&g_async_loader.cond, NULL);
    if (pthread_create(&g_async_loader.thread, NULL, async_loader_thread,
                       NULL) != 0) {
        pthread_mutex_destroy(&g_async_loader.mutex);
        pthread_cond_destroy(&g_async_loader.cond);
        return 0;
    }
#endif
    g_async_loader.initialized = 1;
    g_async_moe_enabled = 1;
    return 1;
}

static int async_loader_submit(Model *m, int layer, ESlot **slots,
                               const int *eids, int n) {
    if (!g_async_loader.initialized || n <= 0) return n == 0;
#ifdef _WIN32
    EnterCriticalSection(&g_async_loader.cs);
    while (g_async_loader.has_job && !g_async_loader.shutdown)
        SleepConditionVariableCS(&g_async_loader.cv, &g_async_loader.cs,
                                 INFINITE);
#else
    pthread_mutex_lock(&g_async_loader.mutex);
    while (g_async_loader.has_job && !g_async_loader.shutdown)
        pthread_cond_wait(&g_async_loader.cond, &g_async_loader.mutex);
#endif
    if (g_async_loader.shutdown) {
#ifdef _WIN32
        LeaveCriticalSection(&g_async_loader.cs);
#else
        pthread_mutex_unlock(&g_async_loader.mutex);
#endif
        return 0;
    }
    g_async_loader.m = m;
    g_async_loader.layer = layer;
    g_async_loader.n = n;
    for (int i = 0; i < n; i++) {
        g_async_loader.slots[i] = slots[i];
        g_async_loader.eids[i] = eids[i];
    }
    g_async_loader.has_job = 1;
    m->async_jobs++;
#ifdef _WIN32
    WakeConditionVariable(&g_async_loader.cv);
    LeaveCriticalSection(&g_async_loader.cs);
#else
    pthread_cond_signal(&g_async_loader.cond);
    pthread_mutex_unlock(&g_async_loader.mutex);
#endif
    return 1;
}

/* Resolve/reserve a routed top-k exactly like cache_load_batch, but return as
 * soon as the misses have been queued. Hits already being filled by predictive
 * prefetch also stay asynchronous instead of forcing a barrier here. */
static void cache_load_batch_async(Model *m, int layer, const int *eids, int n,
                                   ESlot **out) {
    ESlot *pool = m->ecache[layer];
    int miss_i[128], nmiss = 0;
    ESlot *miss_slots[128];
    int miss_eids[128];

    for (int i = 0; i < n; i++) {
        ESlot *found = NULL;
        for (int j = 0; j < m->npin[layer]; j++)
            if (m->pin[layer][j].eid == eids[i]) {
                found = &m->pin[layer][j]; break;
            }
        if (!found)
            for (int j = 0; j < m->ecn[layer]; j++)
                if (pool[j].eid == eids[i]) { found = &pool[j]; break; }
        if (found) {
            found->last_used = ++m->eclock;
            m->hits++;
            out[i] = found;
        } else {
            out[i] = NULL;
            miss_i[nmiss++] = i;
        }
    }

    for (int mi = 0; mi < nmiss; mi++) {
        int i = miss_i[mi];
        ESlot *victim = NULL;
        if (m->ecn[layer] < m->ecap) {
            victim = &pool[m->ecn[layer]++];
        } else {
            uint64_t oldest = UINT64_MAX;
            for (int j = 0; j < m->ecn[layer]; j++) {
                if (slot_loading(&pool[j])) continue;
                if (pool[j].last_used < oldest) {
                    oldest = pool[j].last_used;
                    victim = &pool[j];
                }
            }
            if (!victim) {
                victim = &pool[0];
                slot_wait_ready(victim);
            }
        }
        if (victim->eid >= 0) recycle_slot_buffers(victim);
        victim->eid = eids[i];
        victim->layer = layer;
        victim->last_used = ++m->eclock;
        slot_set_loading(victim, 1);
        m->miss++;
        m->ereq++;
        out[i] = victim;
        miss_slots[mi] = victim;
        miss_eids[mi] = eids[i];
    }

    if (!async_loader_submit(m, layer, miss_slots, miss_eids, nmiss)) {
        double t0 = now_s();
        for (int mi = 0; mi < nmiss; mi++) {
            expert_load(m, layer, miss_eids[mi], miss_slots[mi]);
            slot_set_loading(miss_slots[mi], 0);
        }
        m->t_edisk += now_s() - t0;
    }
}

/* ═══════════════════════════════════════════════════════════
 *  PREFETCH → LRU (separate thread, populates the cache directly)
 *
 *  At the end of layer L, the main thread predicts the top-k of layer L+1
 *  (post-MoE proxy, ~89% accurate on the 20B), RESERVES the LRU slots for those
 *  experts, and hands them to a thread that reads them from disk. During the
 *  attention of L+1 the thread fills the slots; when moe_forward(L+1) requests
 *  them, they are already in cache. Unlike the old PILOT (which only warmed the
 *  OS cache → double read → −11%), here there is no second read.
 *
 *  Safety: only the main thread mutates the LRU (reservation). The thread writes
 *  exclusively into the buffers of the reserved slots and sets loading=0
 *  (release) when the read finishes; the main thread waits for loading==0
 *  (acquire) before using a slot. prefetch_issue reserves at most (ecap − topk)
 *  slots, so there are always enough non-loading slots for the real routing.
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
    uint64_t issued;      /* total experts submitted for prefetch */
    double read_seconds;  /* wall time spent by the prefetch worker */
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
static int g_pilot_enabled = 0;   /* historical name: used by main() */

/* ── Lock/signal primitives, 1 = success for trylock on both platforms ── */
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

/* Worker body: reads each reserved slot from disk and marks it ready. The slots
 * are distinct and expert_load is thread-safe (per-thread handles in st.h), so
 * loading a block uses the same IO_THREADS as the sync path. */
static void prefetch_run_job(void) {
    int n = g_pf.n, layer = g_pf.layer;
    Model *m = g_pf.m;
    double t0 = now_s();
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(g_io_threads) \
            if(g_io_threads > 1 && n > 1)
#endif
    for (int i = 0; i < n; i++) {
        if (!g_pf.shutdown)
            expert_load(m, layer, g_pf.eids[i], g_pf.slots[i]);
        slot_set_loading(g_pf.slots[i], 0);  /* ready (or aborted by shutdown) */
    }
    g_pf.read_seconds += now_s() - t0;
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
        fprintf(stderr, "  ⚠ prefetch: thread creation failed\n");
        return;
    }
#else
    pthread_mutex_init(&g_pf.mutex, NULL);
    pthread_cond_init(&g_pf.cond, NULL);
    if (pthread_create(&g_pf.thread, NULL, prefetch_thread, NULL) != 0) {
        fprintf(stderr, "  ⚠ prefetch: thread creation failed\n");
        return;
    }
#endif
    g_pilot_enabled = 1;
}

/* Main thread: reserves the LRU slots for the predicted experts `pred[0..k)` of
 * `layer` and hands them to the worker. Does not wait: if the worker is still
 * busy or the lock is contended, it skips this round (prefetch is best-effort). */
static void prefetch_issue(Model *m, int layer, const int *pred, int k) {
    if (!g_pilot_enabled || layer < 0 || layer >= m->c.n_layers) return;
    int max_reserve = m->ecap - m->c.topk;   /* always leave topk slots for routing */
    if (max_reserve <= 0) return;
    if (!pf_trylock()) return;
    if (g_pf.has_job) { pf_unlock(); return; }   /* previous job still running */

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
        if (victim->eid >= 0) recycle_slot_buffers(victim);
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
    fprintf(stderr, "prefetch: %llu experts submitted for prefetch\n",
            (unsigned long long)g_pf.issued);
    g_pilot_enabled = 0;
}

/* ═══════════════════════════════════════════════════════════
 *  MOE FORWARD (single token)
 * ═══════════════════════════════════════════════════════════ */

/* Compute router logits. GPU_ROUTER keeps all router matrices resident in VRAM;
 * any backend failure falls back to the reference C loop for this call. */
static void router_scores_into(Model *m, int layer, const float *x,
                               const Cfg *c, float *scores) {
    int D = c->hidden, E = c->n_experts;
    Layer *l = &m->L[layer];
    if (g_gpu_router && g_pgpu_router_scores) {
        double t0 = now_s();
        int rc = g_pgpu_router_scores(scores, x, l->router, l->router_bias,
                                      E, D, l->router);
        g_gpu_router_time += now_s() - t0;
        g_gpu_router_calls++;
        if (rc == 0) return;
    }
    for (int e = 0; e < E; e++) {
        float dot = l->router_bias ? l->router_bias[e] : 0.0f;
        const float *rw = l->router + (int64_t)e * D;
        for (int i = 0; i < D; i++) dot += x[i] * rw[i];
        scores[e] = dot;
    }
}

/* Probe: predicts the top-k of layer `nl`'s router applied to the input `x`
 * (already normalized) and writes it into `out`. Does not alter the math: it
 * only measures the accuracy of two proxies for the prefetch. */
static void predict_topk_into(Model *m, int nl, const float *x,
                              const Cfg *c, int *out) {
    int E = c->n_experts, K = c->topk;
    float scores[512];
    if (E > (int)(sizeof(scores) / sizeof(scores[0]))) {
        for (int k = 0; k < K; k++) out[k] = -1;
        return;
    }
    router_scores_into(m, nl, x, c, scores);
    float best_s[64];
    for (int k = 0; k < K; k++) { out[k] = -1; best_s[k] = -1e30f; }
    for (int e = 0; e < E; e++) {
        float dot = scores[e];
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

/* Turn the selected top-k router logits into mixture weights.
 *   scores[]  : full router logits over E (may be overwritten in the Qwen path).
 *   sel[]     : chosen expert ids (top-k).
 *   weights[] : in/out. For GPT-OSS it already holds the top-k raw logits.
 * GPT-OSS: softmax over the top-k logits. Qwen3: softmax over ALL experts, then
 * renormalize the selected top-k probabilities to sum to 1 (norm_topk_prob). */
static void router_weights(const Cfg *c, float *scores, const int *sel,
                           float *weights, int K, int E) {
    if (c->router_norm) {
        softmax(scores, E);
        float s = 0.0f;
        for (int k = 0; k < K; k++) { weights[k] = scores[sel[k]]; s += weights[k]; }
        if (s > 0.0f) for (int k = 0; k < K; k++) weights[k] /= s;
    } else {
        softmax(weights, K);
    }
}

static void expert_apply(ESlot *es, const float *x, float *dst, float w,
                         const Cfg *c);

static void moe_forward(float *out, const float *x, Model *m,
                        int layer, const Cfg *c) {
    int D = c->hidden;
    int E = c->n_experts;
    int K = c->topk;
    /* 1. Router: compute a score for each expert */
    float *scores = falloc(E);
    router_scores_into(m, layer, x, c, scores);

    /* 2. Top-K selection */
    int sel[64];          /* selected experts */
    float weights[64];    /* corresponding weights */

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

    /* SPEC_PROBE: record this decode token's routing (top-k experts per layer).
     * moe_forward runs only in single-token decode (prefill uses its own batched
     * expert loop), so this captures exactly the decode positions. */
    if (g_spec_probe && g_probe_ntok < g_probe_captok && layer < g_probe_L) {
        int *dst = g_probe_exp + ((size_t)g_probe_ntok * g_probe_L + layer) * g_probe_K;
        for (int k = 0; k < K && k < g_probe_K; k++) dst[k] = sel[k];
    }

    /* Probe: compare this layer's real routing with the two proxies predicted
     * for it during the previous layer. */
    if ((g_predict_probe || g_gpu_prefetch) && m->pred_layer == layer) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < K; j++)
                if (sel[k] == m->pred_next[j])  { m->pred_hit++;  break; }
            for (int j = 0; j < K; j++)
                if (sel[k] == m->pred_next2[j]) { m->pred_hit2++; break; }
        }
        m->pred_total += K;
    }

    /* GPU-guided early prefetch. At this point the current layer's routing has
     * already consumed the previous prediction, while its expert I/O and
     * compute have not started. Use the current pre-MoE activation as the L+1
     * proxy and overlap those predicted reads with the whole current MoE. */
    if (g_gpu_prefetch && layer + 1 < c->n_layers) {
        predict_topk_into(m, layer + 1, x, c, m->pred_next);
        m->pred_layer = layer + 1;
        prefetch_issue(m, layer + 1, m->pred_next, c->topk);
    }

    /* 3. Normalize weights (GPT-OSS: over top-k; Qwen3: over all, renorm top-k) */
    if (g_force_top1 && g_sd_draft_k < K) K = g_sd_draft_k;  /* self-draft width */
    router_weights(c, scores, sel, weights, K, E);
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

    /* 4. Update routing statistics (skip on the transient self-draft forward so
     * the learned hot-store is not polluted by top-1 routing). */
    if (!g_force_top1)
    for (int k = 0; k < K; k++) {
        if (m->eusage[layer])
            m->eusage[layer][sel[k]]++;
        if (m->eheat[layer])
            m->eheat[layer][sel[k]]++;
    }

    /* 5. Load and compute each expert. ASYNC_MOE lets ready experts run while
     * other selected misses are still being read. Validation and GPU modes use
     * the established blocking path. */
    float *expert_out = calloc(D, sizeof(float));
    memset(out, 0, D * sizeof(float));

    ESlot *slots[64];
    int async_done = 0;
    int use_async = g_async_moe_enabled && !g_gpu_on &&
                    !g_oracle_dir && !g_trace_numeric;
    /* When experts run on GPU and the backend can report VRAM residency, skip the
     * host/disk read for experts the GPU already holds — the redundant read the
     * naive offload paid on every hot expert. slots[k]==NULL marks a resident,
     * unread expert; the GPU uses its VRAM copy. Any GPU failure re-reads them
     * below so the CPU fallback stays correct. */
    int gpu_skip_reads = g_gpu_on && g_pgpu_moe && g_pgpu_moe_resident &&
                         !g_oracle_dir && !g_trace_numeric;
    if (use_async) {
        cache_load_batch_async(m, layer, sel, K, slots);
    } else if (gpu_skip_reads) {
        int need_eids[64], need_i[64], nneed = 0;
        for (int k = 0; k < K; k++) {
            slots[k] = NULL;
            if (!g_pgpu_moe_resident(layer, sel[k])) {
                need_eids[nneed] = sel[k];
                need_i[nneed] = k;
                nneed++;
            }
        }
        if (nneed > 0) {
            ESlot *nslots[64];
            cache_load_batch(m, layer, need_eids, nneed, nslots);
            for (int j = 0; j < nneed; j++) slots[need_i[j]] = nslots[j];
        }
    } else {
        cache_load_batch(m, layer, sel, K, slots);
    }

    if (use_async) {
        /* Compute in completion order, but keep each contribution separate.
         * The final reduction below is in router top-k order, preserving the
         * same floating-point accumulation order as the synchronous path. */
        float *contrib = (float *)calloc((size_t)K * D, sizeof(float));
        unsigned char done[64] = {0};
        if (!contrib) { fprintf(stderr, "OOM async MoE contributions\n"); exit(1); }
        int completed = 0;
        while (completed < K) {
            int progress = 0;
            for (int k = 0; k < K; k++) {
                if (done[k] || slot_loading(slots[k])) continue;
                if (slots[k]->eid >= 0)
                    expert_apply(slots[k], x, contrib + (size_t)k * D,
                                 1.0f, c);
                done[k] = 1;
                completed++;
                progress = 1;
            }
            if (!progress && completed < K) {
                double tw = now_s();
#ifdef _WIN32
                Sleep(0);
#else
                usleep(50);
#endif
                m->t_async_wait += now_s() - tw;
            }
        }
        for (int k = 0; k < K; k++) {
            float w = weights[k] * c->routed_scale;
            const float *src = contrib + (size_t)k * D;
            for (int i = 0; i < D; i++) out[i] += w * src[i];
        }
        free(contrib);
        async_done = 1;
    }

    /* G2: compute the whole expert sum on the GPU (weights cached in VRAM),
     * falling back to the CPU loop below on any failure. Skipped in validation
     * modes so the oracle/trace dumps stay on the byte-exact CPU path. */
    int gpu_done = 0;
    if (g_gpu_on && g_pgpu_moe && !g_oracle_dir && !g_trace_numeric) {
        PgpuExpert ge[64];
        float wsum[64];
        int ok = 1, bs = 0;
        for (int k = 0; k < K; k++) {
            ESlot *es = slots[k];
            wsum[k] = weights[k] * c->routed_scale;
            ge[k].eid = sel[k];
            if (es == NULL) {  /* resident in VRAM: no host copy, GPU uses its own */
                ge[k].gu_q4 = NULL; ge[k].gu_s = NULL; ge[k].gu_bias = NULL;
                ge[k].d_q4  = NULL; ge[k].d_s  = NULL; ge[k].d_bias  = NULL;
                continue;
            }
            if (es->eid < 0 ||
                !((es->gu.fmt == 2 && es->d.fmt == 2) ||
                  (es->gu.fmt == 5 && es->d.fmt == 5)) ||
                es->gu.block_size <= 0 || es->gu.block_size != es->d.block_size) {
                ok = 0; break;
            }
            bs = es->gu.block_size;
            ge[k].gu_q4 = es->gu.q4; ge[k].gu_s = es->gu.s; ge[k].gu_bias = es->gu_bias;
            ge[k].d_q4  = es->d.q4;  ge[k].d_s  = es->d.s;  ge[k].d_bias  = es->d_bias;
        }
        if (bs == 0) bs = 64;  /* all experts resident: group size is gs64 for this tier */
        if (ok && g_pgpu_moe(out, x, D, c->moe_inter, layer, ge, wsum, K, bs,
                             c->expert_bits,
                             c->swiglu_clipped, c->swiglu_limit, c->swiglu_alpha) == 0)
            gpu_done = 1;
    }

    /* GPU path failed or was skipped: the CPU loop needs every expert on the
     * host, but resident ones were left unread above. Read them now. */
    if (!gpu_done && !async_done) {
        for (int k = 0; k < K; k++) {
            if (slots[k] == NULL) {
                int e1 = sel[k];
                cache_load_batch(m, layer, &e1, 1, &slots[k]);
            }
        }
    }

    for (int k = 0; !gpu_done && !async_done && k < K; k++) {
        ESlot *es = slots[k];
        if (es->eid < 0) continue;  /* load error */
        double compute_start = now_s();

        int I = c->moe_inter;  /* 5760 = gate_up fused */
        float *gu = falloc(I);

        /* gate_up fused: one matmul for [5760, 2880] × x */
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

        /* Activation over the interleaved (gate, up) pairs.
         * GPT-OSS: clipped SwiGLU  (up+1)*gate*sigmoid(alpha*gate) with clamp.
         * Qwen3:   plain SiLU gate * up = gate*sigmoid(gate) * up. */
        int half = I / 2;
        if (c->swiglu_clipped) {
            for (int i = 0; i < half; i++) {
                float gate = gu[2 * i];
                float up = gu[2 * i + 1];
                if (gate > c->swiglu_limit) gate = c->swiglu_limit;
                if (up > c->swiglu_limit) up = c->swiglu_limit;
                if (up < -c->swiglu_limit) up = -c->swiglu_limit;
                gu[i] = (up + 1.0f) * gate * sigmoidf(c->swiglu_alpha * gate);
            }
        } else {
            for (int i = 0; i < half; i++) {
                float gate = gu[2 * i];
                float up = gu[2 * i + 1];
                gu[i] = gate * sigmoidf(gate) * up;
            }
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

        /* Accumulate weighted */
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
        g_expert_compute_seconds += now_s() - compute_start;
    }

    /* 6. (GPT-OSS has no shared expert — all layers are pure MoE) */
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

/* Sampling parameters (read from the environment) */
static float g_temperature = 1.0f;
static float g_top_p = 0.95f;
static int   g_top_k = 50;
static float g_rep_penalty = 1.1f;

/* Token history for the repetition penalty */
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

/* Comparator for descending sort */
typedef struct { float val; int idx; } IdxVal;
static int cmp_desc(const void *a, const void *b) {
    float fa = ((const IdxVal *)a)->val;
    float fb = ((const IdxVal *)b)->val;
    return (fa < fb) - (fa > fb);
}
static int cmp_desc_f(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
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

    /* 3. Top-K: keep only the K highest logits.
     * The threshold is the K-th largest value, found by sorting a copy of the
     * logits (O(vocab·log vocab), independent of K: no cap on K).
     * Any ties at the threshold stay included, as usual. */
    if (g_top_k > 0 && g_top_k < vocab) {
        float *scratch = (float *)malloc((size_t)vocab * sizeof(float));
        if (scratch) {
            memcpy(scratch, logits, (size_t)vocab * sizeof(float));
            qsort(scratch, vocab, sizeof(float), cmp_desc_f);
            float threshold = scratch[g_top_k - 1];
            free(scratch);
            for (int i = 0; i < vocab; i++)
                if (logits[i] < threshold) logits[i] = -1e30f;
        }
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
        /* Sort by descending probability (only the > 0 ones) */
        /* For efficiency: collect only the non-zero ones */
        int n_active = 0;
        for (int i = 0; i < vocab; i++)
            if (logits[i] > 1e-10f) n_active++;
        
        if (n_active > 1) {
            IdxVal *sorted = (IdxVal *)malloc(n_active * sizeof(IdxVal));
            int si = 0;
            for (int i = 0; i < vocab; i++)
                if (logits[i] > 1e-10f) { sorted[si].val = logits[i]; sorted[si].idx = i; si++; }
            
            qsort(sorted, n_active, sizeof(IdxVal), cmp_desc);
            
            /* Accumulate up to top_p */
            float cum = 0;
            int cutoff = n_active;
            for (int i = 0; i < n_active; i++) {
                cum += sorted[i].val;
                if (cum >= g_top_p) { cutoff = i + 1; break; }
            }

            /* Zero out everything beyond the cutoff */
            for (int i = cutoff; i < n_active; i++)
                logits[sorted[i].idx] = 0.0f;

            /* Renormalize */
            sum = 0;
            for (int i = 0; i < vocab; i++) sum += logits[i];
            if (sum > 0) for (int i = 0; i < vocab; i++) logits[i] /= sum;
            
            free(sorted);
        }
    }

    /* 6. Sample from the distribution */
    float r = rng_float();
    float cum = 0;
    for (int i = 0; i < vocab; i++) {
        cum += logits[i];
        if (cum >= r) {
            if (g_history_len < 4096) g_history[g_history_len++] = i;
            return i;
        }
    }
    
    /* Fallback: last token with probability > 0 */
    for (int i = vocab - 1; i >= 0; i--) {
        if (logits[i] > 0) {
            if (g_history_len < 4096) g_history[g_history_len++] = i;
            return i;
        }
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════
 *  FORWARD PASS (one token, decode)
 * ═══════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════
 *  PIPELINE-SPLITTABLE FORWARD
 *
 *  The residual stream h (D floats) is the only state that crosses a layer
 *  boundary; each layer's KV cache stays local to whichever machine owns that
 *  layer. So the forward can be cut at any layer L: stage A runs [0, cut) and
 *  emits h, stage B runs [cut, n_layers) on the received h. run_layer_range is
 *  the shared primitive; calling it with [0, n_layers) is byte-identical to the
 *  old monolithic loop (pipe_split_check proves this in the self-test).
 * ═══════════════════════════════════════════════════════════ */

/* Run transformer layers [l_lo, l_hi) in place on the residual stream h at
 * position pos. Behaviorally identical to the original forward_token loop. */
static void run_layer_range(Model *m, float *h, int pos, int l_lo, int l_hi) {
    Cfg *c = &m->c;
    int D = c->hidden;
    float *hn = falloc(D);          /* normalized hidden */
    float *attn_out = falloc(D);    /* attention output */
    float *ffn_out = falloc(D);     /* FFN/MoE output */

    for (int l = l_lo; l < l_hi; l++) {
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

        /* MoE (all layers in GPT-OSS are MoE) */
        double tm = now_s();

        moe_forward(ffn_out, hn, m, l, c);

        /* Probe proxy A: predict the top-k of l+1 from hn (pre-MoE norm of l). hn
         * is no longer needed after moe_forward, so it will be reused as scratch. */
        if (g_predict_probe && !g_gpu_prefetch && l + 1 < c->n_layers)
            predict_topk_into(m, l + 1, hn, c, m->pred_next);
        m->t_moe += now_s() - tm;

        /* Residual */
        for (int i = 0; i < D; i++) h[i] += ffn_out[i];

        /* Proxy B: predict the top-k of l+1 from l's post-MoE state normalized
         * with l+1's post_ln (only l+1's attention is missing, ~89% accurate). It
         * is the signal for both the probe and the prefetch→LRU, which reserves
         * those experts and loads them during l+1's attention. */
        if ((g_predict_probe || g_pilot_enabled) && l + 1 < c->n_layers) {
            rmsnorm(hn, h, m->L[l + 1].post_ln, D, c->eps);
            predict_topk_into(m, l + 1, hn, c, m->pred_next2);
            m->pred_layer = l + 1;
            if (g_pilot_enabled && !g_gpu_prefetch)
                prefetch_issue(m, l + 1, m->pred_next2, c->topk);
        }
        trace_vector("post_moe", l, h, D);
        if (g_oracle_dir) {
            snprintf(dump_name, sizeof(dump_name), "layer%d.output", l);
            oracle_dump_vec(dump_name, h, D);
        }
    }

    free(hn); free(attn_out); free(ffn_out);
}

/* ═══════════════════════════════════════════════════════════
 *  OPTIONAL GPU BACKEND (loaded at runtime; CPU path unchanged)
 *
 *  The CUDA kernels live in a separate library (picchio_cuda.dll /
 *  libpicchio_cuda.so) built by nvcc, so the pure-C engine never links against
 *  CUDA and the default build stays dependency-free and byte-identical. The
 *  backend is opt-in (GPU=1) and used only if the library AND a device are
 *  present; any per-call failure transparently falls back to the CPU kernel.
 * ═══════════════════════════════════════════════════════════ */
#ifndef _WIN32
#include <dlfcn.h>
#endif

static void *gpu_sym(void *lib, const char *name) {
#ifdef _WIN32
    return (void *)(uintptr_t)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

static void gpu_backend_shutdown(void) {
    if (g_gpu_backend && g_pgpu_shutdown) g_pgpu_shutdown();
    g_gpu_backend = 0;
    g_gpu_router = 0;
    g_gpu_dense = 0;
    g_gpu_on = 0;
}

static void gpu_backend_load(Model *m) {
    const char *legacy = getenv("GPU");
    const char *wr = getenv("GPU_ROUTER");
    const char *wp = getenv("GPU_PREFETCH");
    const char *wd = getenv("GPU_DENSE");
    const char *we = getenv("GPU_EXPERTS");
    int want_legacy = legacy && atoi(legacy) != 0;
    int want_router = want_legacy || (wr && atoi(wr) != 0) || (wp && atoi(wp) != 0);
    int want_dense = wd && atoi(wd) != 0;
    int want_experts = want_legacy || (we && atoi(we) != 0);
    const char *lh = getenv("GPU_LMHEAD");
    int want_lmhead = lh && atoi(lh) != 0;
    if (!want_router && !want_dense && !want_experts && !want_lmhead) return;

    /* Router/prefetch/dense mode is part of the C engine itself. It loads the
     * installed NVIDIA driver and JITs embedded PTX; no CUDA runtime DLL or
     * separately built picchio_cuda.dll is involved. */
    if ((want_router || want_dense) && !want_experts && !want_lmhead &&
        pgr_native_init()) {
        int uploaded = 0;
        if (want_router) {
            for (int l = 0; l < m->c.n_layers; l++) {
                Layer *ly = &m->L[l];
                if (ly->router && pgr_native_upload(ly->router, ly->router_bias,
                                                    m->c.n_experts, m->c.hidden,
                                                    ly->router) == 0)
                    uploaded++;
            }
        }
        int dense_uploaded = 0, dense_ok = want_dense;
        if (want_dense) {
            for (int l = 0; l < m->c.n_layers && dense_ok; l++) {
                Layer *ly = &m->L[l];
                QT *matrix[4] = {&ly->wq, &ly->wk, &ly->wv, &ly->wo};
                for (int j = 0; j < 4; j++) {
                    QT *q = matrix[j];
                    if (!q->qf || q->fmt != 0) continue; /* unowned pipeline layer */
                    if (pgr_native_dense_upload(q->qf, q->O, q->I, q) != 0) {
                        dense_ok = 0;
                        break;
                    }
                    dense_uploaded++;
                }
            }
            if (!dense_ok || dense_uploaded == 0) {
                pgr_native_dense_clear();
                dense_uploaded = 0;
                fprintf(stderr, "[gpu-dense] resident upload unavailable; using CPU attention\n");
            }
        }
        if (uploaded > 0 || dense_uploaded > 0) {
            g_pgpu_shutdown = pgr_native_shutdown;
            g_pgpu_router_upload = pgr_native_upload;
            g_pgpu_router_scores = pgr_native_scores;
            g_gpu_backend = 1;
            g_gpu_router = uploaded > 0;
            g_gpu_dense = dense_uploaded > 0;
            const char *release_host = getenv("GPU_DENSE_RELEASE_HOST");
            if (g_gpu_dense && release_host && atoi(release_host)) {
                /* Upload is transactional: no host weights are freed until every
                 * eligible projection is resident. Keys are stable QT addresses. */
                for (int l = 0; l < m->c.n_layers; l++) {
                    Layer *ly = &m->L[l];
                    QT *matrix[4] = {&ly->wq, &ly->wk, &ly->wv, &ly->wo};
                    for (int j = 0; j < 4; j++) {
                        QT *q = matrix[j];
                        if (q->fmt != 0 || !q->qf) continue;
                        uint64_t bytes = (uint64_t)q->O * q->I * sizeof(float);
                        free(q->qf);
                        q->qf = NULL;
                        g_gpu_dense_host_released += bytes;
                        m->resident_bytes -= (int64_t)bytes;
                    }
                }
                fprintf(stderr, "[gpu-dense] released %.2f GiB host weights; "
                        "GPU errors now stop inference (no CPU fallback)\n",
                        g_gpu_dense_host_released / 1073741824.0);
            }
            atexit(gpu_backend_shutdown);
            if (uploaded > 0)
                fprintf(stderr, "[gpu] %d routers resident in VRAM (native C backend)\n",
                        uploaded);
            if (dense_uploaded > 0)
                fprintf(stderr, "[gpu] %d attention matrices resident as FP16 "
                        "(%.2f GiB, native C backend)\n", dense_uploaded,
                        pgr_native_dense_bytes() / 1073741824.0);
            return;
        }
        pgr_native_shutdown();
    }
#ifdef _WIN32
    void *lib = (void *)LoadLibraryA("picchio_cuda.dll");
#else
    void *lib = dlopen("libpicchio_cuda.so", RTLD_NOW);
#endif
    if (!lib) {
        fprintf(stderr, "[gpu] backend library not found — using CPU\n");
        return;
    }
    pgpu_init_t init = (pgpu_init_t)(uintptr_t)gpu_sym(lib, "pgpu_init");
    pgpu_moe_t  moe  = (pgpu_moe_t)(uintptr_t)gpu_sym(lib, "pgpu_moe_layer");
    pgpu_moe_resident_t mres = (pgpu_moe_resident_t)(uintptr_t)
                               gpu_sym(lib, "pgpu_moe_resident");
    pgpu_i8_t   i8   = (pgpu_i8_t)(uintptr_t)gpu_sym(lib, "pgpu_matmul_i8");
    pgpu_router_upload_t rup = (pgpu_router_upload_t)(uintptr_t)
                              gpu_sym(lib, "pgpu_router_upload");
    pgpu_router_scores_t rsc = (pgpu_router_scores_t)(uintptr_t)
                              gpu_sym(lib, "pgpu_router_scores");
    g_pgpu_shutdown  = (pgpu_shutdown_t)(uintptr_t)gpu_sym(lib, "pgpu_shutdown");
    if (!init || !g_pgpu_shutdown) {
        fprintf(stderr, "[gpu] backend symbols missing — using CPU\n");
        return;
    }
    if (!init()) {
        fprintf(stderr, "[gpu] no CUDA device — using CPU\n");
        return;
    }
    g_gpu_backend = 1;
    atexit(gpu_backend_shutdown);
    if (want_router && rup && rsc) {
        int uploaded = 0;
        for (int l = 0; l < m->c.n_layers; l++) {
            Layer *ly = &m->L[l];
            if (ly->router && rup(ly->router, ly->router_bias,
                                  m->c.n_experts, m->c.hidden, ly->router) == 0)
                uploaded++;
        }
        if (uploaded > 0) {
            g_pgpu_router_upload = rup;
            g_pgpu_router_scores = rsc;
            g_gpu_router = 1;
            fprintf(stderr, "[gpu] %d routers resident in VRAM\n", uploaded);
        }
    } else if (want_router) {
        fprintf(stderr, "[gpu] router symbols missing — router stays on CPU\n");
    }
    if (want_experts && moe) {
        g_pgpu_moe = moe;
        g_pgpu_moe_resident = mres;  /* NULL with an older DLL → no read-skip */
        g_gpu_on = 1;
        fprintf(stderr, "[gpu] MoE expert offload enabled%s\n",
                mres ? " (VRAM-resident experts skip the disk read)" : "");
    }
    /* lm_head offload is opt-in: on weak GPUs it is memory-bound and slower than
     * the CPU (see DESIGN 0.16), and it is not the bottleneck. */
    if (want_lmhead && i8) {
        g_pgpu_i8 = i8;
        g_gpu_lmhead = 1;
        fprintf(stderr, "[gpu] lm_head also offloaded (GPU_LMHEAD=1)\n");
    }
}

/* Final norm + LM head → logits. The pipeline's last stage, after the layers. */
static void forward_head_logits(Model *m, const float *h, float *logits) {
    Cfg *c = &m->c;
    int D = c->hidden;
    float *hn = falloc(D);
    rmsnorm(hn, h, m->final_norm, D, c->eps);
    if (g_oracle_dir) oracle_dump_vec("final_norm", hn, D);
    double th = now_s();
    /* GPU lm_head (INT8 resident in VRAM), opt-in, with CPU fallback. */
    if (!(g_gpu_lmhead && m->lm_head.fmt == 1 &&
          g_pgpu_i8(logits, hn, m->lm_head.q8, m->lm_head.s,
                    m->lm_head.O, m->lm_head.I, m->lm_head.q8) == 0))
        matmul_qt(logits, hn, &m->lm_head, 1);
    trace_top_logits(logits, c->vocab);
    if (g_oracle_dir) oracle_dump_vec("logits", logits, c->vocab);
    m->t_head += now_s() - th;
    free(hn);
}

/* Post-generation report for SPEC_PROBE: n-gram acceptance + expert-union. */
static void spec_probe_report(void) {
    if (g_self_draft_probe && g_sd_total > 0) {
        g_sd_runhist[g_sd_run < 7 ? g_sd_run : 7]++;   /* flush the final run */
        double p = (double)g_sd_match / g_sd_total;
        fprintf(stderr, "\n── SELF_DRAFT_PROBE (top-1 draft vs top-4 target) ──\n");
        fprintf(stderr, "positions %ld · top-1==top-4: %ld (%.1f%% single-step acceptance)\n",
                g_sd_total, g_sd_match, 100.0 * p);
        fprintf(stderr, "accepted-run hist (consecutive matches):");
        for (int r = 0; r <= 7; r++) fprintf(stderr, " %d:%ld", r, g_sd_runhist[r]);
        fprintf(stderr, "\n  if i.i.d., expected accepted per draft-4 ≈ %.2f tokens "
                "(→ ~%.2f produced/verify with the bonus)\n",
                p + p*p + p*p*p + p*p*p*p, 1 + p + p*p + p*p*p + p*p*p*p);
    }
    if (!g_spec_probe || g_probe_ntok < 2) return;
    const int NKEY = 3, NDRAFT = 4;
    int L = g_probe_L, K = g_probe_K;

    /* n-gram (prompt-lookup) acceptance over the token-id stream. At each decode
     * position, propose the tokens after the most recent match of the last NKEY
     * tokens, then count how many match the real continuation. */
    long offered = 0, acc_sum = 0, dist[8] = {0};
    for (int p = g_probe_prompt_n; p < g_probe_hist_n; p++) {
        if (p < NKEY) continue;
        int found = -1;
        for (int i = p - NKEY - 1; i >= 0; i--) {
            int mt = 1;
            for (int j = 0; j < NKEY; j++)
                if (g_probe_hist[i + j] != g_probe_hist[p - NKEY + j]) { mt = 0; break; }
            if (mt) { found = i + NKEY; break; }
        }
        if (found < 0) continue;
        offered++;
        int a = 0;
        while (a < NDRAFT && found + a < g_probe_hist_n && p + a < g_probe_hist_n &&
               g_probe_hist[found + a] == g_probe_hist[p + a]) a++;
        acc_sum += a; dist[a]++;
    }

    fprintf(stderr, "\n── SPEC_PROBE (n-gram key=%d, max draft=%d) ──\n", NKEY, NDRAFT);
    fprintf(stderr, "decode tokens recorded: %d\n", g_probe_ntok);
    if (offered > 0) {
        fprintf(stderr, "n-gram acceptance: %ld drafts offered, mean %.2f accepted/draft\n",
                offered, (double)acc_sum / offered);
        fprintf(stderr, "  accepted-length hist:");
        for (int a = 0; a <= NDRAFT; a++) fprintf(stderr, " len%d=%ld", a, dist[a]);
        fprintf(stderr, "\n  → ~%.2f tokens per model-forward (accepted + 1 bonus, when drafting)\n",
                (double)(acc_sum + offered) / offered);
    } else {
        fprintf(stderr, "n-gram acceptance: no matches (too little repetition in this text)\n");
    }

    /* Expert-union across W consecutive decode tokens: how many unique experts a
     * batched verify of W tokens must read, vs one token. */
    double base = (double)L * K;
    fprintf(stderr, "expert-union (1 token = %.0f experts: %d layers x top-%d):\n",
            base, L, K);
    for (int W = 2; W <= 5; W++) {
        if (W > g_probe_ntok) break;
        double sum = 0; long wins = 0;
        for (int t = 0; t + W <= g_probe_ntok; t++) {
            long uniq = 0;
            for (int l = 0; l < L; l++) {
                int tmp[64], mct = 0;
                for (int w = 0; w < W; w++)
                    for (int k = 0; k < K; k++) {
                        int e = g_probe_exp[((size_t)(t + w) * L + l) * K + k];
                        int dup = 0;
                        for (int x = 0; x < mct; x++) if (tmp[x] == e) { dup = 1; break; }
                        if (!dup && mct < 64) tmp[mct++] = e;
                    }
                uniq += mct;
            }
            sum += uniq; wins++;
        }
        double avg = sum / wins;
        fprintf(stderr, "  verify %d tokens: %.1f unique experts = %.2fx one token"
                " (no-overlap would be %.0fx)\n", W, avg, avg / base, (double)W);
    }
}

static int forward_token(Model *m, int tok, int pos) {
    Cfg *c = &m->c;
    int D = c->hidden;

    float *h = falloc(D);           /* hidden state (residual stream) */
    embed_token(m, tok, h);
    trace_vector("embedding", -1, h, D);
    if (g_oracle_dir) oracle_dump_vec("embedding", h, D);

    run_layer_range(m, h, pos, 0, c->n_layers);

    float *logits = falloc(c->vocab);
    forward_head_logits(m, h, logits);
    int sampled = sample_logits(logits, c->vocab);

    free(h); free(logits);
    m->n_fw++;
    return sampled;
}

/* Byte-identity gate for the pipeline split: the monolithic forward must produce
 * logits bit-for-bit identical to the same forward cut at layer `cut`, with the
 * residual stream carried through a serialize/deserialize round-trip (the wire
 * format the two machines exchange). Any mismatch is a split/transport bug. */
static int pipe_split_check(Model *m, int tok, int pos, int cut) {
    Cfg *c = &m->c;
    int D = c->hidden, V = c->vocab;

    /* monolithic reference */
    float *hf = falloc(D);
    embed_token(m, tok, hf);
    run_layer_range(m, hf, pos, 0, c->n_layers);
    float *lf = falloc(V);
    forward_head_logits(m, hf, lf);

    /* split: stage A [0,cut) → wire → stage B [cut,L) → head */
    float *hs = falloc(D);
    embed_token(m, tok, hs);
    run_layer_range(m, hs, pos, 0, cut);
    unsigned char *wire = (unsigned char *)malloc((size_t)D * sizeof(float));
    memcpy(wire, hs, (size_t)D * sizeof(float));     /* stage A -> wire */
    memcpy(hs, wire, (size_t)D * sizeof(float));     /* wire -> stage B */
    free(wire);
    run_layer_range(m, hs, pos, cut, c->n_layers);
    float *ls = falloc(V);
    forward_head_logits(m, hs, ls);

    int same = (memcmp(lf, ls, (size_t)V * sizeof(float)) == 0);
    free(hf); free(lf); free(hs); free(ls);
    return same;
}

/* ═══════════════════════════════════════════════════════════
 *  BATCHED PREFILL (expert batch-union)
 *
 *  By processing several positions together, each unique expert of the layer is
 *  read only once instead of once per token. With a 4-slot-per-layer cache and
 *  top-4, the token-by-token path evicts the entire cache at every token.
 *  The math is identical to the sequential path: only the order of the reads
 *  changes, never the routing or the precision.
 * ═══════════════════════════════════════════════════════════ */

/* Apply an expert to x and accumulate w * output into dst. */
static void expert_apply(ESlot *es, const float *x, float *dst, float w,
                         const Cfg *c) {
    double compute_start = now_s();
    int D = c->hidden, I = c->moe_inter, half = I / 2;
    float *gu = falloc(I);
    float *eo = falloc(D);

    matmul_qt(gu, x, &es->gu, 1);
    if (es->gu_bias) for (int i = 0; i < I; i++) gu[i] += es->gu_bias[i];

    if (c->swiglu_clipped) {
        for (int i = 0; i < half; i++) {
            float gate = gu[2 * i], up = gu[2 * i + 1];
            if (gate > c->swiglu_limit) gate = c->swiglu_limit;
            if (up > c->swiglu_limit) up = c->swiglu_limit;
            if (up < -c->swiglu_limit) up = -c->swiglu_limit;
            gu[i] = (up + 1.0f) * gate * sigmoidf(c->swiglu_alpha * gate);
        }
    } else {
        for (int i = 0; i < half; i++) {
            float gate = gu[2 * i], up = gu[2 * i + 1];
            gu[i] = gate * sigmoidf(gate) * up;
        }
    }

    matmul_qt(eo, gu, &es->d, 1);
    if (es->d_bias) for (int i = 0; i < D; i++) eo[i] += es->d_bias[i];
    for (int i = 0; i < D; i++) dst[i] += w * eo[i];

    free(gu); free(eo);
    g_expert_compute_seconds += now_s() - compute_start;
}

/* Prefill n tokens starting at pos_base. Returns the token sampled from the last
 * position, as the last sequential forward_token would. */
/* Run the batched layers [l_lo, l_hi) over n positions, in place on H (n*D
 * residuals). Extracted from forward_prefill so the batched prefill can be split
 * at a layer boundary for the distributed pipeline; calling it with [0, n_layers)
 * is identical to the original monolithic loop. */
static void run_prefill_range(Model *m, float *H, int n, int pos_base,
                              int l_lo, int l_hi) {
    Cfg *c = &m->c;
    int D = c->hidden, E = c->n_experts, K = c->topk;

    float *XN = (float *)falloc((int64_t)n * D);
    float *hn = falloc(D);
    float *attn_out = falloc(D);
    int *sel_all = (int *)malloc((size_t)n * K * sizeof(int));
    float *w_all = (float *)malloc((size_t)n * K * sizeof(float));
    float *scores = falloc(E);
    int *uniq = (int *)malloc((size_t)E * sizeof(int));
    unsigned char *seen = (unsigned char *)malloc((size_t)E);
    if (!XN || !sel_all || !w_all || !uniq || !seen) {
        fprintf(stderr, "error: not enough memory for the batched prefill\n");
        exit(1);
    }

    for (int l = l_lo; l < l_hi; l++) {
        Layer *ly = &m->L[l];

        /* Attention: in position order, so the previous KV is ready. */
        double ta = now_s();
        for (int p = 0; p < n; p++) {
            float *h = H + (int64_t)p * D;
            rmsnorm(hn, h, ly->in_ln, D, c->eps);
            gqa_attention(attn_out, hn, ly, &m->kv, l, pos_base + p, c);
            for (int i = 0; i < D; i++) h[i] += attn_out[i];
        }
        m->t_attn += now_s() - ta;

        double tm = now_s();
        /* Routing of all positions, then the union of the requested experts. */
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
            router_weights(c, scores, sel, wgt, K, E);
            for (int k = 0; k < K; k++) {
                if (m->eusage[l]) m->eusage[l][sel[k]]++;
                if (m->eheat[l]) m->eheat[l][sel[k]]++;
                if (!seen[sel[k]]) { seen[sel[k]] = 1; uniq[n_uniq++] = sel[k]; }
            }
        }

        /* One unique expert → a single read, reused by all positions.
         * The buffer must be zeroed: expert_apply accumulates. The batch's unique
         * experts often exceed the cache capacity, so we proceed in blocks: each
         * block is read from disk in parallel (queue depth > 1) and then applied
         * to all positions.
         *
         * With prefetch enabled (PREFETCH=1) the blocks are halved (two fit in the
         * cache) and a double-buffer pipeline runs: while the current block is
         * computed (matmul over n positions, disk idle), the next block — already
         * known in uniq[], hence an EXACT prefetch with no prediction — is loaded
         * by the prefetch thread. The math does not change. */
        float *moe = (float *)calloc((size_t)n * D, sizeof(float));
        if (!moe) { fprintf(stderr, "error: MoE memory\n"); exit(1); }
        int chunk = m->ecap < 128 ? m->ecap : 128;
        if (g_pilot_enabled) {
            int half = m->ecap / 2;                 /* two blocks in cache */
            int cap  = m->ecap - c->topk;           /* prefetch_issue invariant */
            if (half < 1) half = 1;
            if (cap >= 1 && half > cap) half = cap;
            if (half >= 1) chunk = half;
        }
        ESlot *slots[128];
        for (int base = 0; base < n_uniq; base += chunk) {
            int cnt = n_uniq - base < chunk ? n_uniq - base : chunk;
            /* Make the current block available: synchronous on the first round,
             * otherwise served (with a wait) by the previous round's prefetch. */
            cache_load_batch(m, l, uniq + base, cnt, slots);
            /* Start the async load of the next block, which overlaps with the
             * computation of this block. */
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

    free(XN); free(hn); free(attn_out); free(sel_all); free(w_all);
    free(scores); free(uniq); free(seen);
}

static int forward_prefill(Model *m, const int *ids, int n, int pos_base) {
    Cfg *c = &m->c;
    int D = c->hidden;

    float *H = (float *)falloc((int64_t)n * D);
    if (!H) {
        fprintf(stderr, "error: not enough memory for the batched prefill\n");
        exit(1);
    }
    for (int p = 0; p < n; p++) embed_token(m, ids[p], H + (int64_t)p * D);

    run_prefill_range(m, H, n, pos_base, 0, c->n_layers);

    /* Only the last position produces the useful logits. */
    float *logits = falloc(c->vocab);
    forward_head_logits(m, H + (int64_t)(n - 1) * D, logits);
    int sampled = sample_logits(logits, c->vocab);

    free(H); free(logits);
    m->n_fw += n;
    return sampled;
}

/* Speculative verify: run n tokens through all layers in ONE batched forward
 * (expert-union) and return, in pred[p], the greedy argmax the model predicts
 * AFTER consuming toks[p]. Same math as n sequential forward_token calls; the
 * only difference is the read order (each unique expert loaded once). Writes KV
 * for positions [pos_base, pos_base+n). */
static void forward_verify(Model *m, const int *toks, int n, int pos_base, int *pred) {
    Cfg *c = &m->c; int D = c->hidden, V = c->vocab;
    float *H = falloc((int64_t)n * D);
    for (int p = 0; p < n; p++) embed_token(m, toks[p], H + (int64_t)p * D);
    run_prefill_range(m, H, n, pos_base, 0, c->n_layers);
    float *logits = falloc(V);
    for (int p = 0; p < n; p++) {
        forward_head_logits(m, H + (int64_t)p * D, logits);
        int best = 0; float bs = logits[0];
        for (int v = 1; v < V; v++) if (logits[v] > bs) { bs = logits[v]; best = v; }
        pred[p] = best;
    }
    free(H); free(logits);
    m->n_fw += n;
}

/* Block prefill. Falls back to the sequential path when the oracle dumps, the
 * tracing, or a repetition penalty that depends on the history are needed. */
static int prefill_tokens(Model *m, const int *ids, int n, int pos_base) {
    int batch = 64;
    { const char *v = getenv("PREFILL_BATCH"); if (v) batch = atoi(v); }
    /* The repetition penalty no longer forces the slow sequential path. Prefill
     * only ENCODES a fixed prompt — it never samples — so the penalty is
     * irrelevant to the encoding itself. The only real difference was that the
     * sequential path recorded, in g_history, the token PREDICTED at each prompt
     * position (an arbitrary basis for the penalty), whereas the batched path
     * samples only once. We therefore keep the batched path even under rep
     * penalty and instead seed g_history with the ACTUAL prompt tokens — the
     * standard behavior (penalize repeating the context) and arguably more
     * correct than recording per-position predictions. */
    int sequential = g_oracle_dir || g_trace_numeric || g_predict_probe
                     || batch <= 1 || n <= 1;

    int last = ids[0];
    if (sequential) {
        for (int i = 0; i < n; i++) {
            last = forward_token(m, ids[i], pos_base + i);
            if (last < 0 || last >= m->c.vocab) last = 0;
        }
        return last;
    }

    /* Seed the repetition-penalty history with the real prompt tokens so decode
     * penalizes repeating the context (only relevant when REP>1; harmless
     * otherwise). forward_prefill still appends the sampled next-token per block. */
    if (g_rep_penalty > 1.0f) {
        for (int i = 0; i < n; i++)
            if (g_history_len < 4096 && ids[i] >= 0 && ids[i] < m->c.vocab)
                g_history[g_history_len++] = ids[i];
    }

    for (int off = 0; off < n; off += batch) {
        int len = n - off < batch ? n - off : batch;
        last = forward_prefill(m, ids + off, len, pos_base + off);
        if (last < 0 || last >= m->c.vocab) last = 0;
    }
    return last;
}

/* ═══════════════════════════════════════════════════════════
 *  STATISTICS
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
    if (g_flat.reads > 0)
        fprintf(stderr, "flat reads: %llu (%.2f GB aligned)\n",
                (unsigned long long)g_flat.reads, g_flat.bytes / 1e9);
    if (g_flat.direct_fallbacks)
        fprintf(stderr, "flat direct fallbacks: %llu\n",
                (unsigned long long)g_flat.direct_fallbacks);
    if (g_flat.corruptions)
        fprintf(stderr, "flat corruptions: %llu (safetensors fallback used)\n",
                (unsigned long long)g_flat.corruptions);
    if (m->async_jobs > 0)
        fprintf(stderr, "async MoE:  %llu batches, %.3f s exposed wait\n",
                (unsigned long long)m->async_jobs, m->t_async_wait);
    fprintf(stderr, "t_attn:     %.2f s\n", m->t_attn);
    fprintf(stderr, "t_moe:      %.2f s\n", m->t_moe);
    fprintf(stderr, "t_head:     %.2f s\n", m->t_head);
    fprintf(stderr, "RSS:        %.2f GB\n", rss_gb());
    fprintf(stderr, "resident:   %.2f GB (dense model)\n",
            m->resident_bytes / 1e9);
    if (g_gpu_router_calls > 0)
        fprintf(stderr, "GPU router: %llu calls, %.3f ms/call, %.3f s total\n",
                (unsigned long long)g_gpu_router_calls,
                1000.0 * g_gpu_router_time / g_gpu_router_calls,
                g_gpu_router_time);
    if (g_gpu_dense_calls > 0)
        fprintf(stderr, "GPU dense:  %llu calls, %.3f ms/call, %.3f s total, "
                "%llu fallbacks\n",
                (unsigned long long)g_gpu_dense_calls,
                1000.0 * g_gpu_dense_time / g_gpu_dense_calls,
                g_gpu_dense_time,
                (unsigned long long)g_gpu_dense_fallbacks);
    if (m->pred_total > 0) {
        double accA = 100.0 * m->pred_hit  / m->pred_total;
        double accB = 100.0 * m->pred_hit2 / m->pred_total;
        double baseline = 100.0 * m->c.topk / m->c.n_experts;  /* expected random overlap */
        fprintf(stderr, "prefetch predict (%llu samples, random ~%.1f%%):\n",
                (unsigned long long)m->pred_total, baseline);
        fprintf(stderr, "  proxy A (hn pre-MoE):   %.1f%% correct experts\n", accA);
        fprintf(stderr, "  proxy B (post-MoE norm): %.1f%% correct experts\n", accB);
    }
}

/* Machine-readable cumulative counters for SERVICE clients. Keeping these
 * cumulative makes the protocol cheap: callers snapshot around each TURN and
 * calculate deltas without resetting global profiling state. */
static void service_stats(Model *m, int pos, int cap) {
    uint64_t cache_total = m->hits + m->miss;
    int64_t expert_bytes = expert_slot_estimate_bytes(&m->c);
    uint64_t process_faults = 0;
    double private_gib = 0, peak_rss_gib = 0;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc))) {
        process_faults = pmc.PageFaultCount; /* includes SOFT faults, not disk paging */
        private_gib = pmc.PrivateUsage / 1073741824.0;
        peak_rss_gib = pmc.PeakWorkingSetSize / 1073741824.0;
    }
#endif
    printf("STATS {"
           "\"position\":%d,\"capacity\":%d,"
           "\"prefill_tokens\":%llu,\"decode_tokens\":%llu,"
           "\"prefill_seconds\":%.9f,\"decode_seconds\":%.9f,"
           "\"gpu_dense_host_released_bytes\":%llu,\"expert_slots_per_layer\":%d,"
           "\"available_ram_gib\":%.6f,"
           "\"expert_buffer_allocs\":%llu,\"expert_buffer_reuses\":%llu,"
           "\"expert_compute_seconds\":%.9f,"
           "\"process_page_faults\":%llu,\"private_gib\":%.6f,\"peak_rss_gib\":%.6f,"
           "\"forward\":%llu,\"tokens_emitted\":%llu,"
           "\"cache_hits\":%llu,\"cache_misses\":%llu,"
           "\"cache_requests\":%llu,\"expert_loads\":%llu,"
           "\"expert_bytes_estimate\":%lld,\"disk_seconds\":%.9f,"
           "\"async_batches\":%llu,\"async_wait_seconds\":%.9f,"
           "\"attention_seconds\":%.9f,\"moe_seconds\":%.9f,"
           "\"head_seconds\":%.9f,\"rss_gb\":%.6f,"
           "\"resident_gb\":%.6f,\"gpu_router_calls\":%llu,"
           "\"gpu_router_seconds\":%.9f,\"gpu_dense_calls\":%llu,"
           "\"gpu_dense_seconds\":%.9f,\"gpu_dense_fallbacks\":%llu,"
           "\"prefetch_issued\":%llu,"
           "\"prefetch_read_seconds\":%.9f,\"predict_hits_a\":%llu,"
           "\"predict_hits_b\":%llu,\"predict_total\":%llu}\n",
           pos, cap,
           (unsigned long long)g_service_prefill_tokens,
           (unsigned long long)g_service_decode_tokens,
           g_service_prefill_seconds, g_service_decode_seconds,
           (unsigned long long)g_gpu_dense_host_released, m->ecap,
           available_ram_bytes() / 1073741824.0,
           (unsigned long long)__atomic_load_n(&g_expert_buffer_allocs, __ATOMIC_RELAXED),
           (unsigned long long)__atomic_load_n(&g_expert_buffer_reuses, __ATOMIC_RELAXED),
           g_expert_compute_seconds,
           (unsigned long long)process_faults, private_gib, peak_rss_gib,
           (unsigned long long)m->n_fw,
           (unsigned long long)m->n_emit,
           (unsigned long long)m->hits,
           (unsigned long long)m->miss,
           (unsigned long long)cache_total,
           (unsigned long long)m->ereq,
           (long long)expert_bytes, m->t_edisk,
           (unsigned long long)m->async_jobs, m->t_async_wait,
           m->t_attn, m->t_moe, m->t_head, rss_gb(),
           m->resident_bytes / 1e9,
           (unsigned long long)g_gpu_router_calls, g_gpu_router_time,
           (unsigned long long)g_gpu_dense_calls, g_gpu_dense_time,
           (unsigned long long)g_gpu_dense_fallbacks,
           (unsigned long long)g_pf.issued, g_pf.read_seconds,
           (unsigned long long)m->pred_hit,
           (unsigned long long)m->pred_hit2,
           (unsigned long long)m->pred_total);
    fflush(stdout);
}

/* ═══════════════════════════════════════════════════════════
 *  PERSISTENT HOT-STORE (.picchio_usage)
 *
 *  Saves the routing counters between sessions.
 *  On the next startup, the hottest experts are preloaded.
 * ═══════════════════════════════════════════════════════════ */

static const char *g_model_path_global = NULL;

static void hotstore_save(Model *m) {
    if (!g_model_path_global) return;
    Cfg *c = &m->c;
    
    char path[512];
    snprintf(path, sizeof(path), "%s/.picchio_usage", g_model_path_global);
    
    FILE *f = fopen(path, "wb");
    if (!f) return;
    
    /* Header: magic + version + dimensions */
    uint32_t magic = 0x50494343;  /* "PICC" */
    uint32_t version = 1;
    uint32_t n_layers = (uint32_t)c->n_layers;
    uint32_t n_experts = (uint32_t)c->n_experts;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&n_layers, 4, 1, f);
    fwrite(&n_experts, 4, 1, f);
    
    /* Per-layer counters */
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
    fprintf(stderr, "  hot-store saved: %s\n", path);
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
        fprintf(stderr, "  hot-store: dimensions do not match, ignoring\n");
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
    fprintf(stderr, "  hot-store loaded: %llu routings recorded\n",
            (unsigned long long)total_usage);
}

/* ═══════════════════════════════════════════════════════════
 *  SELF-TEST: synthetic mini-model to validate the forward pass
 * ═══════════════════════════════════════════════════════════ */

/* Generate deterministic random weights (simple LCG) */
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
    t->fmt = 0;  /* F32 for the self-test (no quantization) */
    t->O = O;
    t->I = I;
    t->qf = falloc((int64_t)O * I);
    fill_random_f32(t->qf, (int64_t)O * I);
}

/* Build a synthetic mini-model (2 layers, D=64, 4 experts) fully resident in RAM.
 * Shared by the self-test and the pipeline self-test so both exercise the exact
 * same forward math with no disk or config dependency. */
static void build_synth_model(Model *m) {
    Cfg *c = &m->c;

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
    /* GPT-OSS architecture flags (this synthetic model mirrors GPT-OSS). */
    c->swiglu_limit = 7.0f;
    c->swiglu_alpha = 1.702f;
    c->swiglu_clipped = 1;
    c->use_sinks = 1;
    c->router_norm = 0;
    c->qk_norm = 0;
    c->layer_type[0] = 0;  /* sliding */
    c->layer_type[1] = 1;  /* full */

    int D = c->hidden;
    int H = c->n_heads;
    int KVH = c->n_kv_heads;
    int hd = c->head_dim;

    fprintf(stderr, "  D=%d L=%d H=%d KV=%d hd=%d E=%d top%d vocab=%d\n",
            D, c->n_layers, H, KVH, hd, c->n_experts, c->topk, c->vocab);

    /* Allocate embedding and lm_head */
    fill_random_qt_f32(&m->embed, c->vocab, D);
    fill_random_qt_f32(&m->lm_head, c->vocab, D);
    m->final_norm = falloc(D);
    for (int i = 0; i < D; i++) m->final_norm[i] = 1.0f;  /* norms = 1 for the test */

    /* Allocate layers */
    m->L = calloc(c->n_layers, sizeof(Layer));
    for (int l = 0; l < c->n_layers; l++) {
        Layer *ly = &m->L[l];
        ly->layer_type = c->layer_type[l];

        /* RMSNorm weights (all 1.0 for the test) */
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
    kv_init(&m->kv, c->n_layers, KVH, hd, c->ctx_len);

    /* Expert cache — for the self-test, preload all experts into RAM */
    m->ecap = c->n_experts;  /* enough slots for all */
    m->ecache = calloc(c->n_layers, sizeof(ESlot *));
    m->ecn = calloc(c->n_layers, sizeof(int));
    m->pin = calloc(c->n_layers, sizeof(ESlot *));
    m->npin = calloc(c->n_layers, sizeof(int));
    m->eusage = calloc(c->n_layers, sizeof(uint32_t *));
    m->eheat = calloc(c->n_layers, sizeof(uint32_t *));

    for (int l = 0; l < c->n_layers; l++) {
        m->ecache[l] = calloc(m->ecap, sizeof(ESlot));
        m->ecn[l] = c->n_experts;  /* all preloaded */
        m->eusage[l] = calloc(c->n_experts, sizeof(uint32_t));
        m->eheat[l] = calloc(c->n_experts, sizeof(uint32_t));

        for (int e = 0; e < c->n_experts; e++) {
            ESlot *es = &m->ecache[l][e];
            es->eid = e;
            es->layer = l;
            /* gate_up fused: [moe_inter, D] */
            fill_random_qt_f32(&es->gu, c->moe_inter, D);
            /* down: [D, moe_inter/2] — the input to down is moe_inter/2 post-SwiGLU values */
            fill_random_qt_f32(&es->d, D, c->moe_inter / 2);
            /* Bias */
            es->gu_bias = falloc(c->moe_inter);
            fill_random_f32(es->gu_bias, c->moe_inter);
            es->d_bias = falloc(D);
            fill_random_f32(es->d_bias, D);
        }
    }

    fprintf(stderr, "  structures allocated ✓\n");
}

static int self_test(void) {
    fprintf(stderr, "── self-test: synthetic mini-model ──\n\n");

    /* Mini configuration: 2 layers, D=64, 4 heads, 2 KV heads, 4 experts, top-2 */
    static Model m;
    memset(&m, 0, sizeof(m));
    Cfg *c = &m.c;

    build_synth_model(&m);

    /* ── Forward pass: 8 tokens ── */
    fprintf(stderr, "  forward pass: 8 tokens...\n");

    int tokens[] = {1, 5, 3, 12, 7, 2, 9, 4};
    int n_tok = 8;

    double t0 = now_s();
    for (int i = 0; i < n_tok; i++) {
        int next = forward_token(&m, tokens[i], i);
        fprintf(stderr, "    pos=%d tok_in=%d → tok_out=%d\n", i, tokens[i], next);
        if (next < 0 || next >= c->vocab) {
            fprintf(stderr, "  ✗ ERROR: token out of range [0, %d)\n", c->vocab);
            return 1;
        }
    }
    double elapsed = now_s() - t0;

    fprintf(stderr, "\n  ✓ %d forward in %.3f ms (%.1f tok/s)\n",
            n_tok, elapsed * 1000.0, n_tok / elapsed);

    /* Verify statistics */
    fprintf(stderr, "  cache: hit=%llu miss=%llu (%.0f%%)\n",
            (unsigned long long)m.hits, (unsigned long long)m.miss,
            m.hits + m.miss > 0 ? 100.0 * m.hits / (m.hits + m.miss) : 0.0);
    fprintf(stderr, "  t_attn=%.3fms t_moe=%.3fms t_head=%.3fms\n",
            m.t_attn * 1000, m.t_moe * 1000, m.t_head * 1000);

    /* ── Test individual components ── */
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
            fprintf(stderr, " ✗ expected %.4f, got %.4f\n", expected0, o[0]);
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
            fprintf(stderr, " ✗ sum=%.6f\n", sum);
            return 1;
        }
        if (x[2] < x[1] || x[1] < x[0]) {
            fprintf(stderr, " ✗ wrong order\n");
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
            fprintf(stderr, " ✗ y=[%.2f, %.2f] expected [1, 2]\n", y[0], y[1]);
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
            fprintf(stderr, " ✗ SiLU(1)=%.4f expected %.4f\n", s, expected);
            return 1;
        }
        fprintf(stderr, " ✓\n");
    }

    fprintf(stderr, "  test async MoE canonical reduction...");
    {
        float x[64], sync_out[64], async_out[64];
        for (int i = 0; i < c->hidden; i++)
            x[i] = (float)((i % 13) - 6) * 0.03125f;
        g_async_moe_enabled = 0;
        moe_forward(sync_out, x, &m, 0, c);
        g_async_moe_enabled = 1;  /* all synthetic experts are resident */
        moe_forward(async_out, x, &m, 0, c);
        g_async_moe_enabled = 0;
        if (memcmp(sync_out, async_out,
                   (size_t)c->hidden * sizeof(float)) != 0) {
            fprintf(stderr, " FAIL: async output differs from sync\n");
            return 1;
        }
        fprintf(stderr, " OK (byte-identical)\n");
    }

    fprintf(stderr, "  test RoPE...");
    {
        /* Check that RoPE at pos=0 changes nothing (cos=1, sin=0) */
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
        /* At pos=0: ang=0 for all → cos=1, sin=0 → no change */
        int ok = 1;
        for (int i = 0; i < 16; i++)
            if (fabsf(q[i] - q_orig[i]) > 1e-5f) { ok = 0; break; }
        for (int i = 0; i < 8; i++)
            if (fabsf(k[i] - k_orig[i]) > 1e-5f) { ok = 0; break; }
        if (!ok) {
            fprintf(stderr, " ✗ RoPE pos=0 modified the vectors\n");
            return 1;
        }
        fprintf(stderr, " ✓\n");
    }

    fprintf(stderr, "  test INT4 quant/matmul...");
    {
        /* Quantize an F32 matrix, then matmul and compare */
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

        /* The quantization error should be small */
        float max_err = 0;
        for (int i = 0; i < O; i++) {
            float err = fabsf(y_ref[i] - y_q4[i]);
            if (err > max_err) max_err = err;
        }
        if (max_err > 2.0f) {  /* wide tolerance for INT4 */
            fprintf(stderr, " ✗ max error=%.2f\n", max_err);
            return 1;
        }
        fprintf(stderr, " ✓ (err max=%.4f)\n", max_err);
    }

    fprintf(stderr, "  test INT3 quant/matmul...");
    {
        /* INT3 gs64: quantize an F32 matrix (2 groups/row), matmul, compare. */
        int O = 4, I = 128;                 /* I multiple of 64 */
        int ng = (I + 63) / 64;
        float W[512], x[128];
        for (int i = 0; i < O * I; i++) W[i] = (float)((i % 32) - 16) * 0.1f;
        for (int i = 0; i < I; i++) x[i] = 1.0f;

        uint8_t q3[4 * 2 * 24];             /* O * ng * 24 */
        float scale[4 * 2];                 /* O * ng */
        quantize_rows_i3_gs(W, q3, scale, O, I);

        float y_ref[4], y_q3[4];
        matmul_f32(y_ref, x, W, 1, I, O);
        matmul_i3_gs(y_q3, x, q3, scale, 1, I, O, 64);

        float max_err = 0;
        for (int i = 0; i < O; i++) {
            float err = fabsf(y_ref[i] - y_q3[i]);
            if (err > max_err) max_err = err;
        }
        (void)ng;
        if (max_err > 6.0f) {               /* wide tolerance: int3 is lossy */
            fprintf(stderr, " ✗ max error=%.2f\n", max_err);
            return 1;
        }
        fprintf(stderr, " ✓ (err max=%.4f)\n", max_err);
    }

    fprintf(stderr, "  test pipeline split (byte-identity)...");
    {
        int cut = c->n_layers / 2; if (cut < 1) cut = 1;
        int tks[3] = {6, 11, 2};
        int ok = 1;
        for (int i = 0; i < 3; i++)
            if (!pipe_split_check(&m, tks[i], 8 + i, cut)) { ok = 0; break; }
        if (!ok) {
            fprintf(stderr, " ✗ split logits differ from the monolithic forward\n");
            return 1;
        }
        fprintf(stderr, " ✓ (cut@%d: layers [0,%d)+[%d,%d) == monolithic)\n",
                cut, cut, cut, c->n_layers);
    }

    fprintf(stderr, "\n── self-test PASSED ──\n");
    fprintf(stderr, "  The full forward pass works correctly.\n");
    fprintf(stderr, "  Ready to connect the real model.\n\n");

    /* TODO: free everything (does not matter for the self-test) */
    return 0;
}

/* ═══════════════════════════════════════════════════════════
 *  WEIGHT LOADER: loads the dense weights from the safetensors DB
 * ═══════════════════════════════════════════════════════════ */

/* Helper: load an F32 tensor from safetensors.
 * If the tensor is BF16, convert to F32. */
/* Set while probing optional/alternate tensor names (router aliases, sinks that
 * a family legitimately lacks) so a normal miss doesn't print a scary warning. */
static int g_quiet_missing = 0;

static float *load_f32_tensor(StDB *db, const char *name, int64_t expected_numel) {
    StTensor *t = st_find(db, name);
    if (!t) {
        if (!g_quiet_missing)
            fprintf(stderr, "  ⚠ tensor not found: %s\n", name);
        return NULL;
    }
    int64_t numel = st_numel(t);
    if (expected_numel > 0 && numel != expected_numel) {
        fprintf(stderr, "  ⚠ %s: numel=%lld expected=%lld\n",
                name, (long long)numel, (long long)expected_numel);
    }

    float *out = falloc(numel);
    int64_t nbytes = st_bytes(t);

    if (t->dtype == ST_F32) {
        st_read_raw(db, t, out, nbytes);
    } else if (t->dtype == ST_BF16) {
        /* BF16 → F32: each BF16 is the high 16 bits of an F32 */
        uint16_t *tmp = (uint16_t *)malloc(numel * 2);
        st_read_raw(db, t, tmp, nbytes);
        for (int64_t i = 0; i < numel; i++) {
            uint32_t bits = (uint32_t)tmp[i] << 16;
            memcpy(&out[i], &bits, 4);
        }
        free(tmp);
    } else if (t->dtype == ST_F16) {
        /* F16 → F32 (minimal conversion) */
        uint16_t *tmp = (uint16_t *)malloc(numel * 2);
        st_read_raw(db, t, tmp, nbytes);
        for (int64_t i = 0; i < numel; i++) {
            uint16_t h = tmp[i];
            uint32_t sign = (h & 0x8000) << 16;
            uint32_t exp  = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t f;
            if (exp == 0) {
                f = sign; /* ±0 or denorm → 0 */
            } else if (exp == 31) {
                f = sign | 0x7F800000 | (mant << 13); /* inf/nan */
            } else {
                f = sign | ((exp + 112) << 23) | (mant << 13);
            }
            memcpy(&out[i], &f, 4);
        }
        free(tmp);
    } else {
        fprintf(stderr, "  ⚠ %s: unsupported dtype (%d)\n", name, t->dtype);
        free(out);
        return NULL;
    }

    return out;
}

/* Helper: load a tensor and quantize it to INT4.
 * Loads as F32, then quantizes in-place. Returns a QT. */
static int load_qt_i4(StDB *db, const char *name, QT *qt, int O, int I) {
    StTensor *t = st_find(db, name);
    if (!t) {
        fprintf(stderr, "  ⚠ not found: %s\n", name);
        return -1;
    }

    memset(qt, 0, sizeof(*qt));
    qt->fmt = 2;
    qt->O = O;
    qt->I = I;

    if (t->dtype == ST_U8) {
        /* Already INT4 packed by the converter */
        int64_t rb = (int64_t)O * ((I + 1) / 2);
        qt->q4 = (uint8_t *)malloc(rb);
        st_read_raw(db, t, qt->q4, rb);

        /* Associated scales */
        char sn[300];
        snprintf(sn, sizeof(sn), "%s.qs", name);
        StTensor *ts = st_find(db, sn);
        if (ts) {
            int64_t n_scale = st_numel(ts);
            qt->s = (float *)malloc(n_scale * sizeof(float));
            st_read_raw(db, ts, qt->s, n_scale * sizeof(float));
            /* Determine whether it is group-scaled: n_scale > O means gs */
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
        /* INT8 per row: used for embedding and lm_head, which at INT4 degrade
         * the logit distribution. */
        int64_t n = (int64_t)O * I;
        qt->fmt = 1;
        qt->q8 = (int8_t *)malloc(n);
        if (!qt->q8) return -1;
        st_read_raw(db, t, qt->q8, n);

        char sn[300];
        snprintf(sn, sizeof(sn), "%s.qs", name);
        StTensor *ts = st_find(db, sn);
        if (!ts) {
            fprintf(stderr, "  ⚠ missing INT8 scales: %s\n", sn);
            free(qt->q8); qt->q8 = NULL;
            return -1;
        }
        int64_t n_scale = st_numel(ts);
        if (n_scale != O) {
            fprintf(stderr, "  ⚠ %s: %lld scales for %d rows\n", sn,
                    (long long)n_scale, O);
            free(qt->q8); qt->q8 = NULL;
            return -1;
        }
        qt->s = (float *)malloc((size_t)O * sizeof(float));
        st_read_raw(db, ts, qt->s, (int64_t)O * (int64_t)sizeof(float));
        return 0;
    }

    /* F32/BF16/F16: load */
    float *f = load_f32_tensor(db, name, (int64_t)O * I);
    if (!f) return -1;

    /* If the tensor is F32 in the file, keep it F32 (attention, do not quantize) */
    if (t->dtype == ST_F32) {
        qt->fmt = 0;
        qt->qf = f;
        return 0;
    }

    /* BF16/F16 → quantize to INT4 */
    qt->q4 = (uint8_t *)malloc((int64_t)O * ((I + 1) / 2));
    qt->s = falloc(O);
    quantize_rows_i4(f, qt->q4, qt->s, O, I);
    free(f);
    return 0;
}

/* Load all resident dense weights (embedding, attention, norms, router) */
/* Distributed pipeline: the layer range this process owns, and whether it holds
 * the embedding (stage A) and the head + final norm (stage B). Defaults describe a
 * full single-node model; the PIPE_ROLE setup in main narrows them for a stage so
 * each node loads only its own layers (the RAM-pooling win of Step 2b). */
static int g_pipe_lo = 0;
static int g_pipe_hi = 1 << 30;   /* clamped to n_layers after cfg_load */
static int g_pipe_cut = 0;        /* the run-time split boundary (coord runs [0,cut)) */
static int g_load_embed = 1;
static int g_load_head = 1;

static int load_dense_weights(Model *m, StDB *db) {
    Cfg *c = &m->c;
    int D = c->hidden;
    int H = c->n_heads;
    int KVH = c->n_kv_heads;
    int hd = c->head_dim;
    int64_t loaded = 0;

    fprintf(stderr, "\n  loading dense weights (layers [%d,%d)%s%s)...\n",
            g_pipe_lo, g_pipe_hi < c->n_layers ? g_pipe_hi : c->n_layers,
            g_load_embed ? " +embed" : "", g_load_head ? " +head" : "");

    /* ── Embedding (stage A, or full model) ── */
    if (g_load_embed) {
        fprintf(stderr, "    embed_tokens [%d, %d]...", c->vocab, D);
        if (load_qt_i4(db, "model.embed_tokens.weight", &m->embed, c->vocab, D) == 0) {
            loaded += qt_bytes(&m->embed);
            fprintf(stderr, " ✓ (%.1f MB)\n", qt_bytes(&m->embed) / 1e6);
        } else {
            fprintf(stderr, " ✗\n");
            return -1;
        }
    }

    /* ── LM head + final norm (stage B, or full model) ── */
    if (g_load_head) {
        fprintf(stderr, "    lm_head [%d, %d]...", c->vocab, D);
        if (load_qt_i4(db, "lm_head.weight", &m->lm_head, c->vocab, D) == 0) {
            loaded += qt_bytes(&m->lm_head);
            fprintf(stderr, " ✓ (%.1f MB)\n", qt_bytes(&m->lm_head) / 1e6);
        } else {
            fprintf(stderr, " ✗\n");
            return -1;
        }

        m->final_norm = load_f32_tensor(db, "model.norm.weight", D);
        if (!m->final_norm) {
            fprintf(stderr, "    ⚠ final_norm not found, using 1.0\n");
            m->final_norm = falloc(D);
            for (int i = 0; i < D; i++) m->final_norm[i] = 1.0f;
        }
        loaded += D * 4;
    }

    /* ── Per-layer weights (only the layers this stage owns) ── */
    for (int l = 0; l < c->n_layers; l++) {
        if (l < g_pipe_lo || l >= g_pipe_hi) continue;   /* another stage's layer */
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

        /* QK-Norm weights (Qwen3): one [head_dim] vector shared across heads. */
        if (c->qk_norm) {
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", l);
            ly->q_norm = load_f32_tensor(db, name, hd);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", l);
            ly->k_norm = load_f32_tensor(db, name, hd);
            if (!ly->q_norm || !ly->k_norm) {
                fprintf(stderr, "  error: QK-Norm weights missing at layer %d\n", l);
                return -1;
            }
            loaded += 2 * hd * 4;
        }

        /* Attention sink logits (one per query head). Mandatory for GPT-OSS,
         * absent in Qwen3 (the attention code guards on a NULL sinks pointer), so
         * probe quietly and let the use_sinks check below report a real problem. */
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.sinks", l);
        g_quiet_missing = 1;
        ly->sinks = load_f32_tensor(db, name, H);
        g_quiet_missing = 0;
        if (!ly->sinks && c->use_sinks) {
            fprintf(stderr, "  error: attention sinks missing at layer %d\n", l);
            return -1;
        }
        if (ly->sinks) loaded += H * 4;

        /* Router weights: the tensor name differs by family, so probe the aliases
         * quietly and only complain if none of them exists. */
        g_quiet_missing = 1;
        snprintf(name, sizeof(name), "model.layers.%d.mlp.router.weight", l);
        ly->router = load_f32_tensor(db, name, (int64_t)c->n_experts * D);
        if (!ly->router) {  /* Qwen3: mlp.gate.weight; Mixtral: block_sparse_moe.gate */
            snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.weight", l);
            ly->router = load_f32_tensor(db, name, (int64_t)c->n_experts * D);
        }
        if (!ly->router) {
            snprintf(name, sizeof(name), "model.layers.%d.block_sparse_moe.gate.weight", l);
            ly->router = load_f32_tensor(db, name, (int64_t)c->n_experts * D);
        }
        snprintf(name, sizeof(name), "model.layers.%d.mlp.router.bias", l);
        ly->router_bias = load_f32_tensor(db, name, c->n_experts);  /* GPT-OSS only */
        g_quiet_missing = 0;
        if (!ly->router) {
            fprintf(stderr, "  error: router weights missing at layer %d\n", l);
            return -1;
        }
        loaded += (int64_t)c->n_experts * D * 4;
        if (ly->router_bias) loaded += c->n_experts * 4;

        if (l % 6 == 0 || l == c->n_layers - 1)
            fprintf(stderr, "    layer %d/%d ✓\n", l + 1, c->n_layers);
    }

    m->resident_bytes = loaded;
    fprintf(stderr, "  ✓ dense weights loaded: %.2f GB\n", loaded / 1e9);
    return 0;
}

/* Reads decimal IDs separated by whitespace. The caller frees *out_ids. */
static int read_token_ids_file(const char *path, int max_tokens, int **out_ids) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open INPUT_FILE=%s\n", path);
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
            fprintf(stderr, "error: invalid ID or prompt beyond %d tokens in %s\n",
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
        fprintf(stderr, "error: non-numeric content in INPUT_FILE=%s\n", path);
        free(ids); fclose(f); return -1;
    }
    fclose(f);
    if (n == 0) {
        fprintf(stderr, "error: empty INPUT_FILE: %s\n", path);
        free(ids); return -1;
    }
    *out_ids = ids;
    return n;
}

/* ═══════════════════════════════════════════════════════════
 *  DISTRIBUTED PIPELINE (Step 2): long-lived TCP stages
 *
 *  A model split at a layer boundary. The coordinator (stage A) owns layers
 *  [0, cut): it embeds the token, runs run_layer_range, and ships the residual
 *  stream (D floats) to the worker (stage B), which owns [cut, n_layers), resumes
 *  the forward, applies the head, samples, and returns the token. The residual is
 *  the only thing on the wire; each stage keeps its own layers' KV locally, so the
 *  result is byte-identical to a single node (pipe_self_test proves it over a real
 *  loopback socket). In this step both nodes load the full model and each runs its
 *  own layer range; loading only the owned layers (the RAM-pooling win) is Step 2b.
 * ═══════════════════════════════════════════════════════════ */

#ifdef _WIN32
typedef SOCKET sock_t;
#define BADSOCK INVALID_SOCKET
static void net_init(void) { WSADATA w; WSAStartup(MAKEWORD(2, 2), &w); }
static void net_close(sock_t s) { closesocket(s); }
#else
typedef int sock_t;
#define BADSOCK (-1)
static void net_init(void) {}
static void net_close(sock_t s) { close(s); }
#endif

static sock_t g_pipe_sock = BADSOCK;  /* coordinator's live connection to the worker */
static int g_pipe_coord = 0;          /* 1 = service_loop forwards through the worker */

#define PIPE_MAGIC 0x32504350u   /* 'PCP2' */
enum { PIPE_FWD = 1, PIPE_RES = 2, PIPE_BATCH = 3 };  /* BATCH = prefill block */

typedef struct {
    uint32_t magic;
    uint32_t type;
    int32_t  seq;
    int32_t  pos;
    int32_t  lo;
    int32_t  hi;
    uint32_t nbytes;
} PipeHdr;

static void nodelay(sock_t s) {
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof(one));
}

static int send_all(sock_t s, const void *buf, size_t n) {
    const char *p = (const char *)buf;
    while (n) {
        int chunk = (int)(n > (1u << 20) ? (1u << 20) : n);
        int r = send(s, p, chunk, 0);
        if (r <= 0) return -1;
        p += r; n -= (size_t)r;
    }
    return 0;
}

static int recv_all(sock_t s, void *buf, size_t n) {
    char *p = (char *)buf;
    while (n) {
        int chunk = (int)(n > (1u << 20) ? (1u << 20) : n);
        int r = recv(s, p, chunk, 0);
        if (r <= 0) return -1;
        p += r; n -= (size_t)r;
    }
    return 0;
}

static sock_t tcp_listen(int port) {
    sock_t s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == BADSOCK) return BADSOCK;
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));
    struct sockaddr_in a; memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons((unsigned short)port);
    if (bind(s, (struct sockaddr *)&a, sizeof(a)) != 0) { net_close(s); return BADSOCK; }
    if (listen(s, 4) != 0) { net_close(s); return BADSOCK; }
    return s;
}

static sock_t tcp_connect(const char *host, int port) {
    char portstr[16]; snprintf(portstr, sizeof(portstr), "%d", port);
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, portstr, &hints, &res) != 0 || !res) return BADSOCK;
    sock_t s = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (s == BADSOCK) { freeaddrinfo(res); return BADSOCK; }
    if (connect(s, res->ai_addr, (int)res->ai_addrlen) != 0) {
        net_close(s); freeaddrinfo(res); return BADSOCK;
    }
    freeaddrinfo(res);
    nodelay(s);
    return s;
}

/* Worker (stage B): serve one connection until it closes. For each residual
 * received, run layers [hdr.lo, hdr.hi), apply the head, and return the LOGITS.
 * Sampling stays on the coordinator, the single authority for temperature, seed
 * and repetition history — so the distributed output matches a single node. */
static void pipe_worker_serve(Model *m, sock_t conn) {
    Cfg *c = &m->c;
    int D = c->hidden;
    float *h = falloc(D);                 /* single-token residual */
    float *logits = falloc(c->vocab);
    nodelay(conn);
    for (;;) {
        PipeHdr hd;
        if (recv_all(conn, &hd, sizeof(hd)) != 0) break;
        if (hd.magic != PIPE_MAGIC) break;
        if (hd.lo < g_pipe_lo || hd.hi > g_pipe_hi) {
            fprintf(stderr, "pipe worker: requested range [%d,%d) outside owned "
                    "[%d,%d) — PIPE_CUT mismatch between the two nodes?\n",
                    hd.lo, hd.hi, g_pipe_lo, g_pipe_hi);
            break;
        }
        if (hd.type == PIPE_FWD) {                 /* decode: one token */
            if (hd.nbytes != (uint32_t)D * (uint32_t)sizeof(float)) break;
            if (recv_all(conn, h, hd.nbytes) != 0) break;
            run_layer_range(m, h, hd.pos, hd.lo, hd.hi);
            forward_head_logits(m, h, logits);
        } else if (hd.type == PIPE_BATCH) {        /* prefill: block of positions */
            int n = hd.seq;
            if (n < 1 ||
                hd.nbytes != (uint32_t)((int64_t)n * D * (int64_t)sizeof(float))) break;
            float *H = (float *)falloc((int64_t)n * D);
            if (!H) break;
            if (recv_all(conn, H, hd.nbytes) != 0) { free(H); break; }
            run_prefill_range(m, H, n, hd.pos, hd.lo, hd.hi);
            forward_head_logits(m, H + (int64_t)(n - 1) * D, logits);
            free(H);
        } else break;
        PipeHdr rh = { PIPE_MAGIC, PIPE_RES, hd.seq, hd.pos, hd.lo, hd.hi,
                       (uint32_t)c->vocab * (uint32_t)sizeof(float) };
        if (send_all(conn, &rh, sizeof(rh)) != 0) break;
        if (send_all(conn, logits, rh.nbytes) != 0) break;
        m->n_fw++;
    }
    free(h); free(logits);
}

/* Coordinator (stage A): embed + run layers [0, cut), ship the residual, receive
 * the logits from the worker, and sample locally. Returns the token (or -1). */
static int pipe_coord_step(Model *m, sock_t s, int tok, int pos, int cut) {
    Cfg *c = &m->c;
    int D = c->hidden;
    float *h = falloc(D);
    embed_token(m, tok, h);
    run_layer_range(m, h, pos, 0, cut);
    PipeHdr hd = { PIPE_MAGIC, PIPE_FWD, pos, pos, cut, c->n_layers,
                   (uint32_t)D * (uint32_t)sizeof(float) };
    int rtok = -1;
    if (send_all(s, &hd, sizeof(hd)) == 0 && send_all(s, h, hd.nbytes) == 0) {
        PipeHdr rh;
        if (recv_all(s, &rh, sizeof(rh)) == 0 && rh.type == PIPE_RES
            && rh.nbytes == (uint32_t)c->vocab * (uint32_t)sizeof(float)) {
            float *logits = falloc(c->vocab);
            if (recv_all(s, logits, rh.nbytes) == 0)
                rtok = sample_logits(logits, c->vocab);
            free(logits);
        }
    }
    free(h);
    m->n_fw++;
    return rtok;
}

/* Distributed batched prefill: embed n tokens, run [0,cut) over the whole block,
 * ship all n residuals at once, receive the last position's logits, sample. One
 * network round-trip per block instead of one per token. */
static int pipe_coord_prefill(Model *m, sock_t s, const int *ids, int n,
                              int pos_base, int cut) {
    Cfg *c = &m->c;
    int D = c->hidden;
    float *H = (float *)falloc((int64_t)n * D);
    if (!H) { fprintf(stderr, "error: prefill memory\n"); exit(1); }
    for (int p = 0; p < n; p++) embed_token(m, ids[p], H + (int64_t)p * D);
    run_prefill_range(m, H, n, pos_base, 0, cut);
    PipeHdr hd = { PIPE_MAGIC, PIPE_BATCH, n, pos_base, cut, c->n_layers,
                   (uint32_t)((int64_t)n * D * (int64_t)sizeof(float)) };
    int rtok = -1;
    if (send_all(s, &hd, sizeof(hd)) == 0 && send_all(s, H, hd.nbytes) == 0) {
        PipeHdr rh;
        if (recv_all(s, &rh, sizeof(rh)) == 0 && rh.type == PIPE_RES
            && rh.nbytes == (uint32_t)c->vocab * (uint32_t)sizeof(float)) {
            float *logits = falloc(c->vocab);
            if (recv_all(s, logits, rh.nbytes) == 0)
                rtok = sample_logits(logits, c->vocab);
            free(logits);
        }
    }
    free(H);
    m->n_fw += n;
    return rtok;
}

/* Coordinator run: raw-ID prompt (INPUT or INPUT_FILE) -> sequential distributed
 * prefill -> decode, printing token IDs (the Python bridge owns any rendering). */
static int pipe_coord_run(Model *m, const char *peer, int cut, int max_tokens) {
    Cfg *c = &m->c;
    char host[256] = "127.0.0.1"; int port = 52200;
    { const char *colon = strrchr(peer, ':');
      if (colon) {
          size_t hl = (size_t)(colon - peer);
          if (hl >= sizeof(host)) hl = sizeof(host) - 1;
          memcpy(host, peer, hl); host[hl] = '\0';
          port = atoi(colon + 1);
      } else { strncpy(host, peer, sizeof(host) - 1); } }

    int stackbuf[8192]; int *prompt = stackbuf; int owned = 0, n = 0;
    const char *input_file = getenv("INPUT_FILE");
    if (input_file && *input_file) {
        n = read_token_ids_file(input_file, c->ctx_len, &prompt);
        if (n < 0) return 1;
        owned = 1;
    } else {
        const char *in = getenv("INPUT");
        if (in) { const char *p = in;
            while (*p && n < 8192) {
                while (*p == ' ' || *p == '\n' || *p == '\t') p++;
                if (!*p) break;
                prompt[n++] = (int)strtol(p, (char **)&p, 10);
            } }
    }
    if (n == 0) {
        fprintf(stderr, "pipe coord: empty prompt (set INPUT or INPUT_FILE)\n");
        if (owned) free(prompt);
        return 1;
    }

    net_init();
    sock_t s = tcp_connect(host, port);
    if (s == BADSOCK) {
        fprintf(stderr, "pipe coord: cannot connect to %s:%d\n", host, port);
        if (owned) free(prompt);
        return 1;
    }
    fprintf(stderr, "pipe coord (stage A): connected to %s:%d · cut@%d · "
            "layers [0,%d) here, [%d,%d) on the worker · %d prompt tokens\n",
            host, port, cut, cut, cut, c->n_layers, n);

    double t0 = now_s();
    int last = prompt[0];
    for (int off = 0; off < n; off += 64) {
        int len = n - off < 64 ? n - off : 64;
        last = pipe_coord_prefill(m, s, prompt + off, len, off, cut);
    }
    fprintf(stderr, "prefill %d tokens in %.2fs (next=%d)\n", n, now_s() - t0, last);

    int output_ids = 0;
    { const char *v = getenv("OUTPUT"); if (v && strcmp(v, "ids") == 0) output_ids = 1; }
    int pos = n, tok = last;
    for (int i = 0; i < max_tokens; i++) {
        if (output_ids) { printf("%d\n", tok); fflush(stdout); }
        else { printf("[%d]", tok); fflush(stdout); }
        if (tok == c->stop_ids[0]) break;
        tok = pipe_coord_step(m, s, tok, pos, cut);
        if (tok < 0) { fprintf(stderr, "\npipe coord: worker dropped\n"); break; }
        pos++;
        m->n_emit++;
    }
    printf("\n");
    net_close(s);
    if (owned) free(prompt);
    return 0;
}

/* Loopback self-test: a worker thread and the coordinator on one machine, over a
 * real TCP socket, must produce the same tokens as a single-node forward. */
typedef struct { Model *m; sock_t ls; } PipeThreadArg;

#ifdef _WIN32
static DWORD WINAPI pipe_worker_thread(LPVOID arg) {
#else
static void *pipe_worker_thread(void *arg) {
#endif
    PipeThreadArg *a = (PipeThreadArg *)arg;
    sock_t conn = accept(a->ls, NULL, NULL);
    if (conn != BADSOCK) { pipe_worker_serve(a->m, conn); net_close(conn); }
    net_close(a->ls);
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

static int pipe_self_test(void) {
    fprintf(stderr, "── pipeline self-test: 2 TCP stages over loopback ──\n\n");
    net_init();
    static Model m;
    memset(&m, 0, sizeof(m));
    Cfg *c = &m.c;
    build_synth_model(&m);

    g_temperature = 0.0f;   /* greedy argmax: deterministic on both paths */
    g_rep_penalty = 1.0f;

    int cut = c->n_layers / 2; if (cut < 1) cut = 1;
    int port = 52190;
    sock_t ls = tcp_listen(port);
    if (ls == BADSOCK) { fprintf(stderr, "  ✗ cannot listen on %d\n", port); return 1; }

    PipeThreadArg arg = { &m, ls };
#ifdef _WIN32
    HANDLE th = CreateThread(NULL, 0, pipe_worker_thread, &arg, 0, NULL);
    if (!th) { fprintf(stderr, "  ✗ cannot start worker thread\n"); return 1; }
#else
    pthread_t th;
    if (pthread_create(&th, NULL, pipe_worker_thread, &arg) != 0) {
        fprintf(stderr, "  ✗ cannot start worker thread\n"); return 1; }
#endif

    sock_t s = tcp_connect("127.0.0.1", port);
    if (s == BADSOCK) { fprintf(stderr, "  ✗ coordinator cannot connect\n"); return 1; }

    fprintf(stderr, "  stage A [0,%d)  ──residual over TCP──▶  stage B [%d,%d)\n\n",
            cut, cut, c->n_layers);

    int tks[6] = {1, 5, 3, 12, 7, 2};
    int mism = 0;
    for (int i = 0; i < 6; i++) {
        int mono = forward_token(&m, tks[i], i);            /* single-node reference */
        int dist = pipe_coord_step(&m, s, tks[i], i, cut);  /* distributed */
        fprintf(stderr, "    pos=%d tok=%d  mono=%d  dist=%d  %s\n",
                i, tks[i], mono, dist, mono == dist ? "ok" : "MISMATCH");
        if (mono != dist) mism++;
    }
    net_close(s);
#ifdef _WIN32
    WaitForSingleObject(th, 2000); CloseHandle(th);
#else
    pthread_join(th, NULL);
#endif

    if (mism) {
        fprintf(stderr, "\n  ✗ %d/6 tokens differ between mono and distributed\n", mism);
        return 1;
    }
    fprintf(stderr, "\n── pipeline self-test PASSED ──\n");
    fprintf(stderr, "  distributed tokens == single-node, over a real socket.\n\n");
    return 0;
}

/* ── Service mode: persistent multi-turn session over a pipe ──
 * stdout contains only the protocol, stderr only diagnostics.
 * Invariant: every emitted TOKEN is also consumed by the forward pass, so
 * `pos` always equals the number of valid positions in the KV-cache. */
static int service_loop(Model *m, int cap) {
    Cfg *c = &m->c;
    int pos = 0;

    printf("READY %d %d", cap, c->vocab);
    for (int si = 0; si < c->n_stop; si++) printf(" %d", c->stop_ids[si]);
    printf("\n");
    fflush(stdout);

    char cmd[32];
    while (scanf("%31s", cmd) == 1) {
        if (strcmp(cmd, "SHUTDOWN") == 0) break;

        if (strcmp(cmd, "STATS") == 0) {
            service_stats(m, pos, cap);
            continue;
        }

        if (strcmp(cmd, "RESET") == 0) {
            pos = 0;
            g_history_len = 0;
            m->kv.cur_pos = 0;
            printf("DONE RESET 0 0\n");
            fflush(stdout);
            continue;
        }

        if (strcmp(cmd, "TURN") != 0) {
            printf("ERROR BAD_COMMAND 1 unknown command\n");
            fflush(stdout);
            return 1;
        }

        /* Extended header: per-request sampling (temp, top_p, top_k) between keep and n_ids. */
        long max_new = 0, keep = 0, n_ids = 0;
        float t_temp = 0.0f, t_topp = 0.0f;
        long t_topk = 0;
        if (scanf("%ld %ld %f %f %ld %ld", &max_new, &keep,
                  &t_temp, &t_topp, &t_topk, &n_ids) != 6) {
            printf("ERROR BAD_HEADER 1 unreadable TURN header\n");
            fflush(stdout);
            return 1;
        }
        /* Apply this turn's sampling (defensive clamp). */
        if (t_temp < 0.0f) t_temp = 0.0f;
        if (t_temp > 5.0f) t_temp = 5.0f;
        if (t_topp <= 0.0f || t_topp > 1.0f) t_topp = 1.0f;
        if (t_topk < 0) t_topk = 0;
        g_temperature = t_temp;
        g_top_p = t_topp;
        g_top_k = (int)t_topk;
        if (max_new < 0 || keep < 0 || keep > pos || n_ids <= 0 ||
            keep + n_ids > cap || n_ids > cap) {
            printf("ERROR BAD_RANGE 0 keep/n_ids out of bounds (pos=%d cap=%d)\n", pos, cap);
            fflush(stdout);
            /* No mutation: the session stays usable. Discard the pending IDs. */
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
            printf("ERROR BAD_TOKEN 0 ID out of vocabulary\n");
            fflush(stdout);
            continue;
        }

        /* Reuse the prefix: the subsequent positions will be overwritten. */
        pos = (int)keep;

        /* Reset the repetition-penalty history at the start of every turn.
         * The persistent chat never sends RESET (it reuses the KV prefix via
         * `keep`), so without this g_history would accumulate across turns —
         * and prefill_tokens re-seeds it with the whole re-rendered prompt each
         * turn, so the fixed 4096-token buffer saturates after a few turns and
         * the penalty poisons sampling (output degrades ~turn 3, sooner on the
         * 120B whose prompts are longer). Clearing it here makes the penalty
         * scope this turn's prompt delta plus its generated tokens, which is the
         * intended "penalize repeating the current context" behavior. Done only
         * after validation passed, so a rejected turn leaves the session intact. */
        g_history_len = 0;

        double service_prefill_start = now_s();
        int next;
        if (g_pipe_coord) {   /* distributed: batched prefill through the worker */
            int batch = 64;
            { const char *v = getenv("PREFILL_BATCH"); if (v) batch = atoi(v); }
            if (batch < 1) batch = 1;
            next = ids[0];
            for (long off = 0; off < n_ids; off += batch) {
                int len = (int)(n_ids - off < batch ? n_ids - off : batch);
                next = pipe_coord_prefill(m, g_pipe_sock, ids + off, len,
                                          pos + (int)off, g_pipe_cut);
            }
        } else {
            next = prefill_tokens(m, ids, (int)n_ids, pos);
        }
        pos += (int)n_ids;
        g_service_prefill_tokens += (uint64_t)n_ids;
        g_service_prefill_seconds += now_s() - service_prefill_start;
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
            double service_decode_start = now_s();
            int following = g_pipe_coord
                ? pipe_coord_step(m, g_pipe_sock, tok, pos, g_pipe_cut)
                : forward_token(m, tok, pos);
            g_service_decode_seconds += now_s() - service_decode_start;
            g_service_decode_tokens++;
            pos++;
            m->n_emit++;
            int stopped = 0;
            for (int si = 0; si < c->n_stop; si++) {
                if (tok == c->stop_ids[si]) {
                    /* GPT-OSS distinguishes the two Harmony terminators; other
                     * families (Qwen) have a single generic EOS. */
                    reason = (c->n_stop >= 2) ? (si == 0 ? "RETURN" : "CALL") : "EOS";
                    stopped = 1;
                    break;
                }
            }
            if (stopped) break;
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

static int print_model_plan(const char *model_path) {
    Cfg c;
    if (cfg_load(&c, model_path) != 0) return 1;
    int64_t dense = dense_weight_estimate_bytes(&c);
    int64_t expert = expert_slot_estimate_bytes(&c);
    int64_t minimum_cache = expert * c.n_layers * c.topk;
    int64_t phys = physical_ram_bytes();
    int64_t avail = available_ram_bytes();

    fprintf(stderr, "\nmodel memory plan\n");
    fprintf(stderr, "  path: %s\n", model_path);
    fprintf(stderr, "  architecture: D=%d L=%d E=%d top-%d, expert INT%d\n",
            c.hidden, c.n_layers, c.n_experts, c.topk, c.expert_bits);
    fprintf(stderr, "  dense estimate: %.2f GiB\n", dense / 1073741824.0);
    fprintf(stderr, "  one cached expert: %.2f MiB\n", expert / 1048576.0);
    fprintf(stderr, "  minimum top-k cache: %.2f GiB (%d slots/layer)\n",
            minimum_cache / 1073741824.0, c.topk);
    fprintf(stderr, "  physical RAM: %.2f GiB; available now: %.2f GiB\n",
            phys / 1073741824.0, avail / 1073741824.0);
    fprintf(stderr, "  minimum practical available RAM: %.2f GiB\n",
            (dense + minimum_cache + 512LL * 1024 * 1024) / 1073741824.0);
    if (avail < dense + minimum_cache + 512LL * 1024 * 1024) {
        fprintf(stderr, "  status: INSUFFICIENT NOW - close applications before loading\n");
        return 2;
    }
    fprintf(stderr, "  status: enough RAM for the minimum streaming configuration\n");
    return 0;
}

static int benchmark_int3_expert(int iterations) {
    const int D = 2880;
    const int I = 5760;
    const int H = 2880;
    const int groups = D / I3_GROUP;
    const int64_t gu_qbytes = (int64_t)I * groups * I3_GBYTES;
    const int64_t d_qbytes = (int64_t)D * groups * I3_GBYTES;
    const int64_t gu_scales = (int64_t)I * groups;
    const int64_t d_scales = (int64_t)D * groups;
    QT gu = {0}, down = {0};
    float *x = falloc(D);
    float *gu_out = falloc(I);
    float *hidden = falloc(H);
    float *out = falloc(D);
    uint32_t rng = 0x12345678u;

    if (iterations < 1) iterations = 1;
    gu.fmt = 5; gu.O = I; gu.I = D; gu.block_size = I3_GROUP;
    down.fmt = 5; down.O = D; down.I = H; down.block_size = I3_GROUP;
    gu.q4 = malloc((size_t)gu_qbytes);
    gu.s = falloc(gu_scales);
    down.q4 = malloc((size_t)d_qbytes);
    down.s = falloc(d_scales);
    if (!x || !gu_out || !hidden || !out || !gu.q4 || !gu.s ||
        !down.q4 || !down.s) {
        fprintf(stderr, "INT3 benchmark: allocation failed\n");
        free(x); free(gu_out); free(hidden); free(out);
        free(gu.q4); free(gu.s); free(down.q4); free(down.s);
        return 1;
    }

    for (int i = 0; i < D; i++) x[i] = (float)((i % 101) - 50) / 50.f;
    for (int64_t i = 0; i < gu_qbytes; i++) {
        rng = rng * 1664525u + 1013904223u;
        gu.q4[i] = (uint8_t)(rng >> 24);
    }
    for (int64_t i = 0; i < d_qbytes; i++) {
        rng = rng * 1664525u + 1013904223u;
        down.q4[i] = (uint8_t)(rng >> 24);
    }
    for (int64_t i = 0; i < gu_scales; i++) gu.s[i] = 0.02f;
    for (int64_t i = 0; i < d_scales; i++) down.s[i] = 0.02f;

    /* Warm-up and fault in every packed page before timing. */
    matmul_qt(gu_out, x, &gu, 1);
    for (int i = 0; i < H; i++) hidden[i] = gu_out[i] * gu_out[i + H];
    matmul_qt(out, hidden, &down, 1);

    double start = now_s();
    for (int it = 0; it < iterations; it++) {
        matmul_qt(gu_out, x, &gu, 1);
        for (int i = 0; i < H; i++) hidden[i] = gu_out[i] * gu_out[i + H];
        matmul_qt(out, hidden, &down, 1);
    }
    double elapsed = now_s() - start;
    double bytes = (double)(gu_qbytes + d_qbytes) +
                   (double)(gu_scales + d_scales) * sizeof(float);
    double checksum = 0.0;
    for (int i = 0; i < D; i++) checksum += out[i];

    fprintf(stderr, "\nINT3 gs64 production expert benchmark\n");
    fprintf(stderr, "  shape: gate_up %dx%d + down %dx%d\n", I, D, D, H);
    fprintf(stderr, "  packed+scales: %.2f MiB, OpenMP threads: %d\n",
            bytes / (1024.0 * 1024.0), omp_get_max_threads());
    fprintf(stderr, "  latency: %.3f ms/expert\n",
            elapsed * 1000.0 / iterations);
    fprintf(stderr, "  effective packed bandwidth: %.2f GiB/s\n",
            (bytes * iterations / (1024.0 * 1024.0 * 1024.0)) / elapsed);
    fprintf(stderr, "  kernel-only upper bound: %.3f tok/s (144 experts/token)\n",
            ((double)iterations / elapsed) / 144.0);
    fprintf(stderr, "  checksum: %.6g\n", checksum);

    free(x); free(gu_out); free(hidden); free(out);
    free(gu.q4); free(gu.s); free(down.q4); free(down.s);
    return 0;
}

static int native_router_self_test(void) {
    const int E = 128, D = 2880;
    float *w = falloc((int64_t)E * D), *b = falloc(E), *x = falloc(D);
    float *cpu = falloc(E), *gpu = falloc(E);
    uint32_t rng = 123456789u;
    for (int i = 0; i < E * D; i++) {
        rng = rng * 1664525u + 1013904223u;
        w[i] = ((int)(rng >> 16) - 32768) / 327680.0f;
    }
    for (int i = 0; i < D; i++) {
        rng = rng * 1664525u + 1013904223u;
        x[i] = ((int)(rng >> 16) - 32768) / 32768.0f;
    }
    for (int e = 0; e < E; e++) {
        b[e] = (e - 64) / 1000.0f;
        float a = b[e];
        for (int i = 0; i < D; i++) a += x[i] * w[(int64_t)e * D + i];
        cpu[e] = a;
    }
    if (!pgr_native_init() || pgr_native_scores(gpu, x, w, b, E, D, w) != 0) {
        fprintf(stderr, "native GPU router self-test: backend unavailable\n");
        pgr_native_shutdown();
        free(w); free(b); free(x); free(cpu); free(gpu);
        return 1;
    }
    int ct = 0, gt = 0;
    float max_abs = 0.0f;
    for (int e = 0; e < E; e++) {
        float d = fabsf(cpu[e] - gpu[e]);
        if (d > max_abs) max_abs = d;
        if (cpu[e] > cpu[ct]) ct = e;
        if (gpu[e] > gpu[gt]) gt = e;
    }
    int ok = ct == gt && max_abs < 1e-4f;
    fprintf(stderr, "native GPU router [128x2880]: CPU top1=%d GPU top1=%d "
            "max_abs=%.7g %s\n", ct, gt, max_abs, ok ? "OK" : "FAIL");
    pgr_native_shutdown();
    free(w); free(b); free(x); free(cpu); free(gpu);
    return ok ? 0 : 1;
}

static int native_dense_self_test(int fail_after_release) {
    const int O = 4096, I = 2880;
    float *w = falloc((int64_t)O * I), *x = falloc(I);
    float *cpu = falloc(O), *gpu = falloc(O);
    uint32_t rng = 246813579u;
    for (int64_t i = 0; i < (int64_t)O * I; i++) {
        rng = rng * 1664525u + 1013904223u;
        w[i] = ((int)(rng >> 16) - 32768) / 1048576.0f;
    }
    for (int i = 0; i < I; i++) {
        rng = rng * 1664525u + 1013904223u;
        x[i] = ((int)(rng >> 16) - 32768) / 32768.0f;
    }
    matmul_f32(cpu, x, w, 1, I, O);
    if (!pgr_native_init() ||
        pgr_native_dense_upload(w, O, I, &rng) != 0 ||
        pgr_native_dense_matmul(gpu, x, w, O, I, &rng) != 0) {
        fprintf(stderr, "native GPU dense self-test: backend unavailable\n");
        pgr_native_shutdown();
        free(w); free(x); free(cpu); free(gpu);
        return 1;
    }
    float max_abs = 0.0f;
    double mse = 0.0, ref = 0.0;
    for (int o = 0; o < O; o++) {
        float d = fabsf(cpu[o] - gpu[o]);
        if (d > max_abs) max_abs = d;
        mse += (double)d * d;
        ref += (double)cpu[o] * cpu[o];
    }
    double rel_rms = ref > 0.0 ? sqrt(mse / ref) : 0.0;
    free(w);
    w = NULL; /* Exercise resident lookup after the host allocation is gone. */
    const int iterations = 20;
    int completed = 0;
    double t0 = now_s();
    for (int i = 0; i < iterations; i++) {
        if (pgr_native_dense_matmul(gpu, x, NULL, O, I, &rng) != 0) break;
        completed++;
    }
    double gpu_ms = 1000.0 * (now_s() - t0) / iterations;
    int ok = max_abs < 0.01f && rel_rms < 0.002 && completed == iterations;
    pgr_native_dense_clear();
    if (pgr_native_dense_matmul(gpu, x, NULL, O, I, &rng) == 0) ok = 0;
    if (fail_after_release) {
        QT released = {0};
        released.O = O; released.I = I;
        g_gpu_dense = 1;
        g_gpu_dense_host_released = (uint64_t)O * I * sizeof(float);
        /* No resident matrix and no host weights: dispatch must stop, not
         * dereference NULL or silently emit an incorrect CPU result. */
        attention_projection(gpu, x, &released);
        fprintf(stderr, "FAIL: released-host dispatch did not stop\n");
        return 2;
    }
    fprintf(stderr, "native GPU dense FP16 [4096x2880]: max_abs=%.7g "
            "rel_rms=%.7g %.3f ms/call %s\n", max_abs, rel_rms,
            gpu_ms, ok ? "OK" : "FAIL");
    pgr_native_shutdown();
    free(w); free(x); free(cpu); free(gpu);
    return ok ? 0 : 1;
}

static int expert_io_self_test(Model *m) {
    if (g_db->name_index) {
        for (int i = 0; i < g_db->n_tensors; i++) {
            StTensor *indexed = st_find(g_db, g_db->tensors[i].name);
            int *index = g_db->name_index;
            g_db->name_index = NULL;
            StTensor *linear = st_find(g_db, g_db->tensors[i].name);
            g_db->name_index = index;
            if (indexed != linear) { fprintf(stderr, "tensor index mismatch\n"); return 1; }
        }
        fprintf(stderr, "tensor name index: %d lookups match linear reference\n", g_db->n_tensors);
    }
    ESlot reference = {0}, reused = {0};
    int checked = 0;
    for (int direct = 0; direct <= 1; direct++) {
        st_set_direct(direct);
        for (int i = 0; i < 12; i++) {
            int layer = (i * 7) % m->c.n_layers;
            int eid = (i * 37) % m->c.n_experts;
            if (i == 11) { layer = m->c.n_layers - 1; eid = m->c.n_experts - 1; }
            g_expert_reuse = 0;
            free_slot_buffers(&reference);
            expert_load(m, layer, eid, &reference);
            g_expert_reuse = 1;
            recycle_slot_buffers(&reused);
            expert_load(m, layer, eid, &reused);
            QT *a[2] = {&reference.gu, &reference.d};
            QT *b[2] = {&reused.gu, &reused.d};
            int ok = reference.eid == eid && reused.eid == eid && reused.slab_reusable;
            for (int j = 0; j < 2 && ok; j++) {
                ok = a[j]->fmt == b[j]->fmt && a[j]->O == b[j]->O && a[j]->I == b[j]->I &&
                     a[j]->block_size == b[j]->block_size &&
                     a[j]->block_size == (a[j]->I > 64 ? 64 : 0);
                size_t groups = (size_t)a[j]->O * ((a[j]->I + 63) / 64);
                size_t packed = groups * (a[j]->fmt == 5 ? 24 : 32);
                if (ok) ok = memcmp(a[j]->q4, b[j]->q4, packed) == 0 &&
                             memcmp(a[j]->s, b[j]->s, groups * sizeof(float)) == 0;
            }
            if (ok) ok = !!reference.gu_bias == !!reused.gu_bias && !!reference.d_bias == !!reused.d_bias;
            if (ok && reference.gu_bias) ok = !memcmp(reference.gu_bias, reused.gu_bias, m->c.moe_inter * sizeof(float));
            if (ok && reference.d_bias) ok = !memcmp(reference.d_bias, reused.d_bias, m->c.hidden * sizeof(float));
            if (!ok) {
                fprintf(stderr, "expert I/O test FAIL direct=%d L=%d E=%d\n", direct, layer, eid);
                free_slot_buffers(&reference); free_slot_buffers(&reused);
                return 1;
            }
            checked++;
        }
    }
    free_slot_buffers(&reference); free_slot_buffers(&reused);
    fprintf(stderr, "expert I/O test: %d experts byte-identical, %llu allocations, %llu reuses %s\n",
            checked, (unsigned long long)g_expert_buffer_allocs,
            (unsigned long long)g_expert_buffer_reuses, g_expert_buffer_reuses ? "PASS" : "FAIL");
    return g_expert_buffer_reuses ? 0 : 1;
}

/* ── Speculative decoding: draft model + verify loop ─────────────────────────
 * The draft is a small model (e.g. Qwen3-0.6B as a 1-expert MoE). It is loaded
 * fully resident: every layer's single expert is preloaded, so after load it
 * never reads the global streaming db (which stays the target's). */
static Model g_draft;
static StDB  g_draft_db;

static int load_draft(const char *path, int ctx) {
    memset(&g_draft, 0, sizeof(g_draft));
    g_draft.pred_layer = -1;
    if (cfg_load(&g_draft.c, path) != 0) { fprintf(stderr, "[draft] bad config\n"); return -1; }
    Cfg *dc = &g_draft.c;
    st_init(&g_draft_db);
    char p[512];
    for (int i = 0; i < 200; i++) {
        snprintf(p, sizeof(p), "%s/model-%05d.safetensors", path, i);
        FILE *f = fopen(p, "rb"); if (f) { fclose(f); st_open_file(&g_draft_db, p); }
    }
    if (g_draft_db.n_tensors == 0) { fprintf(stderr, "[draft] no shards at %s\n", path); return -1; }
    int ec = dc->topk < 1 ? 1 : dc->topk;
    if (ec > dc->n_experts) ec = dc->n_experts;
    g_draft.ecap   = ec;
    g_draft.ecache = calloc(dc->n_layers, sizeof(ESlot *));
    g_draft.ecn    = calloc(dc->n_layers, sizeof(int));
    g_draft.pin    = calloc(dc->n_layers, sizeof(ESlot *));
    g_draft.npin   = calloc(dc->n_layers, sizeof(int));
    g_draft.eusage = calloc(dc->n_layers, sizeof(uint32_t *));
    g_draft.eheat  = calloc(dc->n_layers, sizeof(uint32_t *));
    for (int l = 0; l < dc->n_layers; l++) {
        g_draft.ecache[l] = calloc(ec, sizeof(ESlot));
        for (int i = 0; i < ec; i++) g_draft.ecache[l][i].eid = -1;
        g_draft.eusage[l] = calloc(dc->n_experts, sizeof(uint32_t));
        g_draft.eheat[l]  = calloc(dc->n_experts, sizeof(uint32_t));
    }
    g_draft.L = calloc(dc->n_layers, sizeof(Layer));
    kv_init(&g_draft.kv, dc->n_layers, dc->n_kv_heads, dc->head_dim, ctx);
    /* Swap the global streaming db + layer-range/model-path to the draft's for the
     * duration of its load, then restore the target's. */
    StDB *saved_db = g_db;
    int s_lo = g_pipe_lo, s_hi = g_pipe_hi, s_emb = g_load_embed, s_head = g_load_head;
    const char *s_path = g_model_path_global;
    g_db = &g_draft_db;
    g_pipe_lo = 0; g_pipe_hi = dc->n_layers; g_load_embed = 1; g_load_head = 1;
    g_model_path_global = path;
    int rc = load_dense_weights(&g_draft, &g_draft_db);
    if (rc == 0)
        for (int l = 0; l < dc->n_layers; l++) { ESlot *s[8]; int e0 = 0;
            cache_load_batch(&g_draft, l, &e0, 1, s); }  /* preload → fully resident */
    g_db = saved_db; g_pipe_lo = s_lo; g_pipe_hi = s_hi;
    g_load_embed = s_emb; g_load_head = s_head; g_model_path_global = s_path;
    if (rc != 0) { fprintf(stderr, "[draft] weight load failed\n"); return -1; }
    g_have_draft = 1;
    fprintf(stderr, "[draft] %s loaded (D=%d L=%d E=%d) — resident, spec K=%d\n",
            path, dc->hidden, dc->n_layers, dc->n_experts, g_spec_k);
    return 0;
}

/* The safetensors handle cache is per-thread and keyed by file_idx only, so the
 * target and draft (which share file_idx 0..) would clobber each other on one
 * thread. Load the draft on a throwaway thread: its file handles stay in that
 * thread's TLS and never touch the main thread's target handles. The draft is
 * fully resident after preload, so it never reads the db again. */
#ifdef _WIN32
static const char *g_ld_path; static int g_ld_ctx, g_ld_rc;
static DWORD WINAPI load_draft_thread(LPVOID p) { (void)p; g_ld_rc = load_draft(g_ld_path, g_ld_ctx); return 0; }
static int load_draft_isolated(const char *path, int ctx) {
    g_ld_path = path; g_ld_ctx = ctx;
    HANDLE t = CreateThread(NULL, 0, load_draft_thread, NULL, 0, NULL);
    if (!t) return load_draft(path, ctx);
    WaitForSingleObject(t, INFINITE); CloseHandle(t);
    return g_ld_rc;
}
#else
static int load_draft_isolated(const char *path, int ctx) { return load_draft(path, ctx); }
#endif

static void spec_decode(Model *tgt, int *prompt, int n_prompt, int max_tokens,
                        Tokenizer *tok, int has_tokenizer) {
    Cfg *c = &tgt->c;
    StDB *target_db = g_db;                 /* the draft swaps g_db around its forwards */
    int output_ids = 0;
    { const char *v = getenv("OUTPUT"); if (v && strcmp(v, "ids") == 0) output_ids = 1; }
    int K = g_spec_k; if (K < 1) K = 1; if (K > 16) K = 16;

    int last = prefill_tokens(tgt, prompt, n_prompt, 0);   /* target: streams experts */
    g_db = &g_draft_db;
    prefill_tokens(&g_draft, prompt, n_prompt, 0);          /* draft: resident, no stream */
    g_db = target_db;

    uint64_t fw_before = tgt->n_fw;
    long verifies = 0, accepted_sum = 0;
    int cur = last, pos = n_prompt, produced = 0, stop = 0;
    double t0 = now_s();

    /* Emit the first token (target's prediction after the prompt). */
    { if (output_ids) printf("%d\n", cur);
      else if (has_tokenizer) { char b[512]; tok_decode_raw(tok, cur, b, sizeof(b)); printf("%s", b); }
      else printf("[%d]", cur);
      produced = 1;
      for (int si = 0; si < c->n_stop; si++) if (cur == c->stop_ids[si]) stop = 1; }

    while (!stop && produced < max_tokens && pos < c->ctx_len - K - 2) {
        int dt[16], seq[17], pred[17];
        g_db = &g_draft_db;                 /* 1) draft proposes K tokens */
        int d = cur;
        for (int i = 0; i < K; i++) { d = forward_token(&g_draft, d, pos + i); dt[i] = d; }
        g_db = target_db;                   /* 2) target verifies in ONE batched forward */
        seq[0] = cur; for (int i = 0; i < K; i++) seq[1 + i] = dt[i];
        forward_verify(tgt, seq, K + 1, pos, pred);
        int a = 0; while (a < K && pred[a] == dt[a]) a++;   /* 3) accept prefix + bonus */
        verifies++; accepted_sum += a;
        int out[17], nout = 0;
        for (int i = 0; i < a; i++) out[nout++] = dt[i];
        out[nout++] = pred[a];              /* bonus: target's own token (always valid) */
        for (int i = 0; i < nout && produced < max_tokens; i++) {
            int t = out[i];
            if (output_ids) printf("%d\n", t);
            else if (has_tokenizer) { char b[512]; tok_decode_raw(tok, t, b, sizeof(b)); printf("%s", b); }
            else printf("[%d]", t);
            produced++;
            for (int si = 0; si < c->n_stop; si++) if (t == c->stop_ids[si]) { stop = 1; break; }
            if (stop) break;
        }
        fflush(stdout);
        cur = out[nout - 1]; pos += nout;   /* nout = a+1; KV rolls by overwrite next round */
    }
    double el = now_s() - t0;
    printf("\n");
    fprintf(stderr, "\n── SPEC DECODE (K=%d) ──\n", K);
    fprintf(stderr, "produced %d tokens · %ld verifies · mean accepted %.2f/round"
            " · %.2f tokens per target-forward\n", produced, verifies,
            verifies ? (double)accepted_sum / verifies : 0.0,
            verifies ? (double)produced / verifies : 0.0);
    fprintf(stderr, "target forwards this decode: %llu · %.2f tok/s\n",
            (unsigned long long)(tgt->n_fw - fw_before), el > 0 ? produced / el : 0.0);
}

int main(int argc, char **argv) {
    { const char *v = getenv("EXPERT_REUSE"); if (v) g_expert_reuse = atoi(v) != 0; }
    fprintf(stderr, "🪶 picchio v0.6.0 — MoE streaming engine\n");
    fprintf(stderr, "   GPT-OSS/Qwen3-MoE · INT3/INT4 · native CPU/GPU streaming\n\n");

    /* ── Self-test mode ── */
    if (argc > 1 && strcmp(argv[1], "--self-test") == 0) {
        return self_test();
    }
    if (argc > 1 && strcmp(argv[1], "--bench-int3") == 0) {
        int iterations = argc > 2 ? atoi(argv[2]) : 20;
        return benchmark_int3_expert(iterations);
    }
    if (argc > 1 && strcmp(argv[1], "--gpu-router-test") == 0) {
        return native_router_self_test();
    }
    if (argc > 1 && strcmp(argv[1], "--gpu-dense-test") == 0) {
        return native_dense_self_test(argc > 2 && strcmp(argv[2], "fail-after-release") == 0);
    }
    if (argc > 1 && strcmp(argv[1], "--plan") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Usage: picchio --plan <model-directory>\n");
            return 1;
        }
        return print_model_plan(argv[2]);
    }
    /* ── Pipeline self-test (2 TCP stages over loopback, synthetic model) ── */
    if (argc > 1 && strcmp(argv[1], "--pipe-self-test") == 0) {
        return pipe_self_test();
    }

    int test_expert_io = argc > 1 && strcmp(argv[1], "--expert-io-test") == 0;
    if (test_expert_io && argc < 3) {
        fprintf(stderr, "Usage: picchio --expert-io-test <model-directory>\n"); return 1;
    }
    const char *model_path = test_expert_io ? argv[2] : getenv("MODEL");
    if (!model_path) {
        if (argc > 1) model_path = argv[1];
        else {
            fprintf(stderr, "Usage: MODEL=/path/to/gptoss_i4 ./picchio [max_tokens]\n");
            fprintf(stderr, "       or: ./picchio /path/to/gptoss_i4 [max_tokens]\n");
            fprintf(stderr, "       or: ./picchio --self-test\n");
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

    fprintf(stderr, "model:      %s\n", model_path);
    { const char *svc = getenv("SERVICE");
      if (svc && atoi(svc))
          fprintf(stderr, "mode:       service (the token limit arrives with each TURN)\n\n");
      else
          fprintf(stderr, "max tokens: %d\n\n", max_tokens); }

    /* ── 1. Load configuration ── */
    static Model m;
    memset(&m, 0, sizeof(m));
    m.pred_layer = -1;

    if (cfg_load(&m.c, model_path) != 0) {
        fprintf(stderr, "error: cannot load config.json\n");
        return 1;
    }

    /* Distributed pipeline: narrow the owned layer range BEFORE loading, so a
     * stage reads only its own layers (Step 2b). The coordinator holds [0,cut) and
     * the embedding; the worker holds [cut,n_layers), the final norm and the head. */
    g_pipe_hi = m.c.n_layers;
    { const char *prole = getenv("PIPE_ROLE");
      if (prole) {
          int cut = m.c.n_layers / 2;
          const char *v = getenv("PIPE_CUT"); if (v) cut = atoi(v);
          if (cut < 1) cut = 1;
          if (cut >= m.c.n_layers) cut = m.c.n_layers - 1;
          g_pipe_cut = cut;
          /* PIPE_LOAD_FULL: diagnostic — load the whole model on both nodes but
           * still run the split at `cut`, to separate a partial-load bug from a
           * split/transport bug. */
          int load_full = 0;
          { const char *lf = getenv("PIPE_LOAD_FULL"); if (lf && atoi(lf)) load_full = 1; }
          if (strcmp(prole, "worker") == 0) {
              if (load_full) { g_pipe_lo = 0; g_pipe_hi = m.c.n_layers;
                               g_load_embed = 1; g_load_head = 1; }
              else { g_pipe_lo = cut; g_pipe_hi = m.c.n_layers;
                     g_load_embed = 0; g_load_head = 1; }
              fprintf(stderr, "pipe role: worker — runs [%d,%d) + head, loaded [%d,%d)%s\n",
                      cut, m.c.n_layers, g_pipe_lo, g_pipe_hi, load_full ? " (FULL)" : "");
          } else if (strcmp(prole, "coord") == 0) {
              if (load_full) { g_pipe_lo = 0; g_pipe_hi = m.c.n_layers;
                               g_load_embed = 1; g_load_head = 1; }
              else { g_pipe_lo = 0; g_pipe_hi = cut;
                     g_load_embed = 1; g_load_head = 0; }
              fprintf(stderr, "pipe role: coord — runs [0,%d) + embedding, loaded [%d,%d)%s\n",
                      cut, g_pipe_lo, g_pipe_hi, load_full ? " (FULL)" : "");
          }
      } }

    /* ── 2. Open the safetensors files ── */
    StDB db;
    st_init(&db);

    {
        char path[512];
        FILE *test;

        /* Try dense.safetensors */
        snprintf(path, sizeof(path), "%s/dense.safetensors", model_path);
        test = fopen(path, "rb");
        if (test) { fclose(test); st_open_file(&db, path); }

        /* model.safetensors (single HF file) */
        snprintf(path, sizeof(path), "%s/model.safetensors", model_path);
        test = fopen(path, "rb");
        if (test) { fclose(test); st_open_file(&db, path); }

        /* Shards: model-NNNNN-of-NNNNN.safetensors (HF format) */
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

        /* Additional shard files on other disks, separated by ';'.
         * The order matters: st_find uses the first tensor with a given name. */
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
                    else fprintf(stderr, "  ⚠ MODEL_AUX not found: %s\n", cursor);
                }
                if (!sep) break;
                cursor = sep + 1;
            }
        }

        /* Shards: model-NNNNN.safetensors (converted Picchio format) */
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
        fprintf(stderr, "\n⚠ no tensors found in %s\n", model_path);
        fprintf(stderr, "  Use: python3 convert.py --model openai/gpt-oss-120b --output %s\n",
                model_path);
        st_close(&db);
        return 1;
    }

    fprintf(stderr, "✓ %d tensors from %d files\n", db.n_tensors, db.n_files);

    /* Make the DB accessible to the expert loader */
    g_db = &db;
    { const char *v = getenv("TENSOR_INDEX");
      if ((!v || atoi(v)) && !st_build_name_index(&db))
          fprintf(stderr, "warning: tensor name index unavailable; using linear lookup\n"); }
    if (test_expert_io) {
        int rc = expert_io_self_test(&m);
        st_close(&db);
        return rc;
    }

    /* Optional aligned expert store. Auto-detect experts.picchioflat next to
     * the model; FLAT=<path> overrides it and FLAT=0 disables discovery. A
     * partial store is valid and falls back to safetensors for missing layers. */
    {
        const char *fv = getenv("FLAT");
        if (!(fv && strcmp(fv, "0") == 0)) {
            char flat_path[768];
            int explicit_path = fv && *fv && strcmp(fv, "1") != 0;
            if (explicit_path)
                snprintf(flat_path, sizeof(flat_path), "%s", fv);
            else
                snprintf(flat_path, sizeof(flat_path), "%s/experts.picchioflat",
                         model_path);
            FILE *ff = fopen(flat_path, "rb");
            if (ff) {
                fclose(ff);
                const char *vv = getenv("FLAT_VERIFY");
                int verify = vv && atoi(vv) != 0;
                if (flat_open(&g_flat, flat_path, m.c.hidden, m.c.n_layers,
                              m.c.n_experts, m.c.topk, m.c.moe_inter, verify)) {
                    fprintf(stderr, "  flat experts: %u/%u indexed, 4 KiB aligned%s\n",
                            g_flat.n_entries,
                            g_flat.n_layers * g_flat.n_experts,
                            verify ? ", payload verification ON" : "");
                    atexit(flat_global_shutdown);
                } else {
                    fprintf(stderr, "  warning: flat store rejected; using safetensors\n");
                }
            } else if (explicit_path) {
                fprintf(stderr, "  warning: FLAT not found: %s\n", flat_path);
            }
        }
    }

    /* ── 3. Allocate structures and load weights ── */
    Cfg *c = &m.c;
    m.L = calloc(c->n_layers, sizeof(Layer));

    /* KV-cache (512 initial positions to fit in 16 GB RAM) */
    int initial_ctx = 512;
    const char *ctx_env = getenv("CTX");
    if (ctx_env) initial_ctx = atoi(ctx_env);
    if (initial_ctx > c->ctx_len) initial_ctx = c->ctx_len;
    kv_init(&m.kv, c->n_layers, c->n_kv_heads, c->head_dim, initial_ctx);

    /* Per-layer expert cache — RAM budget for the experts (each expert ~12.4 MB).
     * More cache = more hits = fewer bytes from disk, which is the real bottleneck.
     * Sweep measured on the 20B (16 GB RAM, real RSS now available): the historical
     * default of 6 GB gave 21 slots/layer and 88.9% hit; at 8 GB it reaches 28
     * slots/layer and 91.2%, with RSS ~8.5 GB (wide margin over the 15.8 GB
     * physical). Full residency of the 20B (32 experts/layer) is reached around
     * PIN_GB=9, beyond which there is no gain because there are only 32 experts
     * per layer. The default is now adaptive: without PIN_GB it considers both
     * physical and currently available RAM, subtracts the model-specific dense
     * estimate, and uses exact INT3/INT4 slot sizes. It scales up on larger
     * machines and self-limits when other applications occupy RAM. The old estimate
     * that a low PIN_GB would saturate RAM was based on a broken RSS measurement on
     * Windows (see rss_gb), now fixed. */
    /* Exact bytes for one cached expert. INT3 uses 24 packed bytes plus one
     * F32 scale for every group of 64 values (3.5 effective bits/weight). */
    int64_t expert_bytes = expert_slot_estimate_bytes(c);

    double pin_gb_env = 0;
    { const char *v = getenv("PIN_GB");
      if (v) {
          char *end;
          pin_gb_env = strtod(v, &end);
          if (*end || !isfinite(pin_gb_env) || pin_gb_env < 0 || pin_gb_env > 1048576) {
              fprintf(stderr, "error: PIN_GB must be a finite nonnegative GiB budget\n");
              return 1;
          }
      } }
    int64_t GB = 1024LL * 1024 * 1024;
    int64_t avail_bytes;
    if (pin_gb_env > 0) {
        avail_bytes = (int64_t)(pin_gb_env * GB);   /* explicit override */
        fprintf(stderr, "  PIN_GB=%.3g (explicit)\n", pin_gb_env);
    } else {
        int64_t phys = physical_ram_bytes();
        int64_t avail_now = available_ram_bytes();
        if (phys > 0) {
            /* Reserve for dense (~3–4.5 GB) + KV + OS overhead. The rest goes to
             * the experts. The cache self-limits to n_experts×n_layers anyway. */
            int64_t reserve = 6LL * GB;
            avail_bytes = phys - reserve;
            int64_t min_cache = expert_bytes * c->n_layers * c->topk;
            if (avail_now > 0) {
                int64_t dense_est = dense_weight_estimate_bytes(c);
                int64_t live_cap = avail_now - dense_est - 512LL * 1024 * 1024;
                if (live_cap < min_cache) {
                    fprintf(stderr,
                            "  warning: only %.1f GB RAM available; model needs "
                            "~%.1f GB dense + %.1f GB minimum expert cache. "
                            "Close other applications to avoid paging.\n",
                            (double)avail_now / 1e9, (double)dense_est / 1e9,
                            (double)min_cache / 1e9);
                    live_cap = min_cache;
                }
                if (live_cap < avail_bytes) avail_bytes = live_cap;
            }
            if (avail_bytes < min_cache) avail_bytes = min_cache;
            fprintf(stderr, "  RAM currently available: %.1f GB\n",
                    (double)avail_now / 1e9);
            fprintf(stderr, "  physical RAM %.1f GB → expert budget %.1f GB "
                    "(reserve %.0f GB for dense/KV/OS; override with PIN_GB)\n",
                    (double)phys / 1e9, (double)avail_bytes / 1e9,
                    (double)reserve / 1e9);
        } else {
            avail_bytes = 8LL * GB;  /* RAM undetectable: fixed default */
            fprintf(stderr, "  RAM undetectable → expert budget 8 GB\n");
        }
    }
    int total_slots = (int)(avail_bytes / expert_bytes);
    m.ecap = total_slots / c->n_layers;
    if (m.ecap < 4) m.ecap = 4;
    if (m.ecap > 128) m.ecap = 128;
    { const char *v = getenv("ECAP");   /* manual override for tuning/tests */
      if (v) { int e = atoi(v); if (e > 0) m.ecap = e > 128 ? 128 : e; } }
    if (m.ecap > c->n_experts) m.ecap = c->n_experts;  /* never more slots than experts */
    if (m.ecap < c->topk) m.ecap = c->topk;  /* cache_load_batch requires ecap>=topk */
    fprintf(stderr, "  expert cache: %d slots/layer × %d layers (%d experts total, ~%.1f GB)\n",
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

    /* Load dense weights */
    double t0 = now_s();
    g_model_path_global = model_path;
    if (load_dense_weights(&m, &db) != 0) {
        fprintf(stderr, "error: weight loading failed\n");
        st_close(&db);
        return 1;
    }

    /* Load the hot-store from a previous session */
    hotstore_load(&m);

    /* ── 4. Load the tokenizer ──
     * Only the bare-metal text path uses the built-in tokenizer. In SERVICE mode
     * the Python bridge owns tokenization and exchanges raw token IDs, so skip it
     * entirely (and don't print a misleading "not found" warning). */
    Tokenizer tok;
    int has_tokenizer = 0;
    {
        const char *svc = getenv("SERVICE");
        if (!(svc && atoi(svc))) {
            char tok_path[512];
            snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.json", model_path);
            if (tok_load(&tok, tok_path) == 0) {
                has_tokenizer = 1;
            } else {
                fprintf(stderr, "  ⚠ tokenizer.json not found — raw token ID mode\n");
            }
        }
    }

    fprintf(stderr, "✓ loaded in %.1f s · resident %.2f GB\n",
            now_s() - t0, m.resident_bytes / 1e9);

    /* Optional GPU backend. GPU_PREFETCH/GPU_ROUTER use the resident router
     * tier without moving expert compute away from the CPU. */
    gpu_backend_load(&m);
    {
        const char *release_host = getenv("GPU_DENSE_RELEASE_HOST");
        if (release_host && atoi(release_host) && !g_gpu_dense_host_released) {
            fprintf(stderr, "error: GPU_DENSE_RELEASE_HOST requested, but no host "
                    "weights were released; refusing a CPU fallback with this cache budget\n");
            return 1;
        }
    }

    /* ── 5. Sampling parameters ── */
    {
        const char *v;
        v = getenv("TEMPERATURE"); if (v) g_temperature = (float)atof(v);
        v = getenv("TOPP"); if (v) g_top_p = (float)atof(v);
        v = getenv("TOPK"); if (v) g_top_k = atoi(v);
        v = getenv("REP");  if (v) g_rep_penalty = (float)atof(v);
        v = getenv("SEED"); if (v) g_rng = (uint64_t)atoll(v);
        else g_rng = (uint64_t)time(NULL);  /* default random seed */
    }
    fprintf(stderr, "sampling: temp=%.2f top_p=%.2f top_k=%d rep=%.2f\n",
            g_temperature, g_top_p, g_top_k, g_rep_penalty);

    /* Parallel expert reads (queue depth > 1). IO_THREADS controls the number of
     * threads; PIPE=0 forces the serial path for compatibility. */
    {
        const char *v = getenv("IO_THREADS");
        g_io_threads = v ? atoi(v) : 4;
        const char *pipe = getenv("PIPE");
        if (pipe && atoi(pipe) == 0) g_io_threads = 1;
        if (g_io_threads < 1) g_io_threads = 1;
        fprintf(stderr, "expert I/O: %d threads (%s)\n", g_io_threads,
                g_io_threads > 1 ? "parallel reads" : "serial");
    }

    /* ASYNC_MOE=1: overlap current-layer expert reads with CPU expert compute.
     * Kept opt-in until representative disk/CPU combinations are benchmarked. */
    {
        const char *v = getenv("ASYNC_MOE");
        if (v && atoi(v) != 0) {
            if (async_loader_init(&m)) {
                atexit(async_loader_shutdown);
                fprintf(stderr, "expert pipeline: completion-driven async MoE\n");
            } else {
                fprintf(stderr, "  warning: async MoE thread creation failed; using sync path\n");
            }
        }
    }

    /* IDOT=1: faster integer expert kernel (INT8 activation × INT4 weight, AVX2).
     * Approximate (activation quantized to int8), so opt-in; F32 stays default. */
    {
        const char *v = getenv("IDOT");
        if (v && atoi(v) != 0) {
            quant_set_idot(1);
            fprintf(stderr, "expert kernel: IDOT (int8×int4 integer dot, %s)\n",
                    quant_idot_vnni() ? "AVX-VNNI dpbusd" : "AVX2 maddubs");
        }
    }

    /* DROP=1: drop just-read file pages from the OS page cache after each read,
     * so peak RSS stays "dense + our cache" instead of creeping toward the whole
     * model when streaming a model larger than RAM (Linux only). */
    {
        const char *v = getenv("DROP");
        if (v && atoi(v) != 0) {
            st_set_drop_cache(1);
            fprintf(stderr, "page cache: DROP (fadvise DONTNEED after reads)\n");
        }
    }

    /* DIRECT=1: unbuffered expert reads (O_DIRECT / FILE_FLAG_NO_BUFFERING),
     * bypassing the OS page cache — a win on fast internal NVMe. Opt-in, with a
     * buffered fallback per read so correctness is never at risk. */
    {
        const char *v = getenv("DIRECT");
        if (v && atoi(v) != 0) {
            st_set_direct(1);
            fprintf(stderr, "expert I/O: DIRECT (unbuffered, page-cache bypass)\n");
        }
    }

    /* Prefetch → LRU. GPU_PREFETCH=1 runs the resident L+1 router before the
     * current MoE and overlaps reads with current expert load+compute. The CPU
     * PREFETCH/PILOT mode retains the later, more accurate post-MoE proxy. */
    {
        const char *v = getenv("PREFETCH");
        if (!v) v = getenv("PILOT");
        const char *gv = getenv("GPU_PREFETCH");
        int want_gpu_pf = gv && atoi(gv) != 0;
        if ((v && atoi(v)) || want_gpu_pf) {
            pilot_init(&m);
            g_gpu_prefetch = want_gpu_pf && g_gpu_router && g_pilot_enabled;
            if (g_gpu_prefetch)
                fprintf(stderr, "prefetch → LRU enabled (GPU early L+1 proxy)\n");
            else {
                if (want_gpu_pf && !g_gpu_router)
                    fprintf(stderr, "  warning: GPU prefetch unavailable; using CPU post-MoE proxy\n");
                fprintf(stderr, "prefetch → LRU enabled (post-MoE proxy)\n");
            }
        }
    }

    /* ── 6. Generation ── */
    fprintf(stderr, "\n");

    /* Diagnostic: run the split byte-identity check ON THE REAL MODEL in a single
     * process (mono vs run_layer_range[0,cut)+[cut,L), same memory, no network).
     * This isolates a split-math bug (DIFFER) from a network/process bug (IDENTICAL). */
    { const char *v = getenv("PIPE_SPLIT_CHECK");
      if (v) {
        int cut = atoi(v);
        if (cut < 1) cut = 1;
        if (cut >= c->n_layers) cut = c->n_layers - 1;
        g_temperature = 0.0f; g_rep_penalty = 1.0f;
        int toks[4] = {1, 15496, 100, 2323};
        for (int i = 0; i < 4; i++) {
            int ok = pipe_split_check(&m, toks[i], i, cut);
            fprintf(stderr, "PIPE_SPLIT_CHECK tok=%d pos=%d cut=%d: %s\n",
                    toks[i], i, cut, ok ? "IDENTICAL" : "DIFFER");
        }
        pilot_shutdown();
        async_loader_shutdown();
        st_close(&db);
        return 0;
      } }

    /* Distributed pipeline mode: this process is one stage of a 2-machine split.
     * PIPE_ROLE=worker  -> stage B: accept a coordinator, run its requested layer
     *                      range + head, return tokens (long-lived).
     * PIPE_ROLE=coord   -> stage A: connect to PIPE_PEER (host:port), run
     *                      embed + layers [0,PIPE_CUT), stream residuals, decode.
     * Both load the full model here (Step 2); loading only the owned layers is
     * the next step. The worker is stateless about the cut: the coordinator names
     * the layer range in every message. */
    { const char *prole = getenv("PIPE_ROLE");
      if (prole && strcmp(prole, "worker") == 0) {
        net_init();
        int port = 52200;
        { const char *v = getenv("PIPE_PORT"); if (v) port = atoi(v); }
        sock_t ls = tcp_listen(port);
        if (ls == BADSOCK) {
            fprintf(stderr, "pipe worker: cannot listen on port %d\n", port);
            st_close(&db); return 1;
        }
        fprintf(stderr, "pipe worker (stage B): listening on port %d "
                "(owns layers [%d,%d) + head; returns logits, coordinator samples)\n",
                port, g_pipe_lo, g_pipe_hi);
        for (;;) {
            sock_t conn = accept(ls, NULL, NULL);
            if (conn == BADSOCK) break;
            fprintf(stderr, "pipe worker: coordinator connected\n");
            pipe_worker_serve(&m, conn);
            net_close(conn);
            fprintf(stderr, "pipe worker: coordinator disconnected, waiting\n");
        }
        net_close(ls);
        stats_dump(&m); hotstore_save(&m); pilot_shutdown();
        async_loader_shutdown();
        if (has_tokenizer) tok_free(&tok);
        st_close(&db);
        return 0;
      }
      if (prole && strcmp(prole, "coord") == 0) {
        const char *peer = getenv("PIPE_PEER");
        if (!peer) peer = "127.0.0.1:52200";
        const char *svc = getenv("SERVICE");
        if (svc && atoi(svc)) {
            /* Distributed CHAT: connect to the worker once, then drive service_loop
             * (the protocol the Python bridge speaks), forwarding through the worker. */
            char host[256] = "127.0.0.1"; int port = 52200;
            { const char *colon = strrchr(peer, ':');
              if (colon) { size_t hl = (size_t)(colon - peer);
                           if (hl >= sizeof(host)) hl = sizeof(host) - 1;
                           memcpy(host, peer, hl); host[hl] = '\0'; port = atoi(colon + 1); }
              else strncpy(host, peer, sizeof(host) - 1); }
            net_init();
            g_pipe_sock = tcp_connect(host, port);
            if (g_pipe_sock == BADSOCK) {
                fprintf(stderr, "pipe coord: cannot connect to %s:%d\n", host, port);
                st_close(&db); return 1;
            }
            g_pipe_coord = 1;
            fprintf(stderr, "pipe coord (chat): worker %s:%d, cut@%d, layers [0,%d) here\n",
                    host, port, g_pipe_cut, g_pipe_cut);
            int rc = service_loop(&m, initial_ctx);
            net_close(g_pipe_sock);
            stats_dump(&m); hotstore_save(&m); pilot_shutdown();
            async_loader_shutdown();
            if (has_tokenizer) tok_free(&tok);
            st_close(&db);
            return rc;
        }
        int rc = pipe_coord_run(&m, peer, g_pipe_cut, max_tokens);
        stats_dump(&m); hotstore_save(&m); pilot_shutdown();
        async_loader_shutdown();
        if (has_tokenizer) tok_free(&tok);
        st_close(&db);
        return rc;
      } }

    /* Service mode: persistent session, no prompt/decode in C. */
    { const char *v = getenv("SERVICE");
      if (v && atoi(v)) {
        fprintf(stderr, "service: persistent session, KV capacity %d positions\n",
                initial_ctx);
        int rc = service_loop(&m, initial_ctx);
        stats_dump(&m);
        hotstore_save(&m);
        pilot_shutdown();
        async_loader_shutdown();
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

    /* Build the prompt with the legacy Harmony template, or use official IDs from a file. */
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
        fprintf(stderr, "prompt (INPUT_FILE raw ID, %d tokens)\n", n_prompt);
    } else {
        /* Read the prompt from stdin or an environment variable. */
        char prompt[4096] = "";
        const char *user_prompt = getenv("INPUT");
        if (!user_prompt || strlen(user_prompt) == 0) {
            user_prompt = getenv("PROMPT");
            /* Ignore the Windows PROMPT variable ($P$G). */
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
                fprintf(stderr, "prompt (raw ID, %d tokens)\n", n_prompt);
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
                fprintf(stderr, "prompt: \"%s\" → %d tokens (Harmony legacy)\n",
                        prompt, n_prompt);
            }
        } else if (prompt[0] != '\0') {
            const char *p = prompt;
            while (*p && n_prompt < 8192) {
                while (*p == ' ' || *p == '\n') p++;
                if (*p == '\0') break;
                prompt_tokens[n_prompt++] = (int)strtol(p, (char **)&p, 10);
            }
            fprintf(stderr, "prompt (raw ID): %d tokens\n", n_prompt);
        }
    }

    if (n_prompt == 0) {
        fprintf(stderr, "error: empty prompt\n");
        if (prompt_tokens_owned) free(prompt_tokens);
        if (has_tokenizer) tok_free(&tok);
        st_close(&db);
        return 1;
    }
    for (int i = 0; i < n_prompt; i++) {
        if (prompt_tokens[i] < 0 || prompt_tokens[i] >= c->vocab) {
            fprintf(stderr, "error: token ID out of vocabulary at position %d: %d\n",
                    i, prompt_tokens[i]);
            if (prompt_tokens_owned) free(prompt_tokens);
            if (has_tokenizer) tok_free(&tok);
            st_close(&db);
            return 1;
        }
    }

    /* Speculative decoding: load a small draft (DRAFT_MODEL) and run the
     * draft-propose / target-verify loop instead of plain decoding. */
    {
        const char *sk = getenv("SPEC_K"); if (sk && atoi(sk) > 0) g_spec_k = atoi(sk);
        const char *dm = getenv("DRAFT_MODEL");
        if (dm && *dm) load_draft_isolated(dm, m.kv.max_pos);
    }
    if (g_have_draft) {
        spec_decode(&m, prompt_tokens, n_prompt, max_tokens, &tok, has_tokenizer);
        if (prompt_tokens_owned) free(prompt_tokens);
        if (has_tokenizer) tok_free(&tok);
        st_close(&db);
        return 0;
    }

    /* SPEC_VERIFY: validate the batched forward_verify against the sequential
     * forward_token — per-position argmax must be identical. Core correctness
     * check for speculative decoding; exits without generating. */
    if (getenv("SPEC_VERIFY") && atoi(getenv("SPEC_VERIFY"))) {
        int nv = n_prompt < 16 ? n_prompt : 16;
        int *pb = (int *)malloc(nv * sizeof(int)), *ps = (int *)malloc(nv * sizeof(int));
        forward_verify(&m, prompt_tokens, nv, 0, pb);                 /* one batched forward */
        for (int p = 0; p < nv; p++) ps[p] = forward_token(&m, prompt_tokens[p], p); /* sequential */
        int match = 0;
        for (int p = 0; p < nv; p++) if (pb[p] == ps[p]) match++;
        fprintf(stderr, "\n── SPEC_VERIFY: batched forward_verify vs sequential ──\n");
        fprintf(stderr, "positions %d · argmax match %d/%d (%s)\n", nv, match, nv,
                match == nv ? "IDENTICAL - forward_verify correct" : "MISMATCH");
        for (int p = 0; p < nv; p++) if (pb[p] != ps[p])
            fprintf(stderr, "  pos %d: batched=%d sequential=%d\n", p, pb[p], ps[p]);
        free(pb); free(ps);
        return match == nv ? 0 : 1;
    }

    /* Prefill: process all the prompt tokens */
    fprintf(stderr, "prefill %d tokens...\n", n_prompt);
    double t_prefill = now_s();
    int last_tok = prefill_tokens(&m, prompt_tokens, n_prompt, 0);
    fprintf(stderr, "✓ prefill in %.2f s (next=%d)\n\n", now_s() - t_prefill, last_tok);

    /* Decode: generate tokens autoregressively */
    double t_gen = now_s();
    int pos = n_prompt;
    int tok_id = last_tok;
    if (getenv("SPEC_PROBE") && atoi(getenv("SPEC_PROBE"))) {
        g_spec_probe = 1;
        g_probe_L = c->n_layers; g_probe_K = c->topk;
        g_probe_captok = max_tokens < 8192 ? max_tokens : 8192;
        g_probe_exp = (int *)malloc((size_t)g_probe_captok * g_probe_L * g_probe_K * sizeof(int));
        g_probe_hist_cap = g_probe_captok + n_prompt + 8;
        g_probe_hist = (int *)malloc((size_t)g_probe_hist_cap * sizeof(int));
        if (!g_probe_exp || !g_probe_hist) {
            g_spec_probe = 0; free(g_probe_exp); free(g_probe_hist);
        } else {
            for (int i = 0; i < n_prompt; i++) g_probe_hist[g_probe_hist_n++] = prompt_tokens[i];
            g_probe_prompt_n = n_prompt;
            fprintf(stderr, "SPEC_PROBE on: measuring n-gram acceptance + expert-union\n");
        }
    }
    if (getenv("SELF_DRAFT_PROBE") && atoi(getenv("SELF_DRAFT_PROBE"))) {
        g_self_draft_probe = 1;
        { const char *v = getenv("SELF_DRAFT_K"); if (v && atoi(v) > 0) g_sd_draft_k = atoi(v); }
        fprintf(stderr, "  draft width: top-%d\n", g_sd_draft_k);
        fprintf(stderr, "SELF_DRAFT_PROBE on: top-1 (draft) vs top-4 (target) per token"
                        " — generation runs ~2x slower during the measurement\n");
    }
    int in_final = 0;       /* 1 when we are in the "final" channel */
    int skip_header = 1;    /* 1 to skip the header tokens (channel, etc.) */

    /* If no tokenizer, print everything directly (no harmony parsing) */
    if (!has_tokenizer) { in_final = 1; skip_header = 0; }

    /* Debug: also show analysis if VERBOSE=1 */
    int show_all = 0;
    { const char *v = getenv("VERBOSE"); if (v && atoi(v)) show_all = 1; }
    int output_ids = 0;
    { const char *v = getenv("OUTPUT"); if (v && strcmp(v, "ids") == 0) output_ids = 1; }

    for (int i = 0; i < max_tokens; i++) {
        /* The machine-readable protocol always includes the terminator. */
        if (output_ids) {
            printf("%d\n", tok_id);
            fflush(stdout);
        }

        /* Stop of the assistant response: return or call. TOK_END only closes a message. */
        if (tok_id == TOK_RETURN || tok_id == TOK_CALL) break;
        if (tok_id == c->stop_ids[0]) break;  /* generic EOS */

        if (output_ids) {
            /* No C decode or filter: the Harmony bridge owns the rendering. */
        }
        /* Harmony channel handling:
         * - After <|channel|>, read the channel name
         * - If "final", show the content
         * - If "analysis", do not show (internal chain of thought)
         */
        else if (tok_id == TOK_CHANNEL) {
            skip_header = 1;  /* the next tokens are a channel header */
        } else if (tok_id == TOK_MESSAGE) {
            skip_header = 0;  /* after <|message|> the content begins */
        } else if (tok_id == TOK_START) {
            skip_header = 1;  /* start of a new message */
            in_final = 0;
        } else if (tok_id == TOK_END) {
            /* End of message — do not print */
        } else if (skip_header) {
            /* We are in the header — check whether it is "final" or a normal token */
            if (tok_id == TOK_CHANNEL || tok_id == TOK_START || tok_id == TOK_END ||
                tok_id == TOK_MESSAGE) {
                /* Special token in the header — already handled above */
            } else if (has_tokenizer) {
                const char *s = tok_decode(&tok, tok_id);
                if (strstr(s, "final")) { in_final = 1; }
                else if (strstr(s, "analysis")) { in_final = 0; }
                else if (strstr(s, "commentary")) { in_final = 0; }
                else {
                    /* Normal token outside any structure —
                     * the model is generating without a harmony header.
                     * We treat it as direct content. */
                    skip_header = 0;
                    in_final = 1;
                    /* Print this token */
                    char decoded[512];
                    tok_decode_raw(&tok, tok_id, decoded, sizeof(decoded));
                    printf("%s", decoded);
                    fflush(stdout);
                }
            } else {
                /* No tokenizer, normal token: print */
                skip_header = 0;
                in_final = 1;
                printf("[%d]", tok_id);
                fflush(stdout);
            }
        } else {
            /* Message content — print if we are in final (or verbose) */
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

        /* Forward for the next token */
        if (g_self_draft_probe) {
            /* draft: top-1 forward (transient KV), then target: top-4 forward
             * (overwrites KV correctly). Real generation uses the top-4 token. */
            g_force_top1 = 1; int d0 = forward_token(&m, tok_id, pos);
            g_force_top1 = 0; int m0 = forward_token(&m, tok_id, pos);
            g_sd_total++;
            if (d0 == m0) { g_sd_match++; g_sd_run++; }
            else { g_sd_runhist[g_sd_run < 7 ? g_sd_run : 7]++; g_sd_run = 0; }
            tok_id = m0;
        } else {
            tok_id = forward_token(&m, tok_id, pos);
        }
        if (g_spec_probe) {
            if (g_probe_ntok < g_probe_captok) g_probe_ntok++;   /* slot just filled */
            if (g_probe_hist_n < g_probe_hist_cap) g_probe_hist[g_probe_hist_n++] = tok_id;
        }
        pos++;
        m.n_emit++;
    }

    printf("\n");
    if (g_spec_probe || g_self_draft_probe) spec_probe_report();
    if (g_spec_probe) { free(g_probe_exp); free(g_probe_hist); }
    double gen_elapsed = now_s() - t_gen;

    /* Stats */
    fprintf(stderr, "\n");
    if (m.n_emit > 0) {
        fprintf(stderr, "generated %llu tokens in %.2f s (%.1f tok/s)\n",
                (unsigned long long)m.n_emit, gen_elapsed,
                m.n_emit / gen_elapsed);
    }
    stats_dump(&m);

    /* Save the hot-store for the next session */
    hotstore_save(&m);

    /* Cleanup */
    pilot_shutdown();
    async_loader_shutdown();
    if (prompt_tokens_owned) free(prompt_tokens);
    if (has_tokenizer) tok_free(&tok);
    st_close(&db);
    return 0;
}
