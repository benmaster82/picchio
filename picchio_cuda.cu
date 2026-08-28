/* picchio_cuda.cu — optional CUDA backend for Picchio (prototype).
 *
 * Kernels:
 *   lm_head_i8     : one block per output row, coalesced INT8 reads, block reduce.
 *   expert_i4gs    : one thread per output row, INT4 group-scaled dequant matmul.
 *
 * The math mirrors quant.h exactly (matmul_q8 / matmul_i4_gs) so the GPU top-1
 * token matches the CPU oracle; only the FP reduction order differs, so results
 * are NOT byte-identical (that contract belongs to the CPU path).
 *
 * Build the standalone numeric self-test:
 *   nvcc -O3 -arch=sm_75 -DPGPU_TEST -o pgpu_test.exe picchio_cuda.cu
 *   ./pgpu_test.exe
 *
 * Build the runtime DLL for the engine (integration, next step):
 *   nvcc -O3 -arch=sm_75 -shared -o picchio_cuda.dll picchio_cuda.cu
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

#include "picchio_cuda.h"

/* ── error plumbing ─────────────────────────────────────────────────────── */

static int   g_ok = 0;          /* 1 once a device is initialized */
static char  g_name[256] = "";

#define CU_TRY(call) do {                                            \
        cudaError_t _e = (call);                                     \
        if (_e != cudaSuccess) {                                     \
            fprintf(stderr, "[gpu] %s failed: %s\n", #call,          \
                    cudaGetErrorString(_e));                         \
            return -1;                                               \
        }                                                            \
    } while (0)

/* ── device kernels ─────────────────────────────────────────────────────── */

/* lm_head: y[o] = scale[o] * dot(x, W_int8[o]). One block per row o. */
__global__ void lm_head_i8(const int8_t *__restrict__ W,
                           const float *__restrict__ scale,
                           const float *__restrict__ x,
                           float *__restrict__ y, int O, int I) {
    int o = blockIdx.x;
    if (o >= O) return;
    const int8_t *w = W + (long long)o * I;

    float partial = 0.f;
    for (int i = threadIdx.x; i < I; i += blockDim.x)
        partial += x[i] * (float)w[i];

    __shared__ float sm[256];
    sm[threadIdx.x] = partial;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[o] = sm[0] * scale[o];
}

/* expert INT4 group-scaled: y[o] = sum_g sc[o,g] * sum_{i in g} x[i]*(nib(i)-8).
 * One thread per output row (O is small: 5760 gate_up / 2880 down). */
__global__ void expert_i4gs(const uint8_t *__restrict__ Q,
                            const float *__restrict__ S,
                            const float *__restrict__ x,
                            float *__restrict__ y,
                            int O, int I, int gs) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= O) return;
    int rb = (I + 1) / 2;
    int ng = (I + gs - 1) / gs;
    const uint8_t *w = Q + (long long)o * rb;
    const float   *sc = S + (long long)o * ng;

    float a = 0.f;
    for (int g = 0; g < ng; g++) {
        int g0 = g * gs;
        int g1 = g0 + gs; if (g1 > I) g1 = I;
        float ga = 0.f;
        for (int i = g0; i < g1; i++) {
            unsigned char byte = w[i >> 1];
            int nib = (i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
            ga += x[i] * (float)(nib - 8);
        }
        a += ga * sc[g];
    }
    y[o] = a;
}

/* SwiGLU over interleaved (gate, up) pairs, mirroring moe_forward exactly.
 * bias (optional) is added to gate/up before the activation. */
__global__ void swiglu_k(const float *__restrict__ gu, const float *__restrict__ bias,
                         float *__restrict__ h, int half,
                         int clipped, float limit, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;
    float gate = gu[2 * i], up = gu[2 * i + 1];
    if (bias) { gate += bias[2 * i]; up += bias[2 * i + 1]; }
    if (clipped) {
        if (gate > limit) gate = limit;
        if (up > limit) up = limit;
        if (up < -limit) up = -limit;
        h[i] = (up + 1.0f) * gate * (1.0f / (1.0f + expf(-alpha * gate)));
    } else {
        h[i] = gate * (1.0f / (1.0f + expf(-gate))) * up;
    }
}

/* out[i] += w * (eo[i] + d_bias[i]) */
__global__ void accum_k(float *__restrict__ out, const float *__restrict__ eo,
                        const float *__restrict__ dbias, float w, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;
    float v = eo[i];
    if (dbias) v += dbias[i];
    out[i] += w * v;
}

__global__ void zero_k(float *p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.f;
}

/* ── host state ─────────────────────────────────────────────────────────── */

struct DevW { int8_t *w; float *s; };
static std::unordered_map<const void *, DevW> g_wcache;   /* resident weights */

static float   *g_dx   = nullptr;  size_t g_dx_cap   = 0;  /* activation scratch */
static float   *g_dy   = nullptr;  size_t g_dy_cap   = 0;  /* output scratch */
static uint8_t *g_dq4  = nullptr;  size_t g_dq4_cap  = 0;  /* expert weight scratch */
static float   *g_ds   = nullptr;  size_t g_ds_cap   = 0;  /* expert scale scratch */

/* G2 per-layer scratch (activations only; expert weights live in g_ecache). */
static float   *g_dgu  = nullptr;  size_t g_dgu_cap  = 0;  /* gate_up output */
static float   *g_dh   = nullptr;  size_t g_dh_cap   = 0;  /* swiglu output */
static float   *g_deo  = nullptr;  size_t g_deo_cap  = 0;  /* down output */
static float   *g_dout = nullptr;  size_t g_dout_cap = 0;  /* accumulated token out */

/* G2 VRAM expert cache: hot experts kept resident, keyed by (layer,eid), with a
 * byte-bounded LRU (PIN_VRAM_GB). Independent of the CPU RAM cache. */
struct DevExpert {
    uint8_t *gu; float *gu_s; float *gu_bias;
    uint8_t *d;  float *d_s;  float *d_bias;
    size_t   bytes;
    unsigned long long used;
};
static std::unordered_map<uint64_t, DevExpert> g_ecache;
static unsigned long long g_eclock  = 0;
static size_t             g_vram_used   = 0;
static size_t             g_vram_budget = 0;  /* 0 = set at init from PIN_VRAM_GB */
static unsigned long long g_ehit = 0, g_emiss = 0;

static void free_dev_expert(DevExpert &e) {
    cudaFree(e.gu); cudaFree(e.gu_s); cudaFree(e.gu_bias);
    cudaFree(e.d);  cudaFree(e.d_s);  cudaFree(e.d_bias);
}

static int ensure(void **buf, size_t *cap, size_t need) {
    if (*cap >= need) return 0;
    if (*buf) cudaFree(*buf);
    *buf = nullptr; *cap = 0;
    if (cudaMalloc(buf, need) != cudaSuccess) { *buf = nullptr; return -1; }
    *cap = need;
    return 0;
}

/* ── C API ──────────────────────────────────────────────────────────────── */

extern "C" int pgpu_init(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return 0;
    if (cudaSetDevice(0) != cudaSuccess) return 0;
    size_t freeb = 0, totb = 0; cudaMemGetInfo(&freeb, &totb);
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) == cudaSuccess) {
        snprintf(g_name, sizeof(g_name), "%s", p.name);
        fprintf(stderr, "[gpu] %s · %.1f/%.1f GB free · sm_%d%d\n",
                p.name, freeb / 1e9, totb / 1e9, p.major, p.minor);
    }

    /* Expert VRAM budget (PIN_VRAM_GB). Default: most of the free VRAM, minus a
     * ~0.6 GB margin for the display / driver / per-layer scratch. */
    const char *v = getenv("PIN_VRAM_GB");
    double gb = v ? atof(v) : 0.0;
    if (gb <= 0.0) { gb = (double)freeb / 1e9 - 0.6; if (gb < 0.2) gb = 0.2; }
    g_vram_budget = (size_t)(gb * 1e9);
    fprintf(stderr, "[gpu] expert VRAM budget: %.2f GB (~%.0f experts of ~14 MB)\n",
            g_vram_budget / 1e9, g_vram_budget / 14.0e6);

    g_ok = 1;
    return 1;
}

extern "C" int pgpu_available(void) { return g_ok; }

extern "C" void pgpu_shutdown(void) {
    if (g_emiss + g_ehit > 0)
        fprintf(stderr, "[gpu] expert VRAM cache: %llu hit / %llu miss (%.1f%%), %.2f GB resident\n",
                (unsigned long long)g_ehit, (unsigned long long)g_emiss,
                100.0 * g_ehit / (g_ehit + g_emiss), g_vram_used / 1e9);
    for (auto &kv : g_wcache)  { cudaFree(kv.second.w); cudaFree(kv.second.s); }
    for (auto &kv : g_ecache)  { free_dev_expert(kv.second); }
    g_wcache.clear(); g_ecache.clear();
    g_vram_used = 0;
    if (g_dx)   cudaFree(g_dx);
    if (g_dy)   cudaFree(g_dy);
    if (g_dq4)  cudaFree(g_dq4);
    if (g_ds)   cudaFree(g_ds);
    if (g_dgu)  cudaFree(g_dgu);
    if (g_dh)   cudaFree(g_dh);
    if (g_deo)  cudaFree(g_deo);
    if (g_dout) cudaFree(g_dout);
    g_dx = g_dy = g_dgu = g_dh = g_deo = g_dout = nullptr; g_ds = nullptr; g_dq4 = nullptr;
    g_dx_cap = g_dy_cap = g_dq4_cap = g_ds_cap = 0;
    g_dgu_cap = g_dh_cap = g_deo_cap = g_dout_cap = 0;
    g_ok = 0;
}

extern "C" int pgpu_matmul_i8(float *y, const float *x,
                              const int8_t *w, const float *scale,
                              int O, int I, const void *weight_id) {
    if (!g_ok) return -1;

    /* Upload (and cache) the weight the first time we see this weight_id. */
    auto it = g_wcache.find(weight_id);
    if (it == g_wcache.end()) {
        DevW dw{nullptr, nullptr};
        if (cudaMalloc(&dw.w, (size_t)O * I) != cudaSuccess) return -1;
        if (cudaMalloc(&dw.s, (size_t)O * sizeof(float)) != cudaSuccess) {
            cudaFree(dw.w); return -1;
        }
        if (cudaMemcpy(dw.w, w, (size_t)O * I, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(dw.s, scale, (size_t)O * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(dw.w); cudaFree(dw.s); return -1;
        }
        g_wcache[weight_id] = dw;
        it = g_wcache.find(weight_id);
    }

    if (ensure((void **)&g_dx, &g_dx_cap, (size_t)I * sizeof(float)) != 0) return -1;
    if (ensure((void **)&g_dy, &g_dy_cap, (size_t)O * sizeof(float)) != 0) return -1;
    CU_TRY(cudaMemcpy(g_dx, x, (size_t)I * sizeof(float), cudaMemcpyHostToDevice));

    lm_head_i8<<<O, 128>>>(it->second.w, it->second.s, g_dx, g_dy, O, I);
    CU_TRY(cudaGetLastError());
    CU_TRY(cudaMemcpy(y, g_dy, (size_t)O * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int pgpu_matmul_i4gs(float *y, const float *x,
                                const uint8_t *q4, const float *scale,
                                int O, int I, int gs) {
    if (!g_ok) return -1;
    int rb = (I + 1) / 2;
    int ng = (I + gs - 1) / gs;

    if (ensure((void **)&g_dx,  &g_dx_cap,  (size_t)I * sizeof(float)) != 0) return -1;
    if (ensure((void **)&g_dy,  &g_dy_cap,  (size_t)O * sizeof(float)) != 0) return -1;
    if (ensure((void **)&g_dq4, &g_dq4_cap, (size_t)O * rb) != 0) return -1;
    if (ensure((void **)&g_ds,  &g_ds_cap,  (size_t)O * ng * sizeof(float)) != 0) return -1;

    CU_TRY(cudaMemcpy(g_dx,  x,     (size_t)I * sizeof(float),      cudaMemcpyHostToDevice));
    CU_TRY(cudaMemcpy(g_dq4, q4,    (size_t)O * rb,                 cudaMemcpyHostToDevice));
    CU_TRY(cudaMemcpy(g_ds,  scale, (size_t)O * ng * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 128, blocks = (O + threads - 1) / threads;
    expert_i4gs<<<blocks, threads>>>(g_dq4, g_ds, g_dx, g_dy, O, I, gs);
    CU_TRY(cudaGetLastError());
    CU_TRY(cudaMemcpy(y, g_dy, (size_t)O * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

/* Resolve an expert in the VRAM cache, uploading (and evicting LRU) on a miss.
 * Returns a stable pointer into g_ecache, or NULL on allocation failure. */
static DevExpert *get_expert(uint64_t key, const PgpuExpert *e,
                             size_t gu_wb, size_t gu_sb, size_t gu_bb,
                             size_t d_wb, size_t d_sb, size_t d_bb) {
    auto it = g_ecache.find(key);
    if (it != g_ecache.end()) { g_ehit++; return &it->second; }
    g_emiss++;

    size_t need = gu_wb + gu_sb + d_wb + d_sb
                + (e->gu_bias ? gu_bb : 0) + (e->d_bias ? d_bb : 0);

    /* Evict least-recently-used experts until the newcomer fits the budget. */
    while (g_vram_budget > 0 && g_vram_used + need > g_vram_budget && !g_ecache.empty()) {
        auto victim = g_ecache.begin();
        for (auto i = g_ecache.begin(); i != g_ecache.end(); ++i)
            if (i->second.used < victim->second.used) victim = i;
        g_vram_used -= victim->second.bytes;
        free_dev_expert(victim->second);
        g_ecache.erase(victim);
    }

    DevExpert de; memset(&de, 0, sizeof(de)); de.bytes = need;
    bool ok = cudaMalloc(&de.gu,   gu_wb) == cudaSuccess
           && cudaMalloc(&de.gu_s, gu_sb) == cudaSuccess
           && cudaMalloc(&de.d,    d_wb)  == cudaSuccess
           && cudaMalloc(&de.d_s,  d_sb)  == cudaSuccess;
    if (ok && e->gu_bias) ok = cudaMalloc(&de.gu_bias, gu_bb) == cudaSuccess;
    if (ok && e->d_bias)  ok = cudaMalloc(&de.d_bias,  d_bb)  == cudaSuccess;
    if (ok)
        ok = cudaMemcpy(de.gu,   e->gu_q4, gu_wb, cudaMemcpyHostToDevice) == cudaSuccess
          && cudaMemcpy(de.gu_s, e->gu_s,  gu_sb, cudaMemcpyHostToDevice) == cudaSuccess
          && cudaMemcpy(de.d,    e->d_q4,  d_wb,  cudaMemcpyHostToDevice) == cudaSuccess
          && cudaMemcpy(de.d_s,  e->d_s,   d_sb,  cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok && e->gu_bias)
        ok = cudaMemcpy(de.gu_bias, e->gu_bias, gu_bb, cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok && e->d_bias)
        ok = cudaMemcpy(de.d_bias,  e->d_bias,  d_bb,  cudaMemcpyHostToDevice) == cudaSuccess;
    if (!ok) { free_dev_expert(de); return NULL; }

    g_vram_used += need;
    auto res = g_ecache.emplace(key, de);
    return &res.first->second;
}

extern "C" int pgpu_moe_layer(float *out, const float *x, int D, int moe_inter,
                              int layer, const PgpuExpert *experts,
                              const float *wsum, int nexp, int gs,
                              int swiglu_clipped, float swiglu_limit,
                              float swiglu_alpha) {
    if (!g_ok) return -1;
    int half  = moe_inter / 2;
    int gu_rb = (D + 1) / 2,    gu_ng = (D + gs - 1) / gs;
    int d_rb  = (half + 1) / 2, d_ng  = (half + gs - 1) / gs;
    size_t gu_wb = (size_t)moe_inter * gu_rb, gu_sb = (size_t)moe_inter * gu_ng * 4;
    size_t d_wb  = (size_t)D * d_rb,          d_sb  = (size_t)D * d_ng * 4;
    size_t gu_bb = (size_t)moe_inter * 4,     d_bb  = (size_t)D * 4;

    if (ensure((void **)&g_dx,   &g_dx_cap,   (size_t)D * 4)         != 0) return -1;
    if (ensure((void **)&g_dgu,  &g_dgu_cap,  (size_t)moe_inter * 4) != 0) return -1;
    if (ensure((void **)&g_dh,   &g_dh_cap,   (size_t)half * 4)      != 0) return -1;
    if (ensure((void **)&g_deo,  &g_deo_cap,  (size_t)D * 4)         != 0) return -1;
    if (ensure((void **)&g_dout, &g_dout_cap, (size_t)D * 4)         != 0) return -1;

    if (cudaMemcpy(g_dx, x, (size_t)D * 4, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    zero_k<<<(D + 127) / 128, 128>>>(g_dout, D);

    for (int k = 0; k < nexp; k++) {
        const PgpuExpert *e = &experts[k];
        uint64_t key = ((uint64_t)(unsigned)layer << 32) | (unsigned)e->eid;
        DevExpert *de = get_expert(key, e, gu_wb, gu_sb, gu_bb, d_wb, d_sb, d_bb);
        if (!de) return -1;
        de->used = ++g_eclock;

        expert_i4gs<<<(moe_inter + 127) / 128, 128>>>(de->gu, de->gu_s, g_dx, g_dgu, moe_inter, D, gs);
        swiglu_k<<<(half + 127) / 128, 128>>>(g_dgu, de->gu_bias, g_dh, half,
                                              swiglu_clipped, swiglu_limit, swiglu_alpha);
        expert_i4gs<<<(D + 127) / 128, 128>>>(de->d, de->d_s, g_dh, g_deo, D, half, gs);
        accum_k<<<(D + 127) / 128, 128>>>(g_dout, g_deo, de->d_bias, wsum[k], D);
    }
    if (cudaGetLastError() != cudaSuccess) return -1;
    if (cudaMemcpy(out, g_dout, (size_t)D * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  STANDALONE NUMERIC SELF-TEST + MICROBENCH (nvcc -DPGPU_TEST)
 * ══════════════════════════════════════════════════════════════════════════ */
#ifdef PGPU_TEST
#include <time.h>

static double wall(void) { return (double)clock() / CLOCKS_PER_SEC; }

static void ref_i8(float *y, const float *x, const int8_t *w,
                   const float *sc, int O, int I) {
    for (int o = 0; o < O; o++) {
        double a = 0;
        const int8_t *wr = w + (long long)o * I;
        for (int i = 0; i < I; i++) a += (double)x[i] * (double)wr[i];
        y[o] = (float)(a * sc[o]);
    }
}

static void ref_i4gs(float *y, const float *x, const uint8_t *q4,
                     const float *sc, int O, int I, int gs) {
    int rb = (I + 1) / 2, ng = (I + gs - 1) / gs;
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q4 + (long long)o * rb;
        const float *s = sc + (long long)o * ng;
        double a = 0;
        for (int g = 0; g < ng; g++) {
            int g0 = g * gs, g1 = g0 + gs; if (g1 > I) g1 = I;
            double ga = 0;
            for (int i = g0; i < g1; i++) {
                unsigned char byte = w[i >> 1];
                int nib = (i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                ga += (double)x[i] * (double)(nib - 8);
            }
            a += ga * (double)s[g];
        }
        y[o] = (float)a;
    }
}

static int argmax(const float *y, int n) {
    int b = 0; for (int i = 1; i < n; i++) if (y[i] > y[b]) b = i; return b;
}
static float maxabs(const float *a, const float *b, int n) {
    float m = 0; for (int i = 0; i < n; i++) { float d = fabsf(a[i]-b[i]); if (d>m) m=d; } return m;
}

int main(void) {
    if (!pgpu_init()) { fprintf(stderr, "no CUDA device\n"); return 1; }

    const int D = 2880;

    /* ---- lm_head INT8: [vocab, D] ---- */
    const int V = 201088;
    float  *x   = (float  *)malloc((size_t)D * 4);
    int8_t *w8  = (int8_t *)malloc((size_t)V * D);
    float  *s8  = (float  *)malloc((size_t)V * 4);
    float  *yc  = (float  *)malloc((size_t)V * 4);
    float  *yg  = (float  *)malloc((size_t)V * 4);
    srand(1234);
    for (int i = 0; i < D; i++) x[i] = (float)(rand() % 2001 - 1000) / 1000.f;
    for (long long i = 0; i < (long long)V * D; i++) w8[i] = (int8_t)(rand() % 255 - 127);
    for (int o = 0; o < V; o++) s8[o] = 0.002f + (float)(rand() % 100) / 100000.f;

    ref_i8(yc, x, w8, s8, V, D);
    /* warm-up + upload */
    pgpu_matmul_i8(yg, x, w8, s8, V, D, (const void *)w8);
    double t0 = wall();
    int iters = 20;
    for (int k = 0; k < iters; k++) pgpu_matmul_i8(yg, x, w8, s8, V, D, (const void *)w8);
    cudaDeviceSynchronize();
    double tg = (wall() - t0) / iters * 1000.0;

    double t1 = wall(); ref_i8(yc, x, w8, s8, V, D); double tc = (wall() - t1) * 1000.0;

    printf("lm_head INT8 [%d x %d]\n", V, D);
    printf("  top-1  CPU=%d  GPU=%d  %s\n", argmax(yc, V), argmax(yg, V),
           argmax(yc, V) == argmax(yg, V) ? "MATCH" : "MISMATCH");
    printf("  max|dCPU-GPU| = %.4g\n", maxabs(yc, yg, V));
    printf("  GPU %.2f ms/tok (resident) · CPU(1thr) %.1f ms/tok\n\n", tg, tc);

    /* ---- expert INT4 gs64: gate_up [5760, D] ---- */
    const int O = 5760, gs = 64;
    int rb = (D + 1) / 2, ng = (D + gs - 1) / gs;
    uint8_t *q4 = (uint8_t *)malloc((size_t)O * rb);
    float   *se = (float   *)malloc((size_t)O * ng * 4);
    float   *ec = (float   *)malloc((size_t)O * 4);
    float   *eg = (float   *)malloc((size_t)O * 4);
    for (long long i = 0; i < (long long)O * rb; i++) q4[i] = (uint8_t)(rand() & 0xFF);
    for (int i = 0; i < O * ng; i++) se[i] = 0.01f + (float)(rand() % 100) / 100000.f;

    ref_i4gs(ec, x, q4, se, O, D, gs);
    pgpu_matmul_i4gs(eg, x, q4, se, O, D, gs);
    double t2 = wall();
    for (int k = 0; k < iters; k++) pgpu_matmul_i4gs(eg, x, q4, se, O, D, gs);
    cudaDeviceSynchronize();
    double te = (wall() - t2) / iters * 1000.0;

    printf("expert INT4 gs64 [%d x %d] (streamed via scratch)\n", O, D);
    printf("  max|dCPU-GPU| = %.4g\n", maxabs(ec, eg, O));
    printf("  GPU %.3f ms/expert (upload+compute)\n", te);

    pgpu_shutdown();
    return 0;
}
#endif
