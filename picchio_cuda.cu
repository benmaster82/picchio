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

/* ── host state ─────────────────────────────────────────────────────────── */

struct DevW { int8_t *w; float *s; };
static std::unordered_map<const void *, DevW> g_wcache;   /* resident weights */

static float   *g_dx   = nullptr;  size_t g_dx_cap   = 0;  /* activation scratch */
static float   *g_dy   = nullptr;  size_t g_dy_cap   = 0;  /* output scratch */
static uint8_t *g_dq4  = nullptr;  size_t g_dq4_cap  = 0;  /* expert weight scratch */
static float   *g_ds   = nullptr;  size_t g_ds_cap   = 0;  /* expert scale scratch */

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
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) == cudaSuccess) {
        snprintf(g_name, sizeof(g_name), "%s", p.name);
        size_t freeb = 0, totb = 0; cudaMemGetInfo(&freeb, &totb);
        fprintf(stderr, "[gpu] %s · %.1f/%.1f GB free · sm_%d%d\n",
                p.name, freeb / 1e9, totb / 1e9, p.major, p.minor);
    }
    g_ok = 1;
    return 1;
}

extern "C" int pgpu_available(void) { return g_ok; }

extern "C" void pgpu_shutdown(void) {
    for (auto &kv : g_wcache) { cudaFree(kv.second.w); cudaFree(kv.second.s); }
    g_wcache.clear();
    if (g_dx)  cudaFree(g_dx);
    if (g_dy)  cudaFree(g_dy);
    if (g_dq4) cudaFree(g_dq4);
    if (g_ds)  cudaFree(g_ds);
    g_dx = g_dy = nullptr; g_dq4 = nullptr; g_ds = nullptr;
    g_dx_cap = g_dy_cap = g_dq4_cap = g_ds_cap = 0;
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
