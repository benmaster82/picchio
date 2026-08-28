/* g2_bench.cu — microbench for the G2 expert-on-GPU tier (measurement only).
 *
 * Question: is it worth moving the MoE expert matmuls to the GPU? The expensive
 * part per token is the top-K experts' two INT4 matmuls per layer. This bench
 * times, with accurate cudaEvent timing:
 *   (a) RESIDENT : weights already in VRAM (the hot-store ceiling, no upload)
 *   (b) STREAMED : weights uploaded from RAM each use (the cold path, PCIe cost)
 * and compares against the CPU baseline measured in the real engine.
 *
 * Build:  nvcc -O3 -arch=sm_75 -o g2_bench.exe g2_bench.cu
 * Run  :  g2_bench.exe
 *
 * Shapes: GPT-OSS 20B/120B expert — gate_up [5760,2880], down [2880,2880],
 * INT4 group-scaled (gs=64), top-K=4, 24 layers (20B).
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

#define D        2880
#define MOE_INT  5760      /* gate_up output (fused gate+up) */
#define DOWN_I   2880      /* down input = intermediate after SwiGLU */
#define GS       64
#define K        4         /* active experts per layer */
#define LAYERS   24        /* 20B */

/* One thread per output row: y[o] = sum_g sc[o,g] * sum_i x[i]*(nib(i)-8). */
__global__ void i4gs(const uint8_t *__restrict__ Q, const float *__restrict__ S,
                     const float *__restrict__ x, float *__restrict__ y,
                     int O, int I, int gs) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= O) return;
    int rb = (I + 1) / 2, ng = (I + gs - 1) / gs;
    const uint8_t *w = Q + (long long)o * rb;
    const float   *sc = S + (long long)o * ng;
    float a = 0.f;
    for (int g = 0; g < ng; g++) {
        int g0 = g * gs, g1 = g0 + gs; if (g1 > I) g1 = I;
        float ga = 0.f;
        for (int i = g0; i < g1; i++) {
            unsigned char b = w[i >> 1];
            int nib = (i & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
            ga += x[i] * (float)(nib - 8);
        }
        a += ga * sc[g];
    }
    y[o] = a;
}

struct Expert { uint8_t *gu, *dn; float *gu_s, *dn_s; };  /* device pointers */

static uint8_t *dev_u8(size_t n)  { void *p; cudaMalloc(&p, n); return (uint8_t*)p; }
static float   *dev_f32(size_t n) { void *p; cudaMalloc(&p, n*4); return (float*)p; }

int main(void) {
    int nd = 0; if (cudaGetDeviceCount(&nd) != cudaSuccess || nd == 0) {
        printf("no CUDA device\n"); return 1; }
    cudaSetDevice(0);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("[gpu] %s · sm_%d%d · %.0f GB/s mem\n", p.name, p.major, p.minor,
           2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1e6);

    int gu_rb = (D + 1) / 2,      gu_ng = (D + GS - 1) / GS;
    int dn_rb = (DOWN_I + 1) / 2, dn_ng = (DOWN_I + GS - 1) / GS;
    size_t gu_wb = (size_t)MOE_INT * gu_rb, gu_sb = (size_t)MOE_INT * gu_ng;
    size_t dn_wb = (size_t)D * dn_rb,       dn_sb = (size_t)D * dn_ng;
    printf("expert bytes: gate_up %.2f MB + down %.2f MB = %.2f MB/expert\n",
           gu_wb/1e6, dn_wb/1e6, (gu_wb+dn_wb)/1e6);

    /* Host staging (random) for the streamed test. */
    uint8_t *h_gu = (uint8_t*)malloc(gu_wb); for (size_t i=0;i<gu_wb;i++) h_gu[i]=rand()&0xFF;
    uint8_t *h_dn = (uint8_t*)malloc(dn_wb); for (size_t i=0;i<dn_wb;i++) h_dn[i]=rand()&0xFF;

    /* K resident experts + activation + outputs in VRAM. */
    Expert e[K];
    for (int k=0;k<K;k++){ e[k].gu=dev_u8(gu_wb); e[k].dn=dev_u8(dn_wb);
                           e[k].gu_s=dev_f32(gu_sb); e[k].dn_s=dev_f32(dn_sb);
                           cudaMemcpy(e[k].gu,h_gu,gu_wb,cudaMemcpyHostToDevice);
                           cudaMemcpy(e[k].dn,h_dn,dn_wb,cudaMemcpyHostToDevice); }
    float *x  = dev_f32(D), *h1 = dev_f32(MOE_INT), *o1 = dev_f32(D);
    uint8_t *sc_gu = dev_u8(0); (void)sc_gu;
    uint8_t *scratch_gu = dev_u8(gu_wb), *scratch_dn = dev_u8(dn_wb);

    dim3 tb(128);
    auto gu_grid = dim3((MOE_INT+127)/128);
    auto dn_grid = dim3((D+127)/128);

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    const int ITERS = 100;

    /* ---- (a) RESIDENT: K experts (gate_up + down), weights already in VRAM ---- */
    for (int w=0; w<5; w++)  /* warmup */
      for (int k=0;k<K;k++){ i4gs<<<gu_grid,tb>>>(e[k].gu,e[k].gu_s,x,h1,MOE_INT,D,GS);
                             i4gs<<<dn_grid,tb>>>(e[k].dn,e[k].dn_s,h1,o1,D,DOWN_I,GS); }
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int it=0; it<ITERS; it++)
      for (int k=0;k<K;k++){ i4gs<<<gu_grid,tb>>>(e[k].gu,e[k].gu_s,x,h1,MOE_INT,D,GS);
                             i4gs<<<dn_grid,tb>>>(e[k].dn,e[k].dn_s,h1,o1,D,DOWN_I,GS); }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_res=0; cudaEventElapsedTime(&ms_res,t0,t1); ms_res/=ITERS;

    /* ---- (b) STREAMED: upload each expert from RAM, then compute ---- */
    cudaEventRecord(t0);
    for (int it=0; it<ITERS; it++)
      for (int k=0;k<K;k++){
          cudaMemcpy(scratch_gu,h_gu,gu_wb,cudaMemcpyHostToDevice);
          i4gs<<<gu_grid,tb>>>(scratch_gu,e[k].gu_s,x,h1,MOE_INT,D,GS);
          cudaMemcpy(scratch_dn,h_dn,dn_wb,cudaMemcpyHostToDevice);
          i4gs<<<dn_grid,tb>>>(scratch_dn,e[k].dn_s,h1,o1,D,DOWN_I,GS);
      }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_str=0; cudaEventElapsedTime(&ms_str,t0,t1); ms_str/=ITERS;

    printf("\nPer LAYER (K=%d experts, gate_up+down):\n", K);
    printf("  (a) resident : %.3f ms\n", ms_res);
    printf("  (b) streamed : %.3f ms   (+%.3f ms upload of %.1f MB)\n",
           ms_str, ms_str-ms_res, K*(gu_wb+dn_wb)/1e6);

    printf("\nPer TOKEN (x%d layers), experts only:\n", LAYERS);
    printf("  (a) all resident  : %.1f ms/tok  -> %.2f tok/s (experts)\n",
           ms_res*LAYERS, 1000.0/(ms_res*LAYERS));
    printf("  (b) all streamed  : %.1f ms/tok  -> %.2f tok/s (experts)\n",
           ms_str*LAYERS, 1000.0/(ms_str*LAYERS));
    printf("\n  CPU baseline (engine, 20B, 6-core AVX2): t_moe ~663 ms/tok\n");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
