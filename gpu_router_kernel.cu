/* Build-time source for the PTX embedded by gpu_router_native.h.
 * It is not linked into the runtime. Regenerate for Turing with:
 *   nvcc -O3 -arch=sm_75 -ptx gpu_router_kernel.cu -o gpu_router_kernel.ptx
 */
extern "C" __global__ void picchio_router_f32(const float *__restrict__ W,
                                               const float *__restrict__ bias,
                                               const float *__restrict__ x,
                                               float *__restrict__ scores,
                                               int E, int D) {
    int e = blockIdx.x;
    if (e >= E) return;
    const float *w = W + (long long)e * D;
    float a = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        a += x[i] * w[i];
    __shared__ float sm[256];
    sm[threadIdx.x] = a;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[e] = sm[0] + (bias ? bias[e] : 0.0f);
}

/* Resident FP16 weights, FP32 activations/accumulation.  This is the dense
 * attention tier: one block computes one output row.  FP16 halves the complete
 * GPT-OSS-120B Q/K/V/O footprint from ~3.8 GB to ~1.9 GB, which fits beside the
 * routers on a 4 GB GTX 1650. */
#include <cuda_fp16.h>
extern "C" __global__ void picchio_dense_f16(const __half *__restrict__ W,
                                               const float *__restrict__ x,
                                               float *__restrict__ y,
                                               int O, int I) {
    int o = blockIdx.x;
    if (o >= O) return;
    const __half *w = W + (long long)o * I;
    float a = 0.0f;
    for (int i = threadIdx.x; i < I; i += blockDim.x)
        a += x[i] * __half2float(w[i]);
    __shared__ float sm[256];
    sm[threadIdx.x] = a;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[o] = sm[0];
}
