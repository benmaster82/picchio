/* picchio_cuda.h — optional CUDA backend for Picchio (prototype).
 *
 * This header is deliberately free of any dependency on quant.h / picchio.c:
 * the API exchanges only raw pointers + dimensions, so the CUDA translation
 * unit (compiled by nvcc/MSVC) and the engine (compiled by MinGW gcc) share no
 * C++ types and can be built by different toolchains. On Windows the .cu is
 * shipped as an optional DLL loaded at runtime; if it is absent, or if no CUDA
 * device is present, the engine transparently falls back to the CPU kernels.
 *
 * Scope of this first prototype (low-VRAM friendly, GTX 1650 4 GB target):
 *   - pgpu_matmul_i8    : lm_head — INT8 weight kept RESIDENT in VRAM (uploaded
 *                         once, cached by weight_id), recomputed every token.
 *   - pgpu_matmul_i4gs  : expert gate_up / down — INT4 group-scaled, streamed
 *                         through a small reusable VRAM SCRATCH (upload → compute
 *                         → discard). No residency required: even ~200 MB of VRAM
 *                         offloads the expert matmul compute.
 *
 * Every entry point returns 0 on success and -1 to tell the caller to fall back
 * to the CPU path for that single call (e.g. transient cudaMalloc failure).
 */
#ifndef PICCHIO_CUDA_H
#define PICCHIO_CUDA_H

#include <stdint.h>

/* Export macro: only the DLL build (nvcc -DPGPU_BUILD_DLL) exports the symbols.
 * The engine never includes this with the macro set — it resolves the entry
 * points at runtime via GetProcAddress/dlsym — so this only affects the .cu. */
#if defined(_WIN32) && defined(PGPU_BUILD_DLL)
#  define PGPU_API __declspec(dllexport)
#else
#  define PGPU_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the backend. Returns 1 if a usable CUDA device is ready, 0 if not
 * (no device, driver error): the caller must then stay on the CPU path. */
PGPU_API int  pgpu_init(void);

/* 1 if pgpu_init succeeded and the GPU path is active; 0 otherwise. */
PGPU_API int  pgpu_available(void);

/* Free all device buffers and the resident-weight cache. */
PGPU_API void pgpu_shutdown(void);

/* y[O] = scale[o] * sum_i x[i] * w[o*I + i]     (w INT8, per-row scale)
 * `weight_id` is an opaque, stable key (typically the host weight pointer): the
 * device copy is uploaded on the first call and reused afterwards. S = 1. */
PGPU_API int  pgpu_matmul_i8(float *y, const float *x,
                             const int8_t *w, const float *scale,
                             int O, int I, const void *weight_id);

/* y[O] = sum_g scale[o*ngroups + g] * sum_{i in group g} x[i] * (nibble(i) - 8)
 * q4 is INT4 packed (2 nibbles/byte, low nibble = even i), gs = group size (64).
 * Streamed through scratch — nothing is cached (expert weights change per token). */
PGPU_API int  pgpu_matmul_i4gs(float *y, const float *x,
                               const uint8_t *q4, const float *scale,
                               int O, int I, int gs);

#ifdef __cplusplus
}
#endif

#endif /* PICCHIO_CUDA_H */
