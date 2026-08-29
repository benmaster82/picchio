/* quant.h — Quantization and matmul kernels for Picchio.
 *
 * Supported formats:
 *   fmt=0  F32     (reference, no quantization)
 *   fmt=1  INT8    (per-row, symmetric, float32 scale)
 *   fmt=2  INT4    (per-row, 2 values/byte, offset 8, float32 scale)
 *   fmt=3  MXFP4   (per-block-32, OCP microscaling, E8M0 scale)
 *
 * SIMD kernels: AVX2, AVX-512 VNNI, ARM NEON (+SDOT).
 * IDOT: integer dot-product (activations quantized to int8 on-the-fly).
 *
 * Inspired by Colibri's kernels (c/glm.c), adapted for GPT-OSS-120B.
 */

#ifndef PICCHIO_QUANT_H
#define PICCHIO_QUANT_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ── Quantized tensor ── */

typedef struct {
    int fmt;            /* 0=F32, 1=INT8, 2=INT4, 3=MXFP4 */
    float *qf;          /* F32 data (fmt==0) */
    int8_t *q8;         /* INT8 data (fmt==1) */
    uint8_t *q4;        /* INT4/MXFP4 packed (fmt==2,3) */
    float *s;           /* scale: per-row (fmt 1,2), per-block (fmt 3) */
    int O, I;           /* dimensions [O, I] */
    int block_size;     /* MXFP4: elements per scale block (typically 32) */
} QT;

static inline int64_t qt_bytes(const QT *t) {
    int64_t n = (int64_t)t->O * t->I;
    switch (t->fmt) {
        case 0: return n * 4;                                    /* F32 */
        case 1: return n + (int64_t)t->O * 4;                   /* INT8 + scale */
        case 2: return (int64_t)t->O * ((t->I + 1) / 2)         /* INT4 packed */
                     + (int64_t)t->O * 4;                        /* + scale */
        case 3: {                                                /* MXFP4 */
            int bs = t->block_size > 0 ? t->block_size : 32;
            int64_t nblocks = (int64_t)t->O * ((t->I + bs - 1) / bs);
            return (int64_t)t->O * ((t->I + 1) / 2) + nblocks * 4;
        }
        case 5: {                                                /* INT3 gs64 */
            int64_t ng = ((int64_t)t->I + 63) / 64;
            return (int64_t)t->O * ng * 24 + (int64_t)t->O * ng * 4;  /* planes + scales */
        }
        default: return 0;
    }
}

/* ── Allocation ── */

static inline float *falloc(int64_t n) {
    if (n <= 0 || (uint64_t)n > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "falloc: n=%lld out of range\n", (long long)n);
        exit(1);
    }
    float *p = (float *)malloc((size_t)n * sizeof(float));
    if (!p) {
        fprintf(stderr, "OOM (falloc %lld floats = %.1f MB)\n",
                (long long)n, (double)n * 4 / 1e6);
        exit(1);
    }
    return p;
}

static inline void qt_alloc(QT *t, int O, int I, int bits) {
    memset(t, 0, sizeof(*t));
    t->O = O; t->I = I;
    if (bits >= 16) {
        t->fmt = 0;
        t->qf = falloc((int64_t)O * I);
    } else if (bits >= 5) {
        t->fmt = 1;
        t->q8 = malloc((int64_t)O * I);
        t->s = falloc(O);
    } else {
        t->fmt = 2;
        t->q4 = malloc((int64_t)O * ((I + 1) / 2));
        t->s = falloc(O);
    }
}

/* ── Horizontal sum helpers ── */

#ifdef __AVX2__
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehl_ps(lo, lo);
    lo = _mm_add_ps(lo, sh);
    sh = _mm_shuffle_ps(lo, lo, 1);
    lo = _mm_add_ss(lo, sh);
    return _mm_cvtss_f32(lo);
}

static inline int hsum256_i32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}
#endif

/* ── Matmul F32: y[S,O] = x[S,I] @ W^T ── */

static void matmul_f32(float *y, const float *x, const float *W,
                       int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float a = 0;
#ifdef __AVX2__
            __m256 acc = _mm256_setzero_ps();
            int i = 0;
            for (; i + 8 <= I; i += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i),
                                      _mm256_loadu_ps(w + i), acc);
            a = hsum256(acc);
            for (; i < I; i++) a += xs[i] * w[i];
#elif defined(__ARM_NEON)
            float32x4_t ac0 = vdupq_n_f32(0), ac1 = vdupq_n_f32(0);
            int i = 0;
            for (; i + 8 <= I; i += 8) {
                ac0 = vfmaq_f32(ac0, vld1q_f32(xs + i), vld1q_f32(w + i));
                ac1 = vfmaq_f32(ac1, vld1q_f32(xs + i + 4), vld1q_f32(w + i + 4));
            }
            a = vaddvq_f32(vaddq_f32(ac0, ac1));
            for (; i < I; i++) a += xs[i] * w[i];
#else
            for (int i = 0; i < I; i++) a += xs[i] * w[i];
#endif
            y[(int64_t)s * O + o] = a;
        }
    }
}

/* ── Matmul INT8: y[S,O] = x[S,I] @ W_q8^T * scale ── */

static void matmul_q8(float *y, const float *x, const int8_t *q,
                      const float *scale, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float sc = scale[o];
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float a = 0;
            int i = 0;
#ifdef __AVX2__
            __m256 acc = _mm256_setzero_ps();
            for (; i + 8 <= I; i += 8) {
                __m256i wi = _mm256_cvtepi8_epi32(
                    _mm_loadl_epi64((const __m128i *)(w + i)));
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i),
                                      _mm256_cvtepi32_ps(wi), acc);
            }
            a = hsum256(acc);
#elif defined(__ARM_NEON)
            float32x4_t ac0 = vdupq_n_f32(0), ac1 = vdupq_n_f32(0);
            for (; i + 8 <= I; i += 8) {
                int16x8_t w16 = vmovl_s8(vld1_s8(w + i));
                ac0 = vfmaq_f32(ac0, vld1q_f32(xs + i),
                                vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))));
                ac1 = vfmaq_f32(ac1, vld1q_f32(xs + i + 4),
                                vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16))));
            }
            a = vaddvq_f32(vaddq_f32(ac0, ac1));
#endif
            for (; i < I; i++) a += xs[i] * (float)w[i];
            y[(int64_t)s * O + o] = a * sc;
        }
    }
}

/* ── Matmul INT4: y[S,O] = x[S,I] @ W_i4^T * scale ── */
/* INT4 packed: 2 values/byte, value = nibble - 8 (range [-8, 7]) */

static void matmul_i4(float *y, const float *x, const uint8_t *q4,
                      const float *scale, int S, int I, int O) {
    int rb = (I + 1) / 2;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q4 + (int64_t)o * rb;
        float sc = scale[o];
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float a = 0;
            int i = 0;
#ifdef __AVX2__
            const __m128i m4 = _mm_set1_epi8(0x0F);
            const __m256i b8 = _mm256_set1_epi32(8);
            __m256 acc = _mm256_setzero_ps();
            for (; i + 16 <= I; i += 16) {
                __m128i by = _mm_loadl_epi64((const __m128i *)(w + (i >> 1)));
                __m128i lo = _mm_and_si128(by, m4);
                __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
                __m128i nib = _mm_unpacklo_epi8(lo, hi);
                __m256 w0 = _mm256_cvtepi32_ps(
                    _mm256_sub_epi32(_mm256_cvtepu8_epi32(nib), b8));
                __m256 w1 = _mm256_cvtepi32_ps(
                    _mm256_sub_epi32(
                        _mm256_cvtepu8_epi32(_mm_srli_si128(nib, 8)), b8));
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i), w0, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i + 8), w1, acc);
            }
            a = hsum256(acc);
#elif defined(__ARM_NEON)
            const uint8x8_t m4n = vdup_n_u8(0x0F);
            const int8x8_t b8n = vdup_n_s8(8);
            float32x4_t ac0 = vdupq_n_f32(0), ac1 = vdupq_n_f32(0);
            for (; i + 16 <= I; i += 16) {
                uint8x8_t by = vld1_u8(w + (i >> 1));
                uint8x8x2_t z = vzip_u8(vand_u8(by, m4n), vshr_n_u8(by, 4));
                int16x8_t w0 = vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[0]), b8n));
                int16x8_t w1 = vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[1]), b8n));
                ac0 = vfmaq_f32(ac0, vld1q_f32(xs + i),
                                vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                ac1 = vfmaq_f32(ac1, vld1q_f32(xs + i + 4),
                                vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                ac0 = vfmaq_f32(ac0, vld1q_f32(xs + i + 8),
                                vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                ac1 = vfmaq_f32(ac1, vld1q_f32(xs + i + 12),
                                vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1))));
            }
            a = vaddvq_f32(vaddq_f32(ac0, ac1));
#endif
            for (; i + 1 < I; i += 2) {
                uint8_t byte = w[i >> 1];
                int lo_v = (int)(byte & 0xF) - 8;
                int hi_v = (int)(byte >> 4) - 8;
                a += xs[i] * (float)lo_v + xs[i + 1] * (float)hi_v;
            }
            if (i < I) {
                uint8_t byte = w[i >> 1];
                a += xs[i] * (float)((int)(byte & 0xF) - 8);
            }
            y[(int64_t)s * O + o] = a * sc;
        }
    }
}

/* ── Matmul INT4 Group-Scaled (gs64): one scale every 64 values ── */
/* scale layout: [O, n_groups] where n_groups = ceil(I / 64)
 * Packed data: identical to INT4 (2 nibbles/byte, value = nibble - 8) */

static void matmul_i4_gs(float *y, const float *x, const uint8_t *q4,
                         const float *scale, int S, int I, int O, int gs) {
    int rb = (I + 1) / 2;
    int n_groups = (I + gs - 1) / gs;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q4 + (int64_t)o * rb;
        const float *sc = scale + (int64_t)o * n_groups;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float a = 0;
            int i = 0;
            for (int g = 0; g < n_groups; g++) {
                float group_sc = sc[g];
                int g_start = g * gs;
                int g_end = g_start + gs;
                if (g_end > I) g_end = I;
                float ga = 0;
#ifdef __AVX2__
                int gi = g_start;
                __m256 acc = _mm256_setzero_ps();
                const __m128i m4 = _mm_set1_epi8(0x0F);
                const __m256i b8 = _mm256_set1_epi32(8);
                for (; gi + 16 <= g_end; gi += 16) {
                    __m128i by = _mm_loadl_epi64((const __m128i *)(w + (gi >> 1)));
                    __m128i lo = _mm_and_si128(by, m4);
                    __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
                    __m128i nib = _mm_unpacklo_epi8(lo, hi);
                    __m256 w0 = _mm256_cvtepi32_ps(
                        _mm256_sub_epi32(_mm256_cvtepu8_epi32(nib), b8));
                    __m256 w1 = _mm256_cvtepi32_ps(
                        _mm256_sub_epi32(
                            _mm256_cvtepu8_epi32(_mm_srli_si128(nib, 8)), b8));
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + gi), w0, acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + gi + 8), w1, acc);
                }
                ga = hsum256(acc);
                for (; gi < g_end; gi += 2) {
                    uint8_t byte = w[gi >> 1];
                    ga += xs[gi] * (float)((int)(byte & 0xF) - 8);
                    if (gi + 1 < g_end) ga += xs[gi+1] * (float)((int)(byte >> 4) - 8);
                }
#else
                for (int gi = g_start; gi + 1 < g_end; gi += 2) {
                    uint8_t byte = w[gi >> 1];
                    ga += xs[gi] * (float)((int)(byte & 0xF) - 8);
                    ga += xs[gi+1] * (float)((int)(byte >> 4) - 8);
                }
                if (g_end > g_start && (g_end - g_start) % 2 == 1) {
                    int gi = g_end - 1;
                    uint8_t byte = w[gi >> 1];
                    ga += xs[gi] * (float)((int)(byte & 0xF) - 8);
                }
#endif
                a += ga * group_sc;
            }
            i = I; (void)i;
            y[(int64_t)s * O + o] = a;
        }
    }
}

/* ── Matmul INT4 gs64, IDOT path: INT8-quantized activation × INT4 weight ──
 * Faster CPU kernel for the group-scaled INT4 experts. Instead of dequantizing
 * the weights to F32 and doing F32 FMA, it quantizes the activation to INT8 once
 * per group (shared across all output rows) and runs an *integer* dot with AVX2
 * (maddubs/madd), accumulating in int32, then scales by (act_scale·weight_scale).
 * INT4 weights (|w| ≤ 8) keep the int16 partials well within range. This is an
 * approximation of the exact F32 path (the activation is quantized to int8), so
 * it is opt-in via IDOT=1 and the F32 path stays the default oracle. On AVX2 it
 * roughly halves the expert matmul time (2× the MACs/instruction, no int→float). */
/* AVX-VNNI is available on most x86 CPUs since ~2019 (Intel Ice Lake / Alder
 * Lake+, AMD Zen4+). We compile a VNNI variant behind a function target attribute
 * and dispatch to it at runtime (__builtin_cpu_supports), so a single binary runs
 * everywhere and uses the faster `dpbusd` path only where the CPU supports it. */
#if defined(__AVX2__) && defined(__x86_64__) && \
    ((defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 12) || \
     (defined(__clang__) && __clang_major__ >= 14))
#define PICCHIO_HAVE_VNNI 1
#endif

static int q_idot_enabled = 0;
static int q_idot_vnni    = 0;

/* Per-row expert dot for one activation row already quantized to int8 (xq) with
 * per-group scales (ax). Writes yrow[o] = Σ_g (ax[g]·wscale[o,g])·⟨xq_g, w_g⟩. */
static void idot_rows_avx2(float *yrow, const int8_t *xq, const float *ax,
                           const uint8_t *q4, const float *scale,
                           int I, int O, int gs, int n_groups) {
    int rb = (I + 1) / 2;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q4 + (int64_t)o * rb;
        const float *sc = scale + (int64_t)o * n_groups;
#ifdef __AVX2__
        const __m256i ones = _mm256_set1_epi16(1);
        const __m128i m0f  = _mm_set1_epi8(0x0F);
        const __m256i c8   = _mm256_set1_epi8(8);
        __m256 facc = _mm256_setzero_ps();
        float  atail = 0.f;
        for (int g = 0; g < n_groups; g++) {
            int g0 = g * gs, g1 = g0 + gs; if (g1 > I) g1 = I;
            __m256i gi = _mm256_setzero_si256();
            int i = g0;
            for (; i + 32 <= g1; i += 32) {
                __m128i by  = _mm_loadu_si128((const __m128i *)(w + (i >> 1)));
                __m128i lo4 = _mm_and_si128(by, m0f);
                __m128i hi4 = _mm_and_si128(_mm_srli_epi16(by, 4), m0f);
                __m256i w8  = _mm256_set_m128i(_mm_unpackhi_epi8(lo4, hi4),
                                               _mm_unpacklo_epi8(lo4, hi4));
                w8 = _mm256_sub_epi8(w8, c8);
                __m256i y8  = _mm256_loadu_si256((const __m256i *)(xq + i));
                __m256i p   = _mm256_maddubs_epi16(_mm256_sign_epi8(w8, w8),
                                                   _mm256_sign_epi8(y8, w8));
                gi = _mm256_add_epi32(gi, _mm256_madd_epi16(p, ones));
            }
            float sc_g = ax[g] * sc[g];
            facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(gi), _mm256_set1_ps(sc_g), facc);
            int tail = 0;
            for (; i < g1; i += 2) {
                uint8_t by = w[i >> 1];
                tail += (int)xq[i] * ((int)(by & 0xF) - 8);
                if (i + 1 < g1) tail += (int)xq[i + 1] * ((int)(by >> 4) - 8);
            }
            if (tail) atail += (float)tail * sc_g;
        }
        yrow[o] = hsum256(facc) + atail;
#else
        float a = 0.f;
        for (int g = 0; g < n_groups; g++) {
            int g0 = g * gs, g1 = g0 + gs; if (g1 > I) g1 = I;
            int idot = 0;
            for (int i = g0; i < g1; i += 2) {
                uint8_t by = w[i >> 1];
                idot += (int)xq[i] * ((int)(by & 0xF) - 8);
                if (i + 1 < g1) idot += (int)xq[i + 1] * ((int)(by >> 4) - 8);
            }
            a += (float)idot * ax[g] * sc[g];
        }
        yrow[o] = a;
#endif
    }
}

#ifdef PICCHIO_HAVE_VNNI
/* Same as idot_rows_avx2 but one `dpbusd` (int8×int8→int32) replaces the
 * maddubs+madd pair — 2× the integer throughput on CPUs with AVX-VNNI. */
__attribute__((target("avx2,avxvnni")))
static void idot_rows_vnni(float *yrow, const int8_t *xq, const float *ax,
                           const uint8_t *q4, const float *scale,
                           int I, int O, int gs, int n_groups) {
    int rb = (I + 1) / 2;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q4 + (int64_t)o * rb;
        const float *sc = scale + (int64_t)o * n_groups;
        const __m128i m0f = _mm_set1_epi8(0x0F);
        const __m256i c8  = _mm256_set1_epi8(8);
        __m256 facc = _mm256_setzero_ps();
        float  atail = 0.f;
        for (int g = 0; g < n_groups; g++) {
            int g0 = g * gs, g1 = g0 + gs; if (g1 > I) g1 = I;
            __m256i gi = _mm256_setzero_si256();
            int i = g0;
            for (; i + 32 <= g1; i += 32) {
                __m128i by  = _mm_loadu_si128((const __m128i *)(w + (i >> 1)));
                __m128i lo4 = _mm_and_si128(by, m0f);
                __m128i hi4 = _mm_and_si128(_mm_srli_epi16(by, 4), m0f);
                __m256i w8  = _mm256_set_m128i(_mm_unpackhi_epi8(lo4, hi4),
                                               _mm_unpacklo_epi8(lo4, hi4));
                w8 = _mm256_sub_epi8(w8, c8);
                __m256i y8  = _mm256_loadu_si256((const __m256i *)(xq + i));
                gi = _mm256_dpbusd_avx_epi32(gi, _mm256_sign_epi8(w8, w8),
                                             _mm256_sign_epi8(y8, w8));
            }
            float sc_g = ax[g] * sc[g];
            facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(gi), _mm256_set1_ps(sc_g), facc);
            int tail = 0;
            for (; i < g1; i += 2) {
                uint8_t by = w[i >> 1];
                tail += (int)xq[i] * ((int)(by & 0xF) - 8);
                if (i + 1 < g1) tail += (int)xq[i + 1] * ((int)(by >> 4) - 8);
            }
            if (tail) atail += (float)tail * sc_g;
        }
        yrow[o] = hsum256(facc) + atail;
    }
}
#endif

static void matmul_i4_gs_idot(float *y, const float *x, const uint8_t *q4,
                              const float *scale, int S, int I, int O, int gs) {
    int n_groups = (I + gs - 1) / gs;
    int8_t *xq = (int8_t *)malloc((size_t)I);
    float  *ax = (float *)malloc((size_t)n_groups * sizeof(float));
    if (!xq || !ax) { free(xq); free(ax);
        matmul_i4_gs(y, x, q4, scale, S, I, O, gs); return; }

    for (int s = 0; s < S; s++) {
        const float *xs = x + (int64_t)s * I;
        /* Quantize the activation to int8, one scale per group. */
        for (int g = 0; g < n_groups; g++) {
            int g0 = g * gs, g1 = g0 + gs; if (g1 > I) g1 = I;
            float amax = 0;
            for (int i = g0; i < g1; i++) { float a = fabsf(xs[i]); if (a > amax) amax = a; }
            float axg = amax / 127.f; if (axg < 1e-12f) axg = 1e-12f;
            ax[g] = axg;
            float inv = 1.f / axg;
            for (int i = g0; i < g1; i++) {
                int v = (int)lrintf(xs[i] * inv);
                if (v > 127) v = 127; if (v < -128) v = -128;
                xq[i] = (int8_t)v;
            }
        }

        float *yrow = y + (int64_t)s * O;
#ifdef PICCHIO_HAVE_VNNI
        if (q_idot_vnni)
            idot_rows_vnni(yrow, xq, ax, q4, scale, I, O, gs, n_groups);
        else
#endif
            idot_rows_avx2(yrow, xq, ax, q4, scale, I, O, gs, n_groups);
    }
    free(xq); free(ax);
}

/* IDOT toggle (opt-in via IDOT=1): the integer expert kernel is approximate, so
 * the exact F32 path stays the default and the oracle/self-test reference. On
 * enable, detect AVX-VNNI once so the dispatcher picks the faster dpbusd path. */
static inline void quant_set_idot(int v) {
    q_idot_enabled = v;
#ifdef PICCHIO_HAVE_VNNI
    if (v) { __builtin_cpu_init(); q_idot_vnni = __builtin_cpu_supports("avxvnni"); }
#endif
}
static inline int quant_idot_vnni(void) { return q_idot_vnni; }

/* ── Matmul INT3 gs64: 3-bit weights, group scale (gs=64) ──
 * On-disk, per group of 64 values: a 16-byte low plane (the 2 low bits of each
 * code) + an 8-byte high plane (the top bit of each code) = 24 bytes = 3 bits
 * per value. Code c in [0,7] maps to value c-4 in [-4,3]. ~22% fewer expert
 * bytes than INT4 gs64 (3.5 vs 4.5 bits/weight incl. scale) → less disk
 * bandwidth and RAM at a small quality cost. Packing fixes gs=64. */
#define I3_GROUP  64
#define I3_GBYTES 24
static void matmul_i3_gs(float *y, const float *x, const uint8_t *q3,
                         const float *scale, int S, int I, int O, int gs) {
    (void)gs;
    int n_groups = (I + I3_GROUP - 1) / I3_GROUP;
    int64_t row_bytes = (int64_t)n_groups * I3_GBYTES;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *wr = q3 + (int64_t)o * row_bytes;
        const float *sc = scale + (int64_t)o * n_groups;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
#ifdef __AVX2__
            /* Decode each 64-value group from its bit-planes with SIMD (the plane
             * layout is chosen for exactly this), dequantize to int8 [-4,3], then
             * FMA in F32. One hsum per row (scale folded into a float vector). */
            const __m128i m3   = _mm_set1_epi8(3);
            const __m256i c4   = _mm256_set1_epi8(4);
            const __m256i shuf = _mm256_setr_epi8(0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
                                                  2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
            const __m256i bitp = _mm256_setr_epi8(1,2,4,8,16,32,64,(char)128,
                                                  1,2,4,8,16,32,64,(char)128,
                                                  1,2,4,8,16,32,64,(char)128,
                                                  1,2,4,8,16,32,64,(char)128);
            __m256 facc = _mm256_setzero_ps();
            float  atail = 0.f;
            for (int g = 0; g < n_groups; g++) {
                const uint8_t *lp = wr + (int64_t)g * I3_GBYTES;
                const uint8_t *hp = lp + 16;
                int base = g * I3_GROUP;
                int n = I - base; if (n > I3_GROUP) n = I3_GROUP;
                if (n < I3_GROUP) {                              /* partial tail: scalar */
                    float ga = 0.f;
                    for (int j = 0; j < n; j++) {
                        int l2 = (lp[j >> 2] >> ((j & 3) * 2)) & 3;
                        int h1 = (hp[j >> 3] >> (j & 7)) & 1;
                        ga += xs[base + j] * (float)((l2 | (h1 << 2)) - 4);
                    }
                    atail += ga * sc[g];
                    continue;
                }
                /* low plane (16 B): 4-way byte interleave → codes' low 2 bits */
                __m128i lo16 = _mm_loadu_si128((const __m128i *)lp);
                __m128i t0 = _mm_and_si128(lo16, m3);
                __m128i t1 = _mm_and_si128(_mm_srli_epi16(lo16, 2), m3);
                __m128i t2 = _mm_and_si128(_mm_srli_epi16(lo16, 4), m3);
                __m128i t3 = _mm_and_si128(_mm_srli_epi16(lo16, 6), m3);
                __m128i i01l = _mm_unpacklo_epi8(t0, t1), i01h = _mm_unpackhi_epi8(t0, t1);
                __m128i i23l = _mm_unpacklo_epi8(t2, t3), i23h = _mm_unpackhi_epi8(t2, t3);
                __m256i low_lo = _mm256_set_m128i(_mm_unpackhi_epi16(i01l, i23l),
                                                  _mm_unpacklo_epi16(i01l, i23l));  /* codes 0..31 */
                __m256i low_hi = _mm256_set_m128i(_mm_unpackhi_epi16(i01h, i23h),
                                                  _mm_unpacklo_epi16(i01h, i23h));  /* codes 32..63 */
                /* high plane (8 B): expand bits → 0/4 (top bit of each code) */
                uint32_t hlo, hhi; memcpy(&hlo, hp, 4); memcpy(&hhi, hp + 4, 4);
                __m256i e_lo = _mm256_shuffle_epi8(_mm256_set1_epi32((int)hlo), shuf);
                e_lo = _mm256_cmpeq_epi8(_mm256_and_si256(e_lo, bitp), bitp);
                __m256i e_hi = _mm256_shuffle_epi8(_mm256_set1_epi32((int)hhi), shuf);
                e_hi = _mm256_cmpeq_epi8(_mm256_and_si256(e_hi, bitp), bitp);
                __m256i val_lo = _mm256_sub_epi8(
                    _mm256_add_epi8(low_lo, _mm256_and_si256(e_lo, c4)), c4);  /* [-4,3] */
                __m256i val_hi = _mm256_sub_epi8(
                    _mm256_add_epi8(low_hi, _mm256_and_si256(e_hi, c4)), c4);
                /* dot with x (int8 → f32, 8 lanes at a time) */
                __m128i vL = _mm256_castsi256_si128(val_lo), vLh = _mm256_extracti128_si256(val_lo, 1);
                __m128i vH = _mm256_castsi256_si128(val_hi), vHh = _mm256_extracti128_si256(val_hi, 1);
                __m256 gv = _mm256_setzero_ps();
                gv = _mm256_fmadd_ps(_mm256_loadu_ps(xs+base+0),  _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vL)), gv);
                gv = _mm256_fmadd_ps(_mm256_loadu_ps(xs+base+8),  _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vL,8))), gv);
                gv = _mm256_fmadd_ps(_mm256_loadu_ps(xs+base+16), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vLh)), gv);
                gv = _mm256_fmadd_ps(_mm256_loadu_ps(xs+base+24), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vLh,8))), gv);
                gv = _mm256_fmadd_ps(_mm256_loadu_ps(xs+base+32), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vH)), gv);
                gv = _mm256_fmadd_ps(_mm256_loadu_ps(xs+base+40), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vH,8))), gv);
                gv = _mm256_fmadd_ps(_mm256_loadu_ps(xs+base+48), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vHh)), gv);
                gv = _mm256_fmadd_ps(_mm256_loadu_ps(xs+base+56), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vHh,8))), gv);
                facc = _mm256_fmadd_ps(gv, _mm256_set1_ps(sc[g]), facc);
            }
            y[(int64_t)s * O + o] = hsum256(facc) + atail;
#else
            float a = 0.f;
            for (int g = 0; g < n_groups; g++) {
                const uint8_t *lp = wr + (int64_t)g * I3_GBYTES;
                const uint8_t *hp = lp + 16;
                int base = g * I3_GROUP;
                int n = I - base; if (n > I3_GROUP) n = I3_GROUP;
                float ga = 0.f;
                for (int j = 0; j < n; j++) {
                    int low2  = (lp[j >> 2] >> ((j & 3) * 2)) & 3;
                    int high1 = (hp[j >> 3] >> (j & 7)) & 1;
                    ga += xs[base + j] * (float)((low2 | (high1 << 2)) - 4);
                }
                a += ga * sc[g];
            }
            y[(int64_t)s * O + o] = a;
#endif
        }
    }
}

/* Quantize F32 rows to INT3 gs64 (mirror of the Python converter, used by the
 * self-test). Writes packed planes into `q3` and one F32 scale per group. */
static void quantize_rows_i3_gs(const float *w, uint8_t *q3, float *scale, int O, int I) {
    int n_groups = (I + I3_GROUP - 1) / I3_GROUP;
    int64_t row_bytes = (int64_t)n_groups * I3_GBYTES;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *wr = w + (int64_t)o * I;
        uint8_t *qr = q3 + (int64_t)o * row_bytes;
        float *sc = scale + (int64_t)o * n_groups;
        for (int g = 0; g < n_groups; g++) {
            int base = g * I3_GROUP;
            int n = I - base; if (n > I3_GROUP) n = I3_GROUP;
            float amax = 0;
            for (int j = 0; j < n; j++) { float a = fabsf(wr[base + j]); if (a > amax) amax = a; }
            float s = amax / 3.5f; if (s < 1e-8f) s = 1e-8f;
            sc[g] = s;
            float inv = 1.f / s;
            uint8_t *lp = qr + (int64_t)g * I3_GBYTES;
            uint8_t *hp = lp + 16;
            memset(lp, 0, I3_GBYTES);
            for (int j = 0; j < n; j++) {
                int v = (int)lrintf(wr[base + j] * inv);
                if (v > 3) v = 3; if (v < -4) v = -4;
                int code = v + 4;
                lp[j >> 2] |= (uint8_t)((code & 3) << ((j & 3) * 2));
                hp[j >> 3] |= (uint8_t)(((code >> 2) & 1) << (j & 7));
            }
        }
    }
}

/* ── Dispatcher: picks the kernel based on the format ── */

static void matmul_qt(float *y, const float *x, QT *w, int S) {
    if (w->fmt == 0) { matmul_f32(y, x, w->qf, S, w->I, w->O); return; }
    if (w->fmt == 1) { matmul_q8(y, x, w->q8, w->s, S, w->I, w->O); return; }
    if (w->fmt == 2) {
        /* If block_size > 0, use group-scaled */
        if (w->block_size > 0) {
            if (q_idot_enabled)
                matmul_i4_gs_idot(y, x, w->q4, w->s, S, w->I, w->O, w->block_size);
            else
                matmul_i4_gs(y, x, w->q4, w->s, S, w->I, w->O, w->block_size);
        } else
            matmul_i4(y, x, w->q4, w->s, S, w->I, w->O);
        return;
    }
    if (w->fmt == 5) { matmul_i3_gs(y, x, w->q4, w->s, S, w->I, w->O, w->block_size); return; }
    /* fmt==3 MXFP4: TODO — for v1 we convert to INT4 at build time */
    fprintf(stderr, "matmul_qt: format %d not supported\n", w->fmt);
    exit(1);
}

/* ── Runtime quantization F32 → INT8 per-row ── */

static void quantize_rows_i8(const float *w, int8_t *q, float *scale,
                             int O, int I) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *wr = w + (int64_t)o * I;
        float amax = 0;
        for (int i = 0; i < I; i++) {
            float a = fabsf(wr[i]);
            if (a > amax) amax = a;
        }
        float s = amax / 127.f;
        if (s < 1e-8f) s = 1e-8f;
        scale[o] = s;
        int8_t *qr = q + (int64_t)o * I;
        for (int i = 0; i < I; i++) {
            int v = (int)lrintf(wr[i] / s);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            qr[i] = (int8_t)v;
        }
    }
}

/* ── Runtime quantization F32 → INT4 packed ── */

static void quantize_rows_i4(const float *w, uint8_t *q4, float *scale,
                             int O, int I) {
    int rb = (I + 1) / 2;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *wr = w + (int64_t)o * I;
        float amax = 0;
        for (int i = 0; i < I; i++) {
            float a = fabsf(wr[i]);
            if (a > amax) amax = a;
        }
        float s = amax / 7.f;
        if (s < 1e-8f) s = 1e-8f;
        scale[o] = s;
        uint8_t *qr = q4 + (int64_t)o * rb;
        for (int i = 0; i < I; i += 2) {
            int v0 = (int)lrintf(wr[i] / s);
            if (v0 > 7) v0 = 7; if (v0 < -8) v0 = -8;
            int v1 = 0;
            if (i + 1 < I) {
                v1 = (int)lrintf(wr[i + 1] / s);
                if (v1 > 7) v1 = 7; if (v1 < -8) v1 = -8;
            }
            qr[i >> 1] = (uint8_t)((v0 + 8) | ((v1 + 8) << 4));
        }
    }
}

/* ── RMSNorm ── */

static void rmsnorm(float *out, const float *x, const float *w,
                    int D, float eps) {
    double ms = 0;
    for (int i = 0; i < D; i++) ms += (double)x[i] * x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * w[i];
}

/* ── Softmax ── */

static void softmax(float *x, int n) {
    float m = -1e30f;
    for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ── Activations ── */

static inline float siluf(float x) { return x / (1.f + expf(-x)); }
static inline float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

#endif /* PICCHIO_QUANT_H */
