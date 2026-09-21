/* gpu_router_native.h - dependency-free Windows CUDA Driver router backend.
 *
 * Header-only by design: picchio.exe dynamically resolves nvcuda.dll and embeds
 * its sm_75 PTX. No CUDA headers, import library, runtime DLL, or toolkit is
 * required on the target machine; only the installed NVIDIA display driver.
 */
#ifndef PICCHIO_GPU_ROUTER_NATIVE_H
#define PICCHIO_GPU_ROUTER_NATIVE_H

#ifdef _WIN32

typedef int PgrCUresult;
typedef int PgrCUdevice;
typedef unsigned long long PgrCUdeviceptr;
typedef void *PgrCUcontext;
typedef void *PgrCUmodule;
typedef void *PgrCUfunction;
typedef void *PgrCUstream;

typedef PgrCUresult (WINAPI *pgr_init_fn)(unsigned int);
typedef PgrCUresult (WINAPI *pgr_device_get_fn)(PgrCUdevice *, int);
typedef PgrCUresult (WINAPI *pgr_device_name_fn)(char *, int, PgrCUdevice);
typedef PgrCUresult (WINAPI *pgr_ctx_create_fn)(PgrCUcontext *, unsigned int, PgrCUdevice);
typedef PgrCUresult (WINAPI *pgr_ctx_destroy_fn)(PgrCUcontext);
typedef PgrCUresult (WINAPI *pgr_mem_alloc_fn)(PgrCUdeviceptr *, size_t);
typedef PgrCUresult (WINAPI *pgr_mem_free_fn)(PgrCUdeviceptr);
typedef PgrCUresult (WINAPI *pgr_htod_fn)(PgrCUdeviceptr, const void *, size_t);
typedef PgrCUresult (WINAPI *pgr_dtoh_fn)(void *, PgrCUdeviceptr, size_t);
typedef PgrCUresult (WINAPI *pgr_module_load_fn)(PgrCUmodule *, const void *,
                                                 unsigned int, int *, void **);
typedef PgrCUresult (WINAPI *pgr_module_unload_fn)(PgrCUmodule);
typedef PgrCUresult (WINAPI *pgr_function_fn)(PgrCUfunction *, PgrCUmodule, const char *);
typedef PgrCUresult (WINAPI *pgr_launch_fn)(PgrCUfunction,
    unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int, PgrCUstream, void **, void **);
typedef PgrCUresult (WINAPI *pgr_sync_fn)(void);
typedef PgrCUresult (WINAPI *pgr_error_name_fn)(PgrCUresult, const char **);

typedef struct {
    const void *key;
    PgrCUdeviceptr w, bias;
    int E, D;
} PgrRouter;

typedef struct {
    const void *key;
    PgrCUdeviceptr w;
    int O, I;
} PgrDense;

static struct {
    HMODULE dll;
    PgrCUcontext ctx;
    PgrCUmodule module;
    PgrCUfunction kernel, dense_kernel;
    PgrCUdeviceptr dx, dscores, dy;
    size_t dx_cap, dscores_cap, dy_cap;
    PgrRouter routers[128];
    int nrouters;
    PgrDense dense[192];
    int ndense;
    uint64_t dense_bytes;
    pgr_ctx_destroy_fn ctx_destroy;
    pgr_mem_alloc_fn mem_alloc;
    pgr_mem_free_fn mem_free;
    pgr_htod_fn htod;
    pgr_dtoh_fn dtoh;
    pgr_module_unload_fn module_unload;
    pgr_launch_fn launch;
    pgr_sync_fn sync;
    pgr_error_name_fn error_name;
} pgr;

/* Generated from gpu_router_kernel.cu with CUDA 12.6 for sm_75. PTX is JITed
 * by the installed NVIDIA driver when the feature is first enabled. */
static const char pgr_ptx[] =
".version 8.5\n"
".target sm_75\n"
".address_size 64\n"
".visible .entry picchio_router_f32(\n"
" .param .u64 p0, .param .u64 p1, .param .u64 p2, .param .u64 p3,\n"
" .param .u32 p4, .param .u32 p5)\n"
"{\n"
" .reg .pred %p<9>; .reg .f32 %f<19>; .reg .b32 %r<19>; .reg .b64 %rd<22>;\n"
" .shared .align 4 .b8 sm[1024];\n"
" ld.param.u64 %rd5,[p0]; ld.param.u64 %rd6,[p1];\n"
" ld.param.u64 %rd7,[p2]; ld.param.u64 %rd8,[p3];\n"
" ld.param.u32 %r11,[p4]; ld.param.u32 %r10,[p5];\n"
" mov.u32 %r1,%ctaid.x; setp.ge.s32 %p1,%r1,%r11; @%p1 bra DONE;\n"
" cvt.s64.s32 %rd1,%r1; mov.u32 %r2,%tid.x;\n"
" setp.lt.s32 %p2,%r2,%r10; @%p2 bra LOOP_SETUP; bra ZERO;\n"
"LOOP_SETUP: mov.u32 %r3,%ntid.x; cvta.to.global.u64 %rd2,%rd5;\n"
" cvta.to.global.u64 %rd3,%rd7; cvt.s64.s32 %rd9,%r10;\n"
" mul.lo.s64 %rd4,%rd9,%rd1; mov.f32 %f17,0f00000000; mov.u32 %r17,%r2;\n"
"LOOP: cvt.s64.s32 %rd10,%r17; mul.wide.s32 %rd11,%r17,4;\n"
" add.s64 %rd12,%rd3,%rd11; add.s64 %rd13,%rd4,%rd10;\n"
" shl.b64 %rd14,%rd13,2; add.s64 %rd15,%rd2,%rd14;\n"
" ld.global.nc.f32 %f9,[%rd15]; ld.global.nc.f32 %f10,[%rd12];\n"
" fma.rn.f32 %f17,%f10,%f9,%f17; add.s32 %r17,%r17,%r3;\n"
" setp.lt.s32 %p3,%r17,%r10; @%p3 bra LOOP; bra STORE;\n"
"ZERO: mov.f32 %f17,0f00000000;\n"
"STORE: shl.b32 %r12,%r2,2; mov.u32 %r13,sm; add.s32 %r6,%r13,%r12;\n"
" st.shared.f32 [%r6],%f17; bar.sync 0; mov.u32 %r14,%ntid.x;\n"
" shr.u32 %r18,%r14,1; setp.eq.s32 %p4,%r18,0; @%p4 bra REDUCE_DONE;\n"
"REDUCE: setp.ge.u32 %p5,%r2,%r18; @%p5 bra REDUCE_SYNC;\n"
" shl.b32 %r15,%r18,2; add.s32 %r16,%r6,%r15;\n"
" ld.shared.f32 %f11,[%r6]; ld.shared.f32 %f12,[%r16];\n"
" add.f32 %f13,%f12,%f11; st.shared.f32 [%r6],%f13;\n"
"REDUCE_SYNC: bar.sync 0; shr.u32 %r18,%r18,1;\n"
" setp.ne.s32 %p6,%r18,0; @%p6 bra REDUCE;\n"
"REDUCE_DONE: setp.ne.s32 %p7,%r2,0; @%p7 bra DONE;\n"
" ld.shared.f32 %f4,[sm]; setp.eq.s64 %p8,%rd6,0;\n"
" mov.f32 %f18,0f00000000; @%p8 bra WRITE;\n"
" cvta.to.global.u64 %rd16,%rd6; shl.b64 %rd17,%rd1,2;\n"
" add.s64 %rd18,%rd16,%rd17; ld.global.nc.f32 %f18,[%rd18];\n"
"WRITE: cvta.to.global.u64 %rd19,%rd8; shl.b64 %rd20,%rd1,2;\n"
" add.s64 %rd21,%rd19,%rd20; add.f32 %f15,%f4,%f18;\n"
" st.global.f32 [%rd21],%f15;\n"
"DONE: ret; }\n";

/* Regenerated from gpu_router_kernel.cu.  Kept in a separate include so normal
 * C builds need no CUDA Toolkit; generate_native_ptx.py is only a maintainer
 * helper. */
static const char pgr_native_ptx[] =
#include "gpu_native_ptx.inc"
;

static int pgr_ok(PgrCUresult rc, const char *what) {
    const char *name = NULL;
    if (rc == 0) return 1;
    if (pgr.error_name) pgr.error_name(rc, &name);
    fprintf(stderr, "[gpu-router] %s failed: %s (%d)\n", what,
            name ? name : "CUDA error", rc);
    return 0;
}

static FARPROC pgr_symbol(const char *name) {
    FARPROC p = GetProcAddress(pgr.dll, name);
    if (!p) fprintf(stderr, "[gpu-router] nvcuda.dll missing %s\n", name);
    return p;
}

static void pgr_native_shutdown(void) {
    int i;
    if (!pgr.dll) return;
    for (i = 0; i < pgr.nrouters; i++) {
        if (pgr.routers[i].w) pgr.mem_free(pgr.routers[i].w);
        if (pgr.routers[i].bias) pgr.mem_free(pgr.routers[i].bias);
    }
    for (i = 0; i < pgr.ndense; i++)
        if (pgr.dense[i].w) pgr.mem_free(pgr.dense[i].w);
    if (pgr.dx) pgr.mem_free(pgr.dx);
    if (pgr.dscores) pgr.mem_free(pgr.dscores);
    if (pgr.dy) pgr.mem_free(pgr.dy);
    if (pgr.module) pgr.module_unload(pgr.module);
    if (pgr.ctx) pgr.ctx_destroy(pgr.ctx);
    FreeLibrary(pgr.dll);
    memset(&pgr, 0, sizeof(pgr));
}

static int pgr_native_init(void) {
    pgr_init_fn init;
    pgr_device_get_fn device_get;
    pgr_device_name_fn device_name;
    pgr_ctx_create_fn ctx_create;
    pgr_module_load_fn module_load;
    pgr_function_fn get_function;
    PgrCUdevice dev;
    char name[128] = "CUDA device";
    memset(&pgr, 0, sizeof(pgr));
    pgr.dll = LoadLibraryW(L"nvcuda.dll");
    if (!pgr.dll) return 0;
#define PGR_LOAD(var, type, symbol) do { \
    var = (type)(void *)pgr_symbol(symbol); if (!(var)) goto fail; \
} while (0)
    PGR_LOAD(init, pgr_init_fn, "cuInit");
    PGR_LOAD(device_get, pgr_device_get_fn, "cuDeviceGet");
    PGR_LOAD(device_name, pgr_device_name_fn, "cuDeviceGetName");
    PGR_LOAD(ctx_create, pgr_ctx_create_fn, "cuCtxCreate_v2");
    PGR_LOAD(pgr.ctx_destroy, pgr_ctx_destroy_fn, "cuCtxDestroy_v2");
    PGR_LOAD(pgr.mem_alloc, pgr_mem_alloc_fn, "cuMemAlloc_v2");
    PGR_LOAD(pgr.mem_free, pgr_mem_free_fn, "cuMemFree_v2");
    PGR_LOAD(pgr.htod, pgr_htod_fn, "cuMemcpyHtoD_v2");
    PGR_LOAD(pgr.dtoh, pgr_dtoh_fn, "cuMemcpyDtoH_v2");
    PGR_LOAD(module_load, pgr_module_load_fn, "cuModuleLoadDataEx");
    PGR_LOAD(pgr.module_unload, pgr_module_unload_fn, "cuModuleUnload");
    PGR_LOAD(get_function, pgr_function_fn, "cuModuleGetFunction");
    PGR_LOAD(pgr.launch, pgr_launch_fn, "cuLaunchKernel");
    PGR_LOAD(pgr.sync, pgr_sync_fn, "cuCtxSynchronize");
    pgr.error_name = (pgr_error_name_fn)(void *)GetProcAddress(pgr.dll, "cuGetErrorName");
#undef PGR_LOAD
    if (!pgr_ok(init(0), "cuInit") ||
        !pgr_ok(device_get(&dev, 0), "cuDeviceGet") ||
        !pgr_ok(ctx_create(&pgr.ctx, 0, dev), "cuCtxCreate") ||
        !pgr_ok(module_load(&pgr.module, pgr_native_ptx, 0, NULL, NULL), "cuModuleLoadDataEx") ||
        !pgr_ok(get_function(&pgr.kernel, pgr.module, "picchio_router_f32"),
                "cuModuleGetFunction router") ||
        !pgr_ok(get_function(&pgr.dense_kernel, pgr.module, "picchio_dense_f16"),
                "cuModuleGetFunction dense")) goto fail;
    device_name(name, (int)sizeof(name), dev);
    fprintf(stderr, "[gpu-router] native CUDA Driver backend: %s\n", name);
    return 1;
fail:
    pgr_native_shutdown();
    return 0;
}

static int pgr_ensure(PgrCUdeviceptr *p, size_t *cap, size_t need) {
    if (*cap >= need) return 1;
    if (*p) pgr.mem_free(*p);
    *p = 0; *cap = 0;
    if (!pgr_ok(pgr.mem_alloc(p, need), "cuMemAlloc")) return 0;
    *cap = need;
    return 1;
}

static PgrRouter *pgr_find(const void *key) {
    int i;
    for (i = 0; i < pgr.nrouters; i++)
        if (pgr.routers[i].key == key) return &pgr.routers[i];
    return NULL;
}

static int pgr_native_upload(const float *w, const float *bias,
                             int E, int D, const void *key) {
    PgrRouter *r;
    size_t wb = (size_t)E * D * sizeof(float), bb = (size_t)E * sizeof(float);
    if (!pgr.dll || !w || !key || E <= 0 || D <= 0) return -1;
    if (pgr_find(key)) return 0;
    if (pgr.nrouters >= (int)(sizeof(pgr.routers) / sizeof(pgr.routers[0]))) return -1;
    r = &pgr.routers[pgr.nrouters];
    memset(r, 0, sizeof(*r));
    if (!pgr_ok(pgr.mem_alloc(&r->w, wb), "cuMemAlloc router") ||
        (bias && !pgr_ok(pgr.mem_alloc(&r->bias, bb), "cuMemAlloc bias")) ||
        !pgr_ok(pgr.htod(r->w, w, wb), "cuMemcpyHtoD router") ||
        (bias && !pgr_ok(pgr.htod(r->bias, bias, bb), "cuMemcpyHtoD bias"))) {
        if (r->w) pgr.mem_free(r->w);
        if (r->bias) pgr.mem_free(r->bias);
        memset(r, 0, sizeof(*r));
        return -1;
    }
    r->key = key; r->E = E; r->D = D;
    pgr.nrouters++;
    return 0;
}

static int pgr_native_scores(float *scores, const float *x,
                             const float *w, const float *bias,
                             int E, int D, const void *key) {
    PgrRouter *r;
    void *args[6];
    if (pgr_native_upload(w, bias, E, D, key) != 0) return -1;
    r = pgr_find(key);
    if (!r || r->E != E || r->D != D ||
        !pgr_ensure(&pgr.dx, &pgr.dx_cap, (size_t)D * sizeof(float)) ||
        !pgr_ensure(&pgr.dscores, &pgr.dscores_cap, (size_t)E * sizeof(float)) ||
        !pgr_ok(pgr.htod(pgr.dx, x, (size_t)D * sizeof(float)), "cuMemcpyHtoD x"))
        return -1;
    args[0] = &r->w; args[1] = &r->bias; args[2] = &pgr.dx;
    args[3] = &pgr.dscores; args[4] = &E; args[5] = &D;
    if (!pgr_ok(pgr.launch(pgr.kernel, (unsigned)E, 1, 1, 256, 1, 1,
                           0, NULL, args, NULL), "cuLaunchKernel") ||
        !pgr_ok(pgr.sync(), "cuCtxSynchronize") ||
        !pgr_ok(pgr.dtoh(scores, pgr.dscores, (size_t)E * sizeof(float)),
                "cuMemcpyDtoH scores")) return -1;
    return 0;
}

/* Round-to-nearest-even IEEE-754 binary32 -> binary16.  Dense attention weights
 * are finite in supported checkpoints, but infinities/NaNs are preserved for a
 * useful failure signal during validation. */
static uint16_t pgr_f32_to_f16(float value) {
    uint32_t x;
    uint32_t sign, mantissa;
    int exponent;
    memcpy(&x, &value, sizeof(x));
    sign = (x >> 16) & 0x8000u;
    exponent = (int)((x >> 23) & 0xffu) - 127 + 15;
    mantissa = x & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return (uint16_t)sign;
        mantissa = (mantissa | 0x800000u) >> (1 - exponent);
        if (mantissa & 0x1000u) mantissa += 0x2000u;
        return (uint16_t)(sign | (mantissa >> 13));
    }
    if (exponent >= 31) {
        if (((x >> 23) & 0xffu) == 0xffu && mantissa)
            return (uint16_t)(sign | 0x7c00u | (mantissa >> 13) | 1u);
        return (uint16_t)(sign | 0x7c00u);
    }
    if (mantissa & 0x1000u) {
        mantissa += 0x2000u;
        if (mantissa & 0x800000u) {
            mantissa = 0;
            exponent++;
            if (exponent >= 31) return (uint16_t)(sign | 0x7c00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)exponent << 10) | (mantissa >> 13));
}

static PgrDense *pgr_dense_find(const void *key) {
    int i;
    for (i = 0; i < pgr.ndense; i++)
        if (pgr.dense[i].key == key) return &pgr.dense[i];
    return NULL;
}

static void pgr_native_dense_clear(void) {
    int i;
    if (!pgr.dll) return;
    for (i = 0; i < pgr.ndense; i++)
        if (pgr.dense[i].w) pgr.mem_free(pgr.dense[i].w);
    memset(pgr.dense, 0, sizeof(pgr.dense));
    pgr.ndense = 0;
    pgr.dense_bytes = 0;
}

static int pgr_native_dense_upload(const float *w, int O, int I,
                                   const void *key) {
    PgrDense *d;
    uint16_t *half;
    size_t i, n, bytes;
    if (!pgr.dll || !key || O <= 0 || I <= 0) return -1;
    if (pgr_dense_find(key)) return 0;
    if (!w) return -1; /* host storage may be released after a successful upload */
    if (pgr.ndense >= (int)(sizeof(pgr.dense) / sizeof(pgr.dense[0]))) return -1;
    n = (size_t)O * (size_t)I;
    if (n > SIZE_MAX / sizeof(uint16_t)) return -1;
    bytes = n * sizeof(uint16_t);
    half = (uint16_t *)malloc(bytes);
    if (!half) return -1;
    for (i = 0; i < n; i++) half[i] = pgr_f32_to_f16(w[i]);
    d = &pgr.dense[pgr.ndense];
    memset(d, 0, sizeof(*d));
    if (!pgr_ok(pgr.mem_alloc(&d->w, bytes), "cuMemAlloc dense") ||
        !pgr_ok(pgr.htod(d->w, half, bytes), "cuMemcpyHtoD dense")) {
        if (d->w) pgr.mem_free(d->w);
        memset(d, 0, sizeof(*d));
        free(half);
        return -1;
    }
    free(half);
    d->key = key; d->O = O; d->I = I;
    pgr.ndense++;
    pgr.dense_bytes += bytes;
    return 0;
}

static int pgr_native_dense_matmul(float *y, const float *x,
                                   const float *w, int O, int I,
                                   const void *key) {
    PgrDense *d;
    void *args[5];
    if (pgr_native_dense_upload(w, O, I, key) != 0) return -1;
    d = pgr_dense_find(key);
    if (!d || d->O != O || d->I != I ||
        !pgr_ensure(&pgr.dx, &pgr.dx_cap, (size_t)I * sizeof(float)) ||
        !pgr_ensure(&pgr.dy, &pgr.dy_cap, (size_t)O * sizeof(float)) ||
        !pgr_ok(pgr.htod(pgr.dx, x, (size_t)I * sizeof(float)), "cuMemcpyHtoD dense x"))
        return -1;
    args[0] = &d->w; args[1] = &pgr.dx; args[2] = &pgr.dy;
    args[3] = &O; args[4] = &I;
    if (!pgr_ok(pgr.launch(pgr.dense_kernel, (unsigned)O, 1, 1, 256, 1, 1,
                           0, NULL, args, NULL), "cuLaunchKernel dense") ||
        !pgr_ok(pgr.sync(), "cuCtxSynchronize dense") ||
        !pgr_ok(pgr.dtoh(y, pgr.dy, (size_t)O * sizeof(float)),
                "cuMemcpyDtoH dense y")) return -1;
    return 0;
}

static uint64_t pgr_native_dense_bytes(void) { return pgr.dense_bytes; }

#else
static int pgr_native_init(void) { return 0; }
static void pgr_native_shutdown(void) { }
static int pgr_native_upload(const float *w, const float *b, int E, int D, const void *k) {
    (void)w; (void)b; (void)E; (void)D; (void)k; return -1;
}
static int pgr_native_scores(float *s, const float *x, const float *w, const float *b,
                             int E, int D, const void *k) {
    (void)s; (void)x; (void)w; (void)b; (void)E; (void)D; (void)k; return -1;
}
static void pgr_native_dense_clear(void) { }
static int pgr_native_dense_upload(const float *w, int O, int I, const void *key) {
    (void)w; (void)O; (void)I; (void)key; return -1;
}
static int pgr_native_dense_matmul(float *y, const float *x, const float *w,
                                   int O, int I, const void *key) {
    (void)y; (void)x; (void)w; (void)O; (void)I; (void)key; return -1;
}
static uint64_t pgr_native_dense_bytes(void) { return 0; }
#endif
#endif
