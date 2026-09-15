/* flat.h - aligned .picchioflat expert store (format version 2).
 *
 * The dense model remains in safetensors. Expert payloads are addressed by a
 * resident (layer,eid) index and read in one aligned operation. All functions
 * are static so this header keeps Picchio's single-translation-unit build.
 */
#ifndef PICCHIO_FLAT_H
#define PICCHIO_FLAT_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FLAT_MAGIC "PCHIOFL1"
#define FLAT_VERSION 2
#define FLAT_SUPER_SIZE 4096
#define FLAT_INDEX_ENTRY_SIZE 24

typedef struct {
    uint64_t offset;
    uint32_t len;
    uint32_t padded_len;
    uint64_t hash;
} FlatLoc;

typedef struct {
    int active;
    int verify_payload;
    uint32_t block_size;
    uint32_t hidden;
    uint32_t n_layers;
    uint32_t n_experts;
    uint32_t topk;
    uint32_t moe_inter;
    uint32_t n_entries;
    uint64_t file_size;
    FlatLoc *index;
    char path[768];
    uint64_t reads;
    uint64_t bytes;
    uint64_t direct_fallbacks;
    uint64_t corruptions;
#ifdef _WIN32
    HANDLE hFile;
#else
    int fd;
    int fd_direct;
#endif
} FlatStore;

/* Small dependency-free SHA-256, used once for the resident index and only on
 * expert payloads when FLAT_VERIFY=1. Verification is off on the hot path. */
typedef struct {
    uint32_t h[8];
    uint64_t bits;
    uint8_t block[64];
    uint32_t used;
} FlatSha256;

static uint32_t flat_rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

static void flat_sha256_transform(FlatSha256 *s, const uint8_t *p) {
    static const uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)p[i*4] << 24) | ((uint32_t)p[i*4+1] << 16) |
               ((uint32_t)p[i*4+2] << 8) | p[i*4+3];
    for (int i = 16; i < 64; i++) {
        uint32_t a = flat_rotr32(w[i-15],7) ^ flat_rotr32(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t b = flat_rotr32(w[i-2],17) ^ flat_rotr32(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + a + w[i-7] + b;
    }
    uint32_t a=s->h[0], b=s->h[1], c=s->h[2], d=s->h[3];
    uint32_t e=s->h[4], f=s->h[5], g=s->h[6], h=s->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t s1 = flat_rotr32(e,6) ^ flat_rotr32(e,11) ^ flat_rotr32(e,25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + s1 + ch + k[i] + w[i];
        uint32_t s0 = flat_rotr32(a,2) ^ flat_rotr32(a,13) ^ flat_rotr32(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = s0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s->h[0]+=a; s->h[1]+=b; s->h[2]+=c; s->h[3]+=d;
    s->h[4]+=e; s->h[5]+=f; s->h[6]+=g; s->h[7]+=h;
}

static void flat_sha256_init(FlatSha256 *s) {
    static const uint32_t iv[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };
    memcpy(s->h, iv, sizeof(iv));
    s->bits = 0; s->used = 0;
}

static void flat_sha256_update(FlatSha256 *s, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    s->bits += (uint64_t)n * 8;
    while (n) {
        size_t take = 64 - s->used;
        if (take > n) take = n;
        memcpy(s->block + s->used, p, take);
        s->used += (uint32_t)take; p += take; n -= take;
        if (s->used == 64) {
            flat_sha256_transform(s, s->block);
            s->used = 0;
        }
    }
}

static void flat_sha256_final(FlatSha256 *s, uint8_t out[32]) {
    s->block[s->used++] = 0x80;
    if (s->used > 56) {
        memset(s->block + s->used, 0, 64 - s->used);
        flat_sha256_transform(s, s->block);
        s->used = 0;
    }
    memset(s->block + s->used, 0, 56 - s->used);
    for (int i = 0; i < 8; i++)
        s->block[63-i] = (uint8_t)(s->bits >> (i*8));
    flat_sha256_transform(s, s->block);
    for (int i = 0; i < 8; i++) {
        out[i*4]   = (uint8_t)(s->h[i] >> 24);
        out[i*4+1] = (uint8_t)(s->h[i] >> 16);
        out[i*4+2] = (uint8_t)(s->h[i] >> 8);
        out[i*4+3] = (uint8_t)s->h[i];
    }
}

static void flat_sha256(const void *data, size_t n, uint8_t out[32]) {
    FlatSha256 s; flat_sha256_init(&s); flat_sha256_update(&s, data, n);
    flat_sha256_final(&s, out);
}

static uint32_t flat_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t flat_u64(const uint8_t *p) {
    return (uint64_t)flat_u32(p) | ((uint64_t)flat_u32(p + 4) << 32);
}
static uint64_t flat_hash64(const void *data, size_t n) {
    uint8_t h[32]; flat_sha256(data, n, h); return flat_u64(h);
}

static void *flat_aligned_alloc(size_t n) {
#ifdef _WIN32
    return VirtualAlloc(NULL, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    void *p = NULL;
    if (posix_memalign(&p, FLAT_SUPER_SIZE, n) != 0) return NULL;
    return p;
#endif
}
static void flat_aligned_free(void *p) {
    if (!p) return;
#ifdef _WIN32
    VirtualFree(p, 0, MEM_RELEASE);
#else
    free(p);
#endif
}

static int flat_file_seek(FILE *f, uint64_t off) {
#ifdef _WIN32
    return _fseeki64(f, (__int64)off, SEEK_SET);
#else
    return fseeko(f, (off_t)off, SEEK_SET);
#endif
}

static int flat_open(FlatStore *fs, const char *path, uint32_t hidden,
                     uint32_t n_layers, uint32_t n_experts, uint32_t topk,
                     uint32_t moe_inter, int verify_payload) {
    memset(fs, 0, sizeof(*fs));
#ifndef _WIN32
    fs->fd = -1; fs->fd_direct = -1;
#else
    fs->hFile = INVALID_HANDLE_VALUE;
#endif
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    uint8_t sb[FLAT_SUPER_SIZE];
    if (fread(sb, 1, sizeof(sb), f) != sizeof(sb) ||
        memcmp(sb, FLAT_MAGIC, 8) != 0) {
        fprintf(stderr, "flat: invalid superblock: %s\n", path);
        fclose(f); return 0;
    }
    uint32_t version = flat_u32(sb + 8);
    fs->block_size = flat_u32(sb + 12);
    fs->hidden = flat_u32(sb + 16);
    fs->n_layers = flat_u32(sb + 20);
    fs->n_experts = flat_u32(sb + 24);
    fs->topk = flat_u32(sb + 28);
    fs->moe_inter = flat_u32(sb + 32);
    fs->n_entries = flat_u32(sb + 36);
    uint64_t index_offset = flat_u64(sb + 40);
    uint64_t index_len = flat_u64(sb + 48);
    uint64_t data_offset = flat_u64(sb + 56);
    const uint8_t *index_hash = sb + 64;
    if (version != FLAT_VERSION || fs->block_size != FLAT_SUPER_SIZE ||
        fs->hidden != hidden || fs->n_layers != n_layers ||
        fs->n_experts != n_experts || fs->topk != topk ||
        fs->moe_inter != moe_inter || fs->n_entries == 0 ||
        fs->n_entries > n_layers * n_experts ||
        index_len != (uint64_t)fs->n_entries * FLAT_INDEX_ENTRY_SIZE ||
        index_offset % fs->block_size || data_offset % fs->block_size) {
        fprintf(stderr, "flat: format/config mismatch: %s\n", path);
        fclose(f); return 0;
    }
    if (flat_file_seek(f, 0) != 0 || flat_file_seek(f, index_offset) != 0) {
        fclose(f); return 0;
    }
    uint8_t *raw = (uint8_t *)malloc((size_t)index_len);
    if (!raw || fread(raw, 1, (size_t)index_len, f) != index_len) {
        free(raw); fclose(f); return 0;
    }
    uint8_t actual_hash[32];
    flat_sha256(raw, (size_t)index_len, actual_hash);
    if (memcmp(actual_hash, index_hash, 32) != 0) {
        fprintf(stderr, "flat: index SHA-256 mismatch: %s\n", path);
        free(raw); fclose(f); return 0;
    }
    if (flat_file_seek(f, 0) != 0) { free(raw); fclose(f); return 0; }
#ifdef _WIN32
    _fseeki64(f, 0, SEEK_END); fs->file_size = (uint64_t)_ftelli64(f);
#else
    fseeko(f, 0, SEEK_END); fs->file_size = (uint64_t)ftello(f);
#endif
    fs->index = (FlatLoc *)calloc(fs->n_entries, sizeof(FlatLoc));
    if (!fs->index) { free(raw); fclose(f); return 0; }
    for (uint32_t i = 0; i < fs->n_entries; i++) {
        const uint8_t *p = raw + (size_t)i * FLAT_INDEX_ENTRY_SIZE;
        FlatLoc *loc = &fs->index[i];
        loc->offset = flat_u64(p);
        loc->len = flat_u32(p + 8);
        loc->padded_len = flat_u32(p + 12);
        loc->hash = flat_u64(p + 16);
        if (loc->offset < data_offset || loc->offset % fs->block_size ||
            loc->len == 0 || loc->len > loc->padded_len ||
            loc->padded_len % fs->block_size ||
            loc->offset + loc->padded_len > fs->file_size) {
            fprintf(stderr, "flat: invalid index entry %u: %s\n", i, path);
            free(raw); free(fs->index); fs->index = NULL; fclose(f); return 0;
        }
    }
    free(raw); fclose(f);
    snprintf(fs->path, sizeof(fs->path), "%s", path);
#ifdef _WIN32
    fs->hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, NULL);
    if (fs->hFile == INVALID_HANDLE_VALUE) { free(fs->index); fs->index=NULL; return 0; }
#else
    fs->fd = open(path, O_RDONLY);
    if (fs->fd < 0) { free(fs->index); fs->index=NULL; return 0; }
#endif
    fs->verify_payload = verify_payload;
    fs->active = 1;
    return 1;
}

static FlatLoc *flat_loc(FlatStore *fs, int layer, int eid) {
    if (!fs->active || layer < 0 || eid < 0 ||
        layer >= (int)fs->n_layers || eid >= (int)fs->n_experts) return NULL;
    uint32_t i = (uint32_t)layer * fs->n_experts + (uint32_t)eid;
    return i < fs->n_entries ? &fs->index[i] : NULL;
}

#ifdef _WIN32
static __thread HANDLE flat_tls_h = NULL;
static __thread HANDLE flat_tls_hd = NULL;
static __thread const FlatStore *flat_tls_owner = NULL;
static HANDLE flat_win_handle(FlatStore *fs, int direct) {
    if (flat_tls_owner != fs) {
        flat_tls_owner = fs; flat_tls_h = NULL; flat_tls_hd = NULL;
    }
    HANDLE *slot = direct ? &flat_tls_hd : &flat_tls_h;
    if (!*slot) {
        DWORD flags = FILE_FLAG_RANDOM_ACCESS |
                      (direct ? FILE_FLAG_NO_BUFFERING : 0);
        HANDLE h = CreateFileA(fs->path, GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, flags, NULL);
        *slot = h == INVALID_HANDLE_VALUE ? NULL : h;
    }
    return *slot ? *slot : (direct ? NULL : fs->hFile);
}
#else
static int flat_posix_direct(FlatStore *fs) {
    if (fs->fd_direct >= 0) return fs->fd_direct;
    int fd = open(fs->path, O_RDONLY | O_DIRECT);
    fs->fd_direct = fd >= 0 ? fd : fs->fd;
    return fs->fd_direct;
}
#endif

/* dst must be aligned and have loc->padded_len bytes. */
static int flat_read(FlatStore *fs, FlatLoc *loc, void *dst, int direct) {
    int64_t got = -1;
    if (direct) {
#ifdef _WIN32
        HANDLE h = flat_win_handle(fs, 1);
        if (h) got = st_pread_full(h, dst, loc->padded_len, loc->offset);
#else
        got = st_pread_full(flat_posix_direct(fs), dst, loc->padded_len,
                            loc->offset);
#endif
        if (got != loc->padded_len)
            __atomic_add_fetch(&fs->direct_fallbacks, 1, __ATOMIC_RELAXED);
    }
    if (got != loc->padded_len) {
#ifdef _WIN32
        got = st_pread_full(flat_win_handle(fs, 0), dst, loc->padded_len,
                            loc->offset);
#else
        got = st_pread_full(fs->fd, dst, loc->padded_len, loc->offset);
#endif
    }
    if (got != loc->padded_len) return 0;
    if (fs->verify_payload && flat_hash64(dst, loc->len) != loc->hash) {
        __atomic_add_fetch(&fs->corruptions, 1, __ATOMIC_RELAXED);
        return 0;
    }
    __atomic_add_fetch(&fs->reads, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&fs->bytes, loc->padded_len, __ATOMIC_RELAXED);
    return 1;
}

static void flat_close(FlatStore *fs) {
    if (!fs->active) return;
#ifdef _WIN32
    if (fs->hFile != INVALID_HANDLE_VALUE) CloseHandle(fs->hFile);
    fs->hFile = INVALID_HANDLE_VALUE;
#else
    if (fs->fd_direct >= 0 && fs->fd_direct != fs->fd) close(fs->fd_direct);
    if (fs->fd >= 0) close(fs->fd);
    fs->fd = fs->fd_direct = -1;
#endif
    free(fs->index); fs->index = NULL; fs->active = 0;
}

#endif
