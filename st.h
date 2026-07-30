/* st.h — Reader safetensors per Picchio.
 *
 * Il formato safetensors è semplice:
 *   [8 byte LE]  lunghezza header JSON
 *   [header]     JSON con metadati tensori: {nome: {dtype, shape, offsets}}
 *   [dati]       tensori contigui, offset relativo alla fine dell'header
 *
 * Questo reader:
 *   1. Apre il file e legge l'header JSON
 *   2. Parsa l'header per trovare offset e shape di ogni tensore
 *   3. Espone pread() diretto o mmap per accesso ai dati
 *   4. NON carica mai tutto il file — solo ciò che serve, quando serve
 *
 * Ispirato a Colibri st.h, adattato per il layout GPT-OSS.
 */

#ifndef PICCHIO_ST_H
#define PICCHIO_ST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#endif

/* ── Costanti ── */

#define ST_MAX_TENSORS  32768   /* max tensori per file (GPT-OSS ha ~1300/shard × 15) */
#define ST_MAX_NAME     256    /* max lunghezza nome tensore */
#define ST_MAX_FILES    64     /* max file (shard) aperti */

/* ── Tipi dtype ── */

typedef enum {
    ST_F32  = 0,
    ST_F16  = 1,
    ST_BF16 = 2,
    ST_I32  = 3,
    ST_I16  = 4,
    ST_I8   = 5,
    ST_U8   = 6,
    ST_F64  = 7,
    ST_BOOL = 8,
    ST_UNKNOWN = 99
} StDtype;

static int st_dtype_size(StDtype dt) {
    switch (dt) {
        case ST_F32: case ST_I32: return 4;
        case ST_F16: case ST_BF16: case ST_I16: return 2;
        case ST_I8: case ST_U8: case ST_BOOL: return 1;
        case ST_F64: return 8;
        default: return 0;
    }
}

static StDtype st_parse_dtype(const char *s) {
    if (!s) return ST_UNKNOWN;
    if (strcmp(s, "F32") == 0) return ST_F32;
    if (strcmp(s, "F16") == 0) return ST_F16;
    if (strcmp(s, "BF16") == 0) return ST_BF16;
    if (strcmp(s, "I32") == 0) return ST_I32;
    if (strcmp(s, "I16") == 0) return ST_I16;
    if (strcmp(s, "I8") == 0) return ST_I8;
    if (strcmp(s, "U8") == 0) return ST_U8;
    if (strcmp(s, "F64") == 0) return ST_F64;
    if (strcmp(s, "BOOL") == 0) return ST_BOOL;
    return ST_UNKNOWN;
}

/* ── Tensore nel file ── */

typedef struct {
    char name[ST_MAX_NAME];
    StDtype dtype;
    int64_t shape[4];      /* dimensioni (max 4D) */
    int ndim;
    int64_t offset_start;  /* offset dall'inizio dei DATI (dopo header) */
    int64_t offset_end;
    int file_idx;          /* indice del file shard */
} StTensor;

/* ── File shard ── */

typedef struct {
    char path[512];
#ifdef _WIN32
    HANDLE hFile;
    HANDLE hMap;
    uint8_t *mmap_ptr;
    int64_t file_size;
#else
    int fd;
    uint8_t *mmap_ptr;
    int64_t file_size;
#endif
    int64_t data_offset;   /* offset dove iniziano i dati (8 + header_len) */
} StFile;

/* ── Database completo dei tensori ── */

typedef struct {
    StFile files[ST_MAX_FILES];
    int n_files;
    StTensor *tensors;     /* heap-allocated [ST_MAX_TENSORS] */
    int n_tensors;
} StDB;

/* ── Handle per-thread (TLS) per read paralleli sicuri ──
 *
 * Su Windows `st_read_raw` usa un handle sincrono con OVERLAPPED: due ReadFile
 * concorrenti sullo stesso handle non sono sicure. Ogni thread apre quindi il
 * proprio handle in modo lazy (FILE_FLAG_RANDOM_ACCESS, adatto ai read sparsi
 * degli expert). Il thread principale registra in st_open_file l'handle
 * SEQUENTIAL_SCAN già aperto, così il caricamento denso iniziale resta invariato.
 * Su POSIX i read usano pread su fd condiviso, già thread-safe: nessun TLS. */
#ifdef _WIN32
static __thread HANDLE st_tls_h[ST_MAX_FILES];
static __thread int    st_tls_ready = 0;

static HANDLE st_win_handle(StFile *sf, int file_idx) {
    if (!st_tls_ready) {
        for (int i = 0; i < ST_MAX_FILES; i++) st_tls_h[i] = NULL;
        st_tls_ready = 1;
    }
    if (!st_tls_h[file_idx]) {
        HANDLE h = CreateFileA(sf->path, GENERIC_READ,
                               FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                               OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, NULL);
        st_tls_h[file_idx] = (h == INVALID_HANDLE_VALUE) ? sf->hFile : h;
    }
    return st_tls_h[file_idx];
}
#endif

/* ── Inizializzazione ── */

static void st_init(StDB *db) {
    memset(db, 0, sizeof(*db));
    db->tensors = (StTensor *)calloc(ST_MAX_TENSORS, sizeof(StTensor));
}

/* ── Parser dell'header JSON per estrarre tensori ── */
/* L'header ha il formato:
 * {"tensor_name": {"dtype": "F32", "shape": [O, I], "data_offsets": [start, end]}, ...}
 */

static int st_parse_header(StDB *db, const char *header, int64_t header_len,
                           int file_idx) {
    const char *p = header;
    const char *end = header + header_len;
    int count = 0;

    /* Cerca ogni "nome": { ... } */
    while (p < end && db->n_tensors < ST_MAX_TENSORS) {
        /* Trova prossima stringa quotata (nome del tensore) */
        const char *q1 = memchr(p, '"', end - p);
        if (!q1) break;
        q1++;

        /* Trova fine nome */
        const char *q2 = memchr(q1, '"', end - q1);
        if (!q2) break;

        /* Skip __metadata__ */
        size_t nlen = q2 - q1;
        if (nlen >= 10 && memcmp(q1, "__metadata__", 12) == 0) {
            p = q2 + 1;
            /* Skip il valore (oggetto) */
            const char *brace = memchr(p, '{', end - p);
            if (brace) {
                int depth = 1;
                const char *bp = brace + 1;
                while (bp < end && depth > 0) {
                    if (*bp == '{') depth++;
                    else if (*bp == '}') depth--;
                    bp++;
                }
                p = bp;
            }
            continue;
        }

        StTensor *t = &db->tensors[db->n_tensors];
        memset(t, 0, sizeof(*t));
        t->file_idx = file_idx;

        /* Copia nome */
        if (nlen >= ST_MAX_NAME) nlen = ST_MAX_NAME - 1;
        memcpy(t->name, q1, nlen);
        t->name[nlen] = '\0';

        /* Trova l'oggetto {} del tensore */
        p = q2 + 1;
        const char *obj = memchr(p, '{', end - p);
        if (!obj) break;

        /* Trova fine oggetto (non nested) */
        const char *obj_end = memchr(obj + 1, '}', end - obj - 1);
        if (!obj_end) break;

        /* Parse dtype */
        const char *dt = strstr(obj, "\"dtype\"");
        if (dt && dt < obj_end) {
            dt = memchr(dt + 7, '"', obj_end - dt - 7);
            if (dt) {
                dt++;
                char dtype_str[16] = {0};
                int i = 0;
                while (dt < obj_end && *dt != '"' && i < 15)
                    dtype_str[i++] = *dt++;
                t->dtype = st_parse_dtype(dtype_str);
            }
        }

        /* Parse shape */
        const char *sh = strstr(obj, "\"shape\"");
        if (sh && sh < obj_end) {
            const char *arr = memchr(sh + 7, '[', obj_end - sh - 7);
            if (arr) {
                arr++;
                t->ndim = 0;
                while (arr < obj_end && *arr != ']' && t->ndim < 4) {
                    while (arr < obj_end && (*arr == ' ' || *arr == ',')) arr++;
                    if (*arr == ']') break;
                    t->shape[t->ndim++] = strtoll(arr, (char **)&arr, 10);
                }
            }
        }

        /* Parse data_offsets */
        const char *off = strstr(obj, "\"data_offsets\"");
        if (off && off < obj_end) {
            const char *arr = memchr(off + 13, '[', obj_end - off - 13);
            if (arr) {
                arr++;
                while (*arr == ' ') arr++;
                t->offset_start = strtoll(arr, (char **)&arr, 10);
                while (*arr == ' ' || *arr == ',') arr++;
                t->offset_end = strtoll(arr, (char **)&arr, 10);
            }
        }

        db->n_tensors++;
        count++;
        p = obj_end + 1;
    }

    return count;
}

/* ── Apri un file safetensors ── */

static int st_open_file(StDB *db, const char *path) {
    if (db->n_files >= ST_MAX_FILES) {
        fprintf(stderr, "st: troppi file aperti\n");
        return -1;
    }

    int idx = db->n_files;
    StFile *sf = &db->files[idx];
    memset(sf, 0, sizeof(*sf));
    strncpy(sf->path, path, sizeof(sf->path) - 1);

#ifdef _WIN32
    sf->hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                            NULL, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    if (sf->hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "st: impossibile aprire %s\n", path);
        return -1;
    }
    LARGE_INTEGER li;
    GetFileSizeEx(sf->hFile, &li);
    sf->file_size = li.QuadPart;
#else
    sf->fd = open(path, O_RDONLY);
    if (sf->fd < 0) {
        fprintf(stderr, "st: impossibile aprire %s\n", path);
        return -1;
    }
    struct stat st;
    fstat(sf->fd, &st);
    sf->file_size = st.st_size;
#endif

    /* Leggi lunghezza header (8 byte LE) */
    uint64_t header_len = 0;
    {
        uint8_t buf[8];
#ifdef _WIN32
        DWORD rd;
        ReadFile(sf->hFile, buf, 8, &rd, NULL);
#else
        pread(sf->fd, buf, 8, 0);
#endif
        header_len = (uint64_t)buf[0] | ((uint64_t)buf[1] << 8) |
                     ((uint64_t)buf[2] << 16) | ((uint64_t)buf[3] << 24) |
                     ((uint64_t)buf[4] << 32) | ((uint64_t)buf[5] << 40) |
                     ((uint64_t)buf[6] << 48) | ((uint64_t)buf[7] << 56);
    }

    if (header_len > 100 * 1024 * 1024) {
        fprintf(stderr, "st: header troppo grande in %s (%llu)\n",
                path, (unsigned long long)header_len);
        return -1;
    }

    sf->data_offset = 8 + (int64_t)header_len;

    /* Leggi e parsa l'header */
    char *header = (char *)malloc(header_len + 1);
    if (!header) {
        fprintf(stderr, "st: OOM per header (%llu bytes)\n",
                (unsigned long long)header_len);
        return -1;
    }

#ifdef _WIN32
    DWORD rd;
    SetFilePointer(sf->hFile, 8, NULL, FILE_BEGIN);
    ReadFile(sf->hFile, header, (DWORD)header_len, &rd, NULL);
#else
    pread(sf->fd, header, header_len, 8);
#endif
    header[header_len] = '\0';

    int nt = st_parse_header(db, header, (int64_t)header_len, idx);
    free(header);

#ifdef _WIN32
    /* Registra l'handle sequenziale come handle TLS del thread principale. */
    if (!st_tls_ready) {
        for (int i = 0; i < ST_MAX_FILES; i++) st_tls_h[i] = NULL;
        st_tls_ready = 1;
    }
    st_tls_h[idx] = sf->hFile;
#endif

    db->n_files++;
    fprintf(stderr, "  st: %s — %d tensori, %.1f GB\n",
            path, nt, sf->file_size / 1e9);
    return idx;
}

/* ── Trova un tensore per nome ── */

static StTensor *st_find(StDB *db, const char *name) {
    for (int i = 0; i < db->n_tensors; i++) {
        if (strcmp(db->tensors[i].name, name) == 0)
            return &db->tensors[i];
    }
    return NULL;
}

/* Trova con prefisso (utile per expert: "model.layers.5.mlp.experts.") */
static StTensor *st_find_prefix(StDB *db, const char *prefix, const char *suffix) {
    char full[ST_MAX_NAME];
    snprintf(full, sizeof(full), "%s%s", prefix, suffix);
    return st_find(db, full);
}

/* ── Leggi dati grezzi di un tensore (pread) ── */

static int64_t st_read_raw(StDB *db, StTensor *t, void *dst, int64_t max_bytes) {
    StFile *sf = &db->files[t->file_idx];
    int64_t nbytes = t->offset_end - t->offset_start;
    if (nbytes > max_bytes) nbytes = max_bytes;

    int64_t abs_offset = sf->data_offset + t->offset_start;

#ifdef _WIN32
    OVERLAPPED ov = {0};
    ov.Offset = (DWORD)(abs_offset & 0xFFFFFFFF);
    ov.OffsetHigh = (DWORD)(abs_offset >> 32);
    DWORD rd;
    ReadFile(st_win_handle(sf, t->file_idx), dst, (DWORD)nbytes, &rd, &ov);
    return (int64_t)rd;
#else
    return pread(sf->fd, dst, nbytes, abs_offset);
#endif
}

/* Leggi una porzione di un tensore a partire da byte_offset. */
static int64_t st_read_raw_at(StDB *db, StTensor *t, int64_t byte_offset,
                              void *dst, int64_t nbytes) {
    StFile *sf = &db->files[t->file_idx];
    int64_t total = t->offset_end - t->offset_start;
    if (byte_offset < 0 || byte_offset > total) return -1;
    if (nbytes > total - byte_offset) nbytes = total - byte_offset;
    int64_t abs_offset = sf->data_offset + t->offset_start + byte_offset;
#ifdef _WIN32
    OVERLAPPED ov = {0};
    ov.Offset = (DWORD)(abs_offset & 0xFFFFFFFF);
    ov.OffsetHigh = (DWORD)(abs_offset >> 32);
    DWORD rd = 0;
    if (!ReadFile(st_win_handle(sf, t->file_idx), dst, (DWORD)nbytes, &rd, &ov)) return -1;
    return (int64_t)rd;
#else
    return pread(sf->fd, dst, nbytes, abs_offset);
#endif
}

/* ── Leggi un tensore coalescente (offset_start di t1 fino a offset_end di t2) ── */
/* Utile per leggere gate_up + down in una sola pread se contigui */

static int64_t st_read_coalesced(StDB *db, StTensor *t_first, StTensor *t_last,
                                 void *dst, int64_t max_bytes) {
    if (t_first->file_idx != t_last->file_idx) return -1;

    StFile *sf = &db->files[t_first->file_idx];
    int64_t start = t_first->offset_start;
    int64_t end = t_last->offset_end;
    int64_t nbytes = end - start;
    if (nbytes > max_bytes) return -1;

    int64_t abs_offset = sf->data_offset + start;

#ifdef _WIN32
    OVERLAPPED ov = {0};
    ov.Offset = (DWORD)(abs_offset & 0xFFFFFFFF);
    ov.OffsetHigh = (DWORD)(abs_offset >> 32);
    DWORD rd;
    ReadFile(sf->hFile, dst, (DWORD)nbytes, &rd, &ov);
    return (int64_t)rd;
#else
    return pread(sf->fd, dst, nbytes, abs_offset);
#endif
}

/* ── Numero di elementi di un tensore ── */

static int64_t st_numel(StTensor *t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

/* ── Bytes di un tensore ── */

static int64_t st_bytes(StTensor *t) {
    return t->offset_end - t->offset_start;
}

/* ── Chiudi tutti i file ── */

static void st_close(StDB *db) {
    for (int i = 0; i < db->n_files; i++) {
#ifdef _WIN32
        if (db->files[i].hFile != INVALID_HANDLE_VALUE)
            CloseHandle(db->files[i].hFile);
#else
        if (db->files[i].fd >= 0)
            close(db->files[i].fd);
#endif
    }
    free(db->tensors);
    db->tensors = NULL;
    db->n_files = 0;
    db->n_tensors = 0;
}

#endif /* PICCHIO_ST_H */
