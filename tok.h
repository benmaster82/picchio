/* tok.h — Tokenizer BPE minimale per Picchio.
 *
 * Legge il formato HuggingFace tokenizer.json:
 *   - "model.vocab": {"token": id, ...}
 *   - "model.merges": ["a b", "c d", ...]
 *
 * Implementa:
 *   - Decode: token_id → stringa UTF-8 (lookup diretto)
 *   - Encode: stringa UTF-8 → token_id[] (BPE greedy)
 *
 * Il tokenizer di GPT-OSS è basato su o200k (200K+ token).
 * Vocabolario: ~201088 token.
 *
 * Strategia di encode semplificata:
 *   1. Split per whitespace/punteggiatura (pre-tokenizzazione grezza)
 *   2. Byte-fallback: ogni byte non riconosciuto → token byte (0x00..0xFF)
 *   3. BPE merge iterativo (longest match first)
 *
 * Per la v1 usiamo un approccio "decode-only + byte-level encode":
 *   - Decode è banale (lookup table)
 *   - Encode usa longest-prefix match sulla vocab (greedy, non ottimale
 *     ma funzionante per la generazione dove serve solo encode del prompt)
 */

#ifndef PICCHIO_TOK_H
#define PICCHIO_TOK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TOK_MAX_VOCAB 210000
#define TOK_MAX_TOKEN_LEN 256

/* ── Struttura tokenizer ── */

typedef struct {
    /* Vocabolario: id → stringa (per decode) */
    char **vocab;         /* [vocab_size] puntatori a stringhe */
    int *vocab_len;       /* [vocab_size] lunghezza di ogni token in bytes */
    int vocab_size;

    /* Hash table per encode: stringa → id */
    /* Simple open-addressing hash table */
    uint32_t *ht_hash;   /* hash values */
    int *ht_id;           /* token ids (-1 = empty) */
    int ht_cap;           /* capacity (power of 2) */

    /* Special tokens */
    int bos_id;
    int eos_id;
    int pad_id;
} Tokenizer;

/* ── FNV-1a hash ── */

static uint32_t tok_hash(const char *s, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h;
}

/* ── Hash table: insert ── */

static void tok_ht_insert(Tokenizer *t, const char *s, int len, int id) {
    uint32_t h = tok_hash(s, len);
    int idx = h & (t->ht_cap - 1);
    while (t->ht_id[idx] != -1) {
        idx = (idx + 1) & (t->ht_cap - 1);
    }
    t->ht_hash[idx] = h;
    t->ht_id[idx] = id;
}

/* ── Hash table: lookup ── */

static int tok_ht_lookup(Tokenizer *t, const char *s, int len) {
    uint32_t h = tok_hash(s, len);
    int idx = h & (t->ht_cap - 1);
    while (t->ht_id[idx] != -1) {
        if (t->ht_hash[idx] == h) {
            int id = t->ht_id[idx];
            if (id < t->vocab_size && t->vocab_len[id] == len &&
                memcmp(t->vocab[id], s, len) == 0) {
                return id;
            }
        }
        idx = (idx + 1) & (t->ht_cap - 1);
    }
    return -1;
}

/* ── Decode un byte escape (es. "Ġ" → ' ', "Ċ" → '\n') ── */
/* Il tokenizer HF/tiktoken usa una mappatura bytes-to-unicode:
 * byte 0x20 (space) → 'Ġ' (U+0120)
 * byte 0x0A (newline) → 'Ċ' (U+010A)
 * etc. Per il decode dobbiamo invertire questa mappatura. */

static int tok_decode_byte_char(const char *s, int *advance) {
    /* UTF-8 decode */
    uint8_t c = (uint8_t)s[0];
    if (c < 0x80) {
        *advance = 1;
        return c;  /* ASCII diretto (i byte 33-126 sono mappati a se stessi) */
    }
    if ((c & 0xE0) == 0xC0 && s[1]) {
        /* 2-byte UTF-8 */
        int cp = ((c & 0x1F) << 6) | (s[1] & 0x3F);
        *advance = 2;
        /* Mappa Unicode codepoint → byte originale */
        /* La mappatura bytes_to_unicode di tiktoken:
         * byte 0..32 → U+0100..U+0120 (256+byte)
         * byte 127..160 → U+0121..U+014E
         * etc. Approccio semplificato: */
        if (cp >= 0x100 && cp <= 0x1FF) {
            /* Questa è la mappatura inversa approssimata */
            /* I byte 0-32, 127-160, 173 sono mappati a U+0100+ */
            static const int byte_map[] = {
                /* U+0100..U+010F → byte 0..15 */
                0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                /* U+0110..U+011F → byte 16..31 */
                16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
                /* U+0120 → byte 32 (space) */
                32,
                /* U+0121..U+017E → byte 127..160, 173 */
            };
            if (cp - 0x100 < (int)(sizeof(byte_map)/sizeof(byte_map[0])))
                return byte_map[cp - 0x100];
            /* Fallback per i byte alti (127-255) */
            return cp - 0x100 + 0;
        }
        return -1;  /* Non è un byte escape */
    }
    if ((c & 0xF0) == 0xE0 && s[1] && s[2]) {
        *advance = 3;
        return -1;
    }
    *advance = 1;
    return -1;
}

/* ── Parse del tokenizer: formato binario Picchio (picchio_vocab.bin) ── */
/* Molto più veloce del parsing JSON da 28 MB.
 * Generato da export_vocab.py. */

static int tok_load_binary(Tokenizer *t, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    /* Header */
    char magic[4];
    uint32_t total, n_added, eos_id, pad_id;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "PVOC", 4) != 0) {
        fclose(f); return -1;
    }
    fread(&total, 4, 1, f);
    fread(&n_added, 4, 1, f);
    fread(&eos_id, 4, 1, f);
    fread(&pad_id, 4, 1, f);

    t->vocab_size = (int)total;
    t->eos_id = (int)eos_id;
    t->pad_id = (int)pad_id;
    t->bos_id = -1;

    /* Alloca */
    t->vocab = (char **)calloc(total, sizeof(char *));
    t->vocab_len = (int *)calloc(total, sizeof(int));

    /* Hash table */
    t->ht_cap = 1;
    while (t->ht_cap < total * 4) t->ht_cap <<= 1;
    t->ht_hash = (uint32_t *)calloc(t->ht_cap, sizeof(uint32_t));
    t->ht_id = (int *)malloc(t->ht_cap * sizeof(int));
    for (int i = 0; i < t->ht_cap; i++) t->ht_id[i] = -1;

    /* Leggi token */
    for (uint32_t i = 0; i < total; i++) {
        uint16_t len;
        if (fread(&len, 2, 1, f) != 1) break;
        char *s = (char *)malloc(len + 1);
        if (len > 0) fread(s, 1, len, f);
        s[len] = '\0';
        t->vocab[i] = s;
        t->vocab_len[i] = (int)len;
        if (len > 0)
            tok_ht_insert(t, s, (int)len, (int)i);
    }

    fclose(f);
    fprintf(stderr, "  tok: %d token caricati (binario, eos=%d)\n",
            t->vocab_size, t->eos_id);
    return 0;
}

static int tok_load(Tokenizer *t, const char *path) {
    memset(t, 0, sizeof(*t));

    /* Prova prima il formato binario (veloce) */
    {
        /* Cerca picchio_vocab.bin nella stessa directory */
        char bin_path[512];
        /* Estrai directory dal path */
        strncpy(bin_path, path, sizeof(bin_path) - 1);
        char *last_sep = strrchr(bin_path, '/');
        if (!last_sep) last_sep = strrchr(bin_path, '\\');
        if (last_sep) {
            strcpy(last_sep + 1, "picchio_vocab.bin");
        } else {
            strcpy(bin_path, "picchio_vocab.bin");
        }
        
        if (tok_load_binary(t, bin_path) == 0) return 0;
    }

    /* Fallback: parse tokenizer.json (lento, supporto limitato) */
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "tok: impossibile aprire %s\n", path);
        fprintf(stderr, "     Genera il vocab binario: python export_vocab.py\n");
        return -1;
    }
    fclose(f);
    fprintf(stderr, "tok: tokenizer.json trovato ma serve il formato binario.\n");
    fprintf(stderr, "     Esegui: python export_vocab.py %s\n", path);
    return -1;
}

/* ── Decode: token_id → stringa UTF-8 ── */
/* Ritorna puntatore alla stringa del token (non copiare, è nel vocab) */

static const char *tok_decode(Tokenizer *t, int id) {
    if (id < 0 || id >= t->vocab_size || !t->vocab[id])
        return "";
    return t->vocab[id];
}

/* Decode con conversione bytes-to-unicode inversa.
 * Scrive in dst (max dstlen), ritorna bytes scritti. */
static int tok_decode_raw(Tokenizer *t, int id, char *dst, int dstlen) {
    const char *s = tok_decode(t, id);
    int slen = t->vocab_len[id];
    int out = 0;

    for (int i = 0; i < slen && out < dstlen - 1; ) {
        int adv = 0;
        int byte = tok_decode_byte_char(s + i, &adv);
        if (byte >= 0) {
            dst[out++] = (char)byte;
            i += adv;
        } else {
            /* Copia UTF-8 char così com'è */
            uint8_t c = (uint8_t)s[i];
            if (c < 0x80) { dst[out++] = s[i]; i++; }
            else if ((c & 0xE0) == 0xC0) {
                if (out + 2 <= dstlen - 1) { dst[out++] = s[i]; dst[out++] = s[i+1]; }
                i += 2;
            }
            else if ((c & 0xF0) == 0xE0) {
                if (out + 3 <= dstlen - 1) { dst[out++] = s[i]; dst[out++] = s[i+1]; dst[out++] = s[i+2]; }
                i += 3;
            }
            else if ((c & 0xF8) == 0xF0) {
                if (out + 4 <= dstlen - 1) { dst[out++] = s[i]; dst[out++] = s[i+1]; dst[out++] = s[i+2]; dst[out++] = s[i+3]; }
                i += 4;
            }
            else { dst[out++] = s[i]; i++; }
        }
    }
    dst[out] = '\0';
    return out;
}

/* ── Encode: stringa UTF-8 → token_id[] (greedy longest match) ── */
/* Non è BPE ottimale, ma funziona per prompt encoding.
 * Ritorna il numero di token scritti in out_ids (max max_tokens). */

static int tok_encode(Tokenizer *t, const char *text, int *out_ids, int max_tokens) {
    int n = 0;
    int len = (int)strlen(text);
    int pos = 0;

    while (pos < len && n < max_tokens) {
        /* Trova il token più lungo che matcha a partire da pos */
        int best_len = 0;
        int best_id = -1;

        /* Prova da lunghezza massima (TOK_MAX_TOKEN_LEN) in giù */
        int max_try = len - pos;
        if (max_try > TOK_MAX_TOKEN_LEN) max_try = TOK_MAX_TOKEN_LEN;

        for (int try_len = max_try; try_len >= 1; try_len--) {
            int id = tok_ht_lookup(t, text + pos, try_len);
            if (id >= 0) {
                best_len = try_len;
                best_id = id;
                break;
            }
        }

        if (best_id >= 0) {
            out_ids[n++] = best_id;
            pos += best_len;
        } else {
            /* Byte fallback: singolo byte come token */
            /* I token byte in tiktoken/o200k sono i primi 256 ID
             * oppure hanno una mappatura specifica. Per ora skip. */
            pos++;
        }
    }

    return n;
}

/* ── Free ── */

static void tok_free(Tokenizer *t) {
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++)
            free(t->vocab[i]);
        free(t->vocab);
    }
    free(t->vocab_len);
    free(t->ht_hash);
    free(t->ht_id);
    memset(t, 0, sizeof(*t));
}

#endif /* PICCHIO_TOK_H */
