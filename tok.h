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

/* ── Parse del tokenizer.json (HuggingFace format) ── */

static int tok_load(Tokenizer *t, const char *path) {
    memset(t, 0, sizeof(*t));

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "tok: impossibile aprire %s\n", path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(sz + 1);
    if (!buf) { fclose(f); return -1; }
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);

    /* Conta token nel vocabolario per allocare */
    /* Il formato è: "vocab": {"token": id, ...} */
    const char *vocab_start = strstr(buf, "\"vocab\"");
    if (!vocab_start) {
        /* Prova formato alternativo: "model": {"vocab": ...} */
        vocab_start = strstr(buf, "\"model\"");
        if (vocab_start) {
            vocab_start = strstr(vocab_start, "\"vocab\"");
        }
    }

    if (!vocab_start) {
        fprintf(stderr, "tok: 'vocab' non trovato in %s\n", path);
        free(buf);
        return -1;
    }

    /* Stima vocab size (conta le virgolette di apertura dei token) */
    int estimated_size = 0;
    const char *p = vocab_start;
    const char *end = buf + sz;
    int brace_depth = 0;
    int in_vocab = 0;
    while (p < end) {
        if (*p == '{' && !in_vocab) { in_vocab = 1; brace_depth = 1; p++; continue; }
        if (in_vocab) {
            if (*p == '{') brace_depth++;
            if (*p == '}') { brace_depth--; if (brace_depth == 0) break; }
            if (*p == ':') estimated_size++;
        }
        p++;
    }

    if (estimated_size <= 0) estimated_size = 201088;
    t->vocab_size = estimated_size;

    /* Alloca strutture */
    t->vocab = (char **)calloc(t->vocab_size, sizeof(char *));
    t->vocab_len = (int *)calloc(t->vocab_size, sizeof(int));

    /* Hash table: 4x vocab per bassa collisione */
    t->ht_cap = 1;
    while (t->ht_cap < t->vocab_size * 4) t->ht_cap <<= 1;
    t->ht_hash = (uint32_t *)calloc(t->ht_cap, sizeof(uint32_t));
    t->ht_id = (int *)malloc(t->ht_cap * sizeof(int));
    for (int i = 0; i < t->ht_cap; i++) t->ht_id[i] = -1;

    /* Parse del vocabolario */
    p = vocab_start;
    /* Trova il primo '{' dopo "vocab" */
    while (p < end && *p != '{') p++;
    if (p >= end) { free(buf); return -1; }
    p++;  /* skip '{' */

    int max_id = 0;
    int loaded = 0;
    while (p < end && *p != '}') {
        /* Skip whitespace e virgole */
        while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' ||
               *p == '\t' || *p == ',')) p++;
        if (*p == '}') break;

        /* Parse "token": id */
        if (*p != '"') { p++; continue; }
        p++;  /* skip opening quote */

        /* Leggi token string (con gestione escape) */
        char token_buf[TOK_MAX_TOKEN_LEN];
        int tlen = 0;
        while (p < end && *p != '"' && tlen < TOK_MAX_TOKEN_LEN - 1) {
            if (*p == '\\' && p + 1 < end) {
                p++;
                switch (*p) {
                    case 'n': token_buf[tlen++] = '\n'; break;
                    case 'r': token_buf[tlen++] = '\r'; break;
                    case 't': token_buf[tlen++] = '\t'; break;
                    case '"': token_buf[tlen++] = '"'; break;
                    case '\\': token_buf[tlen++] = '\\'; break;
                    case '/': token_buf[tlen++] = '/'; break;
                    case 'u': {
                        /* Unicode escape \uXXXX */
                        if (p + 4 < end) {
                            char hex[5] = {p[1], p[2], p[3], p[4], 0};
                            int cp = (int)strtol(hex, NULL, 16);
                            p += 4;
                            /* Encode codepoint as UTF-8 */
                            if (cp < 0x80) {
                                token_buf[tlen++] = (char)cp;
                            } else if (cp < 0x800) {
                                token_buf[tlen++] = (char)(0xC0 | (cp >> 6));
                                token_buf[tlen++] = (char)(0x80 | (cp & 0x3F));
                            } else {
                                token_buf[tlen++] = (char)(0xE0 | (cp >> 12));
                                token_buf[tlen++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                                token_buf[tlen++] = (char)(0x80 | (cp & 0x3F));
                            }
                        }
                        break;
                    }
                    default: token_buf[tlen++] = *p; break;
                }
            } else {
                token_buf[tlen++] = *p;
            }
            p++;
        }
        if (*p == '"') p++;  /* skip closing quote */

        /* Skip to ':' */
        while (p < end && *p != ':') p++;
        if (p >= end) break;
        p++;  /* skip ':' */

        /* Parse id */
        while (p < end && (*p == ' ' || *p == '\t')) p++;
        int id = (int)strtol(p, (char **)&p, 10);

        /* Store */
        if (id >= 0 && id < t->vocab_size) {
            t->vocab[id] = (char *)malloc(tlen + 1);
            memcpy(t->vocab[id], token_buf, tlen);
            t->vocab[id][tlen] = '\0';
            t->vocab_len[id] = tlen;
            tok_ht_insert(t, token_buf, tlen, id);
            if (id > max_id) max_id = id;
            loaded++;
        }
    }

    t->vocab_size = max_id + 1;

    /* Special tokens */
    t->eos_id = 200002;
    t->bos_id = -1;  /* GPT-OSS non usa BOS esplicito */
    t->pad_id = 199999;

    /* Cerca special tokens nel JSON */
    const char *added = strstr(buf, "\"added_tokens\"");
    if (added) {
        /* Parse added_tokens per trovare EOS/BOS/PAD */
        const char *eos = strstr(added, "\"<|endoftext|>\"");
        if (eos) {
            const char *id_s = strstr(eos, "\"id\"");
            if (id_s) {
                id_s += 4;
                while (*id_s && *id_s != ':') id_s++;
                if (*id_s == ':') t->eos_id = (int)strtol(id_s + 1, NULL, 10);
            }
        }
    }

    free(buf);
    fprintf(stderr, "  tok: %d token caricati (max_id=%d, eos=%d)\n",
            loaded, max_id, t->eos_id);
    return 0;
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
