/* json.h — Parser JSON minimale per Picchio.
 *
 * Legge SOLO ciò che serve: scalari (int, float, string, bool)
 * e array di stringhe/interi. Niente allocazione dinamica per la struttura,
 * niente gestione completa dello standard JSON — solo il subset usato
 * da config.json di HuggingFace.
 *
 * Uso:
 *   char *buf = read_file("config.json");
 *   int hidden = json_int(buf, "hidden_size", 2880);
 *   float eps = json_float(buf, "rms_norm_eps", 1e-5f);
 *   char val[256]; json_str(buf, "model_type", val, sizeof(val), "gpt_oss");
 *   int arr[128]; int n = json_int_array(buf, "layer_types_idx", arr, 128);
 */

#ifndef PICCHIO_JSON_H
#define PICCHIO_JSON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ── Leggi file in un buffer NUL-terminato ── */

static char *json_read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, sz, f);
    buf[rd] = '\0';
    fclose(f);
    return buf;
}

/* ── Trova la chiave "key": nel JSON (livello top o nested) ── */

static const char *json_find_key(const char *json, const char *key) {
    if (!json || !key) return NULL;
    size_t klen = strlen(key);
    const char *p = json;
    while ((p = strstr(p, key)) != NULL) {
        /* Verifica che sia tra virgolette: "key" */
        if (p > json && p[-1] == '"') {
            const char *after = p + klen;
            if (*after == '"') {
                /* Trova il ':' dopo */
                after++;
                while (*after && isspace((unsigned char)*after)) after++;
                if (*after == ':') return after + 1;
            }
        }
        p++;
    }
    return NULL;
}

/* ── Skip whitespace ── */

static const char *json_skip_ws(const char *p) {
    while (p && *p && isspace((unsigned char)*p)) p++;
    return p;
}

/* ── Leggi un intero ── */

static int json_int(const char *json, const char *key, int def) {
    const char *v = json_find_key(json, key);
    if (!v) return def;
    v = json_skip_ws(v);
    if (*v == 'n') return def;  /* null */
    return atoi(v);
}

/* ── Leggi un float ── */

static float json_float(const char *json, const char *key, float def) {
    const char *v = json_find_key(json, key);
    if (!v) return def;
    v = json_skip_ws(v);
    if (*v == 'n') return def;
    return (float)atof(v);
}

/* ── Leggi un booleano ── */

static int json_bool(const char *json, const char *key, int def) {
    const char *v = json_find_key(json, key);
    if (!v) return def;
    v = json_skip_ws(v);
    if (*v == 't') return 1;
    if (*v == 'f') return 0;
    return def;
}

/* ── Leggi una stringa (copia in dst, max dstlen) ── */

static int json_str(const char *json, const char *key,
                    char *dst, int dstlen, const char *def) {
    const char *v = json_find_key(json, key);
    if (!v) { if (def) strncpy(dst, def, dstlen); return 0; }
    v = json_skip_ws(v);
    if (*v != '"') { if (def) strncpy(dst, def, dstlen); return 0; }
    v++; /* skip opening quote */
    int i = 0;
    while (*v && *v != '"' && i < dstlen - 1) {
        if (*v == '\\' && v[1]) { v++; } /* skip escape */
        dst[i++] = *v++;
    }
    dst[i] = '\0';
    return 1;
}

/* ── Leggi un array di stringhe (es. layer_types) ── */
/* Ritorna il numero di elementi letti.
 * Ogni stringa è troncata a maxstr chars. */

static int json_str_array(const char *json, const char *key,
                          char (*dst)[64], int maxn) {
    const char *v = json_find_key(json, key);
    if (!v) return 0;
    v = json_skip_ws(v);
    if (*v != '[') return 0;
    v++;
    int n = 0;
    while (*v && *v != ']' && n < maxn) {
        v = json_skip_ws(v);
        if (*v == '"') {
            v++;
            int i = 0;
            while (*v && *v != '"' && i < 63) {
                if (*v == '\\' && v[1]) v++;
                dst[n][i++] = *v++;
            }
            dst[n][i] = '\0';
            if (*v == '"') v++;
            n++;
        }
        /* skip comma */
        while (*v && *v != '"' && *v != ']') v++;
    }
    return n;
}

/* ── Leggi un array di interi ── */

static int json_int_array(const char *json, const char *key,
                          int *dst, int maxn) {
    const char *v = json_find_key(json, key);
    if (!v) return 0;
    v = json_skip_ws(v);
    if (*v != '[') return 0;
    v++;
    int n = 0;
    while (*v && *v != ']' && n < maxn) {
        v = json_skip_ws(v);
        if (*v == '-' || isdigit((unsigned char)*v)) {
            dst[n++] = atoi(v);
            while (*v && *v != ',' && *v != ']') v++;
        }
        if (*v == ',') v++;
    }
    return n;
}

#endif /* PICCHIO_JSON_H */
