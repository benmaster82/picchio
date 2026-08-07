/* json.h — Minimal JSON parser for Picchio.
 *
 * Reads ONLY what is needed: scalars (int, float, string, bool)
 * and arrays of strings/integers. No dynamic allocation for the structure,
 * no full handling of the JSON standard — just the subset used
 * by HuggingFace's config.json.
 *
 * Usage:
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

/* ── Read a file into a NUL-terminated buffer ── */

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

/* ── Find the key "key": in the JSON (top level or nested) ── */

static const char *json_find_key(const char *json, const char *key) {
    if (!json || !key) return NULL;
    size_t klen = strlen(key);
    const char *p = json;
    while ((p = strstr(p, key)) != NULL) {
        /* Check that it is quoted: "key" */
        if (p > json && p[-1] == '"') {
            const char *after = p + klen;
            if (*after == '"') {
                /* Find the ':' after it */
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

/* ── Read an integer ── */

static int json_int(const char *json, const char *key, int def) {
    const char *v = json_find_key(json, key);
    if (!v) return def;
    v = json_skip_ws(v);
    if (*v == 'n') return def;  /* null */
    return atoi(v);
}

/* ── Read a float ── */

static float json_float(const char *json, const char *key, float def) {
    const char *v = json_find_key(json, key);
    if (!v) return def;
    v = json_skip_ws(v);
    if (*v == 'n') return def;
    return (float)atof(v);
}

/* ── Read a boolean ── */

static int json_bool(const char *json, const char *key, int def) {
    const char *v = json_find_key(json, key);
    if (!v) return def;
    v = json_skip_ws(v);
    if (*v == 't') return 1;
    if (*v == 'f') return 0;
    return def;
}

/* ── Read a string (copy into dst, max dstlen) ── */

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

/* ── Read an array of strings (e.g. layer_types) ── */
/* Returns the number of elements read.
 * Each string is truncated to maxstr chars. */

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

/* ── Read an array of integers ── */

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
