/* minimax_forward_check.c — porting scratch tool, NOT part of the Picchio engine.
 *
 * picchio.c has no model_type="minimax" backend yet. This program validates
 * the *math* of a future MiniMax-M2 forward pass in isolation: it loads the
 * tiny fixture written by make_minimax_test_model.py (config.json +
 * model.safetensors, real MiniMaxM2ForCausalLM weights at toy dimensions) and
 * reimplements the same forward pass described in
 * reference/minimax_m2/modeling_minimax_m2.py, then writes its own logits to
 * c_output.json in the same shape as oracle.json for verify_minimax.py to diff.
 *
 * Architecture notes encoded below (see modeling_minimax_m2.py for the
 * PyTorch original):
 *   - attention: GQA, o_proj input width = num_heads*head_dim (can exceed D)
 *   - QK-norm is ONE RMSNorm over the whole concatenated multi-head vector,
 *     not per-head
 *   - RoPE is PARTIAL: only the first `rotary_dim` of each head_dim rotate,
 *     the rest pass through unchanged; attention scaling still uses the full
 *     head_dim
 *   - MoE router: sigmoid scoring; top-k experts are SELECTED using
 *     (sigmoid_score + e_score_correction_bias), but the mixing weight for
 *     each selected expert is the ORIGINAL unbiased sigmoid score, renormalized
 *     to sum to 1 across the k selected experts
 *   - no shared expert; decoder residual is the plain pre-norm pattern (the
 *     Python's extra `residual` return is always equal to `hidden_states`
 *     itself, so it is a no-op here)
 *
 * Build: gcc -O2 -fopenmp -mavx2 -mfma -o minimax_forward_check.exe minimax_forward_check.c -lm
 * Run:   ./minimax_forward_check.exe minimax_test_model
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "json.h"
#include "quant.h"

/* ───────────────────────── minimal in-memory safetensors reader ───────────────────────── */

#define ST_MAX_T 8192

typedef struct {
    char name[160];
    int shape[4];
    int ndim;
    float *data;
} LiteTensor;

typedef struct {
    uint8_t *buf;
    char *header;               /* null-terminated copy of the JSON header */
    LiteTensor t[ST_MAX_T];
    int n;
} LiteFile;

static uint8_t *read_whole_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f);
    if (len_out) *len_out = (size_t)sz;
    return buf;
}

/* Find the `"name": { ... }` object body for an exact tensor key. Returns a
 * malloc'd, null-terminated copy of the `{...}` body (caller frees), or NULL. */
static char *find_tensor_object(const char *header, const char *name) {
    char needle[192];
    snprintf(needle, sizeof(needle), "\"%s\":", name);
    const char *p = strstr(header, needle);
    if (!p) return NULL;
    p += strlen(needle);
    p = json_skip_ws(p);
    if (*p != '{') return NULL;
    const char *start = p;
    int depth = 0;
    do {
        if (*p == '{') depth++;
        else if (*p == '}') depth--;
        p++;
    } while (depth > 0 && *p);
    size_t len = (size_t)(p - start);
    char *out = (char *)malloc(len + 1);
    memcpy(out, start, len);
    out[len] = '\0';
    return out;
}

static void lite_load(LiteFile *lf, const char *path) {
    size_t flen;
    lf->buf = read_whole_file(path, &flen);
    uint64_t header_len;
    memcpy(&header_len, lf->buf, 8);
    lf->header = (char *)malloc(header_len + 1);
    memcpy(lf->header, lf->buf + 8, header_len);
    lf->header[header_len] = '\0';
    uint8_t *data_base = lf->buf + 8 + header_len;
    lf->n = 0;

    /* Walk top-level keys of the header ourselves (json.h only does keyed
     * lookup, not iteration) so any tensor name the fixture wrote is picked
     * up without having to know it in advance. */
    const char *p = lf->header;
    p = json_skip_ws(p);
    if (*p != '{') { fprintf(stderr, "bad safetensors header\n"); exit(1); }
    p++;
    while (1) {
        p = json_skip_ws(p);
        if (*p == '}' || *p == '\0') break;
        if (*p != '"') { fprintf(stderr, "bad header key\n"); exit(1); }
        const char *kstart = ++p;
        while (*p && *p != '"') p++;
        size_t klen = (size_t)(p - kstart);
        char key[160];
        if (klen >= sizeof(key)) klen = sizeof(key) - 1;
        memcpy(key, kstart, klen);
        key[klen] = '\0';
        p++; /* closing quote */
        p = json_skip_ws(p);
        if (*p != ':') { fprintf(stderr, "bad header colon\n"); exit(1); }
        p++;
        p = json_skip_ws(p);
        const char *objstart = p;
        int depth = 0;
        do {
            if (*p == '{') depth++;
            else if (*p == '}') depth--;
            p++;
        } while (depth > 0 && *p);
        size_t objlen = (size_t)(p - objstart);

        if (strcmp(key, "__metadata__") != 0) {
            char *obj = (char *)malloc(objlen + 1);
            memcpy(obj, objstart, objlen);
            obj[objlen] = '\0';

            char dtype[16];
            json_str(obj, "dtype", dtype, sizeof(dtype), "?");
            int shape[4] = {1, 1, 1, 1};
            int ndim = json_int_array(obj, "shape", shape, 4);
            int offs[2] = {0, 0};
            json_int_array(obj, "data_offsets", offs, 2);

            if (strcmp(dtype, "F32") == 0 && lf->n < ST_MAX_T) {
                LiteTensor *t = &lf->t[lf->n++];
                strncpy(t->name, key, sizeof(t->name) - 1);
                t->ndim = ndim;
                for (int i = 0; i < 4; i++) t->shape[i] = shape[i];
                t->data = (float *)(data_base + offs[0]);
            }
            free(obj);
        }

        p = json_skip_ws(p);
        if (*p == ',') p++;
    }
}

static float *lite_get(LiteFile *lf, const char *name, int *out_shape) {
    for (int i = 0; i < lf->n; i++) {
        if (strcmp(lf->t[i].name, name) == 0) {
            if (out_shape) {
                out_shape[0] = lf->t[i].shape[0];
                out_shape[1] = lf->t[i].ndim > 1 ? lf->t[i].shape[1] : 1;
            }
            return lf->t[i].data;
        }
    }
    fprintf(stderr, "tensor not found: %s\n", name);
    exit(1);
}

/* ───────────────────────── config ───────────────────────── */

typedef struct {
    int D, L, H, KVH, hd, rotary, E, topk, I, V;
    float eps, theta;
} Cfg;

static Cfg load_config(const char *path) {
    size_t len;
    uint8_t *raw = read_whole_file(path, &len);
    char *json = (char *)malloc(len + 1);
    memcpy(json, raw, len);
    json[len] = '\0';
    free(raw);

    Cfg c;
    c.D      = json_int(json, "hidden_size", 0);
    c.L      = json_int(json, "num_hidden_layers", 0);
    c.H      = json_int(json, "num_attention_heads", 0);
    c.KVH    = json_int(json, "num_key_value_heads", 0);
    c.hd     = json_int(json, "head_dim", 0);
    c.rotary = json_int(json, "rotary_dim", 0);
    c.E      = json_int(json, "num_local_experts", 0);
    c.topk   = json_int(json, "num_experts_per_tok", 0);
    c.I      = json_int(json, "intermediate_size", 0);
    c.V      = json_int(json, "vocab_size", 0);
    c.eps    = json_float(json, "rms_norm_eps", 1e-6f);
    c.theta  = json_float(json, "rope_theta", 10000.0f);
    free(json);
    return c;
}

/* ───────────────────────── forward pass ───────────────────────── */

/* Partial RoPE on one head vector (length hd): rotate the first `rotary`
 * dims (standard HF non-interleaved rotate_half), pass the rest through. */
static void rope_partial(float *v, int pos, int rotary, float theta) {
    int half = rotary / 2;
    float tmp[512];
    memcpy(tmp, v, (size_t)rotary * sizeof(float));
    for (int j = 0; j < half; j++) {
        float freq = 1.0f / powf(theta, (2.0f * j) / (float)rotary);
        float ang = pos * freq;
        float cs = cosf(ang), sn = sinf(ang);
        float a = tmp[j], b = tmp[j + half];
        v[j]        = a * cs - b * sn;
        v[j + half] = a * sn + b * cs;
    }
    /* v[rotary:hd] left untouched by design */
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "minimax_test_model";
    char path[1024];

    snprintf(path, sizeof(path), "%s/config.json", dir);
    Cfg c = load_config(path);
    printf("config: D=%d L=%d H=%d KVH=%d hd=%d rotary=%d E=%d top%d I=%d V=%d\n",
           c.D, c.L, c.H, c.KVH, c.hd, c.rotary, c.E, c.topk, c.I, c.V);

    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    LiteFile lf;
    lite_load(&lf, path);
    printf("loaded %d F32 tensors\n", lf.n);

    int group = c.H / c.KVH;
    int qdim = c.H * c.hd, kvdim = c.KVH * c.hd;
    float scale = 1.0f / sqrtf((float)c.hd);

    float *embed  = lite_get(&lf, "model.embed_tokens.weight", NULL);
    float *lmhead = lite_get(&lf, "lm_head.weight", NULL);
    float *finalw = lite_get(&lf, "model.norm.weight", NULL);

    int tokens[] = {1, 5, 3, 7, 2, 9};
    int n_tok = (int)(sizeof(tokens) / sizeof(tokens[0]));

    /* KV cache: [layer][pos][kvdim] */
    float *Kc = (float *)calloc((size_t)c.L * n_tok * kvdim, sizeof(float));
    float *Vc = (float *)calloc((size_t)c.L * n_tok * kvdim, sizeof(float));

    float *h = (float *)malloc(c.D * sizeof(float));

    FILE *out = fopen("c_output.json", "w");
    fprintf(out, "{\n  \"tokens_in\": [");
    for (int i = 0; i < n_tok; i++) fprintf(out, "%d%s", tokens[i], i + 1 < n_tok ? ", " : "");
    fprintf(out, "],\n  \"positions\": [\n");

    for (int pos = 0; pos < n_tok; pos++) {
        memcpy(h, embed + (int64_t)tokens[pos] * c.D, c.D * sizeof(float));

        for (int l = 0; l < c.L; l++) {
            char nb[192];
            #define TN(fmt) (snprintf(nb, sizeof(nb), fmt, l), nb)

            float *w_in_ln = lite_get(&lf, TN("model.layers.%d.input_layernorm.weight"), NULL);
            float hn[512];
            rmsnorm(hn, h, w_in_ln, c.D, c.eps);

            float *Wq = lite_get(&lf, TN("model.layers.%d.self_attn.q_proj.weight"), NULL);
            float *Wk = lite_get(&lf, TN("model.layers.%d.self_attn.k_proj.weight"), NULL);
            float *Wv = lite_get(&lf, TN("model.layers.%d.self_attn.v_proj.weight"), NULL);
            float *Wo = lite_get(&lf, TN("model.layers.%d.self_attn.o_proj.weight"), NULL);
            float *Wqn = lite_get(&lf, TN("model.layers.%d.self_attn.q_norm.weight"), NULL);
            float *Wkn = lite_get(&lf, TN("model.layers.%d.self_attn.k_norm.weight"), NULL);

            float q[512], k[512], v[512];
            matmul_f32(q, hn, Wq, 1, c.D, qdim);
            matmul_f32(k, hn, Wk, 1, c.D, kvdim);
            matmul_f32(v, hn, Wv, 1, c.D, kvdim);

            /* whole-vector QK-norm (one RMSNorm over all heads concatenated) */
            float qn[512], kn[512];
            rmsnorm(qn, q, Wqn, qdim, c.eps);
            rmsnorm(kn, k, Wkn, kvdim, c.eps);
            memcpy(q, qn, qdim * sizeof(float));
            memcpy(k, kn, kvdim * sizeof(float));

            for (int hh = 0; hh < c.H; hh++)  rope_partial(q + hh * c.hd, pos, c.rotary, c.theta);
            for (int hh = 0; hh < c.KVH; hh++) rope_partial(k + hh * c.hd, pos, c.rotary, c.theta);

            memcpy(Kc + ((int64_t)l * n_tok + pos) * kvdim, k, kvdim * sizeof(float));
            memcpy(Vc + ((int64_t)l * n_tok + pos) * kvdim, v, kvdim * sizeof(float));

            float attn_out[512];
            for (int hh = 0; hh < c.H; hh++) {
                int kvh = hh / group;
                float *qh = q + hh * c.hd;
                float scores[64];
                for (int t = 0; t <= pos; t++) {
                    float *kt = Kc + ((int64_t)l * n_tok + t) * kvdim + kvh * c.hd;
                    float s = 0;
                    for (int i = 0; i < c.hd; i++) s += qh[i] * kt[i];
                    scores[t] = s * scale;
                }
                softmax(scores, pos + 1);
                float *ao = attn_out + hh * c.hd;
                for (int i = 0; i < c.hd; i++) ao[i] = 0;
                for (int t = 0; t <= pos; t++) {
                    float *vt = Vc + ((int64_t)l * n_tok + t) * kvdim + kvh * c.hd;
                    for (int i = 0; i < c.hd; i++) ao[i] += scores[t] * vt[i];
                }
            }

            float attn_result[512];
            matmul_f32(attn_result, attn_out, Wo, 1, qdim, c.D);
            for (int i = 0; i < c.D; i++) h[i] += attn_result[i];

            float *w_post_ln = lite_get(&lf, TN("model.layers.%d.post_attention_layernorm.weight"), NULL);
            rmsnorm(hn, h, w_post_ln, c.D, c.eps);

            float *Wgate = lite_get(&lf, TN("model.layers.%d.block_sparse_moe.gate.weight"), NULL);
            float *bias  = lite_get(&lf, TN("model.layers.%d.block_sparse_moe.e_score_correction_bias"), NULL);
            float logits_r[64], scores_r[64], biased[64];
            matmul_f32(logits_r, hn, Wgate, 1, c.D, c.E);
            for (int e = 0; e < c.E; e++) {
                scores_r[e] = sigmoidf(logits_r[e]);
                biased[e] = scores_r[e] + bias[e];
            }

            int sel[16]; float sel_w[16];
            for (int kk = 0; kk < c.topk; kk++) {
                int best = -1; float bestv = -1e30f;
                for (int e = 0; e < c.E; e++) {
                    int used = 0;
                    for (int j = 0; j < kk; j++) if (sel[j] == e) used = 1;
                    if (!used && biased[e] > bestv) { bestv = biased[e]; best = e; }
                }
                sel[kk] = best;
                sel_w[kk] = scores_r[best];   /* UNBIASED score, per the Python */
            }
            float wsum = 0; for (int kk = 0; kk < c.topk; kk++) wsum += sel_w[kk];
            if (wsum < 1e-12f) wsum = 1e-12f;
            for (int kk = 0; kk < c.topk; kk++) sel_w[kk] /= wsum;

            float moe_out[512];
            for (int i = 0; i < c.D; i++) moe_out[i] = 0;
            for (int kk = 0; kk < c.topk; kk++) {
                int e = sel[kk];
                char nb2[224];
                #define TE(fmt) (snprintf(nb2, sizeof(nb2), fmt, l, e), nb2)
                float *W1 = lite_get(&lf, TE("model.layers.%d.block_sparse_moe.experts.%d.w1.weight"), NULL);
                float *W2 = lite_get(&lf, TE("model.layers.%d.block_sparse_moe.experts.%d.w2.weight"), NULL);
                float *W3 = lite_get(&lf, TE("model.layers.%d.block_sparse_moe.experts.%d.w3.weight"), NULL);

                float gate[512], up[512], act[512], eo[512];
                matmul_f32(gate, hn, W1, 1, c.D, c.I);
                matmul_f32(up,   hn, W3, 1, c.D, c.I);
                for (int i = 0; i < c.I; i++) act[i] = siluf(gate[i]) * up[i];
                matmul_f32(eo, act, W2, 1, c.I, c.D);

                for (int i = 0; i < c.D; i++) moe_out[i] += sel_w[kk] * eo[i];
            }
            for (int i = 0; i < c.D; i++) h[i] += moe_out[i];
        }

        float hn_final[512];
        rmsnorm(hn_final, h, finalw, c.D, c.eps);
        float *logits = (float *)malloc(c.V * sizeof(float));
        matmul_f32(logits, hn_final, lmhead, 1, c.D, c.V);

        int argmax = 0; float best = logits[0];
        for (int i = 1; i < c.V; i++) if (logits[i] > best) { best = logits[i]; argmax = i; }
        printf("  pos=%d tok_in=%d -> argmax=%d (logit=%.4f)\n", pos, tokens[pos], argmax, best);

        fprintf(out, "    {\"pos\": %d, \"tok_in\": %d, \"argmax\": %d, \"logits\": [",
                pos, tokens[pos], argmax);
        for (int i = 0; i < c.V; i++) fprintf(out, "%.6f%s", logits[i], i + 1 < c.V ? ", " : "");
        fprintf(out, "]}%s\n", pos + 1 < n_tok ? "," : "");

        free(logits);
    }
    fprintf(out, "  ]\n}\n");
    fclose(out);

    printf("wrote c_output.json\n");
    return 0;
}
