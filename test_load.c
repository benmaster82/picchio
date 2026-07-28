/* test_load.c — test minimale di caricamento expert */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "quant.h"
#include "st.h"

int main(void) {
    StDB db;
    st_init(&db);
    
    /* Apri il shard che contiene layer 0 */
    int idx = st_open_file(&db, "D:/gptoss_i4/model-00009.safetensors");
    if (idx < 0) { fprintf(stderr, "ERRORE: impossibile aprire shard\n"); return 1; }
    
    fprintf(stderr, "Tensori: %d\n", db.n_tensors);
    
    /* Cerca expert 0 layer 0 gate_up */
    const char *name = "model.layers.0.mlp.experts.gate_up_proj.0";
    StTensor *t = st_find(&db, name);
    if (!t) {
        fprintf(stderr, "ERRORE: tensore '%s' non trovato\n", name);
        /* Stampa i primi 10 nomi per debug */
        for (int i = 0; i < 10 && i < db.n_tensors; i++)
            fprintf(stderr, "  [%d] %s\n", i, db.tensors[i].name);
        st_close(&db);
        return 1;
    }
    
    fprintf(stderr, "Trovato: %s\n", t->name);
    fprintf(stderr, "  dtype=%d ndim=%d shape=[%lld", t->dtype, t->ndim, (long long)t->shape[0]);
    for (int i = 1; i < t->ndim; i++) fprintf(stderr, ", %lld", (long long)t->shape[i]);
    fprintf(stderr, "]\n");
    fprintf(stderr, "  offset=[%lld, %lld] (%lld bytes)\n",
            (long long)t->offset_start, (long long)t->offset_end,
            (long long)(t->offset_end - t->offset_start));
    
    /* Leggi i primi 32 byte */
    uint8_t buf[32];
    int64_t rd = st_read_raw(&db, t, buf, 32);
    fprintf(stderr, "  read %lld bytes: ", (long long)rd);
    for (int i = 0; i < 16; i++) fprintf(stderr, "%02x ", buf[i]);
    fprintf(stderr, "\n");
    
    /* Prova scale */
    char sname[256];
    snprintf(sname, sizeof(sname), "%s.qs", name);
    StTensor *ts = st_find(&db, sname);
    if (ts) {
        fprintf(stderr, "Scale trovate: bytes=%lld\n", (long long)st_bytes(ts));
        float sc[4];
        st_read_raw(&db, ts, sc, 16);
        fprintf(stderr, "  scale[0..3] = %.6f %.6f %.6f %.6f\n", sc[0], sc[1], sc[2], sc[3]);
    } else {
        fprintf(stderr, "ERRORE: scale '%s' non trovate\n", sname);
    }
    
    /* Test matmul con dati letti */
    fprintf(stderr, "\nTest matmul INT4...\n");
    int O = 5760, I = 2880;
    int64_t rb = (int64_t)O * ((I + 1) / 2);
    uint8_t *q4 = (uint8_t *)malloc(rb);
    float *s_arr = (float *)malloc(O * sizeof(float));
    if (!q4 || !s_arr) { fprintf(stderr, "OOM\n"); return 1; }
    
    rd = st_read_raw(&db, t, q4, rb);
    fprintf(stderr, "  Letti %lld / %lld bytes di dati\n", (long long)rd, (long long)rb);
    
    if (ts) st_read_raw(&db, ts, s_arr, O * sizeof(float));
    else for (int i = 0; i < O; i++) s_arr[i] = 1.0f;
    
    /* Input x: vettore di 1.0 */
    float *x = (float *)calloc(I, sizeof(float));
    float *y = (float *)calloc(O, sizeof(float));
    for (int i = 0; i < I; i++) x[i] = 0.01f;
    
    matmul_i4(y, x, q4, s_arr, 1, I, O);
    
    fprintf(stderr, "  y[0..4] = %.4f %.4f %.4f %.4f %.4f\n", y[0], y[1], y[2], y[3], y[4]);
    fprintf(stderr, "  OK - nessun crash\n");
    
    free(q4); free(s_arr); free(x); free(y);
    st_close(&db);
    return 0;
}
