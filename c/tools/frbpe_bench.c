/*
 * Speed and memory benchmark of the C encoder (milestone 5).
 *
 *   frbpe_bench MODEL RECORDS_FILE REPEATS
 *
 * RECORDS_FILE holds documents in the same format as frbpe_cli's stdin
 * (4-byte little-endian length + UTF-8 bytes). Everything is read into memory
 * first, so the timed loop only encodes. One untimed pass warms the caches,
 * then REPEATS passes are timed one by one. Prints a single JSON line:
 *
 *   {"load_s": ..., "times_s": [...], "bytes": ..., "tokens": ..., "peak_rss_bytes": ...}
 *
 * Single thread. The token count is checked to be identical on every pass.
 */
#if !defined(_WIN32)
#define _POSIX_C_SOURCE 199309L  /* clock_gettime with -std=c99 */
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "frbpe.h"

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <time.h>
#endif

#define RECORD_HEADER_BYTES 4
#define MAX_REPEATS 1000

typedef struct {
    char *data;     /* all documents, back to back */
    size_t *start;  /* start[i] .. start[i + 1] is document i */
    size_t count;
} corpus;

static double now_seconds(void)
{
#ifdef _WIN32
    LARGE_INTEGER freq, t;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

/* Highest resident memory of this process so far, in bytes. */
static unsigned long long peak_rss_bytes(void)
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc))
        return (unsigned long long)pmc.PeakWorkingSetSize;
    return 0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#ifdef __APPLE__
    return (unsigned long long)usage.ru_maxrss;          /* bytes on macOS */
#else
    return (unsigned long long)usage.ru_maxrss * 1024u;  /* kilobytes on Linux */
#endif
#endif
}

static int load_corpus(const char *path, corpus *c)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) return 0;
    size_t cap_docs = 1024, cap_bytes = 1 << 20, used = 0;
    c->data = malloc(cap_bytes);
    c->start = malloc((cap_docs + 1) * sizeof *c->start);
    c->count = 0;
    int ok = c->data != NULL && c->start != NULL;
    unsigned char h[RECORD_HEADER_BYTES];
    while (ok && fread(h, 1, sizeof h, f) == sizeof h) {
        size_t n = (size_t)h[0] | (size_t)h[1] << 8 | (size_t)h[2] << 16 | (size_t)h[3] << 24;
        while (used + n > cap_bytes) {
            char *bigger = realloc(c->data, cap_bytes * 2);
            if (bigger == NULL) { ok = 0; break; }
            c->data = bigger;
            cap_bytes *= 2;
        }
        if (ok && c->count == cap_docs) {
            size_t *bigger = realloc(c->start, (cap_docs * 2 + 1) * sizeof *c->start);
            if (bigger == NULL) ok = 0;
            else { c->start = bigger; cap_docs *= 2; }
        }
        if (ok && fread(c->data + used, 1, n, f) != n) ok = 0;
        if (ok) {
            c->start[c->count++] = used;
            used += n;
        }
    }
    if (ok && !feof(f)) ok = 0;  /* trailing partial header */
    fclose(f);
    if (ok) c->start[c->count] = used;
    return ok;
}

/* Encode every document once; returns the total token count or -1. */
static long long encode_all(const frbpe_model *m, const corpus *c, uint32_t *ids, size_t cap)
{
    long long total = 0;
    for (size_t i = 0; i < c->count; i++) {
        long n = frbpe_encode(m, c->data + c->start[i], c->start[i + 1] - c->start[i], ids, cap);
        if (n < 0 || (size_t)n > cap) return -1;
        total += n;
    }
    return total;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s MODEL RECORDS_FILE REPEATS\n", argv[0]);
        return 2;
    }
    int repeats = atoi(argv[3]);
    if (repeats < 1 || repeats > MAX_REPEATS) {
        fprintf(stderr, "frbpe_bench: REPEATS must be in 1..%d\n", MAX_REPEATS);
        return 2;
    }
    corpus c = {0};
    if (!load_corpus(argv[2], &c)) {
        fprintf(stderr, "frbpe_bench: cannot read %s\n", argv[2]);
        return 1;
    }
    size_t longest = 0;
    for (size_t i = 0; i < c.count; i++)
        if (c.start[i + 1] - c.start[i] > longest) longest = c.start[i + 1] - c.start[i];
    /* A token covers at least one byte: `longest` ids are always enough. */
    uint32_t *ids = malloc((longest + 1) * sizeof *ids);

    double t0 = now_seconds();
    frbpe_model *m = NULL;
    int rc = frbpe_load(argv[1], &m);
    double load_s = now_seconds() - t0;
    if (rc != FRBPE_OK || ids == NULL) {
        fprintf(stderr, "frbpe_bench: %s: %s\n", argv[1], frbpe_strerror(rc));
        return 1;
    }

    long long tokens = encode_all(m, &c, ids, longest + 1);  /* warm-up, untimed */
    int status = tokens < 0;
    double *times = malloc((size_t)repeats * sizeof *times);
    for (int r = 0; !status && times != NULL && r < repeats; r++) {
        double start = now_seconds();
        long long again = encode_all(m, &c, ids, longest + 1);
        times[r] = now_seconds() - start;
        if (again != tokens) status = 1;  /* encoding must be deterministic */
    }
    if (status || times == NULL) {
        fprintf(stderr, "frbpe_bench: encoding failed\n");
        return 1;
    }
    printf("{\"load_s\": %.6f, \"times_s\": [", load_s);
    for (int r = 0; r < repeats; r++) printf(r ? ", %.6f" : "%.6f", times[r]);
    printf("], \"bytes\": %zu, \"tokens\": %lld, \"peak_rss_bytes\": %llu}\n",
           c.start[c.count], tokens, peak_rss_bytes());
    free(times);
    free(ids);
    free(c.data);
    free(c.start);
    frbpe_free(m);
    return 0;
}
