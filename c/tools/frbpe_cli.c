/*
 * Command line front end, used by the differential tests and the benchmarks.
 *
 *   frbpe_cli encode MODEL   ids of each record, one line per record
 *   frbpe_cli split PATTERN  byte offsets where each chunk ends
 *
 * Records are read from stdin as: 4-byte little-endian length, then that many
 * bytes of UTF-8. A length prefix avoids any escaping problem with newlines
 * or NUL bytes in the text. A record that fails prints "ERROR <code>".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "frbpe.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#define RECORD_HEADER_BYTES 4
#define MAX_RECORD_BYTES (256u << 20)  /* 256 MiB, far above any test document */

static int read_record(char **buf, size_t *cap, size_t *len)
{
    unsigned char header[RECORD_HEADER_BYTES];
    size_t got = fread(header, 1, sizeof header, stdin);
    if (got == 0 && feof(stdin)) return 0;
    if (got != sizeof header) return -1;
    size_t n = (size_t)header[0] | (size_t)header[1] << 8 | (size_t)header[2] << 16 |
               (size_t)header[3] << 24;
    if (n > MAX_RECORD_BYTES) return -1;
    if (n + 1 > *cap) {
        char *bigger = realloc(*buf, n + 1);
        if (bigger == NULL) return -1;
        *buf = bigger;
        *cap = n + 1;
    }
    if (fread(*buf, 1, n, stdin) != n) return -1;
    *len = n;
    return 1;
}

static int run_encode(const char *model_path)
{
    frbpe_model *model;
    int rc = frbpe_load(model_path, &model);
    if (rc != FRBPE_OK) {
        fprintf(stderr, "frbpe_cli: %s: %s\n", model_path, frbpe_strerror(rc));
        return 1;
    }
    char *text = NULL;
    size_t text_cap = 0, len;
    uint32_t *ids = NULL;
    size_t ids_cap = 0;
    int status = 0;
    int got;
    while ((got = read_record(&text, &text_cap, &len)) == 1) {
        /* A token covers at least one byte, so len ids is always enough. */
        if (len > ids_cap) {
            uint32_t *bigger = realloc(ids, len * sizeof *ids);
            if (bigger == NULL) { status = 1; break; }
            ids = bigger;
            ids_cap = len;
        }
        long n = frbpe_encode(model, text, len, ids, ids_cap);
        if (n < 0) {
            printf("ERROR %ld\n", n);
            continue;
        }
        for (long i = 0; i < n; i++) printf(i ? " %u" : "%u", (unsigned)ids[i]);
        putchar('\n');
    }
    if (got < 0) {
        fprintf(stderr, "frbpe_cli: malformed input record\n");
        status = 1;
    }
    free(text);
    free(ids);
    frbpe_free(model);
    return status;
}

static int run_split(const char *pattern)
{
    char *text = NULL;
    size_t text_cap = 0, len;
    size_t *ends = NULL;
    size_t ends_cap = 0;
    int status = 0;
    int got;
    while ((got = read_record(&text, &text_cap, &len)) == 1) {
        if (len > ends_cap) {
            size_t *bigger = realloc(ends, len * sizeof *ends);
            if (bigger == NULL) { status = 1; break; }
            ends = bigger;
            ends_cap = len;
        }
        long n = frbpe_split(pattern, text, len, ends, ends_cap);
        if (n < 0) {
            printf("ERROR %ld\n", n);
            if (n == FRBPE_ERR_PATTERN) { status = 1; break; }
            continue;
        }
        for (long i = 0; i < n; i++) printf(i ? " %zu" : "%zu", ends[i]);
        putchar('\n');
    }
    if (got < 0) {
        fprintf(stderr, "frbpe_cli: malformed input record\n");
        status = 1;
    }
    free(text);
    free(ends);
    return status;
}

int main(int argc, char **argv)
{
#ifdef _WIN32
    /* Without this, Windows turns "\n" into "\r\n" and stops reading at 0x1A. */
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    if (argc == 3 && strcmp(argv[1], "encode") == 0) return run_encode(argv[2]);
    if (argc == 3 && strcmp(argv[1], "split") == 0) return run_split(argv[2]);
    fprintf(stderr, "usage: %s encode MODEL | split PATTERN  (records on stdin)\n", argv[0]);
    return 2;
}
