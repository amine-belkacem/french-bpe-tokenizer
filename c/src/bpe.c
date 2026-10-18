/*
 * Model loading, BPE encoding and decoding.
 *
 * Encoding one chunk follows the Python reference: repeatedly merge the pair
 * with the lowest rank, all its occurrences left to right. Done naively that
 * is O(n^2) per chunk. Here the tokens of a chunk form a doubly linked list,
 * and a min-heap holds (rank, position) of every adjacent pair, so each merge
 * costs O(log n).
 *
 * Why this gives the same result as the Python loop:
 * - equal ranks are popped by increasing position, which is the left to
 *   right order of the Python sweep, and an occurrence that overlaps an
 *   already merged one is dead by then and skipped;
 * - a merge creates pairs that contain the new token 256 + r, whose rank is
 *   necessarily higher than r, so it cannot change the current round.
 * Heap entries are never removed: when popped, an entry is ignored unless
 * its rank still equals the current rank of the pair at that position (a
 * rank identifies a pair, so equal rank means same pair).
 */
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "frbpe.h"
#include "internal.h"

#define N_BYTES 256u
#define FILE_MAGIC "frbpe v1"
#define PATTERN_PREFIX "pattern "
#define MERGES_PREFIX "merges "
#define MAX_PATTERN_NAME 31
#define NO_RANK (-1)
#define NO_NODE (-1)
#define HASH_MULTIPLIER 0x9E3779B97F4A7C15ull  /* 2^64 / golden ratio */

struct frbpe_model {
    frbpe_pattern pattern;
    char pattern_name[MAX_PATTERN_NAME + 1];
    size_t n_merges;
    /* Open-addressing hash table: (a, b) -> rank. */
    uint64_t *keys;
    int32_t *ranks;
    size_t mask;
    /* Bytes of every token, for decoding. */
    uint32_t *tok_offset;
    uint32_t *tok_length;
    unsigned char *tok_bytes;
};

/* ------------------------------------------------------------ hash table */

static uint64_t pair_key(uint32_t a, uint32_t b) { return ((uint64_t)a << 32) | b; }

static size_t slot_of(const frbpe_model *m, uint64_t key)
{
    return (size_t)((key * HASH_MULTIPLIER) >> 32) & m->mask;
}

static int32_t lookup_rank(const frbpe_model *m, uint32_t a, uint32_t b)
{
    uint64_t key = pair_key(a, b);
    for (size_t s = slot_of(m, key);; s = (s + 1) & m->mask) {
        if (m->ranks[s] == NO_RANK) return NO_RANK;
        if (m->keys[s] == key) return m->ranks[s];
    }
}

static void insert_rank(frbpe_model *m, uint32_t a, uint32_t b, int32_t rank)
{
    uint64_t key = pair_key(a, b);
    size_t s = slot_of(m, key);
    while (m->ranks[s] != NO_RANK) s = (s + 1) & m->mask;
    m->keys[s] = key;
    m->ranks[s] = rank;
}

/* ---------------------------------------------------------------- loading */

static char *read_file(const char *path, size_t *size)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) return NULL;
    char *buf = NULL;
    if (fseek(f, 0, SEEK_END) == 0) {
        long end = ftell(f);
        if (end >= 0 && fseek(f, 0, SEEK_SET) == 0) {
            buf = malloc((size_t)end + 1);
            if (buf != NULL && fread(buf, 1, (size_t)end, f) == (size_t)end) {
                buf[end] = '\0';
                *size = (size_t)end;
            } else {
                free(buf);
                buf = NULL;
            }
        }
    }
    fclose(f);
    return buf;
}

/* Cut the next line out of *cursor (in place) and return it, or NULL at the end. */
static char *next_line(char **cursor)
{
    char *line = *cursor;
    if (*line == '\0') return NULL;
    char *eol = strchr(line, '\n');
    if (eol != NULL) {
        *eol = '\0';
        *cursor = eol + 1;
    } else {
        *cursor = line + strlen(line);
    }
    /* Accept CRLF like Python's splitlines() does in BPETokenizer.load. */
    size_t len = strlen(line);
    if (len > 0 && line[len - 1] == '\r') line[len - 1] = '\0';
    return line;
}

/* Strict unsigned parse: digits only, no sign, no overflow past `max`. */
static int parse_uint(const char *s, char **end, unsigned long max, unsigned long *out)
{
    if (*s < '0' || *s > '9') return 0;
    unsigned long v = 0;
    while (*s >= '0' && *s <= '9') {
        unsigned digit = (unsigned)(*s - '0');
        if (v > (max - digit) / 10) return 0;
        v = v * 10 + digit;
        s++;
    }
    *end = (char *)s;
    *out = v;
    return 1;
}

static int parse_merges(frbpe_model *m, char **cursor, uint32_t *merge_a, uint32_t *merge_b)
{
    for (size_t r = 0; r < m->n_merges; r++) {
        char *line = next_line(cursor);
        unsigned long a, b;
        char *end;
        /* A merge can only use tokens that exist before it: ids < 256 + r.
         * n_merges <= INT32_MAX - 256 (checked in parse_model), so this fits
         * even where unsigned long is 32 bits (Windows). */
        unsigned long limit = (unsigned long)(N_BYTES + r - 1);
        if (line == NULL || !parse_uint(line, &end, limit, &a) || *end != ' ' ||
            !parse_uint(end + 1, &end, limit, &b) || *end != '\0')
            return FRBPE_ERR_FORMAT;
        merge_a[r] = (uint32_t)a;
        merge_b[r] = (uint32_t)b;
    }
    return next_line(cursor) == NULL ? FRBPE_OK : FRBPE_ERR_FORMAT;  /* no trailing junk */
}

static int build_tables(frbpe_model *m, const uint32_t *merge_a, const uint32_t *merge_b)
{
    size_t vocab = N_BYTES + m->n_merges;
    size_t capacity = 1;
    while (capacity < 2 * m->n_merges + 1) capacity <<= 1;  /* load factor <= 0.5 */
    m->mask = capacity - 1;
    m->keys = malloc(capacity * sizeof *m->keys);
    m->ranks = malloc(capacity * sizeof *m->ranks);
    m->tok_offset = malloc(vocab * sizeof *m->tok_offset);
    m->tok_length = malloc(vocab * sizeof *m->tok_length);
    if (!m->keys || !m->ranks || !m->tok_offset || !m->tok_length) return FRBPE_ERR_NOMEM;
    for (size_t s = 0; s < capacity; s++) m->ranks[s] = NO_RANK;

    uint64_t total = N_BYTES;
    for (size_t t = 0; t < N_BYTES; t++) m->tok_length[t] = 1;
    for (size_t r = 0; r < m->n_merges; r++) {
        uint64_t len = (uint64_t)m->tok_length[merge_a[r]] + m->tok_length[merge_b[r]];
        if (len > UINT32_MAX || total + len > UINT32_MAX) return FRBPE_ERR_FORMAT;
        m->tok_length[N_BYTES + r] = (uint32_t)len;
        total += len;
        insert_rank(m, merge_a[r], merge_b[r], (int32_t)r);
    }
    m->tok_bytes = malloc((size_t)total);
    if (!m->tok_bytes) return FRBPE_ERR_NOMEM;
    uint32_t offset = 0;
    for (size_t t = 0; t < N_BYTES; t++) {
        m->tok_offset[t] = offset;
        m->tok_bytes[offset++] = (unsigned char)t;
    }
    for (size_t r = 0; r < m->n_merges; r++) {
        size_t t = N_BYTES + r;
        uint32_t la = m->tok_length[merge_a[r]];
        m->tok_offset[t] = offset;
        memcpy(m->tok_bytes + offset, m->tok_bytes + m->tok_offset[merge_a[r]], la);
        memcpy(m->tok_bytes + offset + la, m->tok_bytes + m->tok_offset[merge_b[r]],
               m->tok_length[merge_b[r]]);
        offset += m->tok_length[t];
    }
    return FRBPE_OK;
}

static int parse_model(frbpe_model *m, char *text)
{
    char *cursor = text;
    char *line = next_line(&cursor);
    if (line == NULL || strcmp(line, FILE_MAGIC) != 0) return FRBPE_ERR_FORMAT;

    line = next_line(&cursor);
    if (line == NULL || strncmp(line, PATTERN_PREFIX, strlen(PATTERN_PREFIX)) != 0)
        return FRBPE_ERR_FORMAT;
    const char *name = line + strlen(PATTERN_PREFIX);
    if (strlen(name) > MAX_PATTERN_NAME) return FRBPE_ERR_PATTERN;
    if (frbpe_pattern_from_name(name, &m->pattern) != FRBPE_OK) return FRBPE_ERR_PATTERN;
    strcpy(m->pattern_name, name);

    line = next_line(&cursor);
    unsigned long n;
    char *end;
    if (line == NULL || strncmp(line, MERGES_PREFIX, strlen(MERGES_PREFIX)) != 0 ||
        !parse_uint(line + strlen(MERGES_PREFIX), &end, INT32_MAX - N_BYTES, &n) || *end != '\0')
        return FRBPE_ERR_FORMAT;
    m->n_merges = n;

    uint32_t *merge_a = malloc((n + 1) * sizeof *merge_a);
    uint32_t *merge_b = malloc((n + 1) * sizeof *merge_b);
    int rc = (merge_a && merge_b) ? parse_merges(m, &cursor, merge_a, merge_b) : FRBPE_ERR_NOMEM;
    if (rc == FRBPE_OK) rc = build_tables(m, merge_a, merge_b);
    free(merge_a);
    free(merge_b);
    return rc;
}

int frbpe_load(const char *path, frbpe_model **out)
{
    if (path == NULL || out == NULL) return FRBPE_ERR_ARG;
    *out = NULL;
    size_t size;
    char *text = read_file(path, &size);
    if (text == NULL) return FRBPE_ERR_IO;
    if (strlen(text) != size) {  /* embedded NUL byte */
        free(text);
        return FRBPE_ERR_FORMAT;
    }
    frbpe_model *m = calloc(1, sizeof *m);
    int rc = m ? parse_model(m, text) : FRBPE_ERR_NOMEM;
    free(text);
    if (rc != FRBPE_OK) {
        frbpe_free(m);
        return rc;
    }
    *out = m;
    return FRBPE_OK;
}

void frbpe_free(frbpe_model *m)
{
    if (m == NULL) return;
    free(m->keys);
    free(m->ranks);
    free(m->tok_offset);
    free(m->tok_length);
    free(m->tok_bytes);
    free(m);
}

size_t frbpe_vocab_size(const frbpe_model *m) { return m ? N_BYTES + m->n_merges : 0; }
const char *frbpe_pattern_name(const frbpe_model *m) { return m ? m->pattern_name : ""; }

/* --------------------------------------------------------------- encoding */

typedef struct {
    size_t capacity;   /* in bytes of chunk */
    uint32_t *tok;
    int32_t *prev, *next, *rank;
    uint64_t *heap;    /* (rank << 32) | position, min-heap */
    size_t heap_size;
} workspace;

static int workspace_reserve(workspace *w, size_t n)
{
    if (n <= w->capacity) return FRBPE_OK;
    size_t cap = w->capacity ? w->capacity : 64;
    while (cap < n) cap *= 2;
    /* Every pair is pushed once at the start, then at most 2 more per merge. */
    uint32_t *tok = realloc(w->tok, cap * sizeof *tok);
    if (tok) w->tok = tok;
    int32_t *prev = realloc(w->prev, cap * sizeof *prev);
    if (prev) w->prev = prev;
    int32_t *next = realloc(w->next, cap * sizeof *next);
    if (next) w->next = next;
    int32_t *rank = realloc(w->rank, cap * sizeof *rank);
    if (rank) w->rank = rank;
    uint64_t *heap = realloc(w->heap, 3 * cap * sizeof *heap);
    if (heap) w->heap = heap;
    if (!tok || !prev || !next || !rank || !heap) return FRBPE_ERR_NOMEM;
    w->capacity = cap;
    return FRBPE_OK;
}

static void workspace_free(workspace *w)
{
    free(w->tok);
    free(w->prev);
    free(w->next);
    free(w->rank);
    free(w->heap);
}

static void heap_push(workspace *w, int32_t rank, int32_t pos)
{
    uint64_t item = ((uint64_t)(uint32_t)rank << 32) | (uint32_t)pos;
    size_t i = w->heap_size++;
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (w->heap[parent] <= item) break;
        w->heap[i] = w->heap[parent];
        i = parent;
    }
    w->heap[i] = item;
}

static uint64_t heap_pop(workspace *w)
{
    uint64_t top = w->heap[0];
    uint64_t last = w->heap[--w->heap_size];
    size_t i = 0;
    for (;;) {
        size_t child = 2 * i + 1;
        if (child >= w->heap_size) break;
        if (child + 1 < w->heap_size && w->heap[child + 1] < w->heap[child]) child++;
        if (last <= w->heap[child]) break;
        w->heap[i] = w->heap[child];
        i = child;
    }
    if (w->heap_size > 0) w->heap[i] = last;
    return top;
}

static void set_rank(const frbpe_model *m, workspace *w, int32_t i)
{
    int32_t j = w->next[i];
    w->rank[i] = j == NO_NODE ? NO_RANK : lookup_rank(m, w->tok[i], w->tok[j]);
    if (w->rank[i] != NO_RANK) heap_push(w, w->rank[i], i);
}

/* Encode one chunk; appends its ids to out[*count...] while room remains. */
static int encode_chunk(const frbpe_model *m, workspace *w, const unsigned char *bytes, size_t n,
                        uint32_t *out, size_t cap, size_t *count)
{
    if (n == 1) {  /* most frequent case after spaces: nothing to merge */
        if (*count < cap) out[*count] = bytes[0];
        (*count)++;
        return FRBPE_OK;
    }
    if (n > INT32_MAX) return FRBPE_ERR_ARG;
    int rc = workspace_reserve(w, n);
    if (rc != FRBPE_OK) return rc;

    w->heap_size = 0;
    for (size_t i = 0; i < n; i++) {
        w->tok[i] = bytes[i];
        w->prev[i] = (int32_t)i - 1;
        w->next[i] = i + 1 < n ? (int32_t)i + 1 : NO_NODE;
    }
    for (size_t i = 0; i < n; i++) set_rank(m, w, (int32_t)i);

    while (w->heap_size > 0) {
        uint64_t item = heap_pop(w);
        int32_t rank = (int32_t)(item >> 32);
        int32_t i = (int32_t)(item & 0xFFFFFFFFu);
        if (w->rank[i] != rank) continue;  /* outdated entry */
        int32_t j = w->next[i];
        int32_t k = w->next[j];
        w->tok[i] = N_BYTES + (uint32_t)rank;
        w->next[i] = k;
        if (k != NO_NODE) w->prev[k] = i;
        w->rank[j] = NO_RANK;  /* j is removed from the list */
        set_rank(m, w, i);
        if (w->prev[i] != NO_NODE) set_rank(m, w, w->prev[i]);
    }
    for (int32_t i = 0; i != NO_NODE; i = w->next[i]) {
        if (*count < cap) out[*count] = w->tok[i];
        (*count)++;
    }
    return FRBPE_OK;
}

/* Decode UTF-8 into freshly allocated code point and offset arrays. */
static long decode_text(const char *text, size_t len, uint32_t **cps, size_t **offsets)
{
    *cps = malloc((len + 1) * sizeof **cps);
    *offsets = malloc((len + 1) * sizeof **offsets);
    if (*cps == NULL || *offsets == NULL) return FRBPE_ERR_NOMEM;
    return frbpe_utf8_decode((const unsigned char *)text, len, *cps, *offsets);
}

long frbpe_encode(const frbpe_model *m, const char *text, size_t len, uint32_t *ids, size_t cap)
{
    if (m == NULL || (text == NULL && len > 0) || (ids == NULL && cap > 0)) return FRBPE_ERR_ARG;
    uint32_t *cps;
    size_t *offsets;
    long n = decode_text(text, len, &cps, &offsets);
    workspace w = {0};
    size_t count = 0;
    int rc = n < 0 ? (int)n : FRBPE_OK;
    for (size_t i = 0; rc == FRBPE_OK && i < (size_t)n;) {
        size_t end = frbpe_next_chunk(&m->pattern, cps, (size_t)n, i);
        rc = encode_chunk(m, &w, (const unsigned char *)text + offsets[i],
                          offsets[end] - offsets[i], ids, cap, &count);
        i = end;
    }
    workspace_free(&w);
    free(cps);
    free(offsets);
    if (rc != FRBPE_OK) return rc;
    return count > LONG_MAX ? FRBPE_ERR_ARG : (long)count;
}

long frbpe_decode(const frbpe_model *m, const uint32_t *ids, size_t n, char *out, size_t cap)
{
    if (m == NULL || (ids == NULL && n > 0) || (out == NULL && cap > 0)) return FRBPE_ERR_ARG;
    size_t vocab = N_BYTES + m->n_merges;
    size_t total = 0;
    for (size_t i = 0; i < n; i++) {
        if (ids[i] >= vocab) return FRBPE_ERR_ARG;
        uint32_t len = m->tok_length[ids[i]];
        if (total < cap) {
            size_t room = cap - total;
            memcpy(out + total, m->tok_bytes + m->tok_offset[ids[i]], len < room ? len : room);
        }
        total += len;
    }
    return total > LONG_MAX ? FRBPE_ERR_ARG : (long)total;
}

long frbpe_split(const char *pattern, const char *text, size_t len, size_t *chunk_ends, size_t cap)
{
    frbpe_pattern p;
    if (pattern == NULL || (text == NULL && len > 0) || (chunk_ends == NULL && cap > 0))
        return FRBPE_ERR_ARG;
    if (frbpe_pattern_from_name(pattern, &p) != FRBPE_OK) return FRBPE_ERR_PATTERN;
    uint32_t *cps;
    size_t *offsets;
    long n = decode_text(text, len, &cps, &offsets);
    size_t count = 0;
    for (size_t i = 0; n > 0 && i < (size_t)n;) {
        size_t end = frbpe_next_chunk(&p, cps, (size_t)n, i);
        if (count < cap) chunk_ends[count] = offsets[end];
        count++;
        i = end;
    }
    free(cps);
    free(offsets);
    return n < 0 ? n : (long)count;
}

const char *frbpe_strerror(int code)
{
    switch (code) {
    case FRBPE_OK: return "ok";
    case FRBPE_ERR_IO: return "cannot read model file";
    case FRBPE_ERR_FORMAT: return "invalid model file";
    case FRBPE_ERR_PATTERN: return "unknown pre-tokenization pattern";
    case FRBPE_ERR_NOMEM: return "out of memory";
    case FRBPE_ERR_UTF8: return "input is not valid UTF-8";
    case FRBPE_ERR_ARG: return "invalid argument";
    default: return "unknown error";
    }
}
