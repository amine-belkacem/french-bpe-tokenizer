/*
 * Unit tests of the C library. Expected values are worked out by hand in the
 * comments; the bigger check against Python is tests/test_c_differential.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "frbpe.h"
#include "../src/internal.h"

static int failures = 0;
static int checks = 0;

#define CHECK_EQ_LONG(got, expected)                                                    \
    do {                                                                                \
        long g_ = (long)(got), e_ = (long)(expected);                                   \
        checks++;                                                                       \
        if (g_ != e_) {                                                                 \
            failures++;                                                                 \
            printf("ECHEC %s:%d: %s -> obtenu %ld, attendu %ld\n", __FILE__, __LINE__,  \
                   #got, g_, e_);                                                       \
        }                                                                               \
    } while (0)

static void check_array(const char *what, const long *got, size_t n_got, const long *expected,
                        size_t n_expected, int line)
{
    checks++;
    int same = n_got == n_expected;
    for (size_t i = 0; same && i < n_got; i++) same = got[i] == expected[i];
    if (same) return;
    failures++;
    printf("ECHEC line %d: %s -> obtenu [", line, what);
    for (size_t i = 0; i < n_got; i++) printf(i ? ", %ld" : "%ld", got[i]);
    printf("], attendu [");
    for (size_t i = 0; i < n_expected; i++) printf(i ? ", %ld" : "%ld", expected[i]);
    printf("]\n");
}

static void check_split(const char *pattern, const char *text, const long *expected, size_t n,
                        int line)
{
    size_t ends[64];
    long got = frbpe_split(pattern, text, strlen(text), ends, 64);
    long as_long[64];
    for (long i = 0; i < got && i < 64; i++) as_long[i] = (long)ends[i];
    check_array(text, as_long, got < 0 ? 0 : (size_t)got, expected, n, line);
}

static void test_utf8(void)
{
    uint32_t cps[8];
    size_t offsets[9];
    /* "é" is C3 A9: 0xC3 & 0x1F = 0x03, 0xA9 & 0x3F = 0x29, (0x03 << 6) | 0x29 = 0xE9. */
    CHECK_EQ_LONG(frbpe_utf8_decode((const unsigned char *)"\xC3\xA9", 2, cps, offsets), 1);
    CHECK_EQ_LONG(cps[0], 0xE9);
    CHECK_EQ_LONG(offsets[1], 2);
    /* Rejected like Python's strict codec does. */
    CHECK_EQ_LONG(frbpe_utf8_decode((const unsigned char *)"\xC3", 1, cps, offsets), FRBPE_ERR_UTF8);
    CHECK_EQ_LONG(frbpe_utf8_decode((const unsigned char *)"\xC0\xAF", 2, cps, offsets), FRBPE_ERR_UTF8);
    CHECK_EQ_LONG(frbpe_utf8_decode((const unsigned char *)"\xE0\x80\xAF", 3, cps, offsets), FRBPE_ERR_UTF8);
    CHECK_EQ_LONG(frbpe_utf8_decode((const unsigned char *)"\xED\xA0\x80", 3, cps, offsets), FRBPE_ERR_UTF8);
    CHECK_EQ_LONG(frbpe_utf8_decode((const unsigned char *)"\xF4\x90\x80\x80", 4, cps, offsets), FRBPE_ERR_UTF8);
    CHECK_EQ_LONG(frbpe_utf8_decode((const unsigned char *)"\x80", 1, cps, offsets), FRBPE_ERR_UTF8);
    /* 4-byte emoji U+1F44D: F0 9F 91 8D. */
    CHECK_EQ_LONG(frbpe_utf8_decode((const unsigned char *)"\xF0\x9F\x91\x8D", 4, cps, offsets), 1);
    CHECK_EQ_LONG(cps[0], 0x1F44D);
}

static void test_split(void)
{
    /* GPT-4: "L" | "'école" | " d" | "'été". In bytes: L=1, ' + é(2) + cole(4)
     * = 7, " d" = 2, ' + é(2) + t + é(2) = 6. Ends: 1, 8, 10, 16. */
    static const long gpt4[] = {1, 8, 10, 16};
    check_split("gpt4", "L'\xC3\xA9" "cole d'\xC3\xA9t\xC3\xA9", gpt4, 4, __LINE__);
    /* French: "L'" | "école" | " d'" | "été". Ends: 2, 8, 11, 16. */
    static const long fr[] = {2, 8, 11, 16};
    check_split("fr", "L'\xC3\xA9" "cole d'\xC3\xA9t\xC3\xA9", fr, 4, __LINE__);
    /* "aujourd'hui" stays one chunk: 11 bytes. */
    static const long auj[] = {11};
    check_split("fr", "aujourd'hui", auj, 1, __LINE__);
    /* French numbers: "1" | " 000" | " 000" -> 1, 5, 9. */
    static const long num[] = {1, 5, 9};
    check_split("fr", "1 000 000", num, 3, __LINE__);
    /* Whitespace: "a" | "\n\n" | " " | " b" | "  " -> 1, 3, 4, 6, 8. */
    static const long ws[] = {1, 3, 4, 6, 8};
    check_split("gpt4", "a\n\n  b  ", ws, 5, __LINE__);
    CHECK_EQ_LONG(frbpe_split("klingon", "a", 1, NULL, 0), FRBPE_ERR_PATTERN);
    CHECK_EQ_LONG(frbpe_split("gpt4", "", 0, NULL, 0), 0);
}

static int write_file(const char *path, const char *content)
{
    FILE *f = fopen(path, "wb");
    if (f == NULL) return 0;
    fputs(content, f);
    return fclose(f) == 0;
}

static void test_model(void)
{
    /* Same model as the Python hand example: merges (a,a)->256, (a,b)->257,
     * (256,257)->258, so "aaabdaaabac" encodes to [258, d, 258, a, c]. */
    const char *path = "test_model.tmp";
    if (!write_file(path, "frbpe v1\npattern gpt4\nmerges 3\n97 97\n97 98\n256 257\n")) {
        printf("ECHEC: cannot write %s\n", path);
        failures++;
        return;
    }
    frbpe_model *m = NULL;
    CHECK_EQ_LONG(frbpe_load(path, &m), FRBPE_OK);
    if (m == NULL) return;
    CHECK_EQ_LONG(frbpe_vocab_size(m), 259);

    uint32_t ids[16];
    long n = frbpe_encode(m, "aaabdaaabac", 11, ids, 16);
    long got[16];
    for (long i = 0; i < n && i < 16; i++) got[i] = (long)ids[i];  /* ids < vocab size, fits in long */
    static const long expected[] = {258, 100, 258, 97, 99};
    check_array("encode aaabdaaabac", got, n < 0 ? 0 : (size_t)n, expected, 5, __LINE__);

    /* Buffer too small: total count is still returned, only 2 ids written. */
    uint32_t small[2] = {0, 0};
    CHECK_EQ_LONG(frbpe_encode(m, "aaabdaaabac", 11, small, 2), 5);
    CHECK_EQ_LONG(small[1], 100);

    char text[32];
    long len = frbpe_decode(m, ids, (size_t)n, text, sizeof text);
    CHECK_EQ_LONG(len, 11);
    CHECK_EQ_LONG(memcmp(text, "aaabdaaabac", 11), 0);

    uint32_t bad_id = 259;
    CHECK_EQ_LONG(frbpe_decode(m, &bad_id, 1, text, sizeof text), FRBPE_ERR_ARG);
    CHECK_EQ_LONG(frbpe_encode(m, "\xC3", 1, ids, 16), FRBPE_ERR_UTF8);
    frbpe_free(m);
    remove(path);
}

static void test_load_errors(void)
{
    frbpe_model *m = NULL;
    const char *path = "test_bad.tmp";
    CHECK_EQ_LONG(frbpe_load("does/not/exist.bpe", &m), FRBPE_ERR_IO);

    write_file(path, "not a model\n");
    CHECK_EQ_LONG(frbpe_load(path, &m), FRBPE_ERR_FORMAT);
    write_file(path, "frbpe v1\npattern klingon\nmerges 0\n");
    CHECK_EQ_LONG(frbpe_load(path, &m), FRBPE_ERR_PATTERN);
    /* Merge 0 may only use bytes (ids <= 255): 256 does not exist yet. */
    write_file(path, "frbpe v1\npattern gpt4\nmerges 1\n256 97\n");
    CHECK_EQ_LONG(frbpe_load(path, &m), FRBPE_ERR_FORMAT);
    /* Header announces 2 merges, file has 1. */
    write_file(path, "frbpe v1\npattern gpt4\nmerges 2\n97 97\n");
    CHECK_EQ_LONG(frbpe_load(path, &m), FRBPE_ERR_FORMAT);
    write_file(path, "frbpe v1\npattern gpt4\nmerges 1\n97 -1\n");
    CHECK_EQ_LONG(frbpe_load(path, &m), FRBPE_ERR_FORMAT);
    /* CRLF line endings are accepted, as in Python. */
    write_file(path, "frbpe v1\r\npattern gpt4\r\nmerges 1\r\n97 97\r\n");
    CHECK_EQ_LONG(frbpe_load(path, &m), FRBPE_OK);
    frbpe_free(m);
    remove(path);
}

int main(void)
{
    test_utf8();
    test_split();
    test_model();
    test_load_errors();
    printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
