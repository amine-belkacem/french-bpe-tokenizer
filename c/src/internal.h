/* Declarations shared by the C sources, not part of the public API. */
#ifndef FRBPE_INTERNAL_H
#define FRBPE_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

/*
 * A pre-tokenization pattern is the GPT-4 pattern with up to three pieces
 * replaced by their French version (see src/frbpe/pretokenize.py).
 */
typedef struct {
    int elision;    /* French elision instead of the English contraction */
    int fr_number;  /* number may start with a (no-break) space */
    int fr_punct;   /* punctuation may start with a no-break space */
} frbpe_pattern;

int frbpe_pattern_from_name(const char *name, frbpe_pattern *out);

/*
 * Decode UTF-8 into code points. cps must hold `len` entries and offsets
 * `len + 1`; offsets[i] is the byte position of code point i and
 * offsets[count] == len. Returns the number of code points or FRBPE_ERR_UTF8.
 * Rejects what Python's strict UTF-8 codec rejects: truncated sequences,
 * overlong forms, surrogates and values above U+10FFFF.
 */
long frbpe_utf8_decode(const unsigned char *s, size_t len, uint32_t *cps, size_t *offsets);

/* Index just past the chunk that starts at code point `i` (always > i). */
size_t frbpe_next_chunk(const frbpe_pattern *p, const uint32_t *cp, size_t n, size_t i);

#endif
