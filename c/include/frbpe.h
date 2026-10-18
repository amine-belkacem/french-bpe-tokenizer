/*
 * frbpe: byte-level BPE encoder in C99, compatible with the Python reference
 * (src/frbpe). Loads a model saved by BPETokenizer.save() and produces the
 * same token ids, which tests/test_c_differential.py checks.
 *
 * No dependency besides the C standard library. The pre-tokenizer is written
 * by hand instead of using a regex library; Unicode classes come from tables
 * generated with the same Python regex module as the reference.
 *
 * Input text must be valid UTF-8. Thread safety: a loaded model is read-only,
 * so several threads may encode with the same model at the same time.
 */
#ifndef FRBPE_H
#define FRBPE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    FRBPE_OK = 0,
    FRBPE_ERR_IO = -1,          /* cannot open or read the model file */
    FRBPE_ERR_FORMAT = -2,      /* model file is not a valid frbpe v1 file */
    FRBPE_ERR_PATTERN = -3,     /* unknown pre-tokenization pattern name */
    FRBPE_ERR_NOMEM = -4,       /* allocation failed */
    FRBPE_ERR_UTF8 = -5,        /* input text is not valid UTF-8 */
    FRBPE_ERR_ARG = -6          /* NULL pointer or out-of-range token id */
};

typedef struct frbpe_model frbpe_model;

/* Load a model file. On success *out owns the model; free it with frbpe_free. */
int frbpe_load(const char *path, frbpe_model **out);
void frbpe_free(frbpe_model *model);

size_t frbpe_vocab_size(const frbpe_model *model);
const char *frbpe_pattern_name(const frbpe_model *model);

/*
 * Encode `len` bytes of UTF-8 text. Writes at most `cap` ids into `ids` and
 * returns the total number of tokens, which may be larger than `cap`: call
 * again with a bigger buffer in that case. Returns a negative FRBPE_ERR_*
 * code on error. `ids` may be NULL when `cap` is 0, to only count tokens.
 */
long frbpe_encode(const frbpe_model *model, const char *text, size_t len,
                  uint32_t *ids, size_t cap);

/*
 * Decode token ids into bytes. Same convention: writes at most `cap` bytes
 * and returns the total size, or a negative FRBPE_ERR_* code.
 */
long frbpe_decode(const frbpe_model *model, const uint32_t *ids, size_t n,
                  char *out, size_t cap);

/*
 * Pre-tokenization only, exposed for testing: writes the byte offset where
 * each chunk ends (at most `cap` of them) and returns the number of chunks.
 */
long frbpe_split(const char *pattern, const char *text, size_t len,
                 size_t *chunk_ends, size_t cap);

const char *frbpe_strerror(int code);

#ifdef __cplusplus
}
#endif

#endif /* FRBPE_H */
