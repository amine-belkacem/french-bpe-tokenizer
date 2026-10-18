/*
 * Hand-written equivalent of the regex patterns in src/frbpe/pretokenize.py.
 *
 * A regex alternation tries its branches in order at the current position
 * and keeps the first one that matches. Each match_* function below is one
 * branch: it returns the length of its match at position i, or 0. The
 * comments give the regex piece each function stands for.
 *
 * Possessive quantifiers (?+, ++) never give characters back, so they are
 * plain loops. The few places where the regex can backtrack are written out
 * explicitly and commented. tests/test_c_differential.py checks the result
 * against the Python regex on random and real text.
 */
#include <string.h>

#include "frbpe.h"
#include "internal.h"
#include "unicode_tables.h"

#define CP_LF 0x0Au
#define CP_CR 0x0Du
#define CP_SPACE 0x20u
#define CP_APOSTROPHE 0x27u
#define CP_NBSP 0xA0u
#define CP_RIGHT_QUOTE 0x2019u
#define CP_NARROW_NBSP 0x202Fu
#define GPT4_MAX_DIGITS 3

static int in_ranges(uint32_t cp, const frbpe_range *table, size_t len)
{
    size_t lo = 0, hi = len;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < table[mid].first) hi = mid;
        else if (cp > table[mid].last) lo = mid + 1;
        else return 1;
    }
    return 0;
}

/* Fast path for ASCII, tables for the rest. */
static int is_letter(uint32_t c)
{
    if (c < 0x80) return (c | 0x20) >= 'a' && (c | 0x20) <= 'z';
    return in_ranges(c, FRBPE_LETTER, FRBPE_LETTER_LEN);
}
static int is_number(uint32_t c)
{
    if (c < 0x80) return c >= '0' && c <= '9';
    return in_ranges(c, FRBPE_NUMBER, FRBPE_NUMBER_LEN);
}
static int is_space(uint32_t c) { return in_ranges(c, FRBPE_SPACE, FRBPE_SPACE_LEN); }
static int is_word(uint32_t c) { return in_ranges(c, FRBPE_WORD, FRBPE_WORD_LEN); }
static int is_newline(uint32_t c) { return c == CP_CR || c == CP_LF; }
static int is_apostrophe(uint32_t c) { return c == CP_APOSTROPHE || c == CP_RIGHT_QUOTE; }
static int is_fr_space(uint32_t c) { return c == CP_SPACE || c == CP_NBSP || c == CP_NARROW_NBSP; }

/* [^\r\n\p{L}\p{N}] : optional prefix of a word */
static int is_word_prefix(uint32_t c) { return !is_newline(c) && !is_letter(c) && !is_number(c); }
/* [^\s\p{L}\p{N}] : punctuation and symbols */
static int is_punct(uint32_t c) { return !is_space(c) && !is_letter(c) && !is_number(c); }

/*
 * The ASCII letter that (?i:x) would match `c` against, or 0. In ASCII that is
 * just the lowercase letter (the generator checks it); outside ASCII only the
 * few code points of FRBPE_FOLD_EXTRA qualify (İ, ſ, the Kelvin sign).
 */
static char fold_to_ascii(uint32_t c)
{
    if (c < 0x80) {
        uint32_t lower = c | 0x20u;
        return (lower >= 'a' && lower <= 'z') ? (char)lower : 0;
    }
    for (size_t k = 0; k < FRBPE_FOLD_EXTRA_LEN; k++)
        if (FRBPE_FOLD_EXTRA[k].cp == c) return FRBPE_FOLD_EXTRA[k].letter;
    return 0;
}

static int ci_char(uint32_t c, char lower) { return fold_to_ascii(c) == lower; }

typedef struct {
    const char *text;  /* lowercase ASCII */
    size_t len;
} ascii_word;

#define WORD(s) {s, sizeof(s) - 1}

/* (?i:word) at position i: returns the number of code points matched or 0. */
static size_t ci_word(const uint32_t *cp, size_t n, size_t i, const ascii_word *word)
{
    if (word->len > n - i) return 0;
    for (size_t k = 0; k < word->len; k++)
        if (!ci_char(cp[i + k], word->text[k])) return 0;
    return word->len;
}

/* '(?i:[sdmt]|ll|ve|re) */
static size_t match_contraction(const uint32_t *cp, size_t n, size_t i)
{
    static const ascii_word two_letters[] = {WORD("ll"), WORD("ve"), WORD("re")};
    if (cp[i] != CP_APOSTROPHE || i + 1 >= n) return 0;
    char c = fold_to_ascii(cp[i + 1]);
    if (c == 's' || c == 'd' || c == 'm' || c == 't') return 2;
    for (size_t k = 0; k < sizeof two_letters / sizeof two_letters[0]; k++)
        if (ci_word(cp, n, i + 1, &two_letters[k])) return 3;
    return 0;
}

/* Optional possessive prefix [^\r\n\p{L}\p{N}]?+ : taken whenever present. */
static size_t skip_word_prefix(const uint32_t *cp, size_t n, size_t i)
{
    return (i < n && is_word_prefix(cp[i])) ? i + 1 : i;
}

/* [^\r\n\p{L}\p{N}]?+(?i:aujourd['’]hui)\b */
static size_t match_aujourdhui(const uint32_t *cp, size_t n, size_t i)
{
    static const ascii_word aujourd = WORD("aujourd"), hui = WORD("hui");
    size_t j = skip_word_prefix(cp, n, i);
    size_t k = ci_word(cp, n, j, &aujourd);
    if (k == 0) return 0;
    j += k;
    if (j >= n || !is_apostrophe(cp[j])) return 0;
    j++;
    k = ci_word(cp, n, j, &hui);
    if (k == 0) return 0;
    j += k;
    /* \b: word character before, non-word character (or end) after. */
    int before = is_word(cp[j - 1]);
    int after = j < n && is_word(cp[j]);
    return before != after ? j - i : 0;
}

/*
 * [^\r\n\p{L}\p{N}]?+(?i:jusqu|lorsqu|puisqu|quoiqu|presqu|quelqu|qu|[cdjlmnst])['’](?=\p{L})
 * The alternation is the one place that backtracks: if a word matches but no
 * apostrophe follows, the next word of the list is tried, in the same order.
 *
 * This branch runs at the start of every chunk, so it has to reject fast:
 * a word is only tried if its first letter matches. Most chunks start with a
 * letter that begins no elided word and leave after one fold_to_ascii call.
 * (The first version tried all 15 words each time and made the whole French
 * encoder about 35 % slower than the GPT-4 one.)
 */
static size_t match_elision(const uint32_t *cp, size_t n, size_t i)
{
    static const ascii_word words[] = {
        WORD("jusqu"), WORD("lorsqu"), WORD("puisqu"), WORD("quoiqu"), WORD("presqu"),
        WORD("quelqu"), WORD("qu"), WORD("c"), WORD("d"), WORD("j"), WORD("l"),
        WORD("m"), WORD("n"), WORD("s"), WORD("t"),
    };
    size_t j = skip_word_prefix(cp, n, i);
    if (j >= n) return 0;
    char first = fold_to_ascii(cp[j]);
    if (first == 0) return 0;
    for (size_t w = 0; w < sizeof words / sizeof words[0]; w++) {
        if (words[w].text[0] != first) continue;
        size_t k = ci_word(cp, n, j, &words[w]);
        if (k == 0) continue;
        size_t a = j + k;  /* position of the apostrophe */
        if (a + 1 < n && is_apostrophe(cp[a]) && is_letter(cp[a + 1])) return a + 1 - i;
    }
    return 0;
}

/* [^\r\n\p{L}\p{N}]?+\p{L}++ */
static size_t match_word(const uint32_t *cp, size_t n, size_t i)
{
    size_t j = skip_word_prefix(cp, n, i);
    size_t k = j;
    while (k < n && is_letter(cp[k])) k++;
    return k > j ? k - i : 0;
}

/* GPT-4: \p{N}{1,3}+    French: [   ]?\p{N}{1,3}+ */
static size_t match_number(const uint32_t *cp, size_t n, size_t i, int french)
{
    size_t j = i;
    if (french && is_fr_space(cp[j])) j++;
    size_t k = j;
    while (k < n && k - j < GPT4_MAX_DIGITS && is_number(cp[k])) k++;
    /* If the optional space was taken but no digit follows, backtracking to
     * "no space" cannot help: a space is not a digit. */
    return k > j ? k - i : 0;
}

/* GPT-4: ' ?[^\s\p{L}\p{N}]++[\r\n]*+'   French: same with [   ]? */
static size_t match_punct(const uint32_t *cp, size_t n, size_t i, int french)
{
    size_t j = i;
    if (french ? is_fr_space(cp[j]) : cp[j] == CP_SPACE) j++;
    size_t k = j;
    while (k < n && is_punct(cp[k])) k++;
    if (k == j) return 0;  /* same remark as for numbers: no backtracking helps */
    while (k < n && is_newline(cp[k])) k++;
    return k - i;
}

/* \s++$ | \s*[\r\n] | \s+(?!\S) | \s */
static size_t match_whitespace(const uint32_t *cp, size_t n, size_t i)
{
    if (!is_space(cp[i])) return 0;
    size_t end = i;
    while (end < n && is_space(cp[end])) end++;
    /* \s++$ : the run reaches the end of the text. */
    if (end == n) return end - i;
    /* \s*[\r\n] : greedy, then backtracks to the last newline of the run. */
    for (size_t k = end; k > i; k--)
        if (is_newline(cp[k - 1])) return k - i;
    /* \s+(?!\S) : the run is followed by a non-space, so give one back. */
    if (end - i >= 2) return end - i - 1;
    /* \s */
    return 1;
}

size_t frbpe_next_chunk(const frbpe_pattern *p, const uint32_t *cp, size_t n, size_t i)
{
    size_t len;
    if (p->elision) {
        if ((len = match_aujourdhui(cp, n, i)) != 0) return i + len;
        if ((len = match_elision(cp, n, i)) != 0) return i + len;
    } else if ((len = match_contraction(cp, n, i)) != 0) {
        return i + len;
    }
    if ((len = match_word(cp, n, i)) != 0) return i + len;
    if ((len = match_number(cp, n, i, p->fr_number)) != 0) return i + len;
    if ((len = match_punct(cp, n, i, p->fr_punct)) != 0) return i + len;
    if ((len = match_whitespace(cp, n, i)) != 0) return i + len;
    /* Unreachable: every code point is a letter, a number, a space or
     * punctuation. Emitting one code point keeps the encoder lossless anyway,
     * and the differential test would report the difference. */
    return i + 1;
}

int frbpe_pattern_from_name(const char *name, frbpe_pattern *out)
{
    static const struct { const char *name; frbpe_pattern p; } known[] = {
        {"gpt4", {0, 0, 0}},
        {"fr_elision", {1, 0, 0}},
        {"fr_number", {0, 1, 0}},
        {"fr_punct", {0, 0, 1}},
        {"fr", {1, 1, 1}},
    };
    for (size_t k = 0; k < sizeof known / sizeof known[0]; k++) {
        if (strcmp(name, known[k].name) == 0) {
            *out = known[k].p;
            return FRBPE_OK;
        }
    }
    return FRBPE_ERR_PATTERN;
}
