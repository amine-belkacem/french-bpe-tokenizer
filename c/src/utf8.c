#include "frbpe.h"
#include "internal.h"

/* Number of continuation bytes after a lead byte, or -1 if it cannot lead. */
static int utf8_tail_length(unsigned char lead)
{
    if (lead < 0x80) return 0;
    if (lead >= 0xC2 && lead <= 0xDF) return 1;  /* C0 and C1 would be overlong */
    if (lead >= 0xE0 && lead <= 0xEF) return 2;
    if (lead >= 0xF0 && lead <= 0xF4) return 3;  /* above F4 is past U+10FFFF */
    return -1;
}

long frbpe_utf8_decode(const unsigned char *s, size_t len, uint32_t *cps, size_t *offsets)
{
    /* Smallest code point that needs 1, 2, 3 or 4 bytes: anything below is overlong. */
    static const uint32_t min_for_length[4] = {0x0, 0x80, 0x800, 0x10000};
    size_t pos = 0;
    long count = 0;

    while (pos < len) {
        int tail = utf8_tail_length(s[pos]);
        if (tail < 0) return FRBPE_ERR_UTF8;
        if ((size_t)tail > len - pos - 1) return FRBPE_ERR_UTF8;  /* truncated */
        /* Payload bits of the lead byte: 5, 4 or 3 bits for 2, 3 or 4 bytes. */
        uint32_t cp = tail == 0 ? s[pos] : (uint32_t)(s[pos] & (0x3F >> tail));
        for (int k = 1; k <= tail; k++) {
            unsigned char c = s[pos + (size_t)k];
            if ((c & 0xC0) != 0x80) return FRBPE_ERR_UTF8;
            cp = (cp << 6) | (uint32_t)(c & 0x3F);
        }
        if (cp < min_for_length[tail] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
            return FRBPE_ERR_UTF8;
        cps[count] = cp;
        offsets[count] = pos;
        count++;
        pos += (size_t)tail + 1;
    }
    offsets[count] = len;
    return count;
}
