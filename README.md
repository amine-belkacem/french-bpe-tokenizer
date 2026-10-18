# french-bpe-tokenizer

A byte-level BPE tokenizer that I am writing from scratch, trained on French text,
and measured against the tokenizers of large models (tiktoken, SentencePiece).

Work in progress. This README grows with each milestone; for now it only covers
the data.

## Data

| Split | Source | Documents | Text size |
|---|---|---|---|
| `train_wiki` | French Wikipedia, 2023-11-01 dump, one shard | 36,938 articles | 100.0 MB |
| `test_wiki` | Same shard, held-out articles | 1,989 articles | 5.0 MB |
| `test_gutenberg` | 8 French novels from Project Gutenberg | 351 chunks of ~20k chars | 7.1 MB |

Nothing under `data/` is committed. Two commands rebuild it, and every
downloaded file is checked against a SHA-256 digest:

```bash
python scripts/download_data.py    # ~176 MB
python scripts/prepare_corpus.py   # writes data/processed/*.jsonl + MANIFEST.json
```

A few choices I made, and why:

- **The split is done per article**, using a hash of the article id. Splitting per
  line would put sentences of one article on both sides and make the test set
  look easier than it really is.
- **Text is NFC-normalised.** Otherwise "é" can be stored as one code point or as
  "e" + a combining accent, and the tokenizer would waste vocabulary learning
  both. How each tokenizer copes with NFD input is a separate experiment.
- **The novels are a second test set from another domain.** Training only on
  Wikipedia and testing only on Wikipedia would tell me nothing about how the
  tokenizer behaves on other kinds of French.
- **Gutenberg line breaks are removed inside paragraphs.** The files are wrapped
  at about 70 columns; keeping that would mostly measure newlines.

Known limitations of the data:

- The Hugging Face Wikipedia dump drops the content of templates. Dates and
  some numbers disappear, which leaves sentences like "sa quille est posée le
  par …". It slightly under-represents numbers.
- One shard is not a uniform sample of French Wikipedia; shards follow the
  order of article ids.
- The novels keep their table of contents.

Sources and licences:

- Wikipedia: [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia),
  revision `b04c8d1`, CC BY-SA 3.0 / GFDL.
- Project Gutenberg books 17489, 14155, 5711, 17989, 19657, 798, 4791, 13951
  (Hugo, Flaubert, Zola, Dumas, Stendhal, Verne), public domain.

## The tokenizer (milestone 1)

`src/frbpe/bpe.py` is a byte-level BPE in plain Python: the 256 byte values
are the base vocabulary, so any UTF-8 text can be encoded and nothing is ever
"unknown". Before BPE runs, a regex cuts the text into chunks
(`src/frbpe/pretokenize.py`). For now it is the GPT-4 (`cl100k_base`) pattern,
copied from tiktoken.

```bash
python scripts/train_tokenizer.py --vocab-size 32000 --pattern gpt4
```

```python
from frbpe.bpe import BPETokenizer

tok = BPETokenizer.load("models/gpt4_32k.bpe")
ids = tok.encode("Les élèves mangeaient des pommes.")
assert tok.decode(ids) == "Les élèves mangeaient des pommes."
```

How training stays fast enough in Python: chunks are counted once
(`" maison"` → number of occurrences), then BPE works on the table of unique
chunks. Pair counts are updated only for the chunks touched by each merge, and
the most frequent pair comes from a heap with lazy invalidation. A test checks
that this gives exactly the same merges as a naive version that recounts
everything after each merge.

Ties between pairs with the same count are broken by taking the smallest pair
of ids. Without a fixed rule, two runs could learn different vocabularies.

What I measured so far (one machine, AMD Ryzen, Python 3.12, single process):

| Check | Result |
|---|---|
| Training, 32k vocab, 100 MB of Wikipedia | 148.8 s and 142.3 s (2 runs) |
| Same merges file on both runs | yes, identical SHA-256 |
| `decode(encode(x)) == x` on all 2,340 test documents | 0 failures |

Only 31,744 merges are learned for a 32k vocabulary: 256 ids are the raw bytes.

The first thing the trained model shows is the problem the next milestone deals
with. With the GPT-4 regex, the apostrophe of a French elision is glued to the
*next* word:

```
L | 'école |  d | 'été |  est |  aujourd | 'hui |  fermée | .
```

So `'école`, `'été` and `'hui` become tokens of their own, and the same word
gets a different token depending on what precedes it (`école` vs `'école`).

## Tests

```bash
pip install -r requirements.txt
pytest
```

The tests never touch `data/`; they build tiny corpora whose expected merges
are worked out by hand in the comments (see `tests/test_bpe.py`).
