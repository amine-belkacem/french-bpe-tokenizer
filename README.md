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

## Does a French regex help? (milestone 2)

I wrote three French rules (`src/frbpe/pretokenize.py`), each replacing one
piece of the GPT-4 pattern:

| Rule | GPT-4 regex | French regex |
|---|---|---|
| elision | `L` `'école` | `L'` `école` |
| numbers | `1` `␣` `000` | `1` `␣000` |
| no-break space before `: ; ! ?` | `Prix` `⍽` `:` | `Prix` `⍽:` |

Then I trained one tokenizer per rule, plus one with all of them, all on the
same 100 MB of Wikipedia at 32k, and compared token counts on the two test
sets. The comparison is paired: same documents for both tokenizers, and a
bootstrap over documents (1000 resamples) for the 95 % interval.

![Effect of each rule](figures/ablation_rules_32k.png)

| Change in token count vs GPT-4 regex, 32k | Wikipedia (test) | Novels (Gutenberg) |
|---|---|---|
| elision only | −0.41 % [−0.45, −0.37] | −0.61 % [−0.65, −0.57] |
| numbers only | −2.02 % [−2.15, −1.90] | +0.03 % [+0.02, +0.04] |
| no-break space only | −0.03 % [−0.04, −0.02] | +0.14 % [+0.12, +0.15] |
| **all French rules** | **−2.45 % [−2.57, −2.33]** | **−0.41 % [−0.44, −0.37]** |

The honest summary: the French regex helps, the effect is real (every
interval is far from zero), but it is small. I expected more from the elision
rule. Here is what I understand of the numbers:

- **Numbers carry most of the gain on Wikipedia.** GPT-4 spends a whole token
  on the space before each number (`le` `␣` `14`). Wikipedia is full of dates,
  the novels are not, so the rule does nothing there.
- **Elision matters more in novels**, which have more dialogue and more
  `qu'`, `j'`, `n'`. It is still below 1 %, because BPE copes with the GPT-4
  split on its own: it simply learns `'école` as an extra token. I counted
  them: the GPT-4 32k vocabulary has **1,384 tokens that start with an
  apostrophe**, the French one has 10. The French regex frees those slots, but
  they go to rarer words, which do not save much.
- **The no-break space rule makes the novels slightly worse.** The novels
  contain no no-break space at all; I checked that this rule changes the
  chunking of 0 out of 351 novel documents. The +0.14 % comes from the
  vocabulary alone: slots spent on `⍽:` or `⍽!` learned from Wikipedia are
  useless on Gutenberg.
- **Rules do not add up.** All rules together save less on the novels
  (−0.41 %) than elision alone (−0.61 %), for the same reason: the vocabulary
  is a fixed budget, and what one rule takes, another cannot use.

![Fertility against vocabulary size](figures/fertility_vs_vocab.png)

Across vocabulary sizes, the French regex stays below GPT-4's on both test
sets. On Wikipedia the gap is stable (−2.3 % to −2.5 %), on the novels it
shrinks as the vocabulary grows (−1.03 % at 8k, −0.41 % at 32k). The
intervals of the two curves overlap on this figure, yet the differences are
significant: the per-tokenizer intervals include the variation between
documents, which the paired comparison cancels out.

| Fertility (tokens / word), 32k | Wikipedia | Novels |
|---|---|---|
| GPT-4 regex | 1.476 [1.461, 1.491] | 1.456 [1.450, 1.463] |
| French regex | 1.440 [1.426, 1.454] | 1.450 [1.444, 1.457] |

Full tables: `results/ablation_fertility.csv` and `results/ablation_vs_gpt4.csv`
(both open in a spreadsheet). To rebuild them:

```bash
python scripts/train_tokenizer.py --pattern fr --vocab-size 32000   # and the others
python scripts/evaluate_ablation.py
```

Limits of this experiment:

- A word here is a run of letters and digits, so `l'école` counts as 2 words.
  Absolute fertilities depend on that rule; the relative changes above do not
  (the word count cancels out when two tokenizers are compared).
- One training corpus, one seed. BPE training is deterministic, so a second
  run would give the same numbers, but a different Wikipedia sample could move
  them a little.
- The training times written in `models/*.json` for this milestone were taken
  while other heavy jobs were running on the machine. They are not a speed
  benchmark; that comes in milestone 5.

## Tests

```bash
pip install -r requirements.txt
pytest
```

The tests never touch `data/`; they build tiny corpora whose expected merges
are worked out by hand in the comments (see `tests/test_bpe.py`).
