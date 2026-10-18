#!/usr/bin/env bash
# Rebuild every table and figure of the README from scratch.
#
#   bash scripts/reproduce.sh            use the committed models (about 15 min)
#   bash scripts/reproduce.sh --train    retrain the 9 BPE models and the 2
#                                        SentencePiece models first (+ ~30 min)
#
# Speed numbers are not re-measured here: they depend on the machine. Run
# `python scripts/benchmark.py --tag <machine>` on a quiet machine for that
# (or scripts/run_pi_benchmark.sh on a Raspberry Pi).
#
# Expects the Python packages of requirements.txt and gcc + make on the PATH.
# Everything is deterministic (fixed bootstrap seed, deterministic BPE), so
# with the committed models `git status` should show no change afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
export PYTHONPATH="src:scripts:tests"
[ "${OS:-}" = "Windows_NT" ] && export PYTHONPATH="src;scripts;tests"

TRAIN=0
[ "${1:-}" = "--train" ] && TRAIN=1

step() { echo; echo "== $*"; }

step "Data: download (checked by SHA-256) and split"
$PY scripts/download_data.py
$PY scripts/prepare_corpus.py
$PY scripts/download_reference.py

if [ "$TRAIN" = 1 ]; then
    step "Training 9 BPE tokenizers"
    for spec in "gpt4 8000" "gpt4 16000" "gpt4 32000" "fr 8000" "fr 16000" "fr 32000" \
                "fr_elision 32000" "fr_number 32000" "fr_punct 32000"; do
        set -- $spec
        $PY scripts/train_tokenizer.py --pattern "$1" --vocab-size "$2"
    done
    step "Training SentencePiece BPE and Unigram"
    $PY scripts/train_sentencepiece.py
fi

step "Tests (Python and C)"
make -C c test
$PY -m pytest -q

step "Milestone 2: French regex ablation"
$PY scripts/evaluate_ablation.py

step "Milestone 3: comparison with existing tokenizers"
$PY scripts/compare_tokenizers.py

step "Milestone 4: C encoder against Python on every test document"
$PY scripts/diff_c_python.py

step "Milestone 5: speed figures from the saved measurements"
$PY scripts/plot_speed.py

step "Milestone 6: robustness"
$PY scripts/robustness.py

echo
echo "Done. With the committed models, 'git status' should list no modified file."
