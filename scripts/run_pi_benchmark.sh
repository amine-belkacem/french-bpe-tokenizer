#!/usr/bin/env bash
# Milestone 5 on a Raspberry Pi 5 (or any Linux machine): set everything up
# and run the same benchmark as on the PC.
#
#   git clone https://github.com/amine-belkacem/french-bpe-tokenizer.git
#   cd french-bpe-tokenizer
#   bash scripts/run_pi_benchmark.sh
#
# Nothing is installed system-wide: uv goes into .tools/, Python 3.12 and the
# pinned packages into .venv/. It downloads about 185 MB (corpus + reference
# tokenizers) and checks that the test set is byte-identical to the one used
# on the PC before measuring anything.
set -euo pipefail

TAG="${1:-pi5}"
UV_VERSION="0.12.21"
PYTHON_VERSION="3.12"
# SHA-256 of data/processed/test_wiki.jsonl as built on the PC.
EXPECTED_TEST_SHA="0b31c9205710e89ef6e6cc36849f5d37ac5598d251e48593c32254b8bb584412"

cd "$(dirname "$0")/.."

for tool in git gcc make python3 sha256sum; do
    command -v "$tool" >/dev/null || { echo "missing: $tool (sudo apt install build-essential python3-venv git)"; exit 1; }
done

echo "== 1/5 Python $PYTHON_VERSION environment"
if [ ! -x .tools/bin/uv ]; then
    python3 -m venv .tools
    .tools/bin/pip install --quiet "uv==$UV_VERSION"
fi
[ -x .venv/bin/python ] || .tools/bin/uv venv --python "$PYTHON_VERSION" .venv
.tools/bin/uv pip install --quiet --python .venv/bin/python -r requirements.txt
PY=.venv/bin/python
export PYTHONPATH="src:scripts:tests"

echo "== 2/5 Data (same files and checksums as on the PC)"
$PY scripts/download_data.py
$PY scripts/prepare_corpus.py
$PY scripts/download_reference.py
actual=$(sha256sum data/processed/test_wiki.jsonl | cut -d' ' -f1)
if [ "$actual" != "$EXPECTED_TEST_SHA" ]; then
    echo "test set differs from the PC one: $actual"; exit 1
fi

echo "== 3/5 C encoder"
make -C c clean >/dev/null
make -C c test

echo "== 4/5 C must still match Python here"
$PY -m pytest -q tests/test_c_differential.py

echo "== 5/5 Benchmark (about 15 minutes, do not use the Pi meanwhile)"
governor=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
[ -r "$governor" ] && echo "cpu governor: $(cat "$governor")"
$PY scripts/benchmark.py --tag "$TAG"

echo
echo "Done. Copy these two files back into results/ on the PC:"
echo "  results/speed_${TAG}.csv"
echo "  results/speed_${TAG}_machine.json"
