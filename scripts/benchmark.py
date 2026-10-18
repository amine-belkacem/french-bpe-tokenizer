"""Milestone 5: encoding speed and memory, one tokenizer per process.

    python scripts/benchmark.py --tag pc
    python scripts/benchmark.py --tag pi5

Method:
* Input: every document of data/processed/test_wiki.jsonl (5 MB). Loaded in
  memory before timing; model loading is timed separately.
* One untimed warm-up pass, then --repeats timed passes (default 10). Each
  pass encodes all documents one by one, on a single thread.
* Each tokenizer runs in its own child process, so peak memory is not mixed
  up between tokenizers. A "baseline" child loads the corpus and nothing
  else; for Python tokenizers I also report peak memory minus that baseline.
* frbpe (Python) keeps a cache of encoded chunks. A fresh tokenizer is built
  before each pass, so every pass starts with an empty cache, like a single
  pass over new text would.
* The machine description (CPU, governor, temperature before and after) is
  saved with the results, because a throttling Raspberry Pi would otherwise
  look slower for the wrong reason.

Writes results/speed_<tag>.csv and results/speed_<tag>_machine.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data/processed/test_wiki.jsonl"
RECORDS = ROOT / "data/processed/test_wiki.records"
MODELS = ROOT / "models"
REFERENCE = ROOT / "data/reference"
C_BENCH = ROOT / "c" / "build" / ("frbpe_bench.exe" if os.name == "nt" else "frbpe_bench")
DEFAULT_REPEATS = 10

# name -> (kind, argument). Kinds are handled in run_worker().
TOKENIZERS = {
    "baseline": ("baseline", ""),
    "frbpe-python fr_32k": ("frbpe", "fr_32k.bpe"),
    "frbpe-c fr_32k": ("c", "fr_32k.bpe"),
    "frbpe-c gpt4_32k": ("c", "gpt4_32k.bpe"),
    "tiktoken cl100k_base": ("tiktoken", "cl100k_base"),
    "tiktoken o200k_base": ("tiktoken", "o200k_base"),
    "sentencepiece bpe 32k": ("sentencepiece", "sp_bpe_32k.model"),
    "sentencepiece unigram 32k": ("sentencepiece", "sp_unigram_32k.model"),
}


# ------------------------------------------------------------ child process
def peak_rss_bytes() -> int:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")
            ]

        # Without explicit types, ctypes passes the process handle as a 32-bit
        # int and truncates it on 64-bit Windows (the first version read 0 MB).
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if not get_info(get_process(), ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    # Linux: VmHWM, not ru_maxrss. ru_maxrss survives execve, so a child
    # started by this script inherits the parent's peak if it is higher (the
    # first Pi run showed the same value for the C tool and the baseline).
    for line in (read_text("/proc/self/status") or "").splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def load_docs() -> list[str]:
    with CORPUS.open(encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def run_worker(kind: str, arg: str, repeats: int) -> dict:
    docs = load_docs()
    n_bytes = sum(len(d.encode("utf-8")) for d in docs)
    if kind == "baseline":
        return {"load_s": 0.0, "times_s": [], "bytes": n_bytes, "tokens": 0,
                "peak_rss_bytes": peak_rss_bytes()}

    start = time.perf_counter()
    if kind == "frbpe":
        sys.path.insert(0, str(ROOT / "src"))
        from frbpe.bpe import BPETokenizer

        loaded = BPETokenizer.load(MODELS / arg)
        make = lambda: BPETokenizer(loaded.merges, loaded.pretokenizer.name)  # noqa: E731
    elif kind == "tiktoken":
        os.environ["TIKTOKEN_CACHE_DIR"] = str(REFERENCE / "tiktoken")
        import tiktoken

        enc = tiktoken.get_encoding(arg)
        make = lambda: enc  # noqa: E731
    elif kind == "sentencepiece":
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=str(MODELS / arg))
        make = lambda: sp  # noqa: E731
    else:
        raise ValueError(f"unknown kind {kind}")
    load_s = time.perf_counter() - start

    def encode_all(tok) -> int:
        if kind == "frbpe":
            return sum(len(tok.encode(d)) for d in docs)
        if kind == "tiktoken":
            return sum(len(tok.encode_ordinary(d)) for d in docs)
        return sum(len(tok.encode(d, out_type=int)) for d in docs)

    tokens = encode_all(make())  # warm-up
    times = []
    for _ in range(repeats):
        tok = make()  # outside the timed region
        t0 = time.perf_counter()
        again = encode_all(tok)
        times.append(time.perf_counter() - t0)
        if again != tokens:
            raise RuntimeError("token count changed between passes")
    return {"load_s": load_s, "times_s": times, "bytes": n_bytes, "tokens": tokens,
            "peak_rss_bytes": peak_rss_bytes()}


# ----------------------------------------------------------- parent process
def write_records() -> None:
    with RECORDS.open("wb") as f:
        for doc in load_docs():
            data = doc.encode("utf-8")
            f.write(struct.pack("<I", len(data)) + data)


def single_thread_env() -> dict:
    env = dict(os.environ)
    for var in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = "1"
    return env


def run_child(name: str, repeats: int) -> dict:
    kind, arg = TOKENIZERS[name]
    if kind == "c":
        cmd = [str(C_BENCH), str(MODELS / arg), str(RECORDS), str(repeats)]
    else:
        cmd = [sys.executable, __file__, "--worker", kind, "--arg", arg, "--repeats", str(repeats)]
    out = subprocess.run(cmd, capture_output=True, text=True, env=single_thread_env(), check=False)
    if out.returncode != 0:
        raise RuntimeError(f"{name} failed: {out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def cpu_name() -> str:
    if os.name == "nt":
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    model = read_text("/proc/device-tree/model")  # Raspberry Pi board name
    if model:
        return model.rstrip("\x00")
    for line in (read_text("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def temperature_c() -> float | None:
    raw = read_text("/sys/class/thermal/thermal_zone0/temp")
    return int(raw) / 1000 if raw and raw.isdigit() else None


def machine_info() -> dict:
    import sentencepiece
    import tiktoken

    gcc_path = shutil.which("gcc")
    gcc = subprocess.run([gcc_path or "gcc", "--version"], capture_output=True, text=True,
                         check=False) if gcc_path else None
    return {
        "cpu": cpu_name(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "tiktoken": tiktoken.__version__,
        "sentencepiece": sentencepiece.__version__,
        "gcc": gcc.stdout.splitlines()[0] if gcc and gcc.returncode == 0 else None,
        "cpu_governor": read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cpu_max_freq_khz": read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"),
        "logical_cpus": os.cpu_count(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", help="machine name used in the output file names")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--only", nargs="*", help="subset of tokenizer names")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--arg", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        print(json.dumps(run_worker(args.worker, args.arg, args.repeats)))
        return 0
    if not args.tag:
        parser.error("--tag is required")
    if not C_BENCH.exists():
        parser.error(f"{C_BENCH} not found, run `make -C c` first")
    write_records()

    info = machine_info()
    info["temperature_start_c"] = temperature_c()
    info["repeats"] = args.repeats
    info["corpus"] = str(CORPUS.relative_to(ROOT).as_posix())
    started = time.time()
    rows = []
    baseline_rss = None
    for name in args.only or list(TOKENIZERS):
        res = run_child(name, args.repeats)
        if name == "baseline":
            baseline_rss = res["peak_rss_bytes"]
            continue
        mb_per_s = [res["bytes"] / t / 1e6 for t in res["times_s"]]
        kind = TOKENIZERS[name][0]
        rows.append({
            "tokenizer": name,
            "repeats": len(res["times_s"]),
            "mean_s": statistics.mean(res["times_s"]),
            "std_s": statistics.stdev(res["times_s"]),
            "mb_per_s": statistics.mean(mb_per_s),
            "mb_per_s_std": statistics.stdev(mb_per_s),
            "tokens_per_s": res["tokens"] / statistics.mean(res["times_s"]),
            "tokens": res["tokens"],
            "bytes": res["bytes"],
            "load_s": res["load_s"],
            "peak_rss_mb": res["peak_rss_bytes"] / 1e6,
            "rss_over_baseline_mb": (
                (res["peak_rss_bytes"] - baseline_rss) / 1e6
                if kind != "c" and baseline_rss is not None else None
            ),
        })
        r = rows[-1]
        print(f"{name:28s} {r['mb_per_s']:8.2f} ± {r['mb_per_s_std']:.2f} MB/s  "
              f"{r['tokens_per_s'] / 1e6:6.2f} M tok/s  peak {r['peak_rss_mb']:7.1f} MB")
    info["temperature_end_c"] = temperature_c()
    # A Pi that throttles lowers its frequency: compare with cpu_max_freq_khz.
    info["cpu_freq_end_khz"] = read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    info["duration_s"] = round(time.time() - started, 1)
    info["baseline_python_rss_mb"] = baseline_rss / 1e6 if baseline_rss else None

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    with (results / f"speed_{args.tag}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (results / f"speed_{args.tag}_machine.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
