"""Build a one-page HTML dashboard from benchmark and profiling results.

Runs anywhere with Python (no NPU needed). From the repository root:

    python experiments/profiling/make_dashboard.py --auto --output profile_dashboard.html

--auto picks the newest results/*/repeated-*/summary.json and every
results/**/*.trace.json (the newest one per model/batch/mode wins).
GitHub Actions uses this mode (.github/workflows/dashboard.yml).

Or choose the files yourself:

    python experiments/profiling/make_dashboard.py \
        --summary results/2026-11-19/repeated-20261119T065641Z/summary.json \
        --traces profiling-smallcnn/*.trace.json profiling-resnet/*.trace.json \
        --output profile_dashboard.html

--summary   summary.json from summarize_benchmarks.py (latency numbers)
--traces    optional .trace.json files from profile_modes.py, named
            <model>_b<batch>_<mode>.trace.json (time breakdown)

Open the HTML file in a browser. It is self-contained (no internet needed).
"""
import argparse
import html
import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_trace import clip, merge  # noqa: E402

DEVICE_WORDS = ("npu", "furiosa", "device", "privateuse1", "kernel")
NAME_RE = re.compile(r"(?P<model>.+)_b(?P<batch>\d+)_(?P<mode>eager|compile)\.trace\.json$")


def breakdown(path):
    """Per-iteration ms split into NPU busy / host-only / idle."""
    data = json.loads(Path(path).read_text())
    events = data["traceEvents"] if isinstance(data, dict) else data
    names = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") in ("process_name", "thread_name"):
            key = (e.get("pid"), e.get("tid")) if e["name"] == "thread_name" else (e.get("pid"), None)
            names[key] = str(e.get("args", {}).get("name", ""))
    spans = [e for e in events if e.get("ph") == "X" and "dur" in e]
    iters = sorted((e["ts"], e["ts"] + e["dur"]) for e in spans if str(e.get("name", "")).startswith("iter_"))
    if not spans or not iters:
        return None

    device, host = [], []
    for e in spans:
        if str(e.get("name", "")).startswith("iter_"):
            continue
        key = (e.get("pid"), e.get("tid"))
        text = f"{names.get((key[0], None), '')} {names.get(key, '')} {e.get('cat', '')}".lower()
        (device if any(w in text for w in DEVICE_WORDS) else host).append((e["ts"], e["ts"] + e["dur"]))

    npu = host_only = window = 0.0
    for lo, hi in iters:
        d = merge(clip(device, lo, hi))
        h = merge(clip(host, lo, hi))
        d_busy = sum(b - a for a, b in d)
        both = merge([*map(list, d), *map(list, h)])
        npu += d_busy
        host_only += sum(b - a for a, b in both) - d_busy
        window += hi - lo
    n = len(iters)
    return {
        "has_device_track": bool(device),
        "npu_ms": npu / n / 1000,
        "host_ms": host_only / n / 1000,
        "idle_ms": (window - npu - host_only) / n / 1000,
        "total_ms": window / n / 1000,
    }


parser = argparse.ArgumentParser()
parser.add_argument("--auto", action="store_true", help="find the newest results automatically")
parser.add_argument("--summary", type=Path)
parser.add_argument("--traces", type=Path, nargs="*", default=[])
parser.add_argument("--output", type=Path, default=Path("profile_dashboard.html"))
parser.add_argument("--title", default="Eager vs Compile on Furiosa RNGD")
args = parser.parse_args()

REPO = Path(__file__).resolve().parents[2]
if args.auto:
    if args.summary is None:
        candidates = sorted((REPO / "results").glob("*/repeated-*/summary.json"))
        if not candidates:
            sys.exit("No results/*/repeated-*/summary.json found.")
        args.summary = candidates[-1]
    if not args.traces:
        # Sorted by path, so newer result folders come later and win.
        args.traces = sorted((REPO / "results").rglob("*.trace.json"))
if args.summary is None:
    sys.exit("Give --summary PATH or use --auto.")
print(f"Using summary: {args.summary}")

summary = json.loads(args.summary.read_text())
models = {}
for row in summary["results"]:
    entry = models.setdefault(row["model"], {}).setdefault(str(row["batch_size"]), {})
    entry[row["mode"]] = {
        "mean": row["warm_mean_ms"],
        "p95": row["warm_p95_ms"],
        "throughput": row["throughput_images_per_second"],
        "cv": row["run_mean_cv_percent"],
    }

for path in args.traces:
    match = NAME_RE.search(path.name)
    result = breakdown(path) if match else None
    if not result:
        print(f"skipped {path} (unexpected name or no iter_ labels)")
        continue
    entry = models.setdefault(match["model"], {}).setdefault(match["batch"], {})
    entry.setdefault(match["mode"], {})["breakdown"] = result

ORDER = ["smallcnn", "resnet18_chunked"]
models = dict(sorted(models.items(), key=lambda kv: (ORDER.index(kv[0]) if kv[0] in ORDER else len(ORDER), kv[0])))

payload = {
    "title": args.title,
    "generated": date.today().isoformat(),
    "source": (args.summary.resolve().relative_to(REPO) if args.summary.resolve().is_relative_to(REPO) else args.summary).as_posix(),
    "models": models,
}

TEMPLATE = Path(__file__).with_name("dashboard_template.html").read_text()
page = TEMPLATE.replace("__TITLE__", html.escape(args.title)).replace(
    "__DATA__", json.dumps(payload).replace("</", "<\\/")
)
args.output.write_text(page)
print(f"Wrote {args.output} ({len(models)} models, {len(args.traces)} traces)")
