"""Summarise a Chrome/Perfetto trace: where does each iteration's time go?

Works on traces from profile_modes.py, and on any Chrome-format trace
the Furiosa tools produce. Runs anywhere with Python (no NPU needed):

    python experiments/profiling/analyze_trace.py profiling-results/*.trace.json

For every trace it prints:
- each timeline track (process/thread) with its busy time per iteration
- for each non-host track (e.g. the NPU): busy % of the iteration and
  the idle gaps, which is what the mentor asked us to report
- the longest events, to see which operators dominate

If no iter_<n> labels exist, the whole trace span is treated as one window.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def merge(intervals):
    """Union of [start, end) intervals."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def clip(intervals, lo, hi):
    return [(max(s, lo), min(e, hi)) for s, e in intervals if e > lo and s < hi]


def analyze(path):
    data = json.loads(Path(path).read_text())
    events = data["traceEvents"] if isinstance(data, dict) else data

    names = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") in ("process_name", "thread_name"):
            key = (e.get("pid"), e.get("tid")) if e["name"] == "thread_name" else (e.get("pid"), None)
            names[key] = e.get("args", {}).get("name", "")

    spans = [e for e in events if e.get("ph") == "X" and "dur" in e]
    if not spans:
        print(f"{path}: no duration events found")
        return

    iters = sorted(
        (e["ts"], e["ts"] + e["dur"]) for e in spans if str(e.get("name", "")).startswith("iter_")
    )
    if not iters:
        iters = [(min(e["ts"] for e in spans), max(e["ts"] + e["dur"] for e in spans))]
    total_window = sum(e - s for s, e in iters)
    n = len(iters)

    tracks = defaultdict(list)
    cats = defaultdict(float)
    for e in spans:
        if str(e.get("name", "")).startswith("iter_"):
            continue
        tracks[(e.get("pid"), e.get("tid"))].append(e)
        cats[e.get("cat", "?")] += e["dur"]

    def label(key):
        pid, tid = key
        return f"{names.get((pid, None), pid)} / {names.get(key, tid)}"

    def is_host(key, evs):
        text = (label(key) + " " + " ".join({str(e.get("cat", "")) for e in evs})).lower()
        return not any(word in text for word in ("npu", "furiosa", "device", "privateuse1", "kernel"))

    print(f"\n### {Path(path).name}")
    print(f"iterations: {n}, mean iteration window: {total_window / n / 1000:.3f} ms")
    print("event categories (total us):", {k: round(v) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])})

    print("\n| track | host? | busy ms/iter | busy % | idle gaps/iter | longest gap ms |")
    print("|---|---|---:|---:|---:|---:|")
    found_device = False
    for key, evs in sorted(tracks.items(), key=lambda kv: -sum(e["dur"] for e in kv[1])):
        host = is_host(key, evs)
        found_device |= not host
        busy, gaps, longest = 0.0, 0, 0.0
        for lo, hi in iters:
            merged = merge(clip([(e["ts"], e["ts"] + e["dur"]) for e in evs], lo, hi))
            busy += sum(e - s for s, e in merged)
            for (_, end), (start, _) in zip(merged, merged[1:]):
                gaps += 1
                longest = max(longest, start - end)
        print(
            f"| {label(key)} | {'yes' if host else 'NO'} | {busy / n / 1000:.3f} | "
            f"{100 * busy / total_window:.1f}% | {gaps / n:.1f} | {longest / 1000:.3f} |"
        )
    if not found_device:
        print("\nNo device (NPU) track detected: this trace only has host-side events.")
        print("Ask the mentor which tool/env var produces the NPU timeline for furiosa-torch.")

    print("\nTop 10 events by total time:")
    totals = defaultdict(lambda: [0.0, 0])
    for evs in tracks.values():
        for e in evs:
            totals[e.get("name", "?")][0] += e["dur"]
            totals[e.get("name", "?")][1] += 1
    for name, (dur, count) in sorted(totals.items(), key=lambda kv: -kv[1][0])[:10]:
        print(f"  {dur / n / 1000:8.3f} ms/iter  x{count / n:5.1f}  {name[:80]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: analyze_trace.py TRACE.json [TRACE.json ...]")
    for trace in sys.argv[1:]:
        analyze(trace)
