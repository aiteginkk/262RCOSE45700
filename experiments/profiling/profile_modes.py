"""Record execution traces for NPU Eager and Compile modes.

Run on the NPU server from the repository root:

    python experiments/profiling/profile_modes.py --model smallcnn --batches 1 16 \
        --output-dir profiling-results

For each model/batch/mode this writes:
    <model>_b<batch>_<mode>.trace.json   open in https://ui.perfetto.dev
    <model>_b<batch>_<mode>.ops.txt      top operators by time
plus manifest.json describing the run.

Protocol (same as the benchmark runner): CPU fallback disabled, BF16,
10 warm-up calls that are NOT profiled, then 10 profiled calls. Each
profiled call is labelled iter_<n>, so the analysis script can find
iteration boundaries. Each call includes .cpu() so the call finishes
on the device before the next one starts.

Use a new --output-dir for each experiment.
"""
import argparse
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

from furiosa.torch import config

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["smallcnn", "resnet18_chunked"], default="smallcnn")
parser.add_argument("--batches", type=int, nargs="+", default=[1])
parser.add_argument("--modes", nargs="+", choices=["eager", "compile"], default=["eager", "compile"])
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--iterations", type=int, default=10)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=False)

with config.cpu_fallback(False):
    import torch
    import furiosa.torch
    from torch.profiler import ProfilerActivity, profile, record_function

    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))
    from models.project_models import build_model_pair, make_input

    # Ask torch.profiler for device-side events too, if furiosa-torch
    # registers them. If not, the trace still shows host-side activity.
    activities = [ProfilerActivity.CPU]
    if hasattr(ProfilerActivity, "PrivateUse1"):
        activities.append(ProfilerActivity.PrivateUse1)

    _, model = build_model_pair(args.model, seed=42)
    device = torch.device("furiosa:0")
    model = model.to(device)
    compiled = torch.compile(model, backend=furiosa.torch.backend, mode="default")
    runners = {"eager": model, "compile": compiled}

    runs = []
    for batch in args.batches:
        x = make_input(args.model, batch=batch, seed=43).to(device)

        for mode in args.modes:
            fn = runners[mode]
            stem = f"{args.model}_b{batch}_{mode}"
            print(f"== {stem} ==", flush=True)

            with torch.inference_mode():
                for _ in range(args.warmup):
                    fn(x).detach().cpu()

                wall_ms = []
                used = list(activities)
                try:
                    prof = profile(activities=used, record_shapes=True)
                    prof.__enter__()
                except Exception as error:  # device activity unsupported
                    print(f"  device activity failed ({error}); using CPU only", flush=True)
                    used = [ProfilerActivity.CPU]
                    prof = profile(activities=used, record_shapes=True)
                    prof.__enter__()

                try:
                    for i in range(args.iterations):
                        start = time.perf_counter()
                        with record_function(f"iter_{i}"):
                            fn(x).detach().cpu()
                        wall_ms.append((time.perf_counter() - start) * 1000)
                finally:
                    prof.__exit__(None, None, None)

            trace_path = args.output_dir / f"{stem}.trace.json"
            prof.export_chrome_trace(str(trace_path))
            table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
            (args.output_dir / f"{stem}.ops.txt").write_text(table + "\n")

            mean_ms = sum(wall_ms) / len(wall_ms)
            print(f"  profiled wall mean: {mean_ms:.3f} ms (profiler overhead included)", flush=True)
            runs.append({
                "model": args.model,
                "batch_size": batch,
                "mode": mode,
                "activities": [a.name for a in used],
                "profiled_wall_ms": wall_ms,
                "trace": trace_path.name,
            })

manifest = {
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "furiosa_torch": version("furiosa-torch"),
    },
    "cpu_fallback": False,
    "dtype": "bfloat16",
    "seed": 42,
    "input_seed": 43,
    "warmup_iterations_unprofiled": args.warmup,
    "profiled_iterations": args.iterations,
    "note": "Profiled timings include profiler overhead; use benchmark runs for latency numbers.",
    "runs": runs,
}
(args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Saved {len(runs)} traces to {args.output_dir}")
