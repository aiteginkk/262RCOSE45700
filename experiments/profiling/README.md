# Profiling: where does inference time go on the NPU?

Mentor feedback (11/20): profile the runs, look at how much of each
iteration actually executes on the NPU versus preparation time and
host↔device transfers, and add the traces to the report.

Run everything from the repository root on the NPU server.

## 0. Find the profiler support in our furiosa-torch

    python experiments/profiling/find_profiler.py

Note whether `PrivateUse1` is available and any `FURIOSA_*PROFIL*` /
`*TRACE*` environment variables. Paste the output into the PR.

## 1. Record traces

    python experiments/profiling/profile_modes.py --model smallcnn \
        --batches 1 16 --output-dir profiling-smallcnn

    python experiments/profiling/profile_modes.py --model resnet18_chunked \
        --batches 1 --output-dir profiling-resnet

Each model/batch/mode gets a `.trace.json` and an `.ops.txt` table.
Warm-up calls (10) are not profiled; 10 calls are profiled, labelled
`iter_0` … `iter_9`. CPU fallback is disabled.

If step 0 found a Furiosa trace variable, also run once with it set,
for example `FURIOSA_PROFILER_OUTPUT_PATH=npu-trace.json python ...`,
and keep the file it writes.

## 2. Look at the timeline

Open a `.trace.json` in https://ui.perfetto.dev (drag and drop).
Zoom into one `iter_n` block and screenshot Eager vs Compile.

## 3. Summarise the numbers

    python experiments/profiling/analyze_trace.py profiling-smallcnn/*.trace.json

Prints per-track busy time, busy % of the iteration, idle gaps, and the
longest events. The NPU track's busy % is the "NPU time ÷ total time"
ratio for the report. If it says no device track was found, the trace
only contains host events: ask the mentor which tool produces the NPU
timeline, and run `analyze_trace.py` on that tool's trace instead.

## Limitations

- Profiled timings include profiler overhead. Report latency from the
  benchmark runner, and use traces only to explain where the time goes.
- Random BF16 weights and synthetic inputs, as in all other experiments.
