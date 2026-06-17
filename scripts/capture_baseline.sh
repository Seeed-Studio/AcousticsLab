#!/usr/bin/env bash
# Run the three hot-path benches and write tests/benches/baseline_<sha>.json
# (median/mean/stderr from criterion's estimates.json). Policy: docs/BENCHS.md.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v cargo   >/dev/null || { echo "error: cargo not on PATH" >&2; exit 1; }
command -v python3 >/dev/null || { echo "error: python3 not on PATH" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]] \
    || echo "warning: dirty tree -- baseline won't reproducibly match the named SHA" >&2

GIT_SHA="$(git rev-parse --short=12 HEAD)"
OUTPUT="tests/benches/baseline_${GIT_SHA}.json"
echo "Capturing baseline at SHA ${GIT_SHA}"

for b in audio_buffer_ring_throughput opus_stream_encode_frame inference_engine_run_window; do
    echo "bench: ${b}"
    cargo bench -p acoustics-lab --bench "${b}" -q
done

python3 - "$GIT_SHA" "$OUTPUT" <<'PY'
import json, pathlib, datetime, platform, subprocess, sys

git_sha, output = sys.argv[1], sys.argv[2]
base = pathlib.Path("target/criterion")
benches = {
    "audio_buffer/push_period":  "audio_buffer_push_period/1024_samples",
    "audio_buffer/peek_window":  "audio_buffer_peek_window/44032_samples",
    "opus_stream/encode_frame":  "opus_stream_encode_frame/1024_samples_in_per_call",
    "inference/run_window_burn": "inference_run_window_burn/preproc+backbone+head",
}

def load(rel):
    p = base / rel / "new" / "estimates.json"
    if not p.exists():
        sys.exit(f"missing criterion output: {p} -- bench did not run?")
    with p.open() as f:
        d = json.load(f)
    return {
        "median_ns":          d["median"]["point_estimate"],
        "median_ci_low_ns":   d["median"]["confidence_interval"]["lower_bound"],
        "median_ci_high_ns":  d["median"]["confidence_interval"]["upper_bound"],
        "mean_ns":            d["mean"]["point_estimate"],
        "stderr_ns":          d["mean"]["standard_error"],
    }

doc = {
    "captured_at_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "git_sha":         git_sha,
    "host": {
        "kernel": platform.system() + " " + platform.release(),
        "arch":   platform.machine(),
        "rustc":  subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True
        ).stdout.strip(),
    },
    "regression_threshold_pct": 5.0,
    "comment": "Hot-path baseline. Subsequent changes must not regress median by >5% without a waiver. Re-capture via scripts/capture_baseline.sh.",
    "benches": {name: load(rel) for name, rel in benches.items()},
}

pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
with open(output, "w") as f:
    json.dump(doc, f, indent=2)
    f.write("\n")
print(f"wrote {output}")
PY
