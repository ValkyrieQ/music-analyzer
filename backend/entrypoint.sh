#!/bin/sh
#
# Resolve the thread budget *before* Python starts, then hand off to the real command.
#
# This exists because OpenMP reads OMP_NUM_THREADS once, when the runtime initialises —
# which happens on `import torch`, before any of our code runs. Calling
# torch.set_num_threads() afterwards reports the new value but cannot recover the
# parallelism: measured in this image, a matmul took 1.43s at 1 thread and 1.40s after
# set_num_threads(5), versus 0.34s when the env var was 5 from the start. Demucs was
# therefore pinned to a single core no matter what the Python side asked for.
#
# The count has to be computed at runtime, not baked in: the image is meant to deploy to
# servers with different core counts, and a hardcoded number either wastes cores or
# oversubscribes them.
set -e

if [ -z "$ANALYZER_THREADS" ]; then
    # os.cpu_count() sees the host's cores; a `cpus:` limit in compose shows up only as a
    # cgroup quota, so both have to be consulted and the smaller one wins. `nproc` is not
    # usable here — it honours OMP_NUM_THREADS, which is exactly the value we are trying
    # to compute, and would return 1.
    ANALYZER_THREADS=$(python - <<'PY'
import os

quota = None
try:  # cgroup v2
    with open("/sys/fs/cgroup/cpu.max") as fh:
        allowance, period = fh.read().split()
    if allowance != "max":
        quota = float(allowance) / float(period)
except (OSError, ValueError):
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            allowance = int(fh.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            period = int(fh.read())
        if allowance > 0:
            quota = allowance / period
    except (OSError, ValueError):
        pass

cpus = os.cpu_count() or 2
if quota:
    cpus = min(cpus, int(quota) or 1)
print(max(1, cpus))
PY
)
fi

export ANALYZER_THREADS
export OMP_NUM_THREADS="$ANALYZER_THREADS"
export MKL_NUM_THREADS="$ANALYZER_THREADS"
export OPENBLAS_NUM_THREADS="$ANALYZER_THREADS"
# Torch's own inter-op pool stays small: the work is one big sequential model, so extra
# inter-op threads only add contention on top of the intra-op pool above.
export TORCH_NUM_INTEROP_THREADS=1

echo "entrypoint: ANALYZER_THREADS=$ANALYZER_THREADS (OMP/MKL/OpenBLAS set to match)" >&2

exec "$@"
