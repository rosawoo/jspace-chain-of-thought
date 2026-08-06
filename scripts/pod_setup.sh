#!/usr/bin/env bash
# Pod bootstrap after every start/restart (container disk does not reliably
# survive stop/start; /workspace network volume does). Idempotent.
set -euo pipefail

# Cap CPU threads: torch spawns OMP threads for every visible core; their
# spin-waiting around tiny per-token ops slows BOTH the CPU sampler (~25x)
# and GPU kernel launches (~7x). Diagnosed 2026-08-05 (268ms/tok model,
# 750ms/tok sampler -> ~40/30ms capped).
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
grep -q OMP_NUM_THREADS /root/.bashrc 2>/dev/null || \
    echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4" >> /root/.bashrc

python -m venv --system-site-packages /root/venv
/root/venv/bin/pip install -q transformers datasets math-verify pytest \
    accelerate "jlens @ git+https://github.com/anthropics/jacobian-lens"
/root/venv/bin/python -c "import torch; assert torch.cuda.is_available()"
echo "POD-SETUP-OK $(/root/venv/bin/python -c 'import torch; print(torch.cuda.get_device_name(0))')"
