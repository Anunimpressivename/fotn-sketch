#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-fs/fotn-sketch1
RUN=/root/autodl-fs/fotn-sketch1/runs/ring16_dwcur
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT"
mkdir -p "$RUN"
nohup python -u train_resgan.py --config configs/config_rp_flow32_resgan_gdisc_g025_ring16_dwcur.yaml \
  > "$RUN/train.log" 2>&1 &
echo $! | tee "$RUN/train.pid"
echo "launched ring16_dwcur pid=$(cat "$RUN/train.pid") log=$RUN/train.log"
