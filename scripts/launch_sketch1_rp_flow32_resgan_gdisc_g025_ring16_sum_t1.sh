#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-fs/fotn-sketch1
RUN=/root/autodl-fs/fotn-sketch1/runs/ring16_sum_t1
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT"
mkdir -p "$RUN"
if [ ! -f "$RUN/norm_stats.json" ]; then
  cp /root/autodl-fs/fotn-sketch1/runs/ring16/norm_stats.json "$RUN/norm_stats.json"
fi
nohup python -u train_resgan.py --config configs/config_rp_flow32_resgan_gdisc_g025_ring16_sum_t1.yaml \
  > "$RUN/train.log" 2>&1 &
echo $! | tee "$RUN/train.pid"
echo "launched ring16_sum_t1 pid=$(cat "$RUN/train.pid") log=$RUN/train.log"
