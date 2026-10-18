#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-fs/fotn-sketch1
RUN=/root/autodl-fs/fotn/runs/curriculum_sketch1_rp
TICK_JSON='{"prompt":"Report sketch1_rp progress in Chinese. Sync canvas from metrics.jsonl via scripts/sync_sketch1_rp_canvas.py. Include stage, global/local epoch, live tqdm if any, va_rp vs best, a, train loss, remaining budget. If curriculum finished (done. best va_rp= in train.log) and train.py config_rp is not running: stop this loop, evaluate 202508 test and 202505 val with evaluate.py --config configs/config_rp.yaml, do not invent numbers."}'
while true; do
  sleep 300
  python3 "$ROOT/scripts/sync_sketch1_rp_canvas.py" || true
  echo "AGENT_LOOP_TICK_sketch1_rp ${TICK_JSON}"
  if grep -q "done. best va_rp=" "$RUN/train.log" 2>/dev/null \
     && ! pgrep -af 'python -u train.py --config configs/config_rp.yaml' | grep -v grep >/dev/null; then
    echo 'AGENT_LOOP_STOP_sketch1_rp {"reason":"curriculum finished"}'
    break
  fi
done
