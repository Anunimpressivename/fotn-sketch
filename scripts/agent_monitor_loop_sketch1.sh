#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-fs/fotn-sketch1
RUN=/root/autodl-fs/fotn/runs/curriculum_sketch1
TICK_JSON='{"prompt":"Report fotn-sketch1 progress in Chinese. Sync canvas from metrics.jsonl. Include stage, global/local epoch, live tqdm if any, va_rp vs best 0.0420 at conv e9, a, train loss, remaining res budget. If curriculum finished (done. best va_rp=): stop loop, evaluate 202508 test and 202505 val with evaluate.py, do not invent numbers."}'
while true; do
  sleep 300
  python3 "$ROOT/scripts/sync_sketch1_canvas.py" || true
  echo "AGENT_LOOP_TICK_sketch1 ${TICK_JSON}"
  if grep -q "done. best va_rp=" /root/.cursor/projects/root-autodl-fs/terminals/645704.txt 2>/dev/null \
     && ! pgrep -af 'python -u train.py --config configs/config.yaml' | grep -v grep >/dev/null; then
    echo 'AGENT_LOOP_STOP_sketch1 {"reason":"curriculum finished"}'
    break
  fi
done
