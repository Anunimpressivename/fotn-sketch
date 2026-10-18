#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-fs/fotn-sketch1
RUN=/root/autodl-fs/fotn-sketch1/runs/ring16
SYNC="$ROOT/scripts/sync_flow32_resgan_gdisc_g025_ring16_canvas.py"
CFG=configs/config_rp_flow32_resgan_gdisc_g025_ring16.yaml
echo "loop_pid=$$"
echo $$ > "$RUN/monitor.pid"
while true; do
  sleep 300
  python3 "$SYNC" >"$RUN/canvas_sync.json" 2>&1 || true
  echo 'AGENT_LOOP_TICK_g025_ring16 {"prompt":"Report g025_ring16 progress in Chinese-simplified. Run python3 /root/autodl-fs/fotn-sketch1/scripts/sync_flow32_resgan_gdisc_g025_ring16_canvas.py and rewrite the canvas. Include phase, gan epoch, live tqdm, pool_ent vs log(25)=3.22, weighted 6*adv / 0.25*gadv / 0.01*pool / 0.15*mae, val CSI40 vs best if any, remaining 12-epoch budget. Do not invent numbers. If train.log contains done. best val CSI40= and train_resgan.py for configs/config_rp_flow32_resgan_gdisc_g025_ring16.yaml is not running: stop this 5-minute loop, evaluate 202508 test with the best-val-CSI40 checkpoint, do not overwrite other run dirs."}'
  if grep -q "done. best val CSI40=" "$RUN/train.log" 2>/dev/null \
     && ! pgrep -af "train_resgan.py --config ${CFG}" | grep -v grep >/dev/null; then
    echo "AGENT_LOOP_STOP_g025_ring16 finished"
    break
  fi
done
