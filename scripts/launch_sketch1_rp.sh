#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-fs/fotn-sketch1
export PYTHONUNBUFFERED=1
export PYTHONPATH=/root/autodl-fs/fotn-sketch1
python -u train.py --config configs/config_rp.yaml
