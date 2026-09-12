#!/usr/bin/env bash
# Reproduce everything. ~35 min on two CPU cores.
set -euo pipefail
cd "$(dirname "$0")/src"
python3 test_myadam.py
python3 e01_adam_by_hand.py
python3 e02_bias_correction.py
python3 e02b_tuned_bias_correction.py
python3 e03_update_ratio.py
python3 e04_schedules.py
python3 e05_width_lr_sweep.py
python3 e05b_stability_edge.py
python3 summarize.py | tee ../results/summary.txt
