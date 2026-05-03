#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../src/generative"

python color_transfer_new.py --out_dir ../../results/figure2_color_transfer --flow srmmd --dataset cub --source_idx 7 --target_idx 1007 --split test --kernel gaussian --resize 256 --particles 8192 --step_num 3000 --step_size 0.01 --bandwidth 1.0 --lmbda 1e-2 --alpha 0.8 --metric_n 512 --log_every 25
