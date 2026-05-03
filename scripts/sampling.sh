#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../src/sampling"

python main.py --seed 42 --method srmmd --kernel Stein --particle_num 500 --lmbda 0.5 --alpha 0.8 --dataset mog --step_num 1000 --optimizer sgd --step_size 0.1 --bandwidth 0.3 --save_path ../../results/figure3_sampling_mog/
