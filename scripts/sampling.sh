#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../src/sampling"

python main.py --seed 42 --method mmd --kernel Stein --particle_num 500 --lmbda 0.5 --alpha 0.8 --dataset mog --step_num 1000 --optimizer lbfgs --step_size 0.1 --bandwidth 0.3 --save_path ../../results/figure3_sampling_mog/
python main.py --seed 42 --method srmmd --kernel Stein --particle_num 500 --lmbda 0.5 --alpha 0.8 --dataset mog --step_num 1000 --optimizer lbfgs --step_size 0.1 --bandwidth 0.3 --save_path ../../results/figure3_sampling_mog/
python main.py --seed 42 --method hrmmd --kernel Stein --particle_num 500 --lmbda 0.5 --alpha 0.8 --dataset mog --step_num 1000 --optimizer lbfgs --step_size 0.1 --bandwidth 0.3 --save_path ../../results/figure3_sampling_mog/
python main_svgd.py --seed 42 --method svgd --kernel Gaussian --dataset mog --particle_num 500 --step_num 1000 --step_size 0.01 --bandwidth 0.3 --svgd_bandwidth 0.3 --save_trajectory --save_path ../../results/figure3_sampling_mog/
python main_svgd.py --seed 42 --method rsvgd --kernel Gaussian --dataset mog --particle_num 500 --step_num 1000 --step_size 0.01 --bandwidth 0.3 --svgd_bandwidth 0.3 --nu 0.01 --save_trajectory --save_path ../../results/figure3_sampling_mog/
