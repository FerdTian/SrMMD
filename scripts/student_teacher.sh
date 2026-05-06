#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../src/generative"

python student_teacher/train.py --device -1 --lmbda 0.1 --loss srmmd --lr 0.1 --with_noise false --noise_decay_freq 500 --seed 42 --log_in_file --num_particles 100 --N_train 1000 --N_valid 1000 --total_epochs 15000 --log_dir ../../results/figure2_student_teacher/
