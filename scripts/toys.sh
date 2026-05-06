#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../src/generative"

# Set --flow to mmd, srmmd, drmmd, hrmmd

# Mixture of Gaussians

python run.py --dataset MixGaussian --flow srmmd --kernel Gaussian --lmbda 0.1 --alpha 0.8 --step_size 0.1 --bandwidth 1.0 --step_num 4000 --source_particle_num 500 --target_particle_num 500 --opt sgd --seed 42 --save_path ../../results/figure1_toys/

# Swiss Roll

python run.py --dataset swissroll --flow srmmd --kernel Energy --lmbda 0.1 --alpha 0.8 --step_size 0.1 --bandwidth 1.0 --step_num 4000 --source_particle_num 500 --target_particle_num 500 --opt sgd --seed 42 --save_path ../../results/figure1_toys/
