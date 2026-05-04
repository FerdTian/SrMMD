#!/usr/bin/env bash
set -euo pipefail

# Bayesian logistic regression with SrMMD.
# Override SEEDS, DATASETS, STEP_SIZES, STEP_NUM, PARTICLE_NUM, PYTHON_BIN if needed.
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
DATASETS="${DATASETS:-breast_cancer ionosphere german_credit covtype}"
STEP_SIZES="${STEP_SIZES:-0.1 0.01}"
STEP_NUM="${STEP_NUM:-3000}"
PARTICLE_NUM="${PARTICLE_NUM:-20}"

cd "$(dirname "$0")/../src/sampling/blr"

for seed in ${SEEDS}; do
  for dataset in ${DATASETS}; do
    for step_size in ${STEP_SIZES}; do
      echo "Running SrMMD: seed=${seed}, dataset=${dataset}, step_size=${step_size}"
      "${PYTHON_BIN}" main.py \
        --seed "${seed}" \
        --dataset "${dataset}" \
        --method srmmd \
        --optimizer euler \
        --particle_num "${PARTICLE_NUM}" \
        --step_size "${step_size}" \
        --bandwidth 1.0 \
        --step_num "${STEP_NUM}" \
        --kernel Stein \
        --lmbda 0.1 \
        --save_path ../../../results/figure3_blr/
    done
  done
done
