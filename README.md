# SrMMD Reviewer Reproduction Code

This anonymized repository contains the runnable code extracted from the uploaded archive for the paper's main experiments.

## Repository layout

```text
src/generative/        Toy examples, student-teacher experiment, and color-transfer.
src/sampling/          Sampling experiment.
scripts/               Bash scripts.
results/               Outputs.
```

## Installation

```bash
pip install -r requirements.txt
```

## Reproducing the main available experiments

```bash
# Mixture of Gaussians and Swiss roll
bash scripts/toys.sh

# Student-teacher experiment
bash scripts/student_teacher.sh

# Color transfer
bash scripts/color_transfer.sh

# Sampling experiment
bash scripts/sampling_mog.sh
