#!/bin/bash

nohup python3 training_sweep.py \
    --save_dir /share/u/can/additivity_training/trained_saes \
    --layers 4 \
    --width_exponents 13 \
    --num_tokens 50_000_000 \
    --device cuda:0 \
    > nohup.out 2>&1 &


    # --no_wandb_logging \
    # --dry_run \