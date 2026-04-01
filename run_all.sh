#!/bin/bash
# R3GW: Training, Rendering, and Testing

#### Scene ST ####

ST_SOURCE_PATH="/mnt/dataset/st"
ST_OUTPUT_PATH="/mnt/output/st"

#Train
python train.py dataset.source_path="$ST_SOURCE_PATH" dataset.model_path="$ST_OUTPUT_PATH" "$@"

# Render
python render.py dataset.source_path="$ST_SOURCE_PATH" dataset.model_path="$ST_OUTPUT_PATH" "$@"

# Test
python eval_with_gt_envmaps.py dataset.source_path="$ST_SOURCE_PATH"  dataset.model_path="$ST_OUTPUT_PATH" "$@" \
dataset.test_config_path=configs/test/st


#### Scene LK2 ####

LK2_SOURCE_PATH="/mnt/dataset/lk2"
LK2_OUTPUT_PATH="/mnt/output/lk2"

#Train
python train.py dataset.source_path="$LK2_SOURCE_PATH" dataset.model_path="$LK2_OUTPUT_PATH" "$@"

# Render
python render.py dataset.source_path="$LK2_SOURCE_PATH" dataset.model_path="$LK2_OUTPUT_PATH" "$@"

# Test
python eval_with_gt_envmaps.py dataset.source_path="$LK2_SOURCE_PATH"  dataset.model_path="$LK2_OUTPUT_PATH" "$@" \
dataset.test_config_path=configs/test/lk2


#### Scene LWP ####

LWP_SOURCE_PATH="/mnt/dataset/lwp"
LWP_OUTPUT_PATH="/mnt/output/lwp"

#Train
python train.py dataset.source_path="$LWP_SOURCE_PATH" dataset.model_path="$LWP_OUTPUT_PATH" "$@"

# Render
python render.py dataset.source_path="$LWP_SOURCE_PATH" dataset.model_path="$LWP_OUTPUT_PATH" "$@"

# Test
python eval_with_gt_envmaps.py dataset.source_path="$LWP_SOURCE_PATH" dataset.model_path="$LWP_OUTPUT_PATH" "$@" \
dataset.test_config_path=configs/test/lwp
