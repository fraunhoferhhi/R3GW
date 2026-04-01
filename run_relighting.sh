#!/bin/bash
# RELIGHTING example for scene LK2

LK2_SOURCE_PATH="../data/nerfosr/lk2_colmap"
LK2_OUTPUT_PATH="../R3GW_VISAPP2026/lk2"
# ENVMAP_PATH="../Downloads/lake_pier_4k.hdr"
VIEW_NAME="01-08_07_30_IMG_6660"

# With trained illumination
python relight.py dataset.source_path="$LK2_SOURCE_PATH" \
dataset.model_path="$LK2_OUTPUT_PATH" \
relighting.training_view=False relighting.view_name="$VIEW_NAME" \
relighting.trained_illumination_name="C2_DSC_6_4" \
dataset.eval=True
