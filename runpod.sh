#!/bin/bash
set -e

cd ScreenSpot-Pro-GUI-Grounding

# Single run of the evaluation script
python ./eval_screenspot_pro.py  \
    --model_type "qwen2-5vl"  \
    --model_name_or_path "Qwen/Qwen2-VL-7B-Instruct"  \
    --screenspot_imgs "../data/ScreenSpot-Pro/images"  \
    --screenspot_test "../data/ScreenSpot-Pro/annotations"  \
    --task "all" \
    --language "en" \
    --gt_type "positive" \
    --log_path "./results/qwen2-5vl.json" \
    --inst_style "instruction"

