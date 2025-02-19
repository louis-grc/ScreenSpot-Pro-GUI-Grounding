#!/bin/bash
set -e

# English

models=("cogagent24" "ariaui" "uground" "osatlas-7b" "osatlas-4b" "showui" "seeclick" "qwen1vl" "qwen2vl" "qwen2-5vl" "minicpmv" "cogagent" "gpt4o" )
for model in "${models[@]}"
do
    python eval_screenspot_pro.py.py  \
        --model_type ${model}  \
        --screenspot_imgs "../data/ScreenSpot-Pro/images"  \
        --screenspot_test "../data/ScreenSpot-Pro/annotations"  \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "./results/${model}.json" \
        --inst_style "instruction"

done

