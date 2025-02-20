#!/bin/bash
set -e
export OPENAI_API_KEY=$OPENAI_API_KEY
export HF_TOKEN

echo "OPENAI_API_KEY is set: ${OPENAI_API_KEY}"
echo "HF_TOKEN is set: ${HF_TOKEN}"

cd ScreenSpot-Pro-GUI-Grounding

pip install tqdm
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation

python ./download_screenspot_pro_dataset.py

# Single run of the evaluation script
python ./eval_screenspot_pro.py  \
    --model_type "qwen25vl"  \
    --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct"  \
    --screenspot_imgs "../screenspot_dataset/data/ScreenSpot-Pro/images"  \
    --screenspot_test "../screenspot_dataset/data/ScreenSpot-Pro/annotations"  \
    --task "all" \
    --language "en" \
    --gt_type "positive" \
    --log_path "./results/qwen2-5vl.json" \
    --inst_style "instruction"

