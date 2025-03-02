#!/bin/bash
set -e
export OPENAI_API_KEY=$OPENAI_API_KEY
export HF_TOKEN

echo "OPENAI_API_KEY is set: ${OPENAI_API_KEY}"
echo "HF_TOKEN is set: ${HF_TOKEN}"

# Detect the number of NVIDIA GPUs and create a device string
gpu_count=$(nvidia-smi -L | wc -l)
if [ $gpu_count -eq 0 ]; then
    echo "No NVIDIA GPUs detected. Exiting."
    exit 1
fi
# Construct the CUDA device string
cuda_devices=""
for ((i=0; i<gpu_count; i++)); do
    if [ $i -gt 0 ]; then
        cuda_devices+=","
    fi
    cuda_devices+="$i"
done

echo "GPUs detected : ${cuda_devices}"

cd ScreenSpot-Pro-GUI-Grounding

pip install tqdm
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils
pip install qwen_agent
pip install flash-attn --no-build-isolation

python ./download_screenspot_pro_dataset.py

# Single run of the evaluation script
#--task "all"
python ./eval_screenspot_pro.py  \
    --model_type "qwen25vl"  \
    --model_name_or_path "louisgrc/CU-Qwen-2.5-VL-7B-MacOS-Merged-V2.0"  \
    --screenspot_imgs "./screenspot_dataset/images"  \
    --screenspot_test "./screenspot_dataset/annotations"  \
    --task "word_macos.json, vscode_macos.json, vmware_macos.json, pycharm_macos.json, matlab_macos.json, macos_common_macos.json, excel_macos.json, davinci_macos.json, android_studio_macos.json" \
    --language "en" \
    --gt_type "positive" \
    --log_path "./results/qwen2-5vl.json" \
    --inst_style "instruction"

