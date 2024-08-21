#! /bin/bash
# "llava-hf/llava-v1.6-vicuna-7b-hf" "llava-hf/llava-v1.6-vicuna-13b-hf" "llava-hf/llava-1.5-13b-hf" "llava-hf/llava-1.5-7b-hf" #"llava-hf/llava-v1.6-34b-hf"  
for model in "llava-hf/llava-v1.6-mistral-7b-hf" "llava-hf/llava-v1.6-vicuna-7b-hf" "llava-hf/llava-v1.6-vicuna-13b-hf" "llava-hf/llava-1.5-13b-hf" "llava-hf/llava-1.5-7b-hf" 
do
    for type in "color" "labelled" "labelled_id"
    do
        python test-mcq-2D.py --model $model --prompts_file ../standard_data/depth_synthetic_2D/images-3-shapes-${type}-200.jsonl
        python test-tf-2D.py --model $model --prompts_file ../standard_data/depth_synthetic_2D/images-3-shapes-${type}-200.jsonl
    done
done