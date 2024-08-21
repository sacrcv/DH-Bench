import os
import json
import itertools
import base64
from time import sleep
import re
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, FuyuForCausalLM, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import random
import argparse

# from prompt_formatters import depth_tf_labelled_prompt, depth_tf_color_prompt, height_tf_color_prompt, height_tf_labelled_prompt
from geometric_reasoning_metrics import GeoMCQMetric
from utils import get_prefix_suffix

parser = argparse.ArgumentParser()
parser.add_argument("--prompts_file", type=str, default="/home/yasjain/codebase/geom-bench/data/realworld_depth_height/depth_height_1000_realworld.jsonl")
parser.add_argument("--img_dir", type=str, default="../../data/realworld_depth_height/")
parser.add_argument("--output_dir", type=str, default="../outputs3/")
parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if '1.6' in args.model.split("/")[1]:
    mllm_model = LlavaNextForConditionalGeneration.from_pretrained(args.model, device_map="auto")
elif '1.5' in args.model.split("/")[1]:
    mllm_model = LlavaForConditionalGeneration.from_pretrained(args.model, device_map="auto")
elif 'Bunny' in args.model:
    # if 'v1' in args.model.split("/")[1]:
    mllm_model = AutoModelForCausalLM.from_pretrained(args.model,  trust_remote_code=True)
elif 'fuyu' in args.model:
    mllm_model = FuyuForCausalLM.from_pretrained(args.model, device_map="auto")
elif 'instructblip' in args.model:
    mllm_model = InstructBlipForConditionalGeneration.from_pretrained(args.model, device_map="auto")
processor = AutoProcessor.from_pretrained(args.model) #.to(device)
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
evaluator = GeoMCQMetric()
random.seed(42)

eval_output_path = os.path.join(args.output_dir, f"{args.model.replace('/','-')}-eval.jsonl")
prefix, suffix = get_prefix_suffix(args.model)

def evaluate_model(model, prompts_file, img_dir, file):

    total = 0
    correct = 0

    with open(prompts_file, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
            
        prompt_data = json.loads(json_str)

        answer_set = prompt_data["target_options"]
        answer = prompt_data["target_text"]
        img_path = os.path.join(img_dir, prompt_data["images"][0])

        prompt_text = prompt_data["query_text"]
        if 'fuyu' not in args.model:
            prompt_text = prefix + prompt_text + suffix
        try:
            result = model(img_path, prompt_text, answer_set, answer, file)
        except Exception as e:
            print(e)
            result = 0

        if (result == 1):
            correct += 1
        total += 1
        print(correct, total)
    print("Accuracy: ", correct/total)
    print("Total: ", total)
    print("Correct: ", correct)
    print(args, "TF")

    with open(os.path.join(eval_output_path), "a") as file:
        result = {
            'path': args.prompts_file,
            'type': 'TF',
            'total': total,
            'correct': correct,
            'accuracy': correct/total
        }
        file.write(json.dumps(result) + '\n')

def model_llava(img_path, prompt_text, answer_set, answer, file):
    global mllm_model, evaluator
    img_path = os.path.join(os.getcwd(), img_path)
    encoded_image = Image.open(img_path).convert("RGB")
    inputs = processor(text=prompt_text, images=encoded_image, return_tensors="pt").to(device)
    generate_ids = mllm_model.generate(**inputs,max_new_tokens=50)
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # predicted_answer = predicted_answer.split('ASSISTANT: ')[1]
    if suffix != '':
        try:   
            response = response.split(f'{suffix} ')[1]
        except IndexError as e:
            response = response.split(f'{suffix}')[1]

    predicted_answer = response.replace("'","") # remove single quotes, not sure why it is appearing
    # predicted_answer = predicted_answer.replace(' ','') # remove spaces 
    # answer = answer.replace(' ','') # remove spaces

    # is_correct = re.search(r'Correct', predicted_answer, re.IGNORECASE) and not re.search(r'Incorrect', predicted_answer, re.IGNORECASE)
    # predicted_answer = 'Correct' if is_correct else 'Incorrect'
    
    # judgement = int(predicted_answer == answer)
    judgement = evaluator.__evaluate__(predicted_answer, answer, answer_set, True)
    judgement = 1 if judgement == "correct" else 0
    result = {
        'img_path': img_path,
        'prompt_text': prompt_text,
        "response": response,
        'options': answer_set,
        'ground_truth': answer,
        'prediction': predicted_answer,
        'judgement': judgement
    }
    file.write(json.dumps(result) + '\n')
    
    return judgement

def model_bunny(img_path, prompt_text, answer_set, answer, file):
    global mllm_model, evaluator
    img_path = os.path.join(os.getcwd(), img_path)
    encoded_image = Image.open(img_path)
    text_chunks = [tokenizer(chunk).input_ids for chunk in prompt_text.split('<image>')]
    input_ids = torch.tensor((text_chunks[0]+ [-200] + text_chunks[1]), dtype=torch.long).unsqueeze(0).to(device)
    image_tensor = mllm_model.process_images([encoded_image], model_cfg=mllm_model.config).to(device=device)
    input_ids = input_ids.to(device)
    image_tensor = image_tensor.to(device)
    mllm_model = mllm_model.to(device)
    generate_ids = mllm_model.generate(input_ids, images=image_tensor, max_new_tokens=50)[0]
    # generate_ids = mllm_model.generate(**inputs,max_new_tokens=50)
    # breakpoint()
    predicted_answer = tokenizer.decode(generate_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
    # predicted_answer = predicted_answer.split(f'{suffix} ')[1]


    predicted_answer = predicted_answer.replace("'","") # remove single quotes, not sure why it is appearing
    # predicted_answer = predicted_answer.replace(' ','') # remove spaces 
    # answer = answer.replace(' ','') # remove spaces

    # is_correct = re.search(r'Correct', predicted_answer, re.IGNORECASE) and not re.search(r'Incorrect', predicted_answer, re.IGNORECASE)
    # predicted_answer = 'Correct' if is_correct else 'Incorrect'
    
    # judgement = int(predicted_answer == answer)

    judgement = evaluator.__evaluate__(predicted_answer, answer, answer_set, True)
    judgement = 1 if judgement == "correct" else 0

    result = {
        'img_path': img_path,
        'prompt_text': prompt_text,
        'options': answer_set,
        'ground_truth': answer,
        'prediction': predicted_answer,
        'judgement': judgement
    }
    file.write(json.dumps(result) + '\n')
    
    return judgement


if __name__ == "__main__":

    PATH_TO_FOLDER = args.prompts_file

    IMG_DIR = args.img_dir
    
    last_dir_name = os.path.basename(PATH_TO_FOLDER)
    last_dir_name = last_dir_name.split(".")[0]
    os.makedirs(f"{args.output_dir}/{last_dir_name}/{args.model.replace('/','-')}/", exist_ok=True)
    output_path = f"{args.output_dir}/{last_dir_name}/{args.model.replace('/','-')}/tf.jsonl"   
    if 'Bunny' in args.model:
        model = model_bunny
    elif 'llava' in args.model:
        model = model_llava
    elif 'fuyu' in args.model:
        model = model_llava
    elif 'instructblip' in args.model:
        model = model_llava
    with open(output_path, "a") as file:
        evaluate_model(model, PATH_TO_FOLDER, IMG_DIR, file)