import os
import json
import itertools
import base64
from time import sleep
import re
from transformers import AutoProcessor, FuyuForCausalLM
import torch
from PIL import Image
import random
import argparse

from prompt_formatters import depth_mcq_labelled_prompt, depth_mcq_color_prompt, height_mcq_color_prompt, height_mcq_labelled_prompt

parser = argparse.ArgumentParser()
parser.add_argument("--prompts_file", type=str, default="../inputs/depth_synthetic_2D/prompts-3-shapes-color.jsonl")
parser.add_argument("--img_dir", type=str, default="../../data")
parser.add_argument("--output_dir", type=str, default="outputs")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fuyu_model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b", device_map="auto")
processor = AutoProcessor.from_pretrained("adept/fuyu-8b")

random.seed(42)

def evaluate_model(model, prompts_file, img_dir, file):

    total = 0
    correct = 0

    with open(prompts_file, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
            
        prompt_data = json.loads(json_str)

        prompt_text = prompt_data["prompt"]
        answer_set = prompt_data["options"]
        answer = prompt_data["ground_truth"]
        img_path = os.path.join(img_dir, prompt_data["img_path"])
        random.shuffle(answer_set)
        
        question_items = answer_set[0]

        if "depth" in prompts_file:
            if "color" in prompts_file:
                prompt_text = depth_mcq_color_prompt(question_items, answer_set)
            else:
                prompt_text = depth_mcq_labelled_prompt(question_items, answer_set)
        else:
            if "color" in prompts_file:
                prompt_text = height_mcq_color_prompt(prompt_text, answer_set)
            else:
                prompt_text = height_mcq_labelled_prompt(prompt_text, answer_set)

        result = model(img_path, prompt_text, answer_set, answer, file)

        if (result == 1):
            correct += 1
        total += 1
        print(correct, total)
    print("Accuracy: ", correct/total)
    print("Total: ", total)
    print("Correct: ", correct)
    print(args, "MCQ")


def model(img_path, prompt_text, answer_set, answer, file):
    # print("img_path: ", img_path)
    # print("prompt_text: ", prompt_text)
    # print("answer_set: ", answer_set)
    # print("answer: ", answer)
    img_path = os.path.join(os.getcwd(), img_path)
    encoded_image = Image.open(img_path)
    inputs = processor(text=prompt_text, images=encoded_image, return_tensors="pt").to(device)
    generate_ids = fuyu_model.generate(**inputs,max_new_tokens=50)
    predicted_answer = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    predicted_answer = predicted_answer.split('\x04 ')[1]

    predicted_answer = predicted_answer.replace("'","") # remove single quotes, not sure why it is appearing

    judgement = int(predicted_answer == answer)

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
    '''
    # print("predicted_answer: ", predicted_answer)
    if predicted_answer in answer_set:
        if predicted_answer == answer:
            # print("Correct")
            file.write(f"'{img_path}'\t '{prompt_text}'\t '{answer_set}'\t '{answer}'\t '{predicted_answer}'\t 1\n")
            return 1
    else:
        print("Error for image: ", img_path)
    file.write(f"'{img_path}'\t '{prompt_text}'\t '{answer_set}'\t '{answer}'\t '{predicted_answer}'\t 0\n")
    # print('-----------------------------------')
    return 0 
    '''

if __name__ == "__main__":

    PATH_TO_FOLDER = args.prompts_file

    IMG_DIR = args.img_dir
    
    last_dir_name = os.path.basename(PATH_TO_FOLDER)
    last_dir_name = last_dir_name.split(".")[0]
    os.makedirs(f"{args.output_dir}/{last_dir_name}/fuyu/", exist_ok=True)
    output_path = f"{args.output_dir}/{last_dir_name}/fuyu/mcq.jsonl"   
    with open(output_path, "w") as file:
        evaluate_model(model, PATH_TO_FOLDER, IMG_DIR, file)