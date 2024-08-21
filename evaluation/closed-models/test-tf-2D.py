import os
import json
import itertools
import base64
from time import sleep
import re
import torch
from PIL import Image
import random
import argparse

from prompt_formatters import depth_tf_labelled_prompt, depth_tf_color_prompt, height_tf_color_prompt, height_tf_labelled_prompt
from utils import get_prefix_suffix

parser = argparse.ArgumentParser()
parser.add_argument("--prompts_file", type=str, default="../standard_data/depth_synthetic_2D/images-3-shapes-color-200.jsonl")
parser.add_argument("--img_dir", type=str, default="../../data")
parser.add_argument("--output_dir", type=str, default="../outputs")
parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(42)

if "gemini" in args.model:
    from utils import GeminiModels, gemini_config_vision

    LMMmodels = GeminiModels(gemini_config_vision)

elif "claude" in args.model:
    from utils import CLAUDEModels, claude_config

    LMMmodels = CLAUDEModels(claude_config)

elif "gptv" in args.model:
    from utils import OpenAIModelsAzure, openai_trnllm_gpt4turbov_config

    LMMmodels = OpenAIModelsAzure(openai_trnllm_gpt4turbov_config)

elif "gpt4o" in args.model:
    from utils import OpenAIModelsAzure, openai_config_gpt4o_azure

    LMMmodels = OpenAIModelsAzure(openai_config_gpt4o_azure)


eval_output_path = os.path.join(args.output_dir, f"{args.model.replace('/','-')}-eval.jsonl")
# prefix, suffix = get_prefix_suffix(args.model)

def evaluate_model(model, prompts_file, img_dir, file):

    total = 0
    correct = 0

    with open(prompts_file, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
            
        prompt_data = json.loads(json_str)

        answer_set = prompt_data["tf_options"]
        answer = prompt_data["tf_ground_truth"]
        prompt_answer = prompt_data["tf_prompt_answer"]
        img_path = os.path.join(img_dir, prompt_data["img_path"])
        question_items = prompt_data["question_items"]

        if "depth" in prompts_file:
            if "color" in prompts_file:
                prompt_text = depth_tf_color_prompt(question_items, prompt_answer)
            else:
                prompt_text = depth_tf_labelled_prompt(question_items, prompt_answer)
        else:
            num_stacks = [num for num in prompts_file.split('/')[-1] if num.isdigit()][0]
            num_stacks = int(num_stacks)
            prompt_text = prompt_data["prompt_text"]
            if "color" in prompts_file:
                prompt_text = height_tf_color_prompt(prompt_text, question_items, prompt_answer, num_stacks)
            else:
                prompt_text = height_tf_labelled_prompt(prompt_text, question_items, prompt_answer, num_stacks)
        
        # prompt_text = prefix + prompt_text + suffix
        
        result = model(img_path, prompt_text, answer_set, answer, file)

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


def gptv_model(img_path, prompt_text, answer_set, answer, file):
    img_path = os.path.join(os.getcwd(), img_path)
    encoded_image = base64.b64encode(open(img_path, 'rb').read()).decode('ascii')
    request_data['messages'][1]["content"] = [
            {"image": encoded_image},
            prompt_text
    ]
    is_error = False
    try:
        response = llm_client.send_request('dev-gpt-4v-chat-completions', request_data)
        predicted_answer = response["choices"][0]["message"]["content"]
    except Exception as e:
        print(e)
        sleep(10)
        is_error = True
        

    is_correct = re.search(r'Correct', predicted_answer, re.IGNORECASE) and not re.search(r'Incorrect', predicted_answer, re.IGNORECASE)
    predicted_answer = 'Correct' if is_correct else 'Incorrect'
    predicted_answer = predicted_answer if not is_error else 'Error'

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


#### Gemini model ###

def model_gemini(img_path, prompt_text, answer_set, answer, file):
    img_path = os.path.join(os.getcwd(), img_path)
    encoded_image = base64.b64encode(open(img_path, 'rb').read()).decode('ascii')
    
    response = LMMmodels.generate(prompt_text, [encoded_image]) 

    print(response)
    
    predicted_answer = response[0].strip()
    predicted_answer = predicted_answer.replace("'","") # remove single quotes, not sure why it is appearing
    predicted_answer = predicted_answer.replace(' ','') # remove spaces         
   
    is_correct = re.search(r'Correct', predicted_answer, re.IGNORECASE) and not re.search(r'Incorrect', predicted_answer, re.IGNORECASE)
    predicted_answer = 'Correct' if is_correct else 'Incorrect'
    


    judgement = int(predicted_answer == answer)


    result = {
        'img_path': img_path,
        'prompt_text': prompt_text,
        'options': answer_set,
        'ground_truth': answer,
        'prediction': predicted_answer,
        'response': response[0],
        'judgement': judgement
    }
    file.write(json.dumps(result) + '\n')
    
    return judgement

####################



if __name__ == "__main__":

    PATH_TO_FOLDER = args.prompts_file

    IMG_DIR = args.img_dir
    
    last_dir_name = os.path.basename(PATH_TO_FOLDER)
    last_dir_name = last_dir_name.split(".")[0]
    os.makedirs(f"{args.output_dir}/{last_dir_name}/{args.model.replace('/','-')}/", exist_ok=True)
    output_path = f"{args.output_dir}/{last_dir_name}/{args.model.replace('/','-')}/tf.jsonl"   
     
    if 'gptv' in args.model:
        model = model_gemini
    elif 'gemini' in args.model:
        model = model_gemini
    elif 'claude' in args.model:
        model = model_gemini        
    elif 'gptv' in args.model:
        model = model_gemini    
    elif 'gpt4o' in args.model:
        model = model_gemini

    with open(output_path, "a") as file:
        evaluate_model(model, PATH_TO_FOLDER, IMG_DIR, file)