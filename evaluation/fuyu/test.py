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
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="../../data/depth_synthetic_2D/images-3-shapes")
parser.add_argument("--prompt_label", type=str, default="labelled", choices=["labelled", "labelled_id"])
parser.add_argument("--output_dir", type=str, default="outputs")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fuyu_model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b", device_map="auto")
processor = AutoProcessor.from_pretrained("adept/fuyu-8b")

random.seed(42)

def evaluate_model(model, prompt_dir, img_dir, file, prompt_label="None", prompt_type="descriptive"):

    filenames = os.listdir(prompt_dir)
    filenames.sort()

    total = 0
    correct = 0

    for filename in filenames:
        prompt_file = os.path.join(prompt_dir, filename)
        with open(prompt_file, 'r') as f:
            prompt_list = json.load(f)
        prompts = []
        if (prompt_label == "None"):
            img_path = os.path.join(img_dir, prompt_list["filename"])
            prompts = prompt_list["prompts"]
            print("hello")
        elif (prompt_label == "labelled"):
            img_path = os.path.join(img_dir, prompt_list["filename_labelled"])
            prompts = prompt_list["prompts_labelled"]
        elif (prompt_label == "labelled_id"):
            img_path = os.path.join(img_dir, prompt_list["filename_labelled"])
            prompts = prompt_list["prompts_labelled_id"]

        print("Printing prompts: ")
        prompts = [prompt_data for prompt_data in prompts if prompt_data["type"] == prompt_type]
        print("Number of prompts: ", len(prompts))
        for prompt_data in prompts:

            prompt_text = prompt_data["prompt"]
            answer_set = prompt_data["answerSet"]
            answer_set = [item.strip() for item in answer_set]
            answer = prompt_data["answer"]
            random.shuffle(answer_set)
            question_items = answer_set[0]
            prompt_text = f"""The image shows 2D shapes placed randomly. The shapes overlap each other, creating a depth effect.\
 When two shapes are overlapping, the shape that is complete is defined to be on top of the partially hidden shape.\
 Each 2D shape has a number written over them which we call ShapeID and must be inferred as the label for the corresponding shape.\
 Provide depth ordering from top to bottom for the shapes '{question_items}' in the image. Answer in the format: 'ShapeID, ShapeID, ...'. For eg. '3, 1, 2' is a valid answer format."""


            prompt_text = prompt_text + f" ONLY output the answer in the specified format, no extra text.\n"
            result = model(img_path, prompt_text, answer_set, answer, file)

            if (result == 1):
                correct += 1
            total += 1
            print(correct, total)
    print("Accuracy: ", correct/total)
    print("Total: ", total)
    print("Correct: ", correct)
    print(args, "List")


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

    # file = None

    # PATH_TO_FOLDER = "D:/Rishit/repos/depth_synthetic_2D/images-3-shapes"  #replace with the absolute path to unzipped file subfolder("Eg: images-3-shapes")
    # PATH_TO_FOLDER = "../../data/depth_synthetic_2D/images-3-shapes"
    PATH_TO_FOLDER = args.path
    # PATH_TO_FOLDER = os.getcwd()

    prompt_dir = os.path.join(PATH_TO_FOLDER, "prompts")
    img_dir = os.path.join(PATH_TO_FOLDER, "imgs")
    labelled_dir = os.path.join(PATH_TO_FOLDER, "labelled")
    labelled_id_dir = os.path.join(PATH_TO_FOLDER, "labelled_id")

    prompt_label = args.prompt_label
    if prompt_label == "None":
        input_dir = img_dir
    elif prompt_label == "labelled":
        input_dir = labelled_dir
    elif prompt_label == "labelled_id":
        input_dir = labelled_id_dir
    
    last_dir_name = os.path.basename(PATH_TO_FOLDER)
    os.makedirs(f"{args.output_dir}/{last_dir_name}/fuyu/", exist_ok=True)
    output_path = f"{args.output_dir}/{last_dir_name}/fuyu/list-{prompt_label}.jsonl"
    print(output_path)
    with open(output_path, "w") as file:
        evaluate_model(model, prompt_dir, input_dir, file, prompt_label=prompt_label, prompt_type="descriptive")
