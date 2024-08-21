import os
import json
import itertools
import base64
from time import sleep
import re
# from transformers import AutoProcessor, LlavaForConditionalGeneration
# import torch
from PIL import Image
import random
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="../../data/height_synthetic_2D/images-3-stacks-stepped-colored")
parser.add_argument("--prompt_label", type=str, default="labelled", choices=["labelled", "labelled_id", "color"])
parser.add_argument("--output_dir", type=str, default="../standard_data")
args = parser.parse_args()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# llava_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="auto")
# processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf") #.to(device)
random.seed(42)

def evaluate_model(model, prompt_dir, img_dir, file, prompt_label="None", prompt_type="descriptive"):

    filenames = os.listdir(prompt_dir)
    filenames.sort()

    total = 0
    correct = 0
    breakpoint()
    # filenames = filenames[:200] # first 200 images only
    cnt = 0
    for filename in filenames:
        if cnt == 200:
            break
        prompt_file = os.path.join(prompt_dir, filename)
        with open(prompt_file, 'r') as f:
            prompt_list = json.load(f)
        prompts = []
        if (prompt_label == "color"):
            img_path = os.path.join(img_dir, prompt_list["filename"])
            prompts = prompt_list["prompts"]
        elif (prompt_label == "labelled"):
            img_path = os.path.join(img_dir, prompt_list["filename"])
            prompts = prompt_list["prompts"]
        elif (prompt_label == "labelled_id"):
            img_path = os.path.join(img_dir, prompt_list["filename_labelled"])
            prompts = prompt_list["prompts_labelled_id"]

        print("Printing prompts: ")
        prompts = [prompt_data for prompt_data in prompts if prompt_data["type"] == prompt_type]
        print("Number of prompts: ", len(prompts))
        try:
            prompts = random.sample(prompts, 1)
        except:
            continue
        for prompt_data in prompts:

            prompt_text = prompt_data["prompt"]
            answer_set = prompt_data["answerSet"]
            answer = prompt_data["answer"]
            random.shuffle(answer_set)
            question_items = answer_set[0]

            
#             prompt_text = f"""<image>\nUSER: The image shows 2D shapes placed randomly. The shapes overlap each other, creating a depth effect.\
#  When two shapes are overlapping, the shape that is complete is defined to be on top of the partially hidden shape.\
#  Each 2D shape has a number written over them which we call ShapeID and must be inferred as the label for the corresponding shape.\
#  Provide depth ordering from top to bottom for the shapes '{question_items}' in the image. Answer in the format: 'ShapeID, ShapeID, ...'. For eg. '3,1,2' is a valid answer format."""


            # T/F
            answer_flag = random.choice([True, False])
            if answer_flag:
                pass
                # prompt_text = prompt_text + f"\n Given the predicted depth ordering as '{answer}', evaluate the prediction as Correct or Incorrect.\nASSISTANT:"
            else:
                incorrect_answer = random.choice(list(set(answer_set) - set(answer)))
                # prompt_text = prompt_text + f"\n Given the predicted depth ordering as '{incorrect_answer}', evaluate the prediction as Correct or Incorrect.\nASSISTANT:"
            # result = model(img_path, prompt_text, answer_set, answer, file, answer_flag)
            data = {
                "img_path": '/'.join(img_path.split('/')[3:]),
                "options": answer_set,
                "ground_truth": answer,
                "question_items": question_items,
                "prompt_text": prompt_text,
                "tf_prompt_answer": answer if answer_flag else incorrect_answer,
                "tf_ground_truth": "Correct" if answer_flag else "Incorrect",
                "tf_options": ["Correct", "Incorrect"],
                "prompt_text": prompt_text
            }
            file.write(json.dumps(data) + '\n')
            cnt += 1
           
    print(args, "TF-Random")

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
    if prompt_label == "color":
        input_dir = img_dir
    elif prompt_label == "labelled":
        input_dir = labelled_dir
    elif prompt_label == "labelled_id":
        input_dir = labelled_id_dir
    
    last_dir_name = os.path.basename(PATH_TO_FOLDER)
    # os.makedirs(f"{args.output_dir}/{last_dir_name}/llava/", exist_ok=True)
    output_path = f"{args.output_dir}/{last_dir_name}-{prompt_label}-200.jsonl"

    with open(output_path, "w") as file:
        evaluate_model(None, prompt_dir, input_dir, file, prompt_label=prompt_label, prompt_type="counterfactual")
