import os
import json
import itertools
from gpt4v import llm_client, request_data
import base64
from time import sleep
import re


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
            
            question_items = answer_set[0]
            prompt_text = f"""The image shows 2D shapes placed randomly. The shapes overlap each other, creating a depth effect.\
 When two shapes are overlapping, the shape that is complete is defined to be on top of the partially hidden shape.\
 Each 2D shape has a number written over them which we call ShapeID and must be inferred as the label for the corresponding shape.\
 Provide depth ordering from top to bottom for the shapes '{question_items}' in the image. Answer in the format: 'ShapeID, ShapeID, ...'. For eg. '3, 1, 2' is a valid answer format."""


            # MCQ
            prompt_text = prompt_text + f"\n Answer:"
            result = model(img_path, prompt_text, answer_set, answer, file)

            if (result == 1):
                correct += 1
            total += 1
            print(correct, total)
    print("Accuracy: ", correct/total)
    print("Total: ", total)
    print("Correct: ", correct)


def model(img_path, prompt_text, answer_set, answer, file):
    print("img_path: ", img_path)
    print("prompt_text: ", prompt_text)
    print("answer_set: ", answer_set)
    print("answer: ", answer)
    img_path = os.path.join(os.getcwd(), img_path)
    encoded_image = base64.b64encode(open(img_path, 'rb').read()).decode('ascii')
    request_data['messages'][1]["content"] = [
            {"image": encoded_image},
            prompt_text
    ]
    try:
        response = llm_client.send_request('dev-gpt-4v-chat-completions', request_data)
        predicted_answer = response["choices"][0]["message"]["content"]
    except Exception as e:
        print(e)
        sleep(10)
        file.write(f"{img_path}\t {prompt_text}\t {answer_set}\t {answer}\t {-1}\t {0}\n")
        return 0
    print("predicted_answer: ", predicted_answer)
    if predicted_answer in answer_set:
        if predicted_answer == answer:
            print("Correct")
            file.write(f"{img_path}\t {prompt_text}\t {answer_set}\t {answer}\t {predicted_answer}\t {1}\n")
            return 1
    else:
        print("Error for image: ", img_path)
    file.write(f"{img_path}\t {prompt_text}\t {answer_set}\t {answer}\t {predicted_answer}\t {0}\n")
    print('-----------------------------------')
    return 0 


if __name__ == "__main__":

    # file = None

    # PATH_TO_FOLDER = "D:/Rishit/repos/depth_synthetic_2D/images-3-shapes"  #replace with the absolute path to unzipped file subfolder("Eg: images-3-shapes")
    PATH_TO_FOLDER = "C:\\Users\\yasjain\\Downloads\\depth_synthetic_2D\\depth_synthetic_2D\\images-3-shapes"
    # PATH_TO_FOLDER = os.getcwd()

    prompt_dir = os.path.join(PATH_TO_FOLDER, "prompts")
    img_dir = os.path.join(PATH_TO_FOLDER, "imgs")
    labelled_dir = os.path.join(PATH_TO_FOLDER, "labelled")
    labelled_id_dir = os.path.join(PATH_TO_FOLDER, "labelled_id")

    prompt_label = "labelled_id"
    if prompt_label == "None":
        input_dir = img_dir
    elif prompt_label == "labelled":
        input_dir = labelled_dir
    elif prompt_label == "labelled_id":
        input_dir = labelled_id_dir
    
    last_dir_name = os.path.basename(PATH_TO_FOLDER)
    os.makedirs(f"outputs/{last_dir_name}", exist_ok=True)
    file = open(f"outputs/{last_dir_name}/mcq-{prompt_label}.csv", "w")
    file.write("img_path\t prompt_text\t answer_set\t answer\t prediction\t judgement\n")
    # evaluate_model(model, prompt_dir, img_dir)
    evaluate_model(model, prompt_dir, input_dir, file, prompt_label=prompt_label, prompt_type="descriptive")
    print(prompt_dir)