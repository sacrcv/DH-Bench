"""This file contains the prompt formatters for the different types of questions in the LLAVA dataset.
Note the difference in inputs for height and depth questions. While number of parameters are same, the meanings differ."""


def depth_mcq_labelled_prompt(question_items: str, answer_set: list):
    prompt_text = """The image shows 2D shapes placed randomly. The shapes overlap each other, creating a depth effect.\
When two shapes are overlapping, the shape that is complete is defined to be on top of the partially hidden shape.\
Each 2D shape has a number written over them which we call ShapeID and must be inferred as the label for the corresponding shape.\
Provide depth ordering from top to bottom for the shapes '{question_items}' in the image. Answer in the format: 'ShapeID, ShapeID, ...'. For eg. '3, 1, 2' is a valid answer format."""

    prompt_text = prompt_text + f"\n From the given options: {answer_set}, select the correct answer (ONLY output the answer)."

    return prompt_text


def depth_mcq_color_prompt(question_items: str, answer_set: list):
    prompt_text = f"""The image shows 2D shapes placed randomly. The shapes overlap each other, creating a depth effect.\
When two shapes are overlapping, the shape that is complete is defined to be on top of the partially hidden shape.\
Each 2D shape has a unique color which we call the ShapeColor for the corresponding shape.\
Provide depth ordering from top to bottom for the shapes '{question_items}' in the image. Answer in the format: 'ShapeColor shape, ShapeColor shape, ...'. For eg. 'red triangle, blue circle, green rectangle' is a valid answer format."""

    prompt_text = prompt_text + f"\n From the given options: {answer_set}, select the correct answer (ONLY output the answer)."
    return prompt_text


def depth_tf_labelled_prompt(question_items: str, answer: str):
    prompt_text = f"""The image shows 2D shapes placed randomly. The shapes overlap each other, creating a depth effect.\
 When two shapes are overlapping, the shape that is complete is defined to be on top of the partially hidden shape.\
 Each 2D shape has a number written over them which we call ShapeID and must be inferred as the label for the corresponding shape.\
 Provide depth ordering from top to bottom for the shapes '{question_items}' in the image. Answer in the format: 'ShapeID, ShapeID, ...'. For eg. '3,1,2' is a valid answer format."""
    
    prompt_text = prompt_text + f"\n Given the predicted depth ordering as '{answer}', evaluate the prediction as Correct or Incorrect."

    return prompt_text


def depth_tf_color_prompt(question_items: str, answer: str):
    prompt_text = f"""The image shows 2D shapes placed randomly. The shapes overlap each other, creating a depth effect.\
 When two shapes are overlapping, the shape that is complete is defined to be on top of the partially hidden shape.\
 Each 2D shape has a unique color which we call the ShapeColor for the corresponding shape.\
 Provide depth ordering from top to bottom for the shapes '{question_items}' in the image. Answer in the format: 'ShapeColor shape, ShapeColor shape, ...'. For eg. 'red triangle, blue circle, green rectangle' is a valid answer format."""
    
    prompt_text = prompt_text + f"\n Given the predicted depth ordering as '{answer}', evaluate the prediction as Correct or Incorrect."

    return prompt_text


def depth_list_order_labelled_prompt(question_items: str):
    prompt_text = f"""The image shows 2D shapes placed randomly. The shapes overlap each other, creating a depth effect.\
 When two shapes are overlapping, the shape that is complete is defined to be on top of the partially hidden shape.\
 Each 2D shape has a number written over them which we call ShapeID and must be inferred as the label for the corresponding shape.\
 Provide depth ordering from top to bottom for the shapes '{question_items}' in the image. Answer in the format: 'ShapeID, ShapeID, ...'. For eg. '3, 1, 2' is a valid answer format.\nASSISTANT:"""

    return prompt_text


def depth_list_order_color_prompt(question_items: str):
    prompt_text = f"""The image shows 2D shapes placed randomly. The shapes overlap each other, creating a depth effect.\
 When two shapes are overlapping, the shape that is complete is defined to be on top of the partially hidden shape.\
 Each 2D shape has a unique color which we call the ShapeColor for the corresponding shape.\
 Provide depth ordering from top to bottom for the shapes '{question_items}' in the image. Answer in the format: 'ShapeColor shape, ShapeColor shape, ...'. For eg. 'red triangle, blue circle, green rectangle' is a valid answer format.\nASSISTANT:"""

    return prompt_text


def height_mcq_labelled_prompt(original_prompt: str, answer_set:list, question_items:str, num_stacks=3):
    stack_labels = ", ".join([chr(ord('A') + i) for i in range(int(num_stacks))])

    condition = ""
    if "Swap" in original_prompt:
        pos1 = original_prompt.find("Swap")
        pos2 = original_prompt.find("Order")
        condition = original_prompt[pos1:pos2]


    prompt_text = f"The image shows red 2D rectangles stacked on top of each other There are multiple stacks in the image. \
The black region at the bottom of the image is the ground level, and is where the base of the stack lies. \
The height of each stack is measured from its base. Each 2D shape has a number written over them which we call ShapeID and must be inferred as the label for the corresponding shape. \
The stacks are labelled {stack_labels} from left to right. {condition} Now order the stacks labelled {question_items} from shortest to tallest. \
Answer in the format: 'StackLabel, StackLabel, ...'. For eg. 'B, A, C' is a valid answer format."
    prompt_text = prompt_text + f"\n From the given options: {answer_set}, select the correct answer (ONLY output the answer)."

    return prompt_text


def height_mcq_color_prompt(original_prompt: str, answer_set:list, question_items:str, num_stacks=3):
    stack_labels = ", ".join([chr(ord('A') + i) for i in range(int(num_stacks))])

    condition = ""
    if "Swap" in original_prompt:
        pos1 = original_prompt.find("Swap")
        pos2 = original_prompt.find("Order")
        condition = original_prompt[pos1:pos2]

    prompt_text = f"The image shows 2D rectangles stacked on top of each other There are multiple stacks in the image. \
The black region at the bottom of the image is the ground level, and is where the base of the stack lies. \
The height of each stack is measured from its base. Each 2D shape has a unique color.\
The stacks are labelled {stack_labels} from left to right. {condition} Now order the stacks labelled {question_items} from shortest to tallest. \
Answer in the format: 'StackLabel, StackLabel, ...'. For eg. 'B, A, C' is a valid answer format."
    prompt_text = prompt_text + f"\n From the given options: {answer_set}, select the correct answer (ONLY output the answer)."


    return prompt_text


def height_tf_labelled_prompt(original_prompt: str, question_items:str, answer: str, num_stacks=3):
    stack_labels = ", ".join([chr(ord('A') + i) for i in range(int(num_stacks))])

    condition = ""
    if "Swap" in original_prompt:
        pos1 = original_prompt.find("Swap")
        pos2 = original_prompt.find("Order")
        condition = original_prompt[pos1:pos2]

    prompt_text = f"The image shows red 2D rectangles stacked on top of each other There are multiple stacks in the image. \
The black region at the bottom of the image is the ground level, and is where the base of the stack lies. \
The height of each stack is measured from its base. Each 2D shape has a number written over them which we call ShapeID and must be inferred as the label for the corresponding shape. \
The stacks are labelled {stack_labels} from left to right. {condition} Now order the stacks labelled {question_items} from shortest to tallest. \
Answer in the format: 'StackLabel, StackLabel, ...'. For eg. 'B, A, C' is a valid answer format."
    
    prompt_text = prompt_text + f"\n Given the predicted height ordering as '{answer}', evaluate the prediction as Correct or Incorrect."

    return prompt_text


def height_tf_color_prompt(original_prompt: str, question_items:str, answer: str, num_stacks=3):
    stack_labels = ", ".join([chr(ord('A') + i) for i in range(int(num_stacks))])

    condition = ""
    if "Swap" in original_prompt:
        pos1 = original_prompt.find("Swap")
        pos2 = original_prompt.find("Order")
        condition = original_prompt[pos1:pos2]

    prompt_text = f"The image shows 2D rectangles stacked on top of each other There are multiple stacks in the image. \
The black region at the bottom of the image is the ground level, and is where the base of the stack lies. \
The height of each stack is measured from its base. Each 2D shape has a unique color.\
The stacks are labelled {stack_labels} from left to right. {condition}vNow order the stacks labelled {question_items} from shortest to tallest. \
Answer in the format: 'StackLabel, StackLabel, ...'. For eg. 'B, A, C' is a valid answer format."
    
    prompt_text = prompt_text + f"\n Given the predicted height ordering as '{answer}', evaluate the prediction as Correct or Incorrect."

    return prompt_text


def height_list_order_labelled_prompt(original_prompt: str, question_items:str, num_stacks=3):
    stack_labels = ", ".join([chr(ord('A') + i) for i in range(int(num_stacks))])

    condition = ""
    if "Swap" in original_prompt:
        pos1 = original_prompt.find("Swap")
        pos2 = original_prompt.find("Order")
        condition = original_prompt[pos1:pos2]

    prompt_text = f"The image shows 2D rectangles stacked on top of each other There are multiple stacks in the image. \
The black region at the bottom of the image is the ground level, and is where the base of the stack lies. \
The height of each stack is measured from its base. Each 2D shape has a number written over them which we call ShapeID and must be inferred as the label for the corresponding shape. \
The stacks are labelled {stack_labels} from left to right. {condition} Now order the stacks labelled {question_items} from shortest to tallest. \
Answer in the format: 'StackLabel, StackLabel, ...'. For eg. 'B, A, C' is a valid answer format."
    
    return prompt_text

def height_list_order_color_prompt(original_prompt: str, question_items:str, num_stacks=3):
    stack_labels = ", ".join([chr(ord('A') + i) for i in range(int(num_stacks))])

    condition = ""
    if "Swap" in original_prompt:
        pos1 = original_prompt.find("Swap")
        pos2 = original_prompt.find("Order")
        condition = original_prompt[pos1:pos2]

    prompt_text = f"The image shows 2D rectangles stacked on top of each other There are multiple stacks in the image. \
The black region at the bottom of the image is the ground level, and is where the base of the stack lies. \
The height of each stack is measured from its base. Each 2D shape has a unique color.\
The stacks are labelled {stack_labels} from left to right. {condition} Now order the stacks labelled {question_items} from shortest to tallest. \
Answer in the format: 'StackLabel, StackLabel, ...'. For eg. 'B, A, C' is a valid answer format."
        
    return prompt_text