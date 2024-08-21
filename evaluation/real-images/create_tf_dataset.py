import json
import random
random.seed(42)

def depth_tf_labelled_prompt(question_items: str, answer: str, options: list = None):
    answer_flag = random.choice([True, False])
    if not answer_flag:
        answer = random.choice(list(set(options) - set([answer])))
    
    prompt_text = f"""{question_items} Given the predicted answer as {answer}, evaluate the prediction as Correct or Incorrect. Write answer as Correct or Incorrect ONLY."""

    return prompt_text, answer_flag

def convert_mcq_to_tf(input_file: str, output_file: str):
    with open(input_file, 'r') as infile:
        data = [json.loads(line) for line in infile]

    new_data = []
    for item in data:
        # Extract the regions from the query_text
        query_parts = item['query_text'].split('\nFrom the given options:')
        question_part = query_parts[0]
        regions = question_part # question_part.split("regions")[1].split("in the image")[0].strip()
        answer = item['target_text']
        options = item['target_options']
        # Create the new query_text using the function
        new_query_text, answer_flag = depth_tf_labelled_prompt(regions, answer, options)
        # Update the item
        item['query_text'] = new_query_text
        item['target_options'] = "['Correct', 'Incorrect']"
        item['target_text'] = 'Correct' if answer_flag else 'Incorrect'
        new_data.append(item)

    with open(output_file, 'w') as outfile:
        for item in new_data:
            json.dump(item, outfile)
            outfile.write('\n')

# Usage
input_file = '/home/yasjain/codebase/geom-bench/data/realworld_depth_height/depth_height_1000_realworld.jsonl'
output_file = '/home/yasjain/codebase/geom-bench/data/realworld_depth_height/depth_height_1000_realworld_tf.jsonl'
convert_mcq_to_tf(input_file, output_file)
