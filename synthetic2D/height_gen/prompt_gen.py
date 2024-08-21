import os
import json
import copy
import argparse
import random
import itertools

THRESH = 80

DESCRIPTION = "The image shows 2D red rectangles stacked on top of each other. There are multiple stacks in the image.\
 The black region at the bottom of the image is the ground level, and is where the base of the stack lies.\
 The height of each stack is measured from its base.\
 Each 2D shape has a number written over them which we call ShapeID and must be inferred as the label for the corresponding shape.\
 The stacks are labelled A, B... from left to right."

COLORED_DESCRIPTION = "The image shows 2D red rectangles stacked on top of each other. There are multiple stacks in the image.\
 The black region at the bottom of the image is the ground level, and is where the base of the stack lies.\
 The height of each stack is measured from its base.\
 Each 2D shape has different color.\
 The stacks are labelled A, B... from left to right."

MIRRORED_DESCRIPTION = "The image shows 2D red rectangles stacked on top of each other. There are multiple stacks in the image.\
 The black region at the bottom of the image is the ground level, and is where the base of the stack lies.\
 The height of each stack is measured from its base.\
 Each 2D shape has a number written over them which we call ShapeID and must be inferred as the label for the corresponding shape.\
 The stacks are labelled A, B... from right to left."

MIRRORED_COLORED_DESCRIPTION = "The image shows 2D red rectangles stacked on top of each other. There are multiple stacks in the image.\
 The black region at the bottom of the image is the ground level, and is where the base of the stack lies.\
 The height of each stack is measured from its base.\
 Each 2D shape has different color.\
 The stacks are labelled A, B... from right to left."

FORMAT_INSTRUCTION = " Answer in the format A, B, C... where A, B, C... are the labels of the stacks."

def getData(path):
    try:
        with open(path, "r") as file:
            data = json.load(file)
    except FileNotFoundError as e:
        raise e 

    return data

#Category 1: simple height ordering with no changes
def createPrompts1(truths, maxcount=3, color=False):
    prompts = []

    labels = [stack["id"] for stack in truths["stacks"]]
    groups = list(itertools.combinations(labels, min(3, len(labels)))) # Choosing any 3 stacks (at max) for the prompt 
    random.shuffle(groups)
    cnt = 0
    for group in groups:
        if cnt >= maxcount:
            break
        groupHeights = [(stack["id"], stack["totalHeight"]) for stack in truths["stacks"] if stack["id"] in group]
        groupHeights.sort(key=lambda x: x[1])
        flag = 0
        for i in range(len(groupHeights) - 1):
            if groupHeights[i+1][1] - groupHeights[i][1] < THRESH:
                flag = 1
                break
        if flag == 0:
            answer = ", ".join([stack[0] for stack in groupHeights])
            answerSet = [", ".join(permutation) for permutation in itertools.permutations([stack[0] for stack in groupHeights])]
            random.shuffle(answerSet)
            groundTruth = [stack[0] for stack in groupHeights]
            random.shuffle(groundTruth)
            if color:
                prompt = COLORED_DESCRIPTION + f"Order the stacks labelled {', '.join(groundTruth)} from shortest to tallest." + FORMAT_INSTRUCTION
                mirrored_prompt = MIRRORED_COLORED_DESCRIPTION + f"Order the stacks labelled {', '.join(groundTruth)} from shortest to tallest." + FORMAT_INSTRUCTION
            else:
                prompt = DESCRIPTION + f"Order the stacks labelled {', '.join(groundTruth)} from shortest to tallest." + FORMAT_INSTRUCTION
                mirrored_prompt = MIRRORED_DESCRIPTION + f"Order the stacks labelled {', '.join(groundTruth)} from shortest to tallest." + FORMAT_INSTRUCTION
            
            prompts.append({"prompt": prompt, "mirrored_prompt": mirrored_prompt, "answer": answer, "answerSet": answerSet, "type" : "descriptive"})
            cnt += 1
    
    return prompts

def getStack(truths, id):
    for stack in truths["stacks"]:
        for shape in stack["shapes"]:
            if shape["id"] == id:
                return stack

#Category 2: height ordering with one random replacement
def createPrompts2(truths, maxcount, color=False):
    prompts = []
    totalHeights = []
    for stack in truths["stacks"]:
        totalHeights.append([stack["id"], stack["totalHeight"]])

    labels = []
    for stack in truths["stacks"]:
        for shape in stack["shapes"]:
            labels.append(shape["id"])
    pairs = list(itertools.combinations(labels, 2))
    random.shuffle(pairs)

    diffs = []
    cnt = 0
    for i, swap in enumerate(pairs):
        if cnt >= maxcount:
            break
        stack1 = getStack(truths, swap[0])
        stack2 = getStack(truths, swap[1])
        shape1 = [shape for shape in stack1["shapes"] if shape["id"] == swap[0]][0]
        shape2 = [shape for shape in stack2["shapes"] if shape["id"] == swap[1]][0]

        tempTotalHeights = copy.deepcopy(totalHeights)
        for j in range(len(tempTotalHeights)):
            if stack1["id"] == stack2["id"]:
                break
            if tempTotalHeights[j][0] == stack1["id"]:
                tempTotalHeights[j][1] -= shape1["height"]
                tempTotalHeights[j][1] += shape2["height"]
            elif tempTotalHeights[j][0] == stack2["id"]:
                tempTotalHeights[j][1] -= shape2["height"]
                tempTotalHeights[j][1] += shape1["height"]

        tempTotalHeights.sort(key=lambda x: x[1])        

        groundTruthAll = []
        for i in range(len(tempTotalHeights)):
            groundTruthAll.append(tempTotalHeights[i][0])

        # Choosing any 3 stacks (at max) for the prompt
        indices = [idx for idx in range(len(groundTruthAll))]
        random.shuffle(indices)
        groups = list(itertools.combinations(indices, min(3, len(indices))))
        random.shuffle(groups)
        groups = [sorted(group) for group in groups]

        flag_outer = 0
        groundTruth = []
        for group in groups:
            groundTruth = [groundTruthAll[idx] for idx in group]

            flag = 0
            for k in range(len(groundTruth) - 1):
                h1 = [stack["totalHeight"] for stack in truths["stacks"] if stack["id"] == groundTruth[k]][0]
                h2 = [stack["totalHeight"] for stack in truths["stacks"] if stack["id"] == groundTruth[k+1]][0]
                if h2 - h1 < THRESH:
                    flag = 1
                    break
            if flag == 0:
                flag_outer = 1
                break
        
        if flag_outer == 0:
            continue

        answer = ", ".join(groundTruth)
        answerSet = [", ".join(permutation) for permutation in itertools.permutations(groundTruth)]
        random.shuffle(answerSet)
        random.shuffle(groundTruth)
        if color:
            prompt = COLORED_DESCRIPTION + f" Swap {shape1['color']} rectangle from stack {stack1['id']} with {shape2['color']} rectangle from stack {stack2['id']}. Order the stacks labelled {', '.join(groundTruth)} from shortest to tallest." + FORMAT_INSTRUCTION
            mirrored_prompt = MIRRORED_COLORED_DESCRIPTION + f" Swap {shape1['color']} rectangle from stack {stack1['id']} with {shape2['color']} rectangle from stack {stack2['id']}. Order the stacks labelled {', '.join(groundTruth)} from shortest to tallest." + FORMAT_INSTRUCTION
        else:
            prompt = DESCRIPTION + f" Swap shape {swap[0]} from stack {stack1['id']} with shape {swap[1]} from stack {stack2['id']}. Order the stacks labelled {', '.join(groundTruth)} from shortest to tallest." + FORMAT_INSTRUCTION
            mirrored_prompt = MIRRORED_DESCRIPTION + f" Swap shape {swap[0]} from stack {stack1['id']} with shape {swap[1]} from stack {stack2['id']}. Order the stacks labelled {', '.join(groundTruth)} from shortest to tallest." + FORMAT_INSTRUCTION
        prompts.append({"prompt": prompt, "mirrored_prompt": mirrored_prompt, "answer": answer, "answerSet": answerSet, "type" : "counterfactual"})
        cnt += 1
        diffs.append(((shape1["height"] - shape2["height"])*2, tempTotalHeights))
    
    # print(diffs)
    return prompts


def createPrompts(path, shapeList, i, maxcount, color=False):
    prompts = []
    prompts = createPrompts1(shapeList, color=color)
    promptsNew = createPrompts2(shapeList, maxcount, color=color)
    prompts += promptsNew

    promptsWithMetadata = {}
    promptsWithMetadata["filename"] = f"img{i}.jpg"
    promptsWithMetadata["prompts"] = prompts


    with open(f"{path}/prompts{i}.json", "w") as file:
        json.dump(promptsWithMetadata, file, indent=4)
    
    if len(prompts) == 0:
        with open("empty.json", "r") as file:
            empty = json.load(file)
        empty["indices"].append(i)
        empty["number"] += 1
        with open("empty.json", "w") as file:
            json.dump(empty, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, default="./height_images/truths")
    parser.add_argument('-n', '--maxcount', type=int, default=3)
    parser.add_argument('-p', '--path', type=str, default="prompts")
    parser.add_argument('-t', '--thresh', type=int, default=100)
    parser.add_argument('-r', '--color', action='store_true', default=False)

    args = parser.parse_args()

    THRESH = args.thresh

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    if os.path.exists("empty.json"):
        os.remove("empty.json")
    
    with open("empty.json", "w") as file:
        json.dump({"indices": [], "number" : 0}, file)
    
    files = os.listdir(args.src)
    files.sort()
    for i, file in enumerate(files):
        shapeList = getData(os.path.join(args.src, file))
        print(shapeList)

        createPrompts(args.path, shapeList, i, args.maxcount, color=args.color)
    
    # with open("empty.json", "r") as file:
    #     empty = json.load(file)
    
    # with open("empty.json", "w") as file:
    #     json.dump({"indices": [], "number" : 0}, file)
    
    # for i in empty["indices"]:
    #     shapeList = getData(os.path.join(args.src, f"truth{i}.json"))
    #     print(shapeList)

    #     createPrompts(args.path, shapeList, i, args.maxcount, color=args.color)
        

        
        