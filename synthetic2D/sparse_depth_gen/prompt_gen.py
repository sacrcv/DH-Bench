import json
import argparse
import random
import copy
import os
import itertools

COLORED_DESCRIPTION = "The image shows a few shapes of various colors placed randomly. Some of these are on top of others. "
LABELLED_DESCRIPTION = "The image shows a few shapes placed randomly. Some of these are on top of others. The shapes are labelled with numbers which we call the shapeID. "

def getData(path):
    try:
        with open(path, "r") as file:
            data = json.load(file)
    except FileNotFoundError as e:
        raise e 

    print(data)
    return data


def createGraph(shapeList):
    edges = {}
    for shape in shapeList:
        edges[shape['id']] = shape['below']
    return edges


def dfs(edges, parent, path, prompts):
    if len(path) >= 2 and path not in prompts:
        prompts.append(copy.copy(path))

    for edge in edges[parent]:
        dfs(edges, edge, path + [edge], prompts)
        dfs(edges, edge, path, prompts)

    if path[-1] == parent:
        path.pop(-1)


def getPromptTruths(edges: dict, maxCount=5):
    promptTruths = []
    for shape in edges.keys():
        subTruths = []
        dfs(edges, shape, [shape], subTruths)
        promptTruths += subTruths

    goodPromptTruths = [prompt for prompt in promptTruths if len(prompt) >= 3]
    badPromptTruths = [prompt for prompt in promptTruths if len(prompt) < 3]

    finalPromptTruths = random.sample(goodPromptTruths, min(maxCount, len(goodPromptTruths)))
    finalPromptTruths = finalPromptTruths + random.sample(badPromptTruths, min(maxCount-len(finalPromptTruths), len(badPromptTruths)))

    fullOrdering = [prompt for prompt in promptTruths if len(prompt) == len(edges)]
    if fullOrdering:
        finalPromptTruths[-1] = fullOrdering[0]

    # print(finalPromptTruths)

    return finalPromptTruths


def bringToTop(shapeList, shapeID):
    newShapeList = copy.deepcopy(shapeList)

    above = []
    for shape in newShapeList:
        if shapeID in shape['below']:
            above.append(shape["id"])
            shape['below'].remove(shapeID)
    
    for shape in newShapeList:
        if shape["id"] == shapeID:
            for a in above:
                shape["below"].append(a)
    
    return newShapeList, above

def bringToBottom(shapeList, shapeID):
    newShapeList = copy.deepcopy(shapeList)

    below = newShapeList[shapeID]["below"]
    for belowID in below:
        newShapeList[belowID]["below"].append(shapeID)
    
    newShapeList[shapeID]["below"] = []
    
    return newShapeList, below

def createBringToBottomPrompts(shapeList, maxNum=3, labelled=False):
    prompts = []
    labelled_prompts = []
    labelled_id_prompts = []

    topList = random.sample(range(len(shapeList)),  min(maxNum, len(shapeList)))

    for topShape in topList:
        newShapeList, below = bringToBottom(shapeList, topShape)
        if len(below) == 0:
            continue
        edges = createGraph(newShapeList)
        promptTruths = getPromptTruths(edges, 1)
        condition = "The shape " + newShapeList[topShape]["color"] + " " + newShapeList[topShape]["type"] + " is taken below "
        labelled_condition = "The shape labelled " + str(newShapeList[topShape]["label"]) + " is taken below shapes labelled "
        labelled_condition_id = "The shape labelled " + str(newShapeList[topShape]["id"]) + " is taken below shapes labelled "

        for shapeID in below:
            condition += newShapeList[shapeID]["color"] + " " + newShapeList[shapeID]["type"] + ", "
            labelled_condition += str(newShapeList[shapeID]["label"]) + ", "
            labelled_condition_id += str(newShapeList[shapeID]["id"]) + ", "
        
        condition = condition[:-2] + ". "
        labelled_condition = labelled_condition[:-2] + ". "
        labelled_condition_id = labelled_condition_id[:-2] + ". "

        newAllPrompts = createPrompts(newShapeList, promptTruths, labelled=labelled, condition=condition, labelled_condition=labelled_condition, labelled_id_condition=labelled_condition_id, promptType="counterfactual")
        prompts = prompts + newAllPrompts["prompts"]
        labelled_prompts = labelled_prompts + newAllPrompts["labelled_prompts"]
        labelled_id_prompts = labelled_id_prompts + newAllPrompts["labelled_id_prompts"]

    ret = {}
    ret["prompts"] = prompts
    ret["labelled_prompts"] = labelled_prompts
    ret["labelled_id_prompts"] = labelled_id_prompts

    return ret

def createBringToTopPrompts(shapeList, maxNum=3, labelled=False):
    prompts = []
    labelled_prompts = []
    labelled_id_prompts = []

    topList = random.sample(range(len(shapeList)),  min(maxNum, len(shapeList)))

    for topShape in topList:
        newShapeList, above = bringToTop(shapeList, topShape)
        if len(above) == 0:
            continue
        edges = createGraph(newShapeList)
        promptTruths = getPromptTruths(edges, 1)
        condition = "The shape " + newShapeList[topShape]["color"] + " " + newShapeList[topShape]["type"] + " is brought on top of "
        labelled_condition = "The shape labelled " + str(newShapeList[topShape]["label"]) + " is brought on top of shapes labelled "
        labelled_condition_id = "The shape labelled " + str(newShapeList[topShape]["id"]) + " is brought on top of shapes labelled "

        for shapeID in above:
            condition += newShapeList[shapeID]["color"] + " " + newShapeList[shapeID]["type"] + ", "
            labelled_condition += str(newShapeList[shapeID]["label"]) + ", "
            labelled_condition_id += str(newShapeList[shapeID]["id"]) + ", "
        
        condition = condition[:-2] + ". "
        labelled_condition = labelled_condition[:-2] + ". "
        labelled_condition_id = labelled_condition_id[:-2] + ". "

        newAllPrompts = createPrompts(newShapeList, promptTruths, labelled=labelled, condition=condition, labelled_condition=labelled_condition, labelled_id_condition=labelled_condition_id, promptType="counterfactual")
        prompts = newAllPrompts["prompts"] + prompts
        labelled_prompts = newAllPrompts["labelled_prompts"] + labelled_prompts
        labelled_id_prompts = newAllPrompts["labelled_id_prompts"] + labelled_id_prompts

    ret = {}
    ret["prompts"] = prompts
    ret["labelled_prompts"] = labelled_prompts
    ret["labelled_id_prompts"] = labelled_id_prompts

    return ret

def createPrompts(shapeList, promptTruths, labelled=False, condition ="", labelled_condition="", labelled_id_condition="", promptType="descriptive"):
    prompts = []
    labelled_prompts = []
    labelled_id_prompts = []

    for promptTruth in promptTruths:
        truthCopy = copy.copy(promptTruth)
        random.shuffle(truthCopy)
        
        prompt = COLORED_DESCRIPTION + condition + "Provide depth ordering from top to bottom for: "
        if len(promptTruth) == len(shapeList) :
            prompt += "all shapes."
        else:
            for shapeID in truthCopy[:-1]:
                prompt += shapeList[shapeID]['color'] + " " + shapeList[shapeID]['type'] + ", "
            prompt += shapeList[truthCopy[-1]]['color'] + " " + shapeList[truthCopy[-1]]['type'] + "."

        answerAsList = []
        prompt += " Answer only in the following format: 'color shape, color shape, ...'"
        answer = ""
        for shapeID in promptTruth:
            answer += shapeList[shapeID]['color'] + " " + shapeList[shapeID]['type'] + ", "
            answerAsList.append(shapeList[shapeID]['color'] + " " + shapeList[shapeID]['type'])

        answerList = [", ".join(permutation) for permutation in itertools.permutations(answerAsList)]

        prompts.append(dict({"prompt": prompt, "answer": answer[:-2], "answerSet": answerList, "type": promptType}))

        if labelled:
            prompt = LABELLED_DESCRIPTION + labelled_condition + "Provide depth ordering from top to bottom for: "
            prompt_id = LABELLED_DESCRIPTION + labelled_id_condition + "Provide depth ordering from top to bottom for: "
            if len(promptTruth) == len(shapeList) :
                prompt += "all shapes."
            else:
                for shapeID in truthCopy[:-1]:
                    prompt += str(shapeList[shapeID]['label']) + ", "
                    prompt_id += str(shapeList[shapeID]['id']) + ", "
                prompt += str(shapeList[truthCopy[-1]]['label']) + "."
                prompt_id += str(shapeList[truthCopy[-1]]['id']) + "."

            answerLabelledAsList = []
            answerLabelledIDAsList = []
            prompt += " Answer only in the following format: 'shapeID, shapeID, ...'"
            prompt_id += " Answer only in the following format: 'shapeID, shapeID, ...'"
            answer = ""
            answer_id = ""
            for shapeID in promptTruth:
                answer += str(shapeList[shapeID]['label']) + ", "
                answer_id += str(shapeList[shapeID]['id']) + ", "
                answerLabelledAsList.append(str(shapeList[shapeID]['label']))
                answerLabelledIDAsList.append(str(shapeList[shapeID]['id']))

            answerListLabelled = [", ".join(permutation) for permutation in itertools.permutations(answerLabelledAsList)]
            answerListLabelledID = [", ".join(permutation) for permutation in itertools.permutations(answerLabelledIDAsList)]

            answerReverse = answer[:-2].split(", ")[::-1]
            answerReverse = ", ".join(answerReverse)
            labelled_prompts.append(dict({"prompt": prompt, "answer": answer[:-2], "answerSet": answerListLabelled, "type": promptType}))
            labelled_id_prompts.append(dict({"prompt": prompt_id, "answer": answer_id[:-2], "answerReverse": answerReverse, "answerSet": answerListLabelledID, "type": promptType}))

    ret = {}
    ret["prompts"] = prompts
    ret["labelled_prompts"] = labelled_prompts
    ret["labelled_id_prompts"] = labelled_id_prompts

    return ret

def savePrompts(shapeList, allPrompts, path, i, labelled=False):
    prompts = allPrompts["prompts"]
    labelled_prompts = allPrompts["labelled_prompts"]
    labelled_id_prompts = allPrompts["labelled_id_prompts"]

    promptsWithMetadata = {}
    promptsWithMetadata["shapeList"] = [shape['color'] + " " + shape['type'] for shape in shapeList]
    promptsWithMetadata["filename"] = f"img{i}.jpg"
    promptsWithMetadata['prompts'] = prompts

    if labelled:
        promptsWithMetadata["filename_labelled"] = f"img{i}_labelled.jpg"
        promptsWithMetadata['prompts_labelled'] = labelled_prompts
        promptsWithMetadata['prompts_labelled_id'] = labelled_id_prompts
        
    with open(f"{path}/prompts{i}.json", "w") as file:
        json.dump(promptsWithMetadata, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, default="new_depth_images/truths")
    parser.add_argument('-n', '--maxcount', type=int, default=5)
    parser.add_argument('-p', '--path', type=str, default="new_depth_images/prompts")
    parser.add_argument('-l', '--labelled', action='store_true')
    
    args = parser.parse_args()

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    i = 0
    while True:
        try:
            shapeList = getData(f"{args.src}/truth{i}.json")["shapes"]
        except FileNotFoundError:
            if i == 0:
                print("No files found")
            else:
                break
        
        edges = createGraph(shapeList)
        promptTruths = getPromptTruths(edges)

        try:
            allPrompts = createPrompts(shapeList, promptTruths, args.labelled)
            topAllPrompts = createBringToTopPrompts(shapeList, 3, args.labelled)
            bottomAllPrompts = createBringToBottomPrompts(shapeList, 3, args.labelled)
            allPrompts["prompts"] = allPrompts["prompts"] + topAllPrompts["prompts"] + bottomAllPrompts["prompts"]
            allPrompts["labelled_prompts"] = allPrompts["labelled_prompts"] + topAllPrompts["labelled_prompts"] + bottomAllPrompts["labelled_prompts"]
            allPrompts["labelled_id_prompts"] = allPrompts["labelled_id_prompts"] + topAllPrompts["labelled_id_prompts"] + bottomAllPrompts["labelled_id_prompts"]
            savePrompts(shapeList, allPrompts, args.path, i, args.labelled)
        except Exception as e:
            print(f"Error in image {i}: {e}")
            i += 1
            continue
        i += 1
    