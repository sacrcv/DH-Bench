from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import random
import numpy as np
import json
import copy
from classes import *
import argparse

IMG_SIZE = [1500, 1024]

COLOR_NAMES = [
    "red", "light green", "dark green", "light blue", "dark blue", 
    "yellow", "cyan", "orange", "purple", "gray", "pink", "brown", 
    "mustard", "olive green", "dark pink", "lavender", "maroon", 
    "lime", "white", "dark orange"
]

COLORS = [
    (255, 0, 0), (144, 238, 144), (0, 100, 0), (173, 216, 230), (0, 0, 139), 
    (255, 255, 0), (0, 255, 255), (255, 165, 0), (128, 0, 128), (128, 128, 128), 
    (255, 192, 203), (165, 42, 42), (255, 219, 88), (128, 128, 0), (231, 84, 128), 
    (230, 230, 250), (128, 0, 0), (0, 255, 0), (255, 255, 255), (255, 140, 0)
]

def getBackgroundImage(bgpath=""):  
    if bgpath == "" or bgpath is None:
        return Image.new(mode="RGB", size=IMG_SIZE, color=(255, 255, 255))
    bg = Image.open(bgpath)
    width, height = bg.size
    left = (width - height) // 2
    top = 0
    right = (width + height) // 2
    bottom = height

    # Crop the image to a square and filter
    cropped_img = bg.crop((left, top, right, bottom))
    cropped_img = cropped_img.resize(IMG_SIZE).filter(ImageFilter.GaussianBlur(5))

    return cropped_img

def getShapeStack(coords: list, maxNum=3, stackNum = 2):
    shapeStack = []
    currBase = copy.deepcopy(coords)

    for i in range(maxNum):
        nextShape = getShape(currBase, IMG_SIZE=IMG_SIZE, pos=i, stackNum=stackNum, numShapes=maxNum)
        currBase[1] -= nextShape.height
        shapeStack.append(nextShape)
    return shapeStack

def getShape(currBase: list, IMG_SIZE, pos, stackNum=2, numShapes=3):
    maxWidth = (IMG_SIZE[0] // stackNum)*9//10
    minWidth = (IMG_SIZE[0] // stackNum)*2//10
    maxHeight = (IMG_SIZE[1] // numShapes)*9//10
    minHeight = (IMG_SIZE[1] // numShapes)*3//10

    width = random.randint(minWidth, maxWidth)
    height = random.randint(minHeight, maxHeight)

    center = [currBase[0], currBase[1] - height//2]
    boundsX = [center[0] - width/2, center[0] + width/2]
    boundsY = [currBase[1] - height, currBase[1]]

    return Rectangle(center, boundsX, boundsY, height, width)

def drawRectangle(im, shape, label, color=(255,0,0), thickness=2, mirror=False):
    if mirror:
        color = shape.color
    if not mirror:
        draw = ImageDraw.Draw(im)
        draw.rectangle([(shape.boundsX[0], shape.boundsY[0]), (shape.boundsX[1], shape.boundsY[1])], fill=color, outline=(0, 0, 0), width=thickness)
        font = ImageFont.truetype("gidole.ttf", 40)
        draw.text((shape.center[0] - 20, shape.center[1] - 20), str(label), font=font, fill=(0, 0, 0))
    else:
        width = im.size[0]
        draw = ImageDraw.Draw(im)
        draw.rectangle([(width - shape.boundsX[1], shape.boundsY[0]), (width - shape.boundsX[0], shape.boundsY[1])], fill=color, outline=(0, 0, 0), width=thickness)
        font = ImageFont.truetype("gidole.ttf", 40)
        draw.text((width - shape.center[0] - 20, shape.center[1] - 20), str(label), font=font, fill=(0, 0, 0))


def createImages(path, stackNum, maxNum, id, mirror=False, bgpath="", color=False):
    im = getBackgroundImage(bgpath)
    bg = copy.deepcopy(im)
    startX = (IMG_SIZE[0]//stackNum)//2

    draw = ImageDraw.Draw(im)
    draw.rectangle([(0, IMG_SIZE[1] - 30), (IMG_SIZE[0], IMG_SIZE[1])], fill=(0, 0, 0), outline=(0, 0, 0))

    shapeIdx = 0
    shapeStacks = {}
    
    indices = [i for i in range(len(COLORS))]
    random.shuffle(indices)
    colors = [COLORS[i] for i in indices]
    color_names = [COLOR_NAMES[i] for i in indices]

    shape_cnt = 0
    for j in range (stackNum):
        coords = [startX + j*(IMG_SIZE[0]//stackNum), IMG_SIZE[1] - 30]
        shapeStack = getShapeStack(coords, maxNum=maxNum, stackNum=stackNum) #Bottom to top
        
        for shape in shapeStack:
            label = shapeIdx
            shape_color = (255, 0, 0)
            if color:
                label = ""
                shape_color = colors[shape_cnt]
            drawRectangle(im, shape, label=label, color=shape_color)
            shape.label = label
            shape.color = color_names[shape_cnt]
            shape.id = shapeIdx
            shape.rgb = shape_color
            shape_cnt += 1
            shapeIdx += 1
        
        shapeStacks[j] = shapeStack
    
    im.save(f"{path}/img{id}.jpg")

    if mirror:
        im2 = bg
        draw = ImageDraw.Draw(im2)
        draw.rectangle([(0, IMG_SIZE[1] - 30), (IMG_SIZE[0], IMG_SIZE[1])], fill=(0, 0, 0), outline=(0, 0, 0))
        for j in range(len(shapeStacks)):
            for shape in shapeStacks[j]:
                drawRectangle(im2, shape, label=shape.label, mirror=True)
        im2.save(f"{path}/img{id}_mirror.jpg")

    return shapeStacks

def createSteppedImages(path, shapeStacks, id, bgpath="", color=False):
    # get the tallest stack
    tallStack = max(shapeStacks, key=lambda x: sum([shape.height for shape in shapeStacks[x]]))

    # from tallest stack remove tallest shape
    idx = 0
    maxH = 0
    for i, shape in enumerate(shapeStacks[tallStack]):
        if shape.height > maxH:
            maxH = shape.height
            idx = i
    print(f"Removing shape with height {maxH} from stack {tallStack}")
    shapeStacks[tallStack].pop(idx)

    # decrease height of all shapes before the removed shape by maxH
    for i in range(idx):
        shapeStacks[tallStack][i].boundsY[0] -= maxH
        shapeStacks[tallStack][i].boundsY[1] -= maxH
        shapeStacks[tallStack][i].center[1] -= maxH

    im = getBackgroundImage(bgpath)

    for i in range(len(shapeStacks)):
        for shape in shapeStacks[i]:
            drawRectangle(im, shape, label=shape.label, color=shape.rgb)

    # create ground
    draw = ImageDraw.Draw(im)
    draw.rectangle([(0, IMG_SIZE[1] - 30), (IMG_SIZE[0], IMG_SIZE[1])], fill=(0, 0, 0), outline=(0, 0, 0))
    draw.rectangle([(shapeStacks[tallStack][0].boundsX[0] - 20, shapeStacks[tallStack][0].boundsY[1]), (shapeStacks[tallStack][0].boundsX[1] + 20, IMG_SIZE[1] - 30)], fill=(0, 0, 0), outline=(0, 0, 0))

    im.save(f"{path}/img{id}.jpg")
    print(f"Saved stepped image {id} at {path}/img{id}.jpg")

    return shapeStacks

def createTruths(path, shapeStacks, id, bgfile="", color=False):
    truths = {}
    truths["filename"] = f"img{id}.jpg"
    truths["stacks"] = []
    truths["numShapes"] = 0
    truths["background"] = bgfile

    for i in range(len(shapeStacks)):
        shapeStack = shapeStacks[i]
        stack = {}
        stack["id"] = chr(ord('A') + i)
        stack["shapes"] = []
        stack["ordering"] = []
        stack["totalHeight"] = 0
        for j, shape in enumerate(shapeStack):
            if color:
                stack["shapes"].append({"id": shape.id, "height": shape.height, "width": shape.width, "color": shape.color, "rgb": shape.rgb})
                stack["ordering"].append(shape.color)
            else:
                stack["shapes"].append({"id": shape.label, "height": shape.height, "width": shape.width})
                stack["ordering"].append(shape.label)           
            stack["totalHeight"] += shape.height
            truths["numShapes"] += 1
        
        truths["stacks"].append(stack)
    
    with open(f"{path}/truth{id}.json", 'w') as f:
        json.dump(truths, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stacks', type=int, default=2)
    parser.add_argument('-n', '--num_shapes', type=int, default=4)
    parser.add_argument('-c', '--count', type=int, default=5)
    parser.add_argument('-p', '--path', type=str, default="height_images")
    parser.add_argument('-t', '--stepped', action='store_true', default=False)
    parser.add_argument('-m', '--mirror', action='store_true')
    parser.add_argument('-l', '--put_bg', action='store_true')
    parser.add_argument('-b', '--bg_dir', type=str, default="../real_world_bg")
    parser.add_argument('-r', '--color', action='store_true', default=False)

    args = parser.parse_args()

    stackNum = args.stacks
    maxNum = args.num_shapes
    
    if not os.path.exists(args.path):
        os.makedirs(args.path)
        os.makedirs(f"{args.path}/imgs")
        os.makedirs(f"{args.path}/truths")
    
    if args.stepped == True:
        if not os.path.exists(f"{args.path}/stepped_imgs"):
            os.makedirs(f"{args.path}/stepped_imgs")
        if not os.path.exists(f"{args.path}/stepped_truths"):
            os.makedirs(f"{args.path}/stepped_truths")
            
    # with open("empty.json", "r") as file:
    #     empty = json.load(file)
    # for i in empty["indices"]:
    for i in range(args.count):

        bgpath = ""
        bgfile = ""
        if args.put_bg:
            bgfile = random.choice(os.listdir(args.bg_dir))
            bgpath = f"{args.bg_dir}/{bgfile}"

        shapeStacks = createImages(f"{args.path}/imgs", stackNum, maxNum, i, mirror=args.mirror, bgpath=bgpath, color=args.color)
        createTruths(f"{args.path}/truths", shapeStacks, i, bgfile=bgfile, color=args.color)

        if args.stepped:
            shapeStacks = createSteppedImages(f"{args.path}/stepped_imgs", shapeStacks, i, bgpath=bgpath)
            createTruths(f"{args.path}/stepped_truths", shapeStacks, i, bgfile=bgfile, color=args.color)

