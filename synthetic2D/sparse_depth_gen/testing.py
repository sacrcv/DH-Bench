from PIL import Image, ImageFilter
import os
import random
import argparse
from utils import *
import numpy as np
import csv
import json
from shapely.affinity import translate


IMG_SIZE = [1024, 1024]
SIZE = [512, 512]
COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (192,192,192), (0,255,255), (165, 42, 42), (128, 0, 128), (255, 165, 0)]
COLOR_NAMES = ["red", "green", "blue", "yellow", "magenta", "gray", "cyan", "brown", "purple", "orange"]

def getShapes(n, overlap=1, shape_size=1):
    ShapeList = [None, None]
    ShapeFunctions = [getRectangle, getCircle, getTriangle]
    # ShapeFunctions = [getRectangle]

    for i in range(2, n+2):
        ShapeFunc = random.choice(ShapeFunctions)
        try:
            newShape = ShapeFunc(ShapeList[i-1], ShapeList[i-2], SIZE=SIZE, IMG_SIZE=IMG_SIZE, overlap=overlap, shape_size=shape_size)
        except RecursionError as err:
            print("Caution: use higher overlap/size (ideally 0.7 to 1.5)")
            return None
        ShapeList.append(newShape)


    ShapeList.pop(0)
    ShapeList.pop(0)

    boundY = [10000000, -10000000]
    boundX = [10000000, -10000000]
    for shape in ShapeList:
        boundY[0] = min(boundY[0], shape.boundsY[0])
        boundY[1] = max(boundY[1], shape.boundsY[1])
        boundX[0] = min(boundX[0], shape.boundsX[0])
        boundX[1] = max(boundX[1], shape.boundsX[1])
    center = [(boundX[0] + boundX[1])//2, (boundY[0] + boundY[1])//2]
    diff = [center[0] - IMG_SIZE[0]//2, center[1] - IMG_SIZE[1]//2]

    for shape in ShapeList:
        shape.boundsX = [shape.boundsX[0] - diff[0], shape.boundsX[1] - diff[0]]
        shape.boundsY = [shape.boundsY[0] - diff[1], shape.boundsY[1] - diff[1]]
        shape.center = [shape.center[0] - diff[0], shape.center[1] - diff[1]]
        if shape.type == "rectangle":
            shape.custom = [shape.custom[0] - diff[0], shape.custom[1] - diff[1], shape.custom[2] - diff[0], shape.custom[3] - diff[1]]
        elif shape.type == "circle":
            shape.custom = [shape.custom[0] - diff[0], shape.custom[1] - diff[1], shape.custom[2] - diff[0], shape.custom[3] - diff[1]]
        elif shape.type == "triangle":
            shape.custom = [shape.custom[0] - diff[0], shape.custom[1] - diff[1], shape.custom[2] - diff[0], shape.custom[3] - diff[1], shape.custom[4] - diff[0], shape.custom[5] - diff[1]]

        shape.shapely_obj = translate(shape.shapely_obj, xoff=-diff[0], yoff=-diff[1])

    return ShapeList

def getBackgroundImage(bgpath):
    if bgpath is None:
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


def createMask(shapeList, path, id, bgpath=None):
    
    im = getBackgroundImage(bgpath)

    for shape in shapeList:
        if shape.type == "rectangle":
            drawRectangle(im, shape)
        elif shape.type == "circle":
            drawCircle(im, shape)
        elif shape.type == "triangle":
            drawTriangle(im, shape)

    im.save(f"{path}/mask{id}.jpg")


def createColored(shapeList, path, id, bgpath=None):
    
    im = getBackgroundImage(bgpath)
    colorIndices = random.sample(range(len(COLORS)), len(shapeList))
    for j, shape in enumerate(shapeList):
        color = COLORS[colorIndices[j]]
        if shape.type == "rectangle":
            drawRectangle(im, shape, color)
        elif shape.type == "circle":
            drawCircle(im, shape, color)
        elif shape.type == "triangle":
            drawTriangle(im, shape, color)

        shape.color = COLOR_NAMES[colorIndices[j]]

    im.save(f"{path}/img{id}.jpg")


def createPlot(shapeList):
    fig, ax = plt.subplots()

    for j, shape in enumerate(shapeList):
        ax.add_patch(shape.plt_shape)
    
    ax.axis('equal')
    ax.set_xlim(0, IMG_SIZE[0])
    ax.set_ylim(0, IMG_SIZE[1])

    plt.show()


def createTruths(shapeList, path, id, bgfile=""):
    truths = []
    # shapes are bottom first
    labels = random.sample(range(100), len(shapeList))

    for i, shape1 in enumerate(shapeList):
        truth = {}
        truth["id"] = i
        truth["label"] = labels[i]
        truth["color"] = shape1.color
        truth["type"] = shape1.type

        below = set()
        for j, shape2 in enumerate(shapeList[0:i]):
            if checkOverlap(shape1, shape2) == True:
                below.add(j)

        truth["below"] = list(below)

        truths.append(truth)
    
    print(id, str(truths))

    # with open(f'{path}/truth{id}', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     for truth in truths:
    #         writer.writerow(truth)

    truths = {"shapes": truths, "bg": bgfile, "labels": labels}
    with open(f'{path}/truth{id}.json', 'w') as file:
        json.dump(truths, file, indent=4)

    
def createShapeFiles(shapeList, path, id, bgpath=None):
    for i, shape in enumerate(shapeList):
        im = getBackgroundImage(bgpath)
        if shape.type == "rectangle":
            drawRectangle(im, shape)
        elif shape.type == "circle":
            drawCircle(im, shape)
        elif shape.type == "triangle":
            drawTriangle(im, shape)

        im.save(f"{path}/img{id}_shape{i}.jpg")

def createShapesVisible(shapeList, path, id, bgpath=None):
    for i, shape in enumerate(shapeList):
        im = getBackgroundImage(bgpath)
        result = shape.shapely_obj
        for j, above in enumerate(shapeList[i+1:]):
            result = result.difference(above.shapely_obj)

        try:
            drawPolygon(im, result)
        except Exception as e:
            print(f"Error in image {id} shape {i}: {e}")
        im.save(f"{path}/img{id}_shape{i}.jpg")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_shapes', type=int, default=4)
    parser.add_argument('-c', '--count', type=int, default=5)
    parser.add_argument('-p', '--path', type=str, default="new_depth_images")
    parser.add_argument('-o', '--overlap', type=float, default=1)
    parser.add_argument('-s', '--shape_size', type=float, default=1)
    parser.add_argument('-l', '--put_bg', action='store_true')
    parser.add_argument('-b', '--bg_dir', type=str, default="../real_world_bg")

    args = parser.parse_args()
    if not os.path.exists(args.path):
        os.makedirs(args.path)
        os.makedirs(f"{args.path}/masks")
        os.makedirs(f"{args.path}/imgs")
        os.makedirs(f"{args.path}/truths")
        os.makedirs(f"{args.path}/shapes")
        os.makedirs(f"{args.path}/visible")

    # for id in [99, 269, 321, 341]:
    for id in range(args.count):
        shapeList = getShapes(args.num_shapes, overlap=args.overlap, shape_size=args.shape_size)
        print(id, len(shapeList))
        if(shapeList == None):
            continue
        random.shuffle(shapeList)

        bgpath = None
        bgfile = None
        if args.put_bg:
            bgfile = random.choice(os.listdir(args.bg_dir))
            bgpath = f"{args.bg_dir}/{bgfile}"

        createMask(shapeList, f"{args.path}/masks", id, bgpath=None)
        createColored(shapeList, f"{args.path}/imgs", id, bgpath=bgpath)
        createTruths(shapeList, f"{args.path}/truths", id, bgfile=bgfile)
        # createPlot(shapeList)
        createShapeFiles(shapeList, f"{args.path}/shapes", id, bgpath=None)
        createShapesVisible(shapeList, f"{args.path}/visible", id, bgpath=None)
        




    