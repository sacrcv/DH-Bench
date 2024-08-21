import argparse

from PIL import Image, ImageDraw
import os
import numpy as np
import random
from drawShapes import *


SIZE = [1024, 1024]
COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (192,192,192), (0,0,0)]


def getCenters(n=5):
    coords = []

    for i in range(n):
        coords.append([random.randint(int(SIZE[0]/4), int(3*SIZE[0]/4)), random.randint(int(SIZE[0]/4), int(3*SIZE[0]/4))])

    return coords


def generateShapes(image, n=5, centers=None, colorShape=False):
    if centers is None:
        centers = getCenters(n)

    shapeGen = [drawRectangle, drawTriangle, drawCircle]

    color = (255,0,0)
    for idx, center in enumerate(centers):
        if colorShape is True:
            color = COLORS[idx]
        shapeGen[random.randint(0, len(shapeGen)-1)](image, center, color=color, SIZE=image.size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--count', type=int, default=1)
    parser.add_argument('-o', '--path', type=str, default="depth_images")
    parser.add_argument('--mono', action='store_true')
    args = parser.parse_args()
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    for i in range(args.count):
        im = Image.new(mode="RGB", size=SIZE, color=(255, 255, 255))
        generateShapes(im, colorShape=not args.mono)
        im.save(f"{args.path}/img{i}.png")