from PIL import Image, ImageDraw
import os
import numpy as np
import random
import math

SIZE_DEF = (512, 512)


def drawCircle(image, center, radius=None, color=(255, 0, 0), SIZE=SIZE_DEF):
    if radius is None:
        radius = random.randint(int(SIZE[0]/6), int(SIZE[0]/4.5))

    draw = ImageDraw.Draw(image)

    box = (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)
    draw.ellipse(xy=box, fill=color, outline=(0, 0, 0), width=4)

    return radius


def drawRectangle(image, center, dim=None, color=(255, 0, 0), SIZE=SIZE_DEF):
    if dim is None:
        dim = (random.randint(int(SIZE[0]/4), int(SIZE[0]/2.5)), random.randint(int(SIZE[0]/4), int(SIZE[0]/2.5)))

    draw = ImageDraw.Draw(image)

    box = box = (center[0] - dim[0]/2, center[1] - dim[1]/2, center[0] + dim[0]/2, center[1] + dim[1]/2)

    draw.rectangle(xy=box, fill=color, outline=(0, 0, 0), width=4)


def drawTriangle(image, center, coords=None, color=(255, 0, 0), SIZE=SIZE_DEF):
    if coords is None:
        coords = np.array([[random.randint(int(SIZE[0]/5), int(SIZE[0]/3.5)), 0]])
        coords = np.vstack([coords, [-coords[0][0], -random.randint(0, int(SIZE[1]/6))]])
        coords = np.vstack([coords, [0, random.randint(int(SIZE[0]/5), int(SIZE[0]/3.5))]])

    coords = coords + np.array(center)
    angle = random.randint(0, 180)
    coords = np.array([rotateAroundPivot(coord, center, angle) for coord in coords])

    coordsList = coords.flatten().tolist()

    draw = ImageDraw.Draw(image)

    draw.polygon(xy=coordsList, fill=color, outline=(0, 0, 0), width=4)

    return coords


def rotateAroundPivot(point, center, angle):

    x, y = point
    cx, cy = center

    theta = math.radians(angle)

    # Apply rotation formulas with respect to the pivot point
    x_prime = (x - cx) * math.cos(theta) - (y - cy) * math.sin(theta) + cx
    y_prime = (x - cx) * math.sin(theta) + (y - cy) * math.cos(theta) + cy

    return x_prime, y_prime