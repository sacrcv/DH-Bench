from PIL import Image, ImageDraw
import numpy as np
from classes import *
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, Point
import copy

PI = np.pi

IMG_SIZE = [1024, 1024]
SIZE = [512, 512]

#Creates a new center nearer to prevShape1 than prevShape2, with a hard coded threshold distance
def generateCenter(prevShape1, prevShape2, overlap=1, shape_size=1):
    center1 = prevShape1.center
    center2 = prevShape2.center

    minDist = (prevShape1.width() + prevShape1.height()) / 2
    minDist = shape_size*minDist*1.5/overlap
    maxDist = minDist*2.5 #experiment with ratio

    angle = random.uniform(0, 2 * PI)
    distance = random.uniform(minDist, maxDist)

    offsetX = distance * np.cos(angle)
    offsetY = distance * np.sin(angle)

    randomPoint = (center1[0] + offsetX, center1[1] + offsetY)

    distanceToPrev1 = distance
    distanceToPrev2 = np.sqrt((randomPoint[0] - center2[0])**2 + (randomPoint[1] - center2[1])**2)

    while distanceToPrev1 >= distanceToPrev2:
        
        angle = random.uniform(0, 2 * PI)
        distance = random.uniform(minDist, maxDist)
        offsetX = distance * np.cos(angle)
        offsetY = distance * np.sin(angle)

        randomPoint = (center1[0] + offsetX, center1[1] + offsetY)

        distanceToPrev1 = distance
        distanceToPrev2 = np.sqrt((randomPoint[0] - center2[0])**2 + (randomPoint[1] - center2[1])**2)

    return distance, randomPoint


def getRectangle(prevShape1=None, prevShape2=None, SIZE=SIZE, IMG_SIZE=IMG_SIZE, overlap=1, shape_size=1):

    if prevShape1 is not None and prevShape2 is not None:
        distance, center = generateCenter(prevShape1, prevShape2, overlap=overlap, shape_size=shape_size)

    elif prevShape2 is None and prevShape1 is not None:
        angle = random.uniform(0, 2 * PI)
        distance = random.uniform(SIZE[0]/4.5, SIZE[0]/4)

        offsetX = distance * np.cos(angle)
        offsetY = distance * np.sin(angle)

        center = np.array([int(IMG_SIZE[0]/2) + offsetX, int(IMG_SIZE[0]/2) + offsetY])

    else:
        center = np.array([int(IMG_SIZE[0]/2), int(IMG_SIZE[0]/2)])

    dim = [random.randint(int(SIZE[0]/4), int(SIZE[0]/2.5))*shape_size, random.randint(int(SIZE[0]/4), int(SIZE[0]/2.5))*shape_size]
    bbox = [center[0] - dim[0]/2, center[1] - dim[1]/2, center[0] + dim[0]/2, center[1] + dim[1]/2]

    plt_rect = patches.Rectangle((bbox[0], bbox[1]), dim[0], dim[1], edgecolor='black')
    shapely_rect = create_shapely_rect((bbox[0], bbox[1]), dim[0], dim[1])
    newRect = Shape(type="rectangle",  center=center, boundsX=[bbox[0], bbox[2]], boundsY=[bbox[1], bbox[3]], custom=bbox, plt_shape=plt_rect, shapely_obj=shapely_rect)

    if prevShape1 and not checkOverlapArea(newRect, prevShape1):
        newRect = getRectangle(prevShape1, prevShape2, SIZE, IMG_SIZE, overlap, shape_size)

    return newRect


def drawRectangle(image, rectangle, color=(255, 0, 0), width=4):
    draw = ImageDraw.Draw(image)
    draw.rectangle(xy=rectangle.custom, fill=color, outline=(0, 0, 0), width=width)


def getCircle(prevShape1=None, prevShape2=None, SIZE=SIZE, IMG_SIZE=IMG_SIZE, overlap=1, shape_size=1):
    if prevShape1 is not None and prevShape2 is not None:
        distance, center = generateCenter(prevShape1, prevShape2, overlap=overlap)
    
    elif prevShape2 is None and prevShape1 is not None:
        angle = random.uniform(0, 2 * PI)
        distance = random.uniform(SIZE[0]/(4.5*overlap), SIZE[0]/(3.8*overlap))
        
        offsetX = distance * np.cos(angle)
        offsetY = distance * np.sin(angle)
        
        center = np.array([int(IMG_SIZE[0]/2) + offsetX, int(IMG_SIZE[0]/2) + offsetY])

    else:
        center = np.array([int(IMG_SIZE[0]/2), int(IMG_SIZE[0]/2)])

    radius = random.randint(int(SIZE[0]/5.5), int(SIZE[0]/4))*shape_size
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

    plt_circle = patches.Circle(center, radius, edgecolor='black')
    shapely_circle = Point(center).buffer(radius)
    newCircle = Shape(type="circle", center=center, boundsX=[bbox[0], bbox[2]], boundsY=[bbox[1], bbox[3]], custom=bbox, plt_shape=plt_circle, shapely_obj=shapely_circle)

    if prevShape1 and not checkOverlapArea(newCircle, prevShape1):
        newCircle = getCircle(prevShape1, prevShape2, SIZE, IMG_SIZE, overlap, shape_size)

    return newCircle


def drawCircle(image, circle, color=(255, 0, 0)):
    draw = ImageDraw.Draw(image)
    draw.ellipse(xy=circle.custom, fill=color, outline=(0, 0, 0), width=4)


def getTriangle(prevShape1=None, prevShape2=None, SIZE=SIZE, IMG_SIZE=IMG_SIZE, overlap=1, shape_size=1):
    if prevShape1 is not None and prevShape2 is not None:
        distance, center = generateCenter(prevShape1, prevShape2, overlap=overlap)

    elif prevShape2 is None and prevShape1 is not None:
        angle = random.uniform(0, 2 * PI)
        distance = random.uniform(SIZE[0]/(4.5*overlap), SIZE[0]/(4*overlap))

        offsetX = distance * np.cos(angle)
        offsetY = distance * np.sin(angle)

        center = np.array([int(IMG_SIZE[0]/2) + offsetX, int(IMG_SIZE[0]/2) + offsetY])

    else:
        center = np.array([int(IMG_SIZE[0]/2), int(IMG_SIZE[0]/2)])

    radius = random.randint(int(SIZE[0]/(4.5)), int(SIZE[0]/3.5))*shape_size

    angle1 = random.uniform(-PI/12, PI/12)
    angle2 = 2*PI/3 + random.uniform(-PI/9, PI/9)
    angle3 = -2*PI/3 + random.uniform(-PI/9, PI/9)

    x1 = center[0] + radius * np.cos(angle1)
    y1 = center[1] + radius * np.sin(angle1)
    x2 = center[0] + radius * np.cos(angle2)
    y2 = center[1] + radius * np.sin(angle2)
    x3 = center[0] + radius * np.cos(angle3)
    y3 = center[1] + radius * np.sin(angle3)

    plt_triangle = patches.Polygon([(x1,y1), (x2,y2), (x3,y3)], edgecolor='black')
    shapely_triangle = Polygon([(x1,y1), (x2,y2), (x3,y3)])
    newTriangle = Shape(type="triangle", center=center, boundsX=[min(x1,x2,x3), max(x1,x2,x3)], boundsY=[min(y1,y2,y3), max(y1,y2,y3)], custom=[x1, y1, x2, y2, x3, y3], plt_shape=plt_triangle, shapely_obj=shapely_triangle)

    if prevShape1 and not checkOverlapArea(newTriangle, prevShape1):
        newTriangle = getTriangle(prevShape1, prevShape2, SIZE, IMG_SIZE, overlap, shape_size)

    return newTriangle


def drawTriangle(image, triangle, color=(255, 0, 0)):
    draw = ImageDraw.Draw(image)
    draw.polygon(xy=triangle.custom, fill=color, outline=(0, 0, 0), width=4)


def create_shapely_rect(bottom_left, width, height):
    bottom_right = (bottom_left[0] + width, bottom_left[1])
    top_right = (bottom_left[0] + width, bottom_left[1] + height)
    top_left = (bottom_left[0], bottom_left[1] + height)

    rectangle = Polygon([bottom_left, bottom_right, top_right, top_left, bottom_left])
    return rectangle


def checkOverlapArea(shape1, shape2, thresh=150):
    return (shape1.shapely_obj.intersection(shape2.shapely_obj).area) > thresh

def checkOverlap(shape1, shape2):
    return shape1.shapely_obj.intersects(shape2.shapely_obj)

def drawPolygon(image, shapely_obj, color=(255, 0, 0)):
    draw = ImageDraw.Draw(image)

    try:
        if shapely_obj.geom_type == 'Polygon':
            draw.polygon(shapely_obj.exterior.coords, fill=color, outline=(0, 0, 0), width=4)
        elif shapely_obj.geom_type == 'MultiPolygon':
            for polygon in shapely_obj.geoms:
                draw.polygon(polygon.exterior.coords, fill=color, outline=(0, 0, 0), width=4)
    except Exception as e:
        raise Exception(f"Error in drawing polygon: {e}")