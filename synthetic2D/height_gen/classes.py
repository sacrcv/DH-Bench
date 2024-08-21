import numpy as np

class Rectangle:
    center = None
    boundsX = None
    boundsY = None
    label = None

    def __init__(self, center, boundsX, boundsY, height, width, shapely_obj=None):
        self.center = np.array(center)
        self.boundsX = np.array(boundsX)
        self.boundsY = np.array(boundsY)
        self.shapely_obj = shapely_obj
        self.height = height
        self.width = width