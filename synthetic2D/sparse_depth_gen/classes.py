import numpy as np

class Shape:
    type = None
    center = None
    boundsX = None
    boundsY = None

    def __init__(self, type, center, boundsX, boundsY, custom, plt_shape, shapely_obj, color="red"):
        self.type = type
        self.center = np.array(center)
        self.boundsX = np.array(boundsX)
        self.boundsY = np.array(boundsY)
        self.custom = custom
        self.plt_shape = plt_shape
        self.shapely_obj = shapely_obj
        self.color = color
        
    def width(self):
        return np.absolute(self.boundsX[1] - self.boundsX[0])/2

    def height(self):
        return np.absolute(self.boundsY[1] - self.boundsY[1])/2