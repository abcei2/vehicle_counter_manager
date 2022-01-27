import cv2
import copy
import numpy as np
import json

class ZoneConfig:
    def __init__(self, polys):

        self.polys = polys #{ "name": [ 40 40 100 100]}
    
