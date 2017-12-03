import cv2
import numpy as np

class TLClassifier_SRG(object):

    def __init__(self):
        self.blocksize = 96
        
        """
        light code:
            1 == green light
            2 == yellow light
            3 == red light
            0 == unknow
        """
        self.light = 0 
        pass

    def classifyTL(self, image_data):
        
        
        return self.light