import cv2
import numpy as np
import matplotlib.image as mpimg
import tl_SWRP
import tl_SRG


class TLClassifier():

    def __init__(self):
         print("classification statrted...")
         
         self.UNKNOWN  = 'unknow'
         self.GREEN    = 'green'
         self.YELLOW   = 'yellow'
         self.RED      = 'red'


    def run(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        TL_SWRP = tl_SWRP.TLClassifier_SWRP()
        TL_SRG  = tl_SRG.TLClassifier_SRG()
        # sanity check initialization
        light = self.UNKNOWN


        """
        Run the classifier of Sergio
        _______________________________________________________________________
        """
        DETECTIONS = TL_SRG.classifyTL(image)

        """
        Run the classifier of Swaroop
        _______________________________________________________________________
        """

        lights = []
        for detetcion in DETECTIONS:
            # To modify--- crear crop
            boundingBox = image
            light_out = TL_SWRP.classifyTL(boundingBox)
            lights.append(light_out)

        #### TODO: Logic to take a decision


        light_swrp = TL_SWRP.classifyTL(image)


        """
        Run the classifier of Felix
        _______________________________________________________________________
        """



        #_______________________________________________________________________

        light_type = light_swrp

        if light_type == 0:
            light = self.UNKNOWN

        if light_type == 1:
            light = self.GREEN

        if light_type == 2:
            light = self.YELLOW

        if light_type == 3:
            light = self.RED

        # return light
        return light




#******************************************************************************

TL = TLClassifier()
imagePath = 'image210.jpg'
imageA = mpimg.imread(imagePath)
TL.run(imageA)
