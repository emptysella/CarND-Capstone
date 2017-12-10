import cv2
import tl_SWRP 
import tl_SRG
import numpy as np


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
        light_out = TL_SRG.classifyTL(image)
        
        if light_out:
            print("STOP")
        else:
            print("RUN")
            
            
        #boundingBox = image[crop[1]:crop[3],crop[0]:crop[2],:]*255
        #cv2.imwrite('draw_crop.jpg'  ,boundingBox)
        

        """
        Run the classifier of Swaroop
        _______________________________________________________________________
        """

        #light_out = TL_SWRP.classifyTL(boundingBox.astype(np.uint8))

        #print(light_out)



        """
        Run the classifier of Felix
        _______________________________________________________________________
        """



        #_______________________________________________________________________

        light_type = light_out

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

imagePath = 'green_8.png'
imagePath = 'image210.jpg'
imagePath = 'red_0.png'
imagePath = 'red_0.png'
imagePath = '/Volumes/Samsung_T5/MORE_PROJECTS/SDC-System-Integration/test_images/just_traffic_light_0500.jpg'


imageA = cv2.imread(imagePath)  # uint8 image


TL.run(imageA)
    
