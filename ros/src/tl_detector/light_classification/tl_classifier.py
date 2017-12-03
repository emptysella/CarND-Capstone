from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import rospy
from tl_SWRP import TLClassifier_SWRP
from tl_SRG import TLClassifier_SRG

class TLClassifier(object):
    
    def __init__(self):
         print("classification statrted...")
         self.TL_SWRP = TLClassifier_SWRP
         self.TL_SRG  = TLClassifier_SRG
         
         
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
             
        # sanity check initialization 
        light = TrafficLight.UNKNOWN
              
        """
        Run the classifier of Swaroop
        _______________________________________________________________________
        """
        
        light_swrp = self.TL_SWRP.classifyTL(image)
        
        
        """
        Run the classifier of Sergio
        _______________________________________________________________________
        """
        light_srg = self.TL_SRG.classifyTL(image)

       

        """
        Run the classifier of Felix
        _______________________________________________________________________     
        """
        
        
        
        #_______________________________________________________________________
        
        light_type = light_swrp
        
        if light_type == 0:
            light = TrafficLight.UNKNOWN
            
        if light_type == 1:
            light = TrafficLight.GREEN
            
        if light_type == 2:
            light = TrafficLight.YELLOW
            
        if light_type == 3:
            light = TrafficLight.RED
        
        
        return light



