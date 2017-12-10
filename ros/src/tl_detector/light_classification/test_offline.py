import cv2
import tl_simulator_mode
import tl_carla_mode
import glob
import numpy as np


class TLClassifier():

    def __init__(self):
        
         print("classification statrted...")
         
         self.GREEN = 1
         self.RED   = 0
         
         self.mode =  'CARLA'   # 'SIMULATOR' or 'CARLA'
         
         self.classifier_SIMULATOR  = tl_simulator_mode.TLClassifier_SWRP()
         self.classifier_CARLA      = tl_carla_mode.TLClassifier_SRG()


         
    def get_classification(self, image):
        
        """
        Determines the color of the traffic light in the image

            Args:       image (cv::Mat): image containing the traffic light
            Returns:    int: ID of traffic light color (specified in styx_msgs/TrafficLight)           
        """
        
        
        """
        Classifier in Mode Carla 
        _______________________________________________________________________
        """
        if self.mode == 'CARLA' :
            
            light_out = self.classifier_CARLA.classifyTL(image)
            
            if light_out:
                light = self.RED
                print("STOP")
            else:
                light = self.GREEN
                print("RUN")
            
            
        """
        Classifier in Mode SIMULATOR 
        _______________________________________________________________________
        """  

        if self.mode == 'SIMULATOR' :
            
            light_out = self.classifier_SIMULATOR.classifyTL(image)
            
            if light_out:
                light = self.RED
                print("STOP")
            else:
                light = self.GREEN
                print("RUN")          
                

        # return light
        return light




"""
******************************************************************************

This is a off-line test of the SIMULATOR mode
In this file you can finad an exact copy of the classifier 


To run the script unzip the validation_images.zip in the same folder that the script

"""


TL = TLClassifier()

"""
imagePath = 'red_0.png'
imagePath = './validation_images/*.jpg'
"""

images  = glob.glob('./validation_images/*.jpg')
validations = np.zeros(104) 
validations[75:104] = 1

idx = 0
true_possitives = 0
false_possitive = 0
for image in images:
    
    print(image)
    image_ = cv2.imread(image)  # uint8 image
    light = TL.get_classification(image_)
    
    if light ==  validations[idx]:
        true_possitives += 1
    else:
        false_possitive += 1 
        
    idx +=1

print("*********************************************************")
print("*********************************************************")
print("True Possitives: " + str(true_possitives)) 
print("False Possitives: " + str(false_possitive))    






    
