from styx_msgs.msg import TrafficLight

import tl_SWRP 
import tl_SRG




class TLClassifier(object):
    
    def __init__(self):
        
         print("classification statrted...")
         
         self.mode = 'SIMULATOR'
         
         self.classifier_SIMULATOR  = tl_SWRP.TLClassifier_SWRP()
         self.classifier_CARLA      = tl_SRG.TLClassifier_SRG()


         
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
                # STOP
                light = TrafficLight.RED
                
            else:
                # RUN
                light = TrafficLight.GREEN
            
            
        """
        Classifier in Mode SIMULATOR 
        _______________________________________________________________________
        """  

        if self.mode == 'SIMULATOR' :
            
            light_out = self.classifier_SIMULATOR.classifyTL(image)
            
            if light_out:
                # STOP
                light = TrafficLight.RED

            else:
                # RUN
                light = TrafficLight.GREEN
                      

        # return light
        return light
    
    

