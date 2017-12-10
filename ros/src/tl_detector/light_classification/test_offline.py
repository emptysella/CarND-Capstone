import cv2
import tl_simulator_mode
import tl_carla_mode



class TLClassifier():

    def __init__(self):
        
         print("classification statrted...")
         
         self.GREEN = 'RUN'
         self.RED   = 'STOP'
         
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





#******************************************************************************

TL = TLClassifier()

imagePath = 'green_8.png'
imagePath = 'image210.jpg'
imagePath = 'red_0.png'
imagePath = 'red_0.png'
imagePath = '/Volumes/Samsung_T5/MORE_PROJECTS/SDC-System-Integration/test_images/just_traffic_light_0500.jpg'

imageA = cv2.imread(imagePath)  # uint8 image

TL.get_classification(imageA)
    
