from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import rospy

class TLClassifier(object):
    
    def __init__(self):
         print("classification statrted...")
         
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
		# Colour converstion from BGR to HSV
        Luma = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Run the classifier of Swaroop
        light_type  = self.classifyTL(Luma)
        
        return light_type


    def classifyTL(self, image_data):
        # get the image center geometry
        blocksize = 96 # 64x64 block size
        stride = blocksize/ 2
        nredcircles = 0
        nyellowcircles = 0
        ngreencircles = 0
        # Threshold  colors for HSV block to exact red high , red low  and yellow block bit masks
        redImageLowTh = cv2.inRange(image_data, np.array([0, 100, 100]), np.array([10, 255, 255]))
        redImageHighTh = cv2.inRange(image_data, np.array([150, 100, 100]), np.array([179, 255, 255]))
        yellowImage = cv2.inRange(image_data, np.array([20, 100, 100]), np.array([40, 255, 255]))
        nprocessblockrows = (image_data.shape[0]) - (image_data.shape[0]/4)
        #Loop through each 64x64 block in selected ROI
        
        for row in xrange(0, (nprocessblockrows), stride):
            for col in xrange((image_data.shape[1]/4), ((image_data.shape[1]) - stride), stride):
                #Extract block data for detection of colors
                YellowRoi = yellowImage[row:row + blocksize, col:col + blocksize]
                RedHRoi = redImageHighTh[row:row + blocksize, col:col + blocksize]
                RedLRoi = redImageLowTh[row:row + blocksize, col:col + blocksize]

                #combine high and low range red circle masks to single bit mask
                RedROImask = cv2.addWeighted(RedLRoi, 1.0, RedHRoi, 1.0, 0.0)
                
                #Apply gaussian blur filter on each color bit masks to remove unwanted noise and Apply Hough circle detection on each induvisual masks to get count of number of circles
                redcircles = None
                yellowcircles = None
                # optimization to reduce processing time check any red pixel before applying hough circles
                Ysum = np.sum(YellowRoi)
                Rsum = np.sum(RedROImask)
                
                if Rsum > 0 :
                    RedROImask = cv2.GaussianBlur(RedROImask,(9, 9), 2, 2)
                    redcircles = cv2.HoughCircles(RedROImask, cv2.HOUGH_GRADIENT, 1, 20,param1=100,param2=25,minRadius=0,maxRadius=blocksize)

                if Ysum > 0 :
                    YellowRoi = cv2.GaussianBlur( YellowRoi, (9, 9), 2, 2)
                    yellowcircles = cv2.HoughCircles(YellowRoi, cv2.HOUGH_GRADIENT, 1, 20,param1=100,param2=25,minRadius=0,maxRadius=blocksize)
                    
                #Count number of circles detected in each bit masks
                if redcircles is not None:
                    nredcircles = nredcircles + len(redcircles)

                if yellowcircles is not None:
                    nyellowcircles = nyellowcircles + len(yellowcircles)
                    

        # classify color of detected circles based on number of circles detected
        print(nredcircles,nyellowcircles,ngreencircles)
        
        if (nredcircles > 0) and (nredcircles > nyellowcircles):
            rospy.loginfo("TrafficLight: RED-STOP")
            return TrafficLight.RED
        
        elif (nyellowcircles > 0) and (nyellowcircles > nredcircles):
            rospy.loginfo("TrafficLight: YELLOW-SLOW")
            return TrafficLight.YELLOW
        else:
            rospy.loginfo("TrafficLight: UNKNOWN")
            return TrafficLight.UNKNOWN
