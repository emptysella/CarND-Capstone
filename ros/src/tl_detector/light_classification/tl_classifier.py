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
        Luma  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        light_type = self.classifyTL(Luma)
        return light_type


    def classifyTL(self, image_data):
        # get the image center geometry
	blocksize = 64 # 64x64 block size
	stride = 32
        nredcircles = 0
        nyellowcircles = 0
        ngreencircles = 0
        for row in xrange(image_data.shape[0]/4, (image_data.shape[0] - stride), stride):
            for col in xrange(0, ((image_data.shape[1]/2) - stride), stride):
                Roi = image_data[row:row + blocksize, col:col + blocksize]
                redImageLowTh = cv2.inRange(Roi, np.array([0, 100, 100]), np.array([10, 255, 255]))
                redImageHighTh = cv2.inRange(Roi, np.array([150, 100, 100]), np.array([179, 255, 255]))
                yellowImage = cv2.inRange(Roi, np.array([20, 100, 100]), np.array([40, 255, 255]))
                greenImage = cv2.inRange(Roi, np.array([37, 38, 70]), np.array([85, 255, 200]));
                redpixelimage = cv2.addWeighted(redImageLowTh, 1.0, redImageHighTh, 1.0, 0.0)
                redpixelimage = cv2.GaussianBlur(redpixelimage,(9, 9), 2, 2)
                yellowImage = cv2.GaussianBlur( yellowImage, (9, 9), 2, 2)
                greenImage = cv2.GaussianBlur(greenImage, (9, 9), 2, 2)
                redcircles = cv2.HoughCircles(redpixelimage, cv2.HOUGH_GRADIENT, 1, 20,param1=100,param2=30,minRadius=0,maxRadius=blocksize)
                yellowcircles = cv2.HoughCircles(yellowImage, cv2.HOUGH_GRADIENT, 1, 20,param1=100,param2=30,minRadius=0,maxRadius=blocksize)
                greencircles = cv2.HoughCircles(greenImage, cv2.HOUGH_GRADIENT, 1, 20,param1=100,param2=30,minRadius=0,maxRadius=blocksize)
                if redcircles is not None:
                    nredcircles = nredcircles + len(redcircles)
        
                if yellowcircles is not None:
                    nyellowcircles = nyellowcircles + len(yellowcircles)

                if greencircles is not None:
                    ngreencircles = ngreencircles + len(greencircles)

        # perform simple brightness comparisons and print for humans
        print(nredcircles,nyellowcircles,ngreencircles)
        if (nredcircles > 0) and (nredcircles > nyellowcircles):
            rospy.loginfo("TrafficLight: RED-STOP")
            return TrafficLight.RED
        elif (nyellowcircles > 0) and (nyellowcircles > nredcircles):
            rospy.loginfo("TrafficLight: YELLOW-SLOW")
            return TrafficLight.YELLOW
        elif (ngreencircles > 0):
            rospy.loginfo("TrafficLight: GREEN-GO")
            return TrafficLight.GREEN
        else:
	    rospy.loginfo("TrafficLight: UNKNOWN")
            return TrafficLight.UNKNOWN
