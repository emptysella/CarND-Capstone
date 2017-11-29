from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import rospy

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.red_threshold = 100

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        red_channel = image[:,:,2]

        red_area = np.sum(red_channel == red_channel.max())

        if red_area > self.red_threshold:
            detection = TrafficLight.RED
            rospy.loginfo("TrafficLight: RED-STOP")
        else:
            detection = TrafficLight.UNKNOWN

        return detection
