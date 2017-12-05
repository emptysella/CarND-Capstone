#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import sys
import math

car_position =  0
STATE_COUNT_THRESHOLD = 3

DISTANCE_LIMIT = 200
class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.stop_line_distance_pub = rospy.Publisher('stop_line_distance', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.slmindistance = 10000
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def eucledien_distance(self,refx,refy,refz,curx,cury,curz):
        distance = math.sqrt((curx - refx) ** 2 + (cury - refy) ** 2 + (curz - refz) ** 2)
	return distance

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        if (self.waypoints == None):
            return None
        else:
            waypt = self.waypoints.waypoints

        # Create variables for nearest distance and neighbour
        closewpindex = None
        mindistance = sys.maxsize

        # Find Neighbour
        for i in range(len(waypt)):
            curwppos = waypt[i].pose.pose.position
            dist  = self.eucledien_distance(pose.position.x,pose.position.y,pose.position.z,curwppos.x,curwppos.y,curwppos.z)
            if dist < mindistance:
                closewpindex = i
                mindistance = dist

        return closewpindex

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        rate = rospy.Rate(2)
        self.stop_line_distance_pub.publish(Int32(self.slmindistance))	
	if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:

            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            #print("publishing.....................")
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1



    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

        #TODO find the closest visible traffic light (if one exists)
        mindistance = sys.maxsize
	light_wp_idx = None

	# update waypoint poisition close to car_position
        if (self.waypoints == None) or (car_position == None):
            return -1, TrafficLight.UNKNOWN
        else:
            carpose = self.waypoints.waypoints[car_position].pose.pose.position
        car_position = 0
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

        #TODO find the closest visible traffic light (if one exists)
	numlightIdx = len(self.lights)
        for i in range(numlightIdx):
            ltpos = self.lights[i].pose.pose.position
            dist  = self.eucledien_distance(carpose.x,carpose.y,carpose.z,ltpos.x,ltpos.y,ltpos.z)
            if dist < mindistance:
	        light_wp_idx 	= i
                mindistance    	= dist
        #print(light_wp_idx,carpose, self.lights[light_wp_idx].pose.pose.position, mindistance)

        #rospy.loginfo("car_position: %d", car_position)
	#return -1, TrafficLight.UNKNOWN
	stop_wp_idx = None
        
        mindist = sys.maxsize
        closelightIdx  =  self.get_closest_waypoint(self.lights[light_wp_idx].pose.pose)
        #print(closelightIdx , car_position)
        if ((light_wp_idx is not None) and (closelightIdx > (car_position + 1))):
            light = self.lights[light_wp_idx].pose.pose.position
            state = self.get_light_state(light)

	    #find stop line waypoint close to closest traffic light
            for i in range(0, len(stop_line_positions)):
                stoplinepose = PoseStamped()
                stoplinepose.pose.position.x = stop_line_positions[i][0]
                stoplinepose.pose.position.y = stop_line_positions[i][1]
                stoplinepose.pose.position.z = 0
                closeWpIdx  =  self.get_closest_waypoint(stoplinepose.pose)
                stoplpos    = self.waypoints.waypoints[closeWpIdx].pose.pose.position
                #print("dist = " , dist)
                dist  = self.eucledien_distance(stoplpos.x,stoplpos.y,stoplpos.z,light.x,light.y,0)
                 
                if (dist < mindist) and (closeWpIdx > (car_position + 1)):
	            stop_wp_idx 	= closeWpIdx
                    mindist             = dist
    
            #print("light state ",state)
            if stop_wp_idx is not None:
                #rospy.loginfo("Trafficdistance: %d  %d %d", mindist, stop_wp_idx, car_position)
                # only update traffic light if min distance is close to diatnace limit
                stoplinepos = self.waypoints.waypoints[stop_wp_idx].pose.pose.position
                self.slmindistance = self.eucledien_distance(carpose.x,carpose.y,carpose.z,stoplinepos.x,stoplinepos.y,stoplinepos.z)
                #rospy.loginfo("self.slmindistance = %d", self.slmindistance) 
                if (mindist < DISTANCE_LIMIT) :
                    return stop_wp_idx, state
                else:
                    return -1, TrafficLight.UNKNOWN
            else:
                return light_wp_idx, state
        #self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
