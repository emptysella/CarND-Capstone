#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import math
import numpy as np
import tf

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.
Once you have created dbw_node, you will update this node to use the status of traffic lights too.
Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
dropVelRatioAhead = 0
dropVelRatioActual = 0
dropVelRatioEmrgency = 0


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        #rospy.Subscriber('/current_velocity', TwistStamped, self.twist_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
	self.traffic_waypoint_sub = rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.base_waypoints      = Lane()
        self.current_pose        = PoseStamped()
        self.current_twist       = TwistStamped()
        self.wp_num              = 0

        self.base_waypoints_flag = False
        self.current_pose_flag  = False

        self.initial_velocity = 5.0
        self.stop_wayoint  = 0

        # Vehicle Pose variables
        self.car_pose_x = 0.0
        self.car_pose_y = 0.0
        self.car_yaw    = 0.0

        """
        @ Brief In-Loop --> publish trajectory and update waypoints
        -----------------------------------------------------------
        """
        self.publish_trajectory()


    """
    @ Brief
    ***************************************************************************
        Publish trajectory and update waypoints.
    ***************************************************************************
    """
    def publish_trajectory(self):

        rate = rospy.Rate(50)

        while not rospy.is_shutdown():

            isWaypoints = len(self.base_waypoints.waypoints)
            #if( isWaypoints > 0 & self.base_waypoints_flag & self.current_pose_flag):
	    if( isWaypoints > 0 ):

                # STAGE 1. find closer Waypoint to the car
                closer_waypoint = self.closer_waypoint()

                # STAGE 2. Popullate waypoints trajectory buffer
                trajectory_waypoints = self.populate_trajectory(closer_waypoint)

                # STAGE 3. Publish waypoints for vehicle trajectory
                self.final_waypoints_pub.publish(trajectory_waypoints)

            rate.sleep()
    def velocity_update(self, closer_waypoint,idx):
        if self.stop_wayoint > 0 :
	#distance_to_stopLine = self.distance(self.base_waypoints.waypoints, closer_waypoint, self.stop_wayoint)
            carpose = self.base_waypoints.waypoints[closer_waypoint].pose.pose.position
            ltpos =  self.base_waypoints.waypoints[self.stop_wayoint].pose.pose.position
            distance_to_stopLine = self.eucledien_distance(carpose.x,carpose.y,carpose.z,ltpos.x,ltpos.y,ltpos.z)
            #rospy.loginfo("Distance: %d", distance_to_stopLine)
            if(distance_to_stopLine < 100):
                dropVelRatioAhead = distance_to_stopLine/ (100)
                self.set_waypoint_velocity(self.base_waypoints.waypoints, idx,  (dropVelRatioAhead * self.initial_velocity))
                if distance_to_stopLine > 60 :
                    print (distance_to_stopLine,(dropVelRatioAhead * self.initial_velocity))
		if (distance_to_stopLine < 60):
		    dropVelRatioActual = distance_to_stopLine/ (100 + distance_to_stopLine)
	            self.set_waypoint_velocity(self.base_waypoints.waypoints, idx,  (dropVelRatioActual * self.initial_velocity))
                    print (distance_to_stopLine,(dropVelRatioActual * self.initial_velocity))
		if(distance_to_stopLine < 3):
	            dropVelRatioEmrgency = 0
	            self.set_waypoint_velocity(self.base_waypoints.waypoints, idx,  (dropVelRatioEmrgency * self.initial_velocity))    
			    
        else:
            self.previousVelocity = self.initial_velocity
            self.set_waypoint_velocity(self.base_waypoints.waypoints, idx, (self.initial_velocity) )
    """
    @ Brief
    ***************************************************************************
        Here all the waypoints belonging to the car projectory are popullated
        in trajectory_waypoints.
    ***************************************************************************
    """
    def populate_trajectory(self, closer_waypoint):

        trajectory_waypoints        = Lane()
        trajectory_waypoints.header = self.base_waypoints.header

        initial_wp = closer_waypoint
        final_wp   = closer_waypoint + LOOKAHEAD_WPS

	#rospy.loginfo("closer_waypoint: %d", closer_waypoint)
        
        for i in range(initial_wp, final_wp):
            idx = i % self.wp_num
            ### NOTE: Here we update the velovity for each waypoint to make it move it
            ### Alternative way to do it
            ### trajectory_waypoints.waypoints[idx].twist.twist.linear.x = self.initial_velocity
            ### using the method of the class
	    
            self.set_waypoint_velocity(self.base_waypoints.waypoints, idx, self.initial_velocity )
            self.velocity_update(closer_waypoint,idx)
            trajectory_waypoints.waypoints.append(self.base_waypoints.waypoints[idx])

        return trajectory_waypoints

    def eucledien_distance(self,refx,refy,refz,curx,cury,curz):
        distance = math.sqrt((curx - refx) ** 2 + (cury - refy) ** 2 + (curz - refz) ** 2)
	return distance
    """
    @ Brief
    ***************************************************************************
        Finding closer waypoint to the vehicle
    ***************************************************************************
    """

    def closer_waypoint(self):

        """
        @ Brief This method have two main stages.

            Stage1: Find the closet waypoint (wp) to the car that we will use to
                    populate a list of wp to set the cehicle trajectory_waypoints.

            Stage2: Double-check if the closer wp is ahehea of the car, so then,
                    aligned with the vehicle pose.
        """

        ### Stage1
        ###----------------------------------------------------------------------
        init_wp_x = self.base_waypoints.waypoints[0].pose.pose.position.x
        init_wp_y = self.base_waypoints.waypoints[0].pose.pose.position.y
        init_wp = np.array((init_wp_x,init_wp_y))

        car_pos_x = self.current_pose.pose.position.x
        car_pos_y = self.current_pose.pose.position.y
        car_pose = np.array((car_pos_x, car_pos_y))

        distance = np.linalg.norm(init_wp - car_pose)

        wp_index = 0
        all_wp_lenght = len(self.base_waypoints.waypoints)
	#rospy.loginfo("all_wp_lenght: %d", all_wp_lenght)
	min_dist = float('inf')
        for i in range( 1, all_wp_lenght):

            curr_wp_x = self.base_waypoints.waypoints[i].pose.pose.position.x
            curr_wp_y = self.base_waypoints.waypoints[i].pose.pose.position.y
            current_waypoint = np.array((curr_wp_x,curr_wp_y))
   
	    dist = np.linalg.norm(current_waypoint - car_pose)
	    if (dist < min_dist):
		min_dist = dist
		wp_index = i

        ### Stage2
        ###----------------------------------------------------------------------
        x_wp = self.base_waypoints.waypoints[wp_index].pose.pose.position.x
        y_wp = self.base_waypoints.waypoints[wp_index].pose.pose.position.y

        car_wp_orientation = np.arctan2(
                                        (y_wp - self.car_pose_y) ,
                                        (x_wp - self.car_pose_x) )

        orientation_alignement = np.abs(self.car_yaw - car_wp_orientation)
        # check if the car pose is aligned with the car-wp pose
        # if not aligned, we step forward one wp
        if orientation_alignement > np.pi/2.0:
            wp_index += 1
        # sanity check: control if we already are in the last wp
        if wp_index >= all_wp_lenght:
            wp_index = 0

        return wp_index

    """
    @ Brief
    ***************************************************************************
        Pose call-back fuction. Also use to set the car pose orientation
    ***************************************************************************
    """
    def pose_cb(self, msg):

        self.current_pose_flag = True
        self.current_pose = msg

        self.car_pose_x = msg.pose.position.x
        self.car_pose_y = msg.pose.position.y
        orientation = msg.pose.orientation
        _, _, self.car_yaw = tf.transformations.euler_from_quaternion([
                                                                    orientation.x,
                                                                    orientation.y,
                                                                    orientation.z,
                                                                    orientation.w])
	#rospy.loginfo("self.car_pose_x: %d, self.car_pose_y: %d, self.car_yaw: %d", self.car_pose_x, self.car_pose_y, self.car_yaw)


    def waypoints_cb(self, msg):

        self.base_waypoints_flag = True
        self.base_waypoints = msg
        self.wp_num = len(msg.waypoints)


    def twist_cb(self, msg):
        self.current_twist = msg

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stop_wayoint = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
