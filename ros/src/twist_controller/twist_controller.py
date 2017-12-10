from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        # TODO: Implement
	wheel_base = kwargs["wheel_base"]
	steer_ratio = kwargs["steer_ratio"]
	self.min_speed = 0.0
	max_lat_accel = kwargs["max_lat_accel"]
	max_steer_angle = kwargs["max_steer_angle"]
	decel_limit = kwargs["decel_limit"]
	accel_limit = kwargs["accel_limit"]
	self.sample_rate = kwargs["sample_rate"]
        self.deadband    = kwargs['deadband']
	self.pid = PID(0.5, 0.012, 0.1, decel_limit, accel_limit)
	self.yaw_controller = YawController(wheel_base, steer_ratio, self.min_speed, max_lat_accel, max_steer_angle)
	self.filter_throttlebrake = LowPassFilter(0.07, 0.02)

    def control(self, target_linear_velocity, target_angular_velocity, current_linear_velocity, dbwenabled):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        brake = 0.0
        throttle = 0.0
        #sampling fequency as suggested by udacity
	throttle_brake = self.pid.step(target_linear_velocity - current_linear_velocity, 1.0/50)
	# handling boundary conditions for throttle 
	if throttle_brake > 0.:
            throttle = throttle_brake
            throttle = self.filter_throttlebrake.filt(throttle)
        elif throttle_brake < -self.deadband:
            throttle = 0.
            brake = -throttle_brake

	steer = self.yaw_controller.get_steering(target_linear_velocity, target_angular_velocity, current_linear_velocity)
        # handling boundary conditions for steer  based on deadband  
        if self.deadband < 0.1:
            steer = target_angular_velocity * steer_ratio 

        # handling boundary conditions for brake based on deadband at deadstop condition   
        if (target_linear_velocity <= 0.01) and (brake < self.deadband):     
            brake = self.deadband
        # reset PID when DBW not enabled
        if (not dbwenabled):
	    self.pid.reset()

	#rospy.loginfo('Controller - error:%f, throttle:%f, brake:%f, steer:%f', target_linear_velocity - current_linear_velocity, throttle, brake, steer)
	return throttle, brake, steer
