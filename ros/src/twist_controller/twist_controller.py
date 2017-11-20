from yaw_controller import YawController
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        # TODO: Implement
	wheel_base = kwargs["wheel_base"]
	steer_ratio = kwargs["steer_ratio"]
	min_speed = 5.0
	max_lat_accel = kwargs["max_lat_accel"]
	max_steer_angle = kwargs["max_steer_angle"]
	decel_limit = kwargs["decel_limit"]
	accel_limit = kwargs["accel_limit"]
	self.pid = PID(0.1, 0.001, 0.6, decel_limit, accel_limit)
	self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

    def control(self, target_linear_velocity, target_angular_velocity, current_linear_velocity, dbwenabled):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
	throttle_brake = self.pid.step(target_linear_velocity - current_linear_velocity, 0.02)
	throttle = max(throttle_brake, 0)
	brake = min(throttle_brake, 0)
	steer = self.yaw_controller.get_steering(target_linear_velocity, target_angular_velocity, current_linear_velocity)
        #return 1., 0., 0.
	return throttle, brake, steer
