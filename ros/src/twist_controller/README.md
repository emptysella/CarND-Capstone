# DBW (Drive By Wire) Node

This node is responsible for acquiring velocity, drive by wire command from the __Car system__ and twist command from __Waypoint Follower__ node, and its objective is to send brake, throttle and steering signals to the car.
To do so these files were modified:
* dbw_node.py
* twist_controller.py

# dbw_node.py
It starts reading configuration parameters:
```
vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
brake_deadband = rospy.get_param('~brake_deadband', .1)
decel_limit = rospy.get_param('~decel_limit', -5)
accel_limit = rospy.get_param('~accel_limit', 1.)
wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
wheel_base = rospy.get_param('~wheel_base', 2.8498)
steer_ratio = rospy.get_param('~steer_ratio', 14.8)
max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
max_steer_angle = rospy.get_param('~max_steer_angle', 8.)
```
then we define the publishers for the output information we generate in this module:
* steering
* throttle
* brake
```
self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                 SteeringCmd, queue_size=1)
self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                    ThrottleCmd, queue_size=1)
self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                    BrakeCmd, queue_size=1)
```
Variable initialization is made, setting sampling rate to 50 Hz:
```
self.current_linear_velocity = 0.0
self.current_angular_velocity = 0.0
self.target_linear_velocity = 0.0
self.target_angular_velocity = 0.0
self.dbwenabled = False
self.sample_rate = 50 # 50Hz
```
Then we subscribe to the input topics so that we have all available information to work with:
* current velocity (*current_velocity* topic)
* target velocity (*twist_cmd* topic)
* drive by wire enabling command (*dbw_enabled* topic)

To acquire information some callback functions are provided (*read_current_velocity*, *read_target_velocity*, *read_dbwenabled* respectively):
```
rospy.Subscriber("/current_velocity", TwistStamped, self.read_current_velocity)
rospy.Subscriber("/twist_cmd", TwistStamped, self.read_target_velocity)
rospy.Subscriber("/vehicle/dbw_enabled", Bool, self.read_dbwenabled)
```
The callback functions save information into class variables.
Now we define the controller with some input variables:
```
controller_params = {
	'wheel_base':	   wheel_base,
	'steer_ratio':     steer_ratio,
	'max_lat_accel':   max_lat_accel,
	'max_steer_angle': max_steer_angle,
	'decel_limit':     decel_limit,
	'accel_limit':     accel_limit,
	'sample_rate':	   self.sample_rate
}
self.controller = Controller(**controller_params)
```
and call *loop()* method to call the controller, retrieve the commands generated from it and publish them:
```
def loop(self):
    rate = rospy.Rate(self.sample_rate) 
    while not rospy.is_shutdown():
        throttle, brake, steer = self.controller.control(self.target_linear_velocity, self.target_angular_velocity, self.current_linear_velocity, self.dbwenabled)
        if self.dbwenabled:            	
	        self.publish(throttle, brake, steer)
        rate.sleep()
```
# twist_controller.py
This file defines the class Controller to generate throttle, brake and steer based on PID.
In the *init()* method the variables are initialized, some are passed from *dbw_node.py*:
```
wheel_base = kwargs["wheel_base"]
steer_ratio = kwargs["steer_ratio"]
self.min_speed = 0.0
max_lat_accel = kwargs["max_lat_accel"]
max_steer_angle = kwargs["max_steer_angle"]
decel_limit = kwargs["decel_limit"]
accel_limit = kwargs["accel_limit"]
self.sample_rate = kwargs["sample_rate"]
```
2 PIDs are defined for this project: one for velocities lower than 15 kph and one for those greater than 15 kph. 
This has been done to optimize behaviour for Carla, which moves at 10 kph, and for the simulator, running at 40 kph.
Each PID is defined with its configuration parameters (see *PID configuration* section below):
```
self.pid_lowvel = PID(1, 0.5, 0.1, decel_limit, accel_limit) 		# PID for 10 kph
self.pid_highvel = PID(1.0, 0.012, 0.1, decel_limit, accel_limit)   # PID for 40 kph	
```
Class *YawController* is used to calculate steering:
```
self.yaw_controller = YawController(wheel_base, steer_ratio, self.min_speed, max_lat_accel, max_steer_angle)
```
Then a low pass filter has been used to attenuate rapid variations for steer command:
```
self.filter_steer = LowPassFilter(50)
```
This filter is created with a parameter = 50 because the class LowFiler has been modified to calculate last N valuees (N=50) and then giving the mean for these values.
Now the *control()* method reset the value for PID controllers if drive by wire command is not enabled:
```
if (not dbwenabled):
	self.pid_lowvel.reset()
    self.pid_highvel.reset()    
```
The correct PID is selected depending on the target velocity, then calculates throttle/brake values using PID *step()* method, passing the error value (linear - current velocity) and the sampling time (0.02 seconds for 50 Hz sampling rate):
```
if (target_linear_velocity <= self.LOWVEL_LIMIT):
		pid = self.pid_lowvel
	else:
        pid = self.pid_highvel
throttle_brake = self.pid.step(target_linear_velocity - current_linear_velocity, 1.0/self.sample_rate)
```
Throttle/brake values are extracted from the *step()* returned value (throttle is positive and brake is negative) and filtered with the low pass previously defined:
```
throttle_brake = self.filter_throttlebrake.filt(throttle_brake)
throttle = max(throttle_brake, 0)
brake = min(throttle_brake, 0)
```
At the end, steer is calculated and its value is low pass filtered only when velocity is greater than minimum speed:
```
steer = self.yaw_controller.get_steering(target_linear_velocity, target_angular_velocity, current_linear_velocity)
if (current_linear_velocity > self.min_speed):
    steer = self.filter_steer.filt(steer)
```

## PID configuration
PID has been used to control velocity, while steering is calculated by YawController *get_steering()* method.
PID manual tuning method has been used, following rules defined in this Wikipedia page: [PID controller - manual tuning](https://en.wikipedia.org/wiki/PID_controller#Manual_tuning).
At first the proportional term has been tuned (*kp*), setting integral and derivative part to zero (*ki=kd=0*).
kp has been increased until a quarter amplitude decay has been reached, then it has been set to half that value (for the calculations the error values have been logged).
Then the integral term has been tuned to reduce the offset respect to the desired velocity.
Last the derivative term has been increased until the oscillations have been reduced.
At the end a further minor tuning has been done.