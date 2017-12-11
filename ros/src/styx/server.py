#!/usr/bin/env python

import socketio
import eventlet
import eventlet.wsgi
import time
import rospy
from flask import Flask, render_template

from bridge import Bridge
from conf import conf

sio = socketio.Server()
app = Flask(__name__)

msgs = {}
# Reduce simulator lag
# https://github.com/udacity/self-driving-car-sim/issues/53

dbw_enable = False

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)

def send(topic, data):
    msgs[topic] = data

bridge = Bridge(conf, send)

@sio.on('telemetry')
def telemetry(sid, data):
    global dbw_enable
    if data["dbw_enable"] != dbw_enable:
        dbw_enable = data["dbw_enable"]
        bridge.publish_dbw_status(dbw_enable)
    bridge.publish_odometry(data)
    for i in range(len(msgs)):
        topic, data = msgs.popitem()
        sio.emit(topic, data=data, skip_sid=True)

#@sio.on('control')
def control(sid, data):
    #bridge.publish_controls(data)
    pass

#@sio.on('obstacle')
def obstacle(sid, data):
    #bridge.publish_obstacles(data)
    #rospy.loginfo("obstacle callback")
    pass

#@sio.on('lidar')
def obstacle(sid, data):
    #bridge.publish_lidar(data)
    pass
    
@sio.on('trafficlights')
def trafficlights(sid, data):
    bridge.publish_traffic(data)

framecount = 0
framesample = 2
@sio.on('image')
def image(sid, data):
    global count
    framecount += 1
    if framecount%(framesample+1)==0:
        bridge.publish_camera(data)
    #rospy.loginfo("imagesize: %d", len(data["image"]) )
    #pass

if __name__ == '__main__':

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
