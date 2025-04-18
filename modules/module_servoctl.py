"""
module_servoctl.py

Servo Control Module for TARS-AI Application.

This module provides a comprehensive set of functions to control servos in TARS, enabling complex movements such as:
- Torso height adjustment.
- Torso rotation to forward/backward positions.
- Controlled arm and hand movements.
- Synchronized servo adjustments for smooth motion transitions.
"""

from __future__ import division
import time
import Adafruit_PCA9685
from threading import Thread
from datetime import datetime

from modules.module_messageQue import queue_message
from module_config import load_config

config = load_config()


try:
    # Attempt to initialize the PCA9685 using I2C
    pwm = Adafruit_PCA9685.PCA9685(busnum=1)
    pwm.set_pwm_freq(60)  # Set frequency to 60 Hz for servos
except FileNotFoundError as e:
    queue_message(f"ERROR: I2C device not found. Ensure that /dev/i2c-1 exists. Details: {e}")
    pwm = None  # Fallback if hardware is unavailable
except Exception as e:
    queue_message(f"ERROR: Unexpected error during PCA9685 initialization: {e}")
    pwm = None  # Fallback if hardware is unavailable

# Servo Configuration Mapping with Integer Casting
portMain = int(config["SERVO"]["portMain"])
starMain = int(config["SERVO"]["starMain"])
portForarm = int(config["SERVO"]["portForarm"])
starForarm = int(config["SERVO"]["starForarm"])
portHand = int(config["SERVO"]["portHand"])
starHand = int(config["SERVO"]["starHand"])

# Center Lift Servo (0) Values
upHeight = int(config["SERVO"]["upHeight"])
neutralHeight = int(config["SERVO"]["neutralHeight"])
downHeight = int(config["SERVO"]["downHeight"])

# Port Drive Servo (1) Values
forwardPort = int(config["SERVO"]["forwardPort"])
neutralPort = int(config["SERVO"]["neutralPort"])
backPort = int(config["SERVO"]["backPort"])
perfectportoffset = int(config["SERVO"]["perfectportoffset"])  # Use this to fine-tune if it's off by a few degrees

# Starboard Drive Servo (2) Values
forwardStarboard = int(config["SERVO"]["forwardStarboard"])
neutralStarboard = int(config["SERVO"]["neutralStarboard"])
backStarboard = int(config["SERVO"]["backStarboard"])
perfectStaroffset = int(config["SERVO"]["perfectStaroffset"])  # Use this to fine-tune if it's off by a few degrees


# moves the torso from a neutral position upwards, allowing the torso to pivot forwards or backwards
def height_neutral_to_up():
	height = neutralHeight
	#queue_message('setting center servo (0) Neutral --> Up position')
	while (height > upHeight):
		height = height - 1
		pwm.set_pwm(0, 0, height)
		time.sleep(0.001)
	#queue_message('center servo (0) set to: Up position\n ')

# rotates the torso outwards, enough so that when TARS pivots and lands, the bottom of the torso is 
# flush with the ground. Making the torso flush with the ground is an intentional improvement from
# previous programs, where TARS would land and then slide a little on smooth surfaces, which while
# allowing for a simple walking program, inhibited TARS' ability to walk on surfaces with different 
# coefficients of friction
def torso_neutral_to_forwards():
	port = neutralPort
	starboard = neutralStarboard
	#queue_message('setting port and starboard servos (1)(2) Neutral --> Forward')
	while (port < forwardPort):
		port = port + 1
		starboard = starboard - 1
		pwm.set_pwm(1, 1, port)
		pwm.set_pwm(2, 2, starboard)
		time.sleep(0.0001)
	#queue_message('port and starboard servos (1)(2) set to: Forward position\n ')

def torso_neutral_to_backwards():
	port = neutralPort
	starboard = neutralStarboard
	#queue_message('setting port and starboard servos (1)(2) Neutral --> Forward')
	while (port > backPort):
		port = port - 1
		starboard = starboard + 1
		pwm.set_pwm(1, 1, port)
		pwm.set_pwm(2, 2, starboard)
		time.sleep(0.0001)
	#queue_message('port and starboard servos (1)(2) set to: Forward position\n ')

# rapidly shifts the torso height from UP --> DOWN and then returns --> UP, which should cause TARS 
# to pivot and land on it's torso
def torso_bump():
	height = upHeight
	#queue_message('performing a torso bump\nsetting center servo (0) Up --> Down position FAST')
	while (height < downHeight):
		height = height + 2
		pwm.set_pwm(0, 0, height)
		time.sleep(0.000001)
	#queue_message('setting center servo (0) Down --> Up position FAST')
	while (height > upHeight):
		height = height - 1
		pwm.set_pwm(0, 0, height)
		time.sleep(0.0001)
	#queue_message('center servo (0) returned to Up position\n')
	
# returns the torso's vertical height and rotation to centered values from up height and forward 
# rotation. Activates two external functions so movement in both axes can occur in parallel.
def torso_return():
	t1 = Thread(target = torso_return_rotation)
	t2 = Thread(target = torso_return_vertical)
	
	t1.start()
	t2.start()

# returns torso's rotation to neutral from forward
def torso_return_rotation():
	port = forwardPort
	starboard = forwardStarboard
	#queue_message('setting port and starboard servos (1)(2) Forward --> Neutral position')
	while (port > neutralPort):
		port = port - 1
		starboard = starboard + 1
		pwm.set_pwm(1, 1, port)
		pwm.set_pwm(2, 2, starboard)
		time.sleep(0.005)
	#queue_message('port and starboard servos (1)(2) set to: Neutral position\n ')

# returns torso's vertical to neutral from up	
def torso_return_vertical():
	height = upHeight
	#queue_message('setting center servo (0) Up --> Down position')
	# moving the torso down to create clearance for the rotation of the legs
	while (height < downHeight):
		height = height + 1
		pwm.set_pwm(0, 0, height)
		time.sleep(0.00005)
	# moving the torso up from down to neutral
	#time.sleep(.2)
	while (height > neutralHeight):
		height = height - 1
		pwm.set_pwm(0, 0, height)
		time.sleep(0.00001)
	#queue_message('center servo (0) set to: Neutral position\n ')

def torso_return2():
	t1 = Thread(target = torso_return_rotation2)
	t2 = Thread(target = torso_return_vertical2)
	
	t1.start()
	t2.start()

# returns torso's rotation to neutral from forward
def torso_return_rotation2():
	port = backPort
	starboard = backStarboard
	#queue_message('setting port and starboard servos (1)(2) Forward --> Neutral position')
	while (port < neutralPort):
		port = port + 1
		starboard = starboard - 1
		pwm.set_pwm(1, 1, port)
		pwm.set_pwm(2, 2, starboard)
		time.sleep(0.01)
	#queue_message('port and starboard servos (1)(2) set to: Neutral position\n ')

# returns torso's vertical to neutral from up	
def torso_return_vertical2():
	height = upHeight
	#queue_message('setting center servo (0) Up --> Down position')
	# moving the torso down to create clearance for the rotation of the legs
	while (height < downHeight):
		height = height + 1
		pwm.set_pwm(0, 0, height)
		time.sleep(0.001)
	# moving the torso up from down to neutral
	time.sleep(.25)
	while (height > neutralHeight):
		height = height - 1
		pwm.set_pwm(0, 0, height)
		time.sleep(0.001)
	#queue_message('center servo (0) set to: Neutral position\n ')


# moves the torso from neutral position to down
def neutral_to_down():
    height = neutralHeight
    #queue_message('setting center servo (0) Neutral --> Down position')
    while (height < downHeight):
        height = height + 1
        pwm.set_pwm(0, 0, height)
        time.sleep(0.001)
        
def down_to_up():
    height = downHeight
    #queue_message('setting center servo (0) Down --> Neutral position')
    while (height > upHeight):
        height = height - 1
        pwm.set_pwm(0, 0, height)
        time.sleep(0.001)

def down_to_neutral():
    height = downHeight
    #queue_message('setting center servo (0) Down --> Neutral position')
    while (height > neutralHeight):
        height = height - 1
        pwm.set_pwm(0, 0, height)
        time.sleep(0.001)

def neutral_to_down():
    height = neutralHeight
    #queue_message('setting center servo (0) Down --> Neutral position')
    while (height < downHeight):
        height = height + 1
        pwm.set_pwm(0, 0, height)
        time.sleep(0.001)


def turn_right():
    port = neutralPort
    starboard = neutralStarboard
    while (port < forwardPort):
        port = port + 1
        starboard = starboard + 1
        pwm.set_pwm(1, 1, port)
        pwm.set_pwm(2, 2, starboard)
        time.sleep(0.001)
        
def turn_left():
    port = neutralPort
    starboard = neutralStarboard
    while (port > backPort):
        port = port - 1
        starboard = starboard - 1
        pwm.set_pwm(1, 1, port)
        pwm.set_pwm(2, 2, starboard)
        time.sleep(0.001)
        
def neutral_from_right():
    port = forwardPort
    starboard = backStarboard
    while (port > neutralPort):
        port = port - 1
        starboard = starboard - 1
        pwm.set_pwm(1, 1, port)
        pwm.set_pwm(2, 2, starboard)
        time.sleep(0.005)
    pwm.set_pwm(1, 1, neutralPort)
    pwm.set_pwm(2, 2, neutralStarboard)
        
def neutral_from_left():
    port = backPort
    starboard = forwardStarboard
    while (port < neutralPort):
        port = port + 1
        starboard = starboard + 1
        pwm.set_pwm(1, 1, port)
        pwm.set_pwm(2, 2, starboard)
        time.sleep(0.005)
    pwm.set_pwm(1, 1, neutralPort)
    pwm.set_pwm(2, 2, neutralStarboard)
# Arm shenanigans
# port Main Arm
def portMainPlus():
    global portMain
    portMain = portMain - 10
    pwm.set_pwm(3, 3, portMain)
    time.sleep(0.0001)
    queue_message("increase starMain")
    queue_message(portMain)

def portMainMinus():
    global portMain
    portMain = portMain + 10
    pwm.set_pwm(3, 3, portMain)
    time.sleep(0.0001)
    queue_message("decrease starMain")
    queue_message(portMain) 

# port Forarm
def portForarmPlus():
    global portForarm
    portForarm = portForarm - 10
    pwm.set_pwm(4, 4, portForarm)
    time.sleep(0.0001)
    queue_message("increase starForarm")
    queue_message(portForarm)

def portForarmMinus():
    global portForarm
    portForarm = portForarm + 10
    pwm.set_pwm(4, 4, portForarm)
    time.sleep(0.0001)
    queue_message("decrease starForarm")
    queue_message(portForarm) 

# port Hand
def portHandPlus():
    global portHand
    portHand = portHand - 10
    pwm.set_pwm(5, 5, portHand)
    time.sleep(0.0001)
    queue_message("increase starHand")
    queue_message(portHand)

def portHandMinus():
    global portHand
    portHand = portHand + 10
    pwm.set_pwm(5, 5, portHand)
    time.sleep(0.0001)
    queue_message("decrease starHand")
    queue_message(portHand)
    
# starboard Main Arm
def starMainPlus():
    global starMain
    starMain = starMain + 10
    pwm.set_pwm(6, 6, starMain)
    time.sleep(0.0001)
    queue_message("increase starMain")
    queue_message(starMain)

def starMainMinus():
    global starMain
    starMain = starMain - 10
    pwm.set_pwm(6, 6, starMain)
    time.sleep(0.0001)
    queue_message("decrease starMain")
    queue_message(starMain) 

# port Forarm
def starForarmPlus():
    global starForarm
    starForarm = starForarm + 10
    pwm.set_pwm(7, 7, starForarm)
    time.sleep(0.0001)
    queue_message("increase starForarm")
    queue_message(starForarm)

def starForarmMinus():
    global starForarm
    starForarm = starForarm - 10
    pwm.set_pwm(7, 7, starForarm)
    time.sleep(0.0001)
    queue_message("decrease starForarm")
    queue_message(starForarm) 

# port Hand
def starHandPlus():
    global starHand
    starHand = starHand + 10
    pwm.set_pwm(8, 8, starHand)
    time.sleep(0.0001)
    queue_message("increase starHand")
    queue_message(starHand)

def starHandMinus():
    global starHand
    starHand = starHand - 10
    pwm.set_pwm(8, 8, starHand)
    time.sleep(0.0001)
    queue_message("decrease starHand")
    queue_message(starHand) 