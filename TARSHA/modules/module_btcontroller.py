"""
module_btcontroller.py

Provides functionality for managing and interpreting Bluetooth gamepad input 
to control servos and execute specific actions in the TARS-AI system.

This module listens to gamepad events such as button presses, joystick movements, 
and D-pad directions, mapping these events to corresponding robotic movements or 
in-app commands. 
"""

import evdev
import time
from datetime import datetime
from evdev import InputDevice, categorize, ecodes, list_devices
import Adafruit_PCA9685

from modules.module_config import load_config
from modules.module_servoctl import *
from modules.module_messageQue import queue_message

config = load_config()
controller_name = config["CONTROLS"]["controller_name"]

global posevar

try:
    pwm = Adafruit_PCA9685.PCA9685(busnum=1)  # Specify I2C bus 1
    pwm.set_pwm_freq(60)  # Set frequency to 60 Hz
except FileNotFoundError as e:
    queue_message(f"ERROR: I2C device not found. Ensure that /dev/i2c-1 exists. Details: {e}")
    pwm = None  # Fallback if hardware is unavailable
except Exception as e:
    queue_message(f"ERROR: Unexpected error during PCA9685 initialization: {e}")
    pwm = None  # Fallback if hardware is unavailable



# Set initial servo positions
if pwm:
    try:
        # Port
        pwm.set_pwm(3, 3, 610)
        pwm.set_pwm(4, 4, 570)
        pwm.set_pwm(5, 5, 570)
        # Starboard
        pwm.set_pwm(6, 6, 200)
        pwm.set_pwm(7, 7, 200)
        pwm.set_pwm(8, 8, 240)
    except Exception as e:
        queue_message(f"Error setting initial servo positions: {e}")

lTrg = 37
rTrg = 50
upBtn = 46
downBtn = 32
lBtn = 18
rBtn = 33
xBtn = 23
yBtn = 35
aBtn = 36
bBtn = 34
minusBtn = 49
plusBtn = 24

global gamepad_path
toggle = True
posevar = False

SECRET_CODE = [
    "up", "up", "down", "down", "left", "right", "left", "right", "B", "A Button", "Start Button"
]

# Track user input
input_sequence = []

def find_controller(controller_name):
    global gamepad_path
    """
    Search for a controller by its name.
    """
    devices = [InputDevice(path) for path in list_devices()]
    for device in devices:
        if controller_name.lower() in device.name.lower():
            queue_message(f"LOAD: Controller found: {device.name} at {device.path}")
            gamepad_path = device.path
            return device
    queue_message(f"LOAD: {controller_name} Not found, Searching...")
    return None

def check_secret_code(button_name):
    global input_sequence
    input_sequence.append(button_name)

    # Check if the current sequence matches the start of the secret code
    if input_sequence == SECRET_CODE[:len(input_sequence)]:
        # If the sequence matches the full secret code
        if len(input_sequence) == len(SECRET_CODE):
            from module_secrets import play_video_fullscreen
            play_video_fullscreen("secrets/secret.mp4", rotation_angle=90)
            input_sequence = []  # Reset the sequence after the code is entered
    else:
        # If the sequence doesn't match, reset it
        #queue_message(f"Invalid sequence detected: {input_sequence}. Resetting...")
        input_sequence = []

#functions to move
def stepForward():
    queue_message("MOVE: FWD")
    height_neutral_to_up()
    torso_neutral_to_forwards()
    torso_bump()
    torso_return()

def turnRight():
    queue_message("MOVE: TurnRight")
    neutral_to_down()
    turn_right()
    down_to_neutral()
    neutral_from_right()

def turnLeft():
    queue_message("MOVE: TurnLeft")
    neutral_to_down()
    turn_left()
    down_to_neutral()
    neutral_from_left()

def poseaction():
    queue_message("MOVE: Pose")
    neutral_to_down()
    torso_neutral_to_backwards()
    down_to_up()

def unposeaction():
    queue_message("MOVE: UnPose")
    torso_return2()  
        
        
# D-Pad Actions (pressed and released)
def action_dpad_up_pressed():
    #queue_message(f"CTRL: D-Pad Up pressed! Let's move up!")
    stepForward()

def action_dpad_down_pressed():
    queue_message(f"CTRL: D-Pad Down pressed! Let's move down!")
    global posevar
    
    if posevar == False:
        poseaction()
        posevar = True
    elif posevar == True:
        unposeaction()
        posevar = False

def action_dpad_left_pressed():
    #queue_message(f"CTRL: D-Pad Left pressed! Moving left!")
    turnLeft()

def action_dpad_right_pressed():
    #queue_message(f"CTRL: D-Pad Right pressed! Moving right!")
    turnRight()

def action_dpad_up_released():
    #queue_message(f"CTRL: D-Pad Up released! Stopping move up.")
    pass

def action_dpad_down_released():
    #queue_message(f"CTRL: D-Pad Down released! Stopping move down.")
    pass

def action_dpad_left_released():
    #queue_message(f"CTRL: D-Pad Left released! Stopping move left.")
    pass

def action_dpad_right_released():
    #queue_message(f"CTRL: D-Pad Right released! Stopping move right.")
    pass

# Joystick Actions (show values when moved)
def action_left_stick_move(x_value, y_value):
    #queue_message(f"CTRL: Left Stick moved to X: {x_value}, Y: {y_value}")
    pass

def action_right_stick_move(x_value, y_value):
    #queue_message(f"CTRL: Right Stick moved to X: {x_value}, Y: {y_value}")
    pass

# Define custom actions for specific buttons (pressed)
def action_a_button_pressed():
    #queue_message(f"CTRL: A Button? Are you trying to jump?")
    global toggle
    if toggle == True:
        starHandPlus()
    elif toggle == False:
        starHandMinus()

def action_b_button_pressed():
    #queue_message(f"CTRL: Oh no, the B! Self-destruct initiated... just kidding!")
    global toggle
    if toggle == True:
        portHandPlus()
    elif toggle == False:
        portHandMinus()

def action_x_button_pressed():
    #queue_message(f"CTRL: Hey, stop pushing my X Button!")
    global toggle
    if toggle == True:
        starForarmPlus()
    elif toggle == False:
        starForarmMinus()

def action_y_button_pressed():
    #queue_message(f"CTRL: Y Button? I hope you know what youre doing!")
    global toggle
    if toggle == True:
        portForarmPlus()
    elif toggle == False:
        portForarmMinus()

def action_r1_button_pressed():
    #queue_message(f"CTRL: R1 Button pressed! Thats the turbo button!")
    global toggle
    if toggle == True:
        starMainPlus()
    elif toggle == False:
        starMainMinus()

def action_l1_button_pressed():
    #queue_message(f"CTRL: L1 Button activated! Shields up!")
    global toggle
    if toggle == True:
        portMainPlus()
    elif toggle == False:
        portMainMinus()

def action_r2_button_pressed():
    #queue_message(f"CTRL: R2 Button? Are we accelerating now?")
    pass

def action_l2_button_pressed():
    #queue_message(f"CTRL: L2 Button pressed! Steady... dont crash!")
    pass

def action_bottom_button_pressed():
    #queue_message(f"CTRL: Bottom Button? What kind of mischief is this?")
    pass

def action_select_button_pressed():
    #queue_message(f"CTRL: Select Button pressed. Are you opening a menu?")
    pass

def action_start_button_pressed():
    #queue_message(f"CTRL: Start Button pressed. Game on!")
    pass

def LJoyStick_button_pressed():
    #queue_message(f"CTRL: L JoyStick Pressed. HAHAHAHAHA")
    pass

def RJoyStick_button_pressed():
    #queue_message(f"CTRL: R JoyStick Pressed. Be Careful!")
    pass

# Define custom actions for specific buttons (released)
def action_a_button_released():
    #queue_message(f"CTRL: Okay, you stopped jumping. Good!")
    pass

def action_b_button_released():
    #queue_message(f"CTRL: B released. Crisis averted!")
    pass

def action_x_button_released():
    #queue_message(f"CTRL: Thats better. Leave my X Button alone!")
    pass

def action_y_button_released():
    #queue_message(f"CTRL: Y Button released. Thank you for being cautious!")
    pass

def action_r1_button_released():
    #queue_message(f"CTRL: Turbo disengaged. R1 Button safe!")
    pass

def action_l1_button_released():
    #queue_message(f"CTRL: Shields down. L1 Button released!")
    pass

def action_r2_button_released():
    #queue_message(f"CTRL: R2 Button released. No more speeding!")
    global toggle
    queue_message("+")
    toggle = True

def action_l2_button_released():
    #queue_message(f"CTRL: L2 Button released. Smooth landing!")
    global toggle
    queue_message("-")
    toggle = False

def action_bottom_button_released():
    #queue_message(f"CTRL: Bottom Button released. Mischief managed!")
    pass

def action_select_button_released():
    #queue_message(f"CTRL: Select Button released. Menu closed!")
    pass

def action_start_button_released():
    #queue_message(f"CTRL: Start Button released. Lets pause for a moment.")
    pass

def LJoyStick_button_released():
    #queue_message(f"CTRL: L JoyStick released. That tickled.")
    pass

def RJoyStick_button_released():
    #queue_message(f"CTRL: R JoyStick released. Whew!.")
    pass

def start_controls():
    """
    Listen to gamepad events and execute actions based on button presses/releases and analog movements.
    """
    # Retry loop for detecting the gamepad
    gamepad = None
    while gamepad is None:
        try:
            # Try to connect to the gamepad
            gamepad = evdev.InputDevice(gamepad_path)
            queue_message(f"LOAD: {gamepad.name} connected.")
        except FileNotFoundError:
            time.sleep(5)  # Wait before retrying

    # Define mappings for button events with actions
    button_map = {
        evdev.ecodes.BTN_SOUTH: ("A Button", action_a_button_pressed, action_a_button_released),
        evdev.ecodes.BTN_EAST: ("B", action_b_button_pressed, action_b_button_released),
        evdev.ecodes.BTN_NORTH: ("X Button", action_x_button_pressed, action_x_button_released),
        evdev.ecodes.BTN_WEST: ("Y Button", action_y_button_pressed, action_y_button_released),
        311: ("R1 Button", action_r1_button_pressed, action_r1_button_released),
        310: ("L1 Button", action_l1_button_pressed, action_l1_button_released),
        313: ("R2 Button", action_r2_button_pressed, action_r2_button_released),
        312: ("L2 Button", action_l2_button_pressed, action_l2_button_released),
        306: ("Bottom Button", action_bottom_button_pressed, action_bottom_button_released),
        314: ("Select Button", action_select_button_pressed, action_select_button_released),
        315: ("Start Button", action_start_button_pressed, action_start_button_released),
        317: ("L JoyStick", LJoyStick_button_pressed, LJoyStick_button_released),
        318: ("R JoyStick", RJoyStick_button_pressed, RJoyStick_button_released),
    }

    # Define mappings for analog events
    analog_map = {
        evdev.ecodes.ABS_HAT0Y: {"up": action_dpad_up_pressed, "down": action_dpad_down_pressed},
        evdev.ecodes.ABS_HAT0X: {"left": action_dpad_left_pressed, "right": action_dpad_right_pressed},
        evdev.ecodes.ABS_X: "Left Stick X",
        evdev.ecodes.ABS_Y: "Left Stick Y",
        evdev.ecodes.ABS_Z: "Right Stick X",
        evdev.ecodes.ABS_RZ: "Right Stick Y",
        9: "Trigger Axis",  # Example label for Unknown Axis 9
    }

    queue_message(f"LOAD: Controls Listening...")
    try:
        dpad_state = {"up": False, "down": False, "left": False, "right": False}
        
        for event in gamepad.read_loop():
            if event.type == evdev.ecodes.EV_KEY:  # Button press/release
                button_info = button_map.get(event.code)
                if button_info:
                    button_name, button_action_pressed, button_action_released = button_info
                    if event.value == 1:  # Button pressed
                        button_action_pressed()  # Call the associated pressed action
                        check_secret_code(button_name)
                    elif event.value == 0:  # Button released
                        button_action_released()  # Call the associated released action
                else:
                    queue_message(f"MOVE: Unknown Button {event.code}")
            elif event.type == evdev.ecodes.EV_ABS:  # Analog stick or D-pad movement
                #queue_message(f"MOVE: Event Code: {event.code}, Event Value: {event.value}")

                if event.code == evdev.ecodes.ABS_HAT0Y:
                    if event.value < 0 and not dpad_state["up"]:  # Up pressed
                        action_dpad_up_pressed()
                        check_secret_code("up")
                        dpad_state["up"] = True
                        dpad_state["down"] = False
                    elif event.value > 0 and not dpad_state["down"]:  # Down pressed
                        action_dpad_down_pressed()
                        check_secret_code("down")
                        dpad_state["down"] = True
                        dpad_state["up"] = False
                    elif event.value == 0:  # Released
                        if dpad_state["up"]:
                            action_dpad_up_released()
                        if dpad_state["down"]:
                            action_dpad_down_released()
                        dpad_state["up"] = False
                        dpad_state["down"] = False


                # Handle Left and Right (ABS_HAT0X)
                elif event.code == evdev.ecodes.ABS_HAT0X:
                    if event.value < 0 and not dpad_state["left"]:  # Left pressed
                        action_dpad_left_pressed()
                        check_secret_code("left")
                        dpad_state["left"] = True
                        dpad_state["right"] = False
                    elif event.value > 0 and not dpad_state["right"]:  # Right pressed
                        action_dpad_right_pressed()
                        check_secret_code("right")
                        dpad_state["right"] = True
                        dpad_state["left"] = False
                    elif event.value == 0:  # Released
                        if dpad_state["left"]:
                            action_dpad_left_released()
                        if dpad_state["right"]:
                            action_dpad_right_released()
                        dpad_state["left"] = False
                        dpad_state["right"] = False



                elif event.code == evdev.ecodes.ABS_X:  # Left Stick X Axis
                    action_left_stick_move(event.value, 0)  # Y value isn't used here
                elif event.code == evdev.ecodes.ABS_Y:  # Left Stick Y Axis
                    action_left_stick_move(0, event.value)  # X value isn't used here
                elif event.code == evdev.ecodes.ABS_Z:  # Right Stick X Axis
                    action_right_stick_move(event.value, 0)  # Y value isn't used here
                elif event.code == evdev.ecodes.ABS_RZ:  # Right Stick Y Axis
                    action_right_stick_move(0, event.value)  # X value isn't used here

    except KeyboardInterrupt:
        queue_message("\nExiting...")

    # Clean up
    gamepad.close()

    # Clean up
    gamepad.close()

device = find_controller(controller_name)

#Delete this is for testing
if __name__ == "__main__":
    while True:
        try:
            start_controls()
        except Exception as e:
            queue_message(f"An error occurred: {e}")
            # Optionally add a small delay to prevent tight infinite loops in case of failure
            import time
            time.sleep(1)
