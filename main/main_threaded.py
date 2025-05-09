from xarm.wrapper import XArmAPI
import cv2
import mediapipe as mp
import threading
import time

def shift_range(old_value, old_min, old_max, new_min, new_max):
    new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
    return new_value

# Initialize arm
arm = XArmAPI('192.168.1.160', is_radian=True)

arm.clean_error()
arm.clean_warn()

arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

arm.reset(wait=True)
arm.move_gohome()

# Physical arm values
MIN_X = -230
MAX_X = 230

Z_MIN = 100
Z_MAX = 300

MIN_Y = 60
MAX_Y = 200

MIN_ARM_X = 100
MAX_ARM_X = 270

MIN_AREA = 5000
MAX_AREA = 90000

# Coordinates for constant arm position
arm_x = MAX_X / 2
arm_y = MAX_Y / 2
OFFSET_X = 0
OFFSET_Y = 0

# Shared position for arm movement
latest_position = {"x": 250, "y": 0, "z": 120}
lock = threading.Lock()

def arm_mover():
    """Thread to move the arm independently from the main loop"""
    while True:
        with lock:
            x = latest_position["x"]
            y = latest_position["y"]
            z = latest_position["z"]
        arm.set_position(x=x, y=y, z=z, wait=True)
        time.sleep(0.1)

# Start arm movement in a separate thread
threading.Thread(target=arm_mover, daemon=True).start()

# Start video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Initialize Mediapipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

while True:
    success, img = cap.read()
    if not success:
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        # Get the last detected hand
        first_hand = results.multi_hand_landmarks[len(results.multi_hand_landmarks)-1]

        # Get wrist coordinates
        wrist = first_hand.landmark[0]
        relative_x, relative_y = wrist.x, wrist.y
        relative_y = max(relative_y, 0.01)  # Prevent division by zero

        # arm_theoretic_x = shift_range(relative_x, 0, 1, MIN_X, MAX_X)
        arm_theoretic_x = shift_range(1 - relative_x, 0, 1, MIN_X, MAX_X)
        arm_theoretic_z = shift_range(relative_y, 0, 1, Z_MAX, Z_MIN)

        # Compute bounding box around hand
        h, w, _ = img.shape
        x_coords = [lm.x * w for lm in first_hand.landmark]
        y_coords = [lm.y * h for lm in first_hand.landmark]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        area = (x_max - x_min) * (y_max - y_min)
        area = max(min(area, MAX_AREA), MIN_AREA)
        arm_x = shift_range(area, MIN_AREA, MAX_AREA, MIN_ARM_X, MAX_ARM_X)
        arm_x = round(arm_x, 3)

        # Update shared position
        with lock:
            latest_position["x"] = arm_x
            latest_position["y"] = round(arm_theoretic_x + OFFSET_X, 3)
            latest_position["z"] = round(arm_theoretic_z + OFFSET_Y, 3)

        # Draw rectangle around hand
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Optionally draw hand landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            img, first_hand, mpHands.HAND_CONNECTIONS)

    # Display the video feed with annotation
    display_img = cv2.resize(img, (960, 540))  # downscale for faster display
    cv2.imshow('Hand Tracking', display_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
