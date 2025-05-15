from xarm.wrapper import XArmAPI
import cv2
import mediapipe as mp
import threading
import time
import math

def shift_range(old_value, old_min, old_max, new_min, new_max):
    return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

# Initialize arm
arm = XArmAPI('192.168.1.160', is_radian=True)
arm.clean_error()
arm.clean_warn()
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)
arm.reset(wait=True)
arm.move_gohome()

# Ranges
MIN_X, MAX_X = -250, 20
Z_MIN, Z_MAX = 120, 420
MIN_Y, MAX_Y = -20, 250
MIN_ARM_X, MAX_ARM_X = 100, 270
MIN_AREA, MAX_AREA = 5000, 90000
OFFSET_X, OFFSET_Y = 0, 0


# arm.set_position(x=250, y=0, z=120, wait=True)
# arm.set_position(x=250, y=0, z=120, wait=True, roll=0, pitch=-90, yaw=180)

# Shared position
latest_position = {"x": 250, "y": 0, "z": 120}
lock = threading.Lock()
arm.set_position(x=250, y=0, z=120, wait=True)
# arm.set_servo_angle(servo_id=4 ,angle=-90, speed=100, wait=True)

JOINT5_FIXED = math.radians(-90)
def move_xyz_fix_joint5(x, y, z, 
                        roll=0, pitch=math.radians(-90), yaw=math.radians(180),
                        speed=100):
    # 1) Forțează joint 5 la 90°
    ret = arm.set_servo_angle(
        servo_id=5,               # servo_id=5 corespunde joint 5
        angle=JOINT5_FIXED,       # unghi în radiani
        speed=speed,
        wait=True
    )
    if ret != 0:
        print(f"❌ Eroare la set_servo_angle joint5 (cod {ret})")
        return

    # 2) Mută doar TCP-ul pe X, Y, Z, menținând orientarea dată
    ret = arm.set_position(
        x=x+20, y=y, z=z+15,
        roll=roll, pitch=pitch, yaw=yaw,
        speed=speed,
        wait=True
    )
    if ret != 0:
        print(f"❌ Eroare la set_position (cod {ret})")
    else:
        print(f"✅ Moved to X={x:.1f}, Y={y:.1f}, Z={z:.1f} with joint5≈90°")
def arm_mover():
    while True:
        with lock:
            x = latest_position["x"]
            y = latest_position["y"]
            z = latest_position["z"]
            
        arm.set_position(x=x, y=-y, z=z, wait=True)
        # move_xyz_fix_joint5(x, -y, z)
        time.sleep(0.1)

threading.Thread(target=arm_mover, daemon=True).start()

# Video + Mediapipe init
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

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
        hand = results.multi_hand_landmarks[-1]
        wrist = hand.landmark[0]
        rx, ry = wrist.x, max(wrist.y, 0.01)

        # Inverted X mapping
        arm_theoretic_x = shift_range(1 - rx, 0, 1, MIN_X, MAX_X)
        arm_theoretic_z = shift_range(ry, 0, 1, Z_MAX, Z_MIN)

        # Bounding box
        h, w, _ = img.shape
        xs = [lm.x * w for lm in hand.landmark]
        ys = [lm.y * h for lm in hand.landmark]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        area = (x_max - x_min) * (y_max - y_min)
        area = max(min(area, MAX_AREA), MIN_AREA)
        arm_x = round(shift_range(area, MIN_AREA, MAX_AREA, MIN_ARM_X, MAX_ARM_X), 3)

        with lock:
            latest_position["x"] = arm_x
            latest_position["y"] = round(arm_theoretic_x + OFFSET_X, 3)
            latest_position["z"] = round(arm_theoretic_z + OFFSET_Y, 3)

        # Draw thick red rectangle
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), thickness=3)

        # Draw directional arrows
        mid_x, mid_y = (x_min + x_max) // 2, (y_min + y_max) // 2
        arrow_len = 40
        arrow_gap = 10
        color = (0, 0, 255)
        thickness = 3

        # Up
        cv2.arrowedLine(img, (mid_x, y_min - arrow_gap), (mid_x, y_min - arrow_gap - arrow_len),
                        color, thickness, tipLength=0.3)
        # Down
        cv2.arrowedLine(img, (mid_x, y_max + arrow_gap), (mid_x, y_max + arrow_gap + arrow_len),
                        color, thickness, tipLength=0.3)
        # Left
        cv2.arrowedLine(img, (x_min - arrow_gap, mid_y), (x_min - arrow_gap - arrow_len, mid_y),
                        color, thickness, tipLength=0.3)
        # Right
        cv2.arrowedLine(img, (x_max + arrow_gap, mid_y), (x_max + arrow_gap + arrow_len, mid_y),
                        color, thickness, tipLength=0.3)

        # Draw landmarks
        mp.solutions.drawing_utils.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

    # Display resized output
    disp = cv2.resize(img, (1920, 1080))
    cv2.imshow('Hand Tracking', disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
