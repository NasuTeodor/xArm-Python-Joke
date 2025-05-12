from xarm.wrapper import XArmAPI
import cv2
import mediapipe as mp

def shift_range(old_value, old_min, old_max, new_min, new_max):
    new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
    return new_value



# # Create pipeline
# pipeline = dai.Pipeline()
# # Create DepthAI node for color camera
# colorCam = pipeline.createColorCamera()
# colorCam.setPreviewSize(1133, 377)  # Set preview size (width, height)
# # Define output stream
# xout = pipeline.createXLinkOut()
# xout.setStreamName("video")
# # Link nodes
# colorCam.preview.link(xout.input)



arm = XArmAPI('192.168.1.160', is_radian=True)

arm.clean_error()
arm.clean_warn()

arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

arm.reset(wait=True)
arm.move_gohome()

#VALORI FIZICE PENTRU BRAT
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

#COORDONATE LA CARE SA FIE BRATUL CONSTANT
arm_x = MAX_X / 2
arm_y = MAX_Y / 2

OFFSET_X = 0
OFFSET_Y = 0

# START VIDEO CAPTURE
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

arm.set_position(x=250, y=0, z=120, wait=True)

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

        arm_theoretic_x = shift_range(relative_x, 0, 1, MIN_X, MAX_X)
        # arm_theoretic_y = shift_range(1 / relative_y, 0, 1, MIN_Y, MAX_Y)
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

        arm.set_position(x=arm_x,
                         y=round(arm_theoretic_x + OFFSET_X, 3),
                         z=round(arm_theoretic_z + OFFSET_Y, 3),
                         wait=False)
        # , roll=0, pitch=-90, yaw=180

        # Draw rectangle
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Optional: Draw landmarks too
        mp.solutions.drawing_utils.draw_landmarks(
            img, first_hand, mpHands.HAND_CONNECTIONS)

    # Show the video with annotation
    display_img = cv2.resize(img, (1920, 1080))  
    cv2.imshow('Hand Tracking', display_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# while True:

#     success, img = cap.read()
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(imgRGB)


#     if results.multi_hand_landmarks:

#         #get the first hand
#         #last hand that is detected is always placed on the index 0
#         first_hand = results.multi_hand_landmarks[len(results.multi_hand_landmarks)-1]

#         # get the wrist from the hand
#         wrist = first_hand.landmark[0]
#         relative_x, relative_y = wrist.x, wrist.y
#         relative_y = max(relative_y, 0.01)

#         arm_theoretic_x = shift_range(relative_x, 0, 1, MIN_X, MAX_X)
#         arm_theoretic_y = shift_range(1 / relative_y, 0, 1, MIN_Y, MAX_Y)

#         arm_theoretic_x = round(arm_theoretic_x, 3)
#         arm_theoretic_y = round(arm_theoretic_y, 3)

#         # arm_theoretic_x = -arm_theoretic_x
#         # arm_theoretic_y = -arm_theoretic_y

#         # x ar trebui sa fie adancimea
#         # y ar fi stanga - dreapta
#         # z ar fi sus - jos
#         arm.set_position(x=250, y=arm_theoretic_x + OFFSET_X, z=arm_theoretic_y + OFFSET_Y, wait=True, roll=0, pitch=-90, yaw=180)

#         # cv2.imshow('frame', img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

cv2.destroyAllWindows(0)