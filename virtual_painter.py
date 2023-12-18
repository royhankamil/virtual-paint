import cv2
import numpy as np
import mediapipe as mp
import time
import os
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# for webcam input
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 5)
width = 1280
height = 720
cap.set(3, width)
cap.set(4, height)

# image contain drawing
img_canvas = np.zeros((height, width, 3), np.uint8)

# get all header image in list
folder_path = os.path.dirname(os.path.realpath(__file__)) + '\Header'
my_list = os.listdir(folder_path)
overlay_list = []
for imPath in my_list:
    image = cv2.imread(f'{folder_path}/{imPath}')
    overlay_list.append(image)

# presetting
header = overlay_list[0]
draw_color = (0, 0, 255)
thickness = 20
tipIds = [4, 8, 12, 16, 20]
xp, yp = [0, 0]

with mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("ignoring empty camera frame")
            break
        # flip horizontally for a later selfie-view display, convert bgr to rgb
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # to improve performance, optionally mark the image
        image.flags.writeable = False
        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # getting all hand points coordinates
                points = []
                for lm in hand_landmarks.landmark:
                    points.append([int(lm.x * width), int(lm.y * height)])

                # only go through the code when hand detected
                if len(points) != 0:
                    x1, y1 = points[8]  # finger
                    x2, y2 = points[12]  # middle finger
                    x3, y3 = points[4]  # thumb
                    x4, y4 = points[20]  # pinky

                    # checking which fingers are up
                    fingers = []
                    # checking the thumb
                    if points[tipIds[0]][0] < points[tipIds[0] - 1][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    # the rest of the fingers
                    for id in range(1, 5):
                        if points[tipIds[id]][1] < points[tipIds[id] - 2][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    # selection mode - two finger up
                    non_sel = [0, 3, 4]  # indexes of the fingers that need to be down in the selection mode
                    if (fingers[1] and fingers[2]) and all(fingers[i] == 0 for i in non_sel):
                        xp, yp = [x1, y1]

                        # selecting the color or eraser
                        if y1 < 125:
                            if 170 < x1 < 295:
                                header = overlay_list[0]
                                draw_color = (0, 0, 255)
                            elif 436 < x1 < 561:
                                header = overlay_list[1]
                                draw_color = (255, 0, 0)
                            elif 700 < x1 < 825:
                                header = overlay_list[2]
                                draw_color = (0, 255, 0)
                            elif 980 < x1 < 1105:
                                header = overlay_list[3]
                                draw_color = (0, 0, 0)

                        cv2.rectangle(image, (x1 - 10, y1 - 15), (x2 + 10, y2 + 23), draw_color, cv2.FILLED)

                    # stand by mode
                    non_stand = [0, 2, 3]  # indexes of the fingers that need to be down in the Stand Mode
                    if (fingers[1] and fingers[4]) and all(fingers[i] == 0 for i in non_stand):
                        cv2.line(image, (xp, yp), (x4, y4), draw_color, 5)
                        xp, yp = [x1, y1]

                    # draw mode - one finger up
                    non_draw = [0, 2, 3, 4]
                    if fingers[1] and all(fingers[i] == 0 for i in non_draw):
                        cv2.circle(image, (x1, y1), int(thickness / 2), draw_color, cv2.FILLED)
                        if xp == 0 and yp == 0:
                            xp, yp = [x1, y1]

                        # draw a line between the current position
                        cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, thickness)
                        # update last position
                        xp, yp = [x1, y1]

                    # clear canvas when hand closed
                    if cv2.waitKey(3) & 0xFF == ord('c'):
                        img_canvas = np.zeros((height, width, 3), np.uint8)
                        xp, yp = [x1, y1]

                    # adjust the thickness of the line
                    selecting = [1, 1, 0, 0, 0]  # selecting thickness of line
                    setting = [1, 1, 0, 0, 1]  # setting the thickness chosen

                    if all(fingers[i] == j for i, j in zip(range(0, 5), selecting)) or all(
                            fingers[i] == j for i, j in zip(range(0, 5), selecting)):
                        # getting radius circle
                        r = int(math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2) / 3)

                        # get middle point
                        x0, y0 = [(x1 + x3) / 2, (y1 + y3) / 2]

                        # get vector these two fingers
                        v1, v2 = [x1 - x3, y1 - y3]
                        v1, v2 = [-v2, v1]

                        # normalize
                        mod_v = math.sqrt(v1 ** 2 + v2 ** 2)
                        v1, v2 = [v1 / mod_v, v2 / mod_v]

                        # draw circle and translated c units
                        c = 3 + r
                        x0, y0 = [int(x0 - v1 * c), int(y0 - v2 * c)]
                        cv2.circle(image, (x0, y0), int(r / 2), draw_color, -1)

                        # setting the thickness
                        if fingers[4]:
                            thickness = r
                            cv2.putText(image, 'Check', (x4 - 25, y4 - 8), cv2.FONT_HERSHEY_TRIPLEX, 0.8,
                                        (0, 0, 0), 1)

                        xp, yp = [x1, y1]

        # Setting the header in the video
        header_height, header_width, _ = header.shape
        image[0:header_height, 0:width] = header

        # Image preprocessing to produce final image
        img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(image, img_inv)
        img = cv2.bitwise_or(img, img_canvas)

        # Adding shadow effect to text
        shadow = cv2.putText(img.copy(), 'Virtual Painter', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5,
                             cv2.LINE_AA)
        img = cv2.putText(img, 'Virtual Painter', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5,
                          cv2.LINE_AA)

        # Combine the original image with the shadow
        img = cv2.addWeighted(img, 1, shadow, 0.5, 0)

        # Adding user guide text in the bottom right corner
        guide_text = [
            "1. Angkat jari telunjuk untuk menggambar.",
            "2. Angkat dua jari (telunjuk dan tengah) untuk memilih warna atau penghapus.",
            "3. Gunakan jempol dan jari telunjuk untuk mengatur ketebalan, konfirmasi dengan metal (jari kelingking diangkat).",
            "4. Klik tombol 'c' untuk menghapus semua gambar."
        ]

        # Adding background behind the guide text
        bg_height = len(guide_text) * 30 + 20
        bg = np.zeros((bg_height, 450, 3), np.uint8)
        bg[:] = (192, 192, 192)  # Gray background color

        # Adding guide text to the background
        for i, text in enumerate(guide_text):
            cv2.putText(bg, text, (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Combining the background with the main image
        img[height - bg_height:height, width - 450:width] = bg

        cv2.imshow('Hand Tracker', img)
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
