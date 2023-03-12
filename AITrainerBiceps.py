import math
from util import *
import cv2
import numpy as np
import time
import PoseModule as pm

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Hammer.mp4')
# cap = cv2.VideoCapture('http://192.168.1.59:8080/video')

# def drawArmContours(frame, x12, y12, x14, y14, x16, y16):
#     if (x12, y12, x14, y14, x16, y16):
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         lower_skin = np.array([0, 20, 70], dtype=np.uint8)
#         upper_skin = np.array([20, 255, 255], dtype=np.uint8)
#         mask = cv2.inRange(hsv, lower_skin, upper_skin)
#         masked_img = cv2.bitwise_and(frame, frame, mask=mask)
#         x_min = int(min(x12, x14, x16)*0.95)
#         x_max = int(max(x12, x14, x16)*1.05)
#         y_min = int(min(y12, y14, y16)*0.95)
#         y_max = int(max(y12, y14, y16)*1.05)
#         arm_img = masked_img[y_min:y_max, x_min:x_max]
#
#         # Extract arm contour
#         if arm_img.size > 0:
#             gray = cv2.cvtColor(arm_img, cv2.COLOR_BGR2GRAY)
#             blur = cv2.GaussianBlur(gray, (5, 5), 0)
#             edges = cv2.Canny(blur, 50, 150)
#             contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#             # Draw the contours on the original image
#             cv2.drawContours(frame[y_min:y_max, x_min:x_max], contours, -1, (0, 0, 255), 2)
#             # cv2.imshow("Image", arm_img)
#             # cv2.waitKey(1)


def main():
    detector = pm.poseDetector()
    count = 0
    dir = 0
    x1, x2, x3 = 0, 0, 0
    y1, y2, y3 = 0, 0, 0
    angle = 0
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (1280, 720))
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        # print(lmList)
        if len(lmList) != 0:
            # Right Arm
            angle = detector.findAngle(img, 12, 14, 16)
            x1, y1 = lmList[12][1:]
            x2, y2 = lmList[14][1:]
            x3, y3 = lmList[16][1:]
            # print(x12 ,x14)
            performance = 0
            if (y1 > y2):
                if (abs(x1 - x2) < 30):
                    performance = 2
                elif (abs(x1 - x2) < 60):
                    performance = 1

            # Draw performance
            if (abs(x1 - x2) < 30):
                cv2.putText(img, "Good!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 0), 8)
            elif (abs(x1 - x2) < 100):
                cv2.putText(img, "Ok!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 0), 8)
            else:
                cv2.putText(img, "Bad!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 8)

        # Check if there was a frame to read
        if not success:
            break

        per = np.interp(angle, (70, 160), (100, 0))
        bar = np.interp(angle, (70, 160), (100, 650))

        # Check for the dumbbell curls
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 1
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0
                dir = 0

        # draw Contours
        drawArmContours2(img, x1, y1, x2, y2, x3, y3)

        # draw Joints
        detector.drawJoints(img, 12, 14, 16, angle)

        # Draw Bar
        draw_bar(img, color, bar, per)

        # Draw Curl Count
        drawCounter(img, count)

        # cv2.imshow("Image", img)
        # cv2.waitKey(1)


if __name__ == "__main__":
    main()
