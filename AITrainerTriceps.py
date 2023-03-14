import math

import cv2
import numpy as np
import time
import PoseModule as pm
from util import *
cap = cv2.VideoCapture('Triceps.MOV')


# cap = cv2.VideoCapture('http://192.168.1.31:8080/video')

# def drawArmContours(frame,x12, y12 ,x14, y14 ,x16, y16):
#     if (x12, y12 ,x14, y14 ,x16, y16):
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         lower_skin = np.array([0, 20, 70], dtype=np.uint8)
#         upper_skin = np.array([15, 165, 255], dtype=np.uint8)
#         mask = cv2.inRange(hsv, lower_skin, upper_skin)
#         masked_img = cv2.bitwise_and(frame, frame, mask=mask)
#         x_min = int(min(x12, x14, x16)*0.95)
#         x_max = int(max(x12, x14, x16)*1.05)
#         y_min = int(min(y12, y14, y16)*0.95)
#         y_max = int(max(y12, y14, y16)*1.05)
#         arm_img = masked_img[y_min:y_max, x_min:x_max]
#
#         # Extract arm contour
#         if arm_img.size>0:
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
    pTime = 0
    per = 0
    count = 0
    bar = 0
    color = 0
    while True:
        black_frame = np.zeros((1280, 1280, 3), dtype=np.uint8)
        success, img = cap.read()
        img = cv2.resize(img, (720, 1280))
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        # print(lmList)
        if len(lmList) != 0:
            # Right Arm
            angle = detector.findAngle(img, 15, 13, 11)
            x1, y1 = lmList[11][1:]
            x2, y2 = lmList[13][1:]
            x3, y3 = lmList[15][1:]
            # # Left Arm

            per = np.interp(angle, (90, 155), (0, 100))
            bar = np.interp(angle, (90, 155), (650, 100))

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
            # print(count)

            # Draw performance
            if(abs(x1-x2) < 30):
                # cv2.rectangle(img, (0, 450), (250, 720), (255, 255, 0), cv2.FILLED)
                # print("Good")
                cv2.putText(img, "Good!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 10)
            elif (abs(x2-x3) < 100):
                # cv2.rectangle(img, (0, 450), (250, 720), (255, 255, 0), cv2.FILLED)
                # print("OK")
                cv2.putText(img, "Ok!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 10)
            else:
                # cv2.rectangle(img, (0, 450), (250, 720), (255, 255, 0), cv2.FILLED)
                # print("bad")
                cv2.putText(img, "Bad!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 10)
        drawArmContours(img, x1, y1, x2, y2, x3, y3)
        black_frame[:, 280:1000] = img
        draw_bar(black_frame, color, bar, per)
        drawCounter(black_frame, count)
        cv2.imshow("Image", black_frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    triceps_extenstions()
