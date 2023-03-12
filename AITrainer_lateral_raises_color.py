import math
from util import *
import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture('How to do a Dumbbell Lateral Raise.mp4')


# cap = cv2.VideoCapture('http://192.168.1.59:8080/video')

def main():
    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    performance = 0
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (1280, 720))
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        if len(lmList) != 0:
            # Right arm
            angle1 = detector.findAngle(img, 24, 12, 14)
            # Left arm
            angle2 = detector.findAngle(img, 13, 11, 23)

            x1, y1 = lmList[12][1:]
            x2, y2 = lmList[14][1:]
            x3, y3 = lmList[16][1:]
            x4, y4 = lmList[11][1:]
            x5, y5 = lmList[13][1:]
            x6, y6 = lmList[15][1:]

            # # Left Arm
            per1 = np.interp(angle2, (30, 85), (0, 100))
            bar1 = np.interp(angle2, (30, 85), (650, 100))
            # Right Arm
            per2 = np.interp(angle1, (30, 85), (0, 100))
            bar2 = np.interp(angle1, (30, 85), (650, 100))

            # Check for the lateral raises
            color = (255, 0, 255)
            if min(per1, per2) == 100:
                color = (0, 255, 0)
                if dir == 0:
                    count += 1
                    dir = 1
            if min(per1, per2) == 0:
                color = (0, 255, 0)
                if dir == 1:
                    count += 0
                    dir = 0

            # Draw performance
            if (abs(per1 - per2) < 5):
                cv2.putText(img, "Good!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 6)
            elif (abs(per1 - per2) < 15):
                cv2.putText(img, "Ok!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 6)
            else:
                cv2.putText(img, "Bad!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 6)

        drawArmsContours(img, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6)
        draw_bar(img, color, bar1, per1)
        draw_bar2(img, color, bar2, per2)
        drawCounter(img,count)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
