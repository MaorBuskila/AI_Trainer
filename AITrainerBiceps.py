import math

import cv2
import numpy as np
import time
import PoseModule as pm

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('http://192.168.1.59:8080/video')


def main():
    detector = pm.poseDetector()

    count = 0
    dir = 0
    pTime = 0
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (1280, 720))
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        # print(lmList)
        if len(lmList) != 0:
            # Right Arm
            angle = detector.findAngle(img, 12, 14, 16)
            x12, y12 = lmList[12][1:]
            x14, y14 = lmList[14][1:]
            # print(x12 ,x14)
            performance = 0
            if(y12 > y14):
                if (abs(x12-x14) < 30):
                    performance = 2
                elif(abs(x12-x14) < 60):
                    performance = 1



            # performanceAngle = detector.findAngle(img, 11, 12, 14)
            # # Left Arm
            # angle = detector.findAngle(img, 11, 13, 15,False)
            per = np.interp(angle, (70, 160), (100, 0))
            bar = np.interp(angle, (70, 160), (100, 650))
            # straight = np.interp(performanceAngle, (210, 310), (0, 100))
            # print(angle, per)

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

            # Draw Bar
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_DUPLEX, 2, color, 2)

            # Draw Curl Count
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_DUPLEX, 5,
                        (51, 255, 255), 25)

            # Draw performance
            # if(abs(x12-x14) < 30):
            #     # cv2.rectangle(img, (0, 450), (250, 720), (255, 255, 0), cv2.FILLED)
            #     # print("Good")
            #     cv2.putText(img, "Good!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 10)
            # elif (abs(x12-x14) < 100):
            #     # cv2.rectangle(img, (0, 450), (250, 720), (255, 255, 0), cv2.FILLED)
            #     # print("OK")
            #     cv2.putText(img, "Ok!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 10)
            # else:
            #     # cv2.rectangle(img, (0, 450), (250, 720), (255, 255, 0), cv2.FILLED)
            #     # print("bad")
            #     cv2.putText(img, "Bad!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 10)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()