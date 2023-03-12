import math
from util import *
import cv2
import numpy as np
import time
import PoseModule as pm


cap = cv2.VideoCapture('How to do a Dumbbell Lateral Raise.mp4')
backSub = cv2.bgsegm.createBackgroundSubtractorMOG()
backSub = cv2.createBackgroundSubtractorMOG2()



# cap = cv2.VideoCapture('http://192.168.1.59:8080/video')
def drawArmsContours(frame,x12, y12 ,x14, y14 ,x16, y16,x13, y13 ,x15, y15 ,x17, y17):
    if (x12, y12 ,x14, y14 ,x16, y16):
        frame2 = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([65, 76, 80], dtype=np.uint8)
        upper_skin = np.array([144, 169, 200], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        fgMask = backSub.apply(frame2)
        masked_img = cv2.bitwise_and(frame, frame, mask=fgMask)

        cv2.imshow("fg",fgMask)
        margin_x = 30
        margin_y = 20
        x1_min = min(x12, x14, x16) - margin_x
        x1_max = max(x12, x14, x16) + margin_x
        y1_min = min(y12, y14, y16) - margin_y
        y1_max = max(y12, y14, y16) + margin_y
        arm1_img = masked_img[y1_min:y1_max, x1_min:x1_max]

        x2_min = min(x13, x15, x17) - margin_x
        x2_max = max(x13, x15, x17) + margin_x
        y2_min = min(y13, y15, y17) - margin_y
        y2_max = max(y13, y15, y17) + margin_y
        arm2_img = masked_img[y2_min:y2_max, x2_min:x2_max]

        # Extract arm contour
        if arm1_img.size>0:
            gray = cv2.cvtColor(arm1_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            contours1, hierarchy1 = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if arm2_img.size > 0:
            gray = cv2.cvtColor(arm2_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            contours2, hierarchy2 = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the contours on the original image
            cv2.drawContours(frame[y1_min:y1_max, x1_min:x1_max], contours1, -1, (0, 0, 255), 2)
            cv2.drawContours(frame[y2_min:y2_max, x2_min:x2_max], contours2, -1, (0, 0, 255), 2)
            cv2.imshow("Image", frame)
            cv2.waitKey(1)


def main():
    detector = pm.poseDetector()
    flag_3_s = True
    start = time.time()
    time_now = start
    while time_now - start < 3:
        print(time.time())
        success, bg = cap.read()
        backSub.apply(bg)
        time_now = time.time()

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
            # Right arm
            angle1 = detector.findAngle(img, 24, 12, 14)
            # Left arm
            angle2 = detector.findAngle(img, 13, 11, 23)
            # Right elbow
            # angle = detector.findAngle(img, 12, 14, 16)
            # left elbow
            # angle = detector.findAngle(img, 15, 13, 11)
            # print(x12 ,x14)
            performance = 0
            # if (abs(x12-x14) < 30):
            #     performance = 2
            x1, y1 = lmList[12][1:]
            x2, y2 = lmList[14][1:]
            x3, y3 = lmList[16][1:]
            x4, y4 = lmList[11][1:]
            x5, y5 = lmList[13][1:]
            x6, y6 = lmList[15][1:]

            # performanceAngle = detector.findAngle(img, 11, 12, 14)
            # # Left Arm
            # angle = detector.findAngle(img, 11, 13, 15,False)
            per1 = np.interp(angle2, (30, 85), (0, 100))
            bar1 = np.interp(angle2, (30, 85), (650, 100))
            # Right Arm
            per2 = np.interp(angle1, (30, 85), (0, 100))
            bar2 = np.interp(angle1, (30, 85), (650, 100))

            # Check for the dumbbell curls
            color = (255, 0, 255)
            if min(per1,per2) == 100:
                color = (0, 255, 0)
                if dir == 0:
                    count += 1
                    dir = 1
            if min(per1, per2) == 0:
                color = (0, 255, 0)
                if dir == 1:
                    count += 0
                    dir = 0
            # print(count)
            # left arm 335 to 280
            # right arm 20 to 85



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
        drawArmsContours(img,x1, y1 ,x2, y2 ,x3, y3,x4, y4,x5,y5,x6,y6)
        # Draw Bar1
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar1)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per1)} %', (1100, 75), cv2.FONT_HERSHEY_DUPLEX, 2, color, 2)

        # Draw Bar2
        cv2.rectangle(img, (900, 100), (975, 650), color, 3)
        cv2.rectangle(img, (900, int(bar2)), (975, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per2)} %', (900, 75), cv2.FONT_HERSHEY_DUPLEX, 2, color, 2)

        # Draw Curl Count
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_DUPLEX, 5,
                    (51, 255, 255), 25)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":

    main()