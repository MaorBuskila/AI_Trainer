import math

import cv2
from util import *
import numpy as np
import time
import PoseModule as pm


def draw_bar(img, color, bar, per):
    # Draw Bar1
    cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
    cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
    cv2.putText(img, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_DUPLEX, 2, color, 2)


def draw_bar2(img, color, bar2, per2):
    # Draw Bar2
    cv2.rectangle(img, (900, 100), (975, 650), color, 3)
    cv2.rectangle(img, (900, int(bar2)), (975, 650), color, cv2.FILLED)
    cv2.putText(img, f'{int(per2)}%', (900, 75), cv2.FONT_HERSHEY_DUPLEX, 2, color, 2)


def drawCounter(img, count):
    # Draw Curl Count
    cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_DUPLEX, 5,
                (51, 255, 255), 25)


def drawArmContours(frame, x12, y12, x14, y14, x16, y16):
    if (x12, y12, x14, y14, x16, y16):
        img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # skin color range for hsv color space
        HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
        HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # converting from gbr to YCbCr color space
        img_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        # skin color range for hsv color space
        YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
        YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # merge skin detection (YCbCr and hsv)
        global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
        global_mask = cv2.medianBlur(global_mask, 3)
        global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

        # HSV_result = cv2.bitwise_not(HSV_mask)
        # YCrCb_result = cv2.bitwise_not(YCrCb_mask)
        global_result = cv2.bitwise_not(global_mask)
        margin_x = 80
        margin_y = 50
        x_min = min(x12, x14, x16) - margin_x
        x_max = max(x12, x14, x16) + margin_x
        y_min = min(y12, y14, y16) - margin_y
        y_max = max(y12, y14, y16) + margin_y

        arm_img = global_result[y_min:y_max, x_min:x_max]

        if arm_img.size > 0:
            edges = cv2.Canny(arm_img, 50, 150)
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the contours on the original image
            cv2.drawContours(frame[y_min:y_max, x_min:x_max], contours, -1, (0, 0, 255), 2)
            # cv2.imshow("Image", frame)
            # cv2.waitKey(1)


def drawArmsContours(frame, x12, y12, x14, y14, x16, y16, x13, y13, x15, y15, x17, y17):
    if (x12, y12, x14, y14, x16, y16):
        img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # skin color range for hsv color space
        HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
        HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # converting from gbr to YCbCr color space
        img_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        # skin color range for hsv color space
        YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
        YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # merge skin detection (YCbCr and hsv)
        global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
        global_mask = cv2.medianBlur(global_mask, 3)
        global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

        margin_x = 25
        margin_y = 20
        x1_min = min(x12, x14, x16) - margin_x
        x1_max = max(x12, x14, x16) + margin_x
        y1_min = min(y12, y14, y16) - margin_y
        y1_max = max(y12, y14, y16) + margin_y

        arm1_img = global_mask[y1_min:y1_max, x1_min:x1_max]

        x2_min = min(x13, x15, x17) - margin_x
        x2_max = max(x13, x15, x17) + margin_x
        y2_min = min(y13, y15, y17) - margin_y
        y2_max = max(y13, y15, y17) + margin_y
        arm2_img = global_mask[y2_min:y2_max, x2_min:x2_max]

        # Extract arm contour
        if arm1_img.size > 0:
            blur = cv2.GaussianBlur(arm1_img, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            contours1, hierarchy1 = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if arm2_img.size > 0:
            blur = cv2.GaussianBlur(arm2_img, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            contours2, hierarchy2 = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the contours on the original image
            cv2.drawContours(frame[y1_min:y1_max, x1_min:x1_max], contours1, -1, (0, 0, 255), 2)
            cv2.drawContours(frame[y2_min:y2_max, x2_min:x2_max], contours2, -1, (0, 0, 255), 2)
            # cv2.imshow("Image", skinYCrCb)
            # cv2.waitKey(1)


def incCounter(per, color, end, count, badRep):
    if per == 100:
        color = (0, 255, 0)
        if end[0] == 0:
            if not badRep[0]:
                count[0] += 1
                end[0] = 1
    if per == 0:
        color = (0, 255, 0)
        if end[0] == 1:
            end[0] = 0


def incCounterLateralRaises(per1, per2, end, count, color, badRep):
    if min(per1, per2) == 100:
        color = (0, 255, 0)
        if end[0] == 0:
            if not badRep[0]:
                count[0] += 1
                end[0] = 1
    if min(per1, per2) == 0:
        color = (0, 255, 0)
        if end[0] == 1:
            end[0] = 0


def drawPerformanceBiceps(frame, x1, x2, badRep):
    if (abs(x1 - x2) < 20):
        cv2.putText(frame, "Good!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 0), 8)
    elif (abs(x1 - x2) < 40):
        cv2.putText(frame, "Ok!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 0), 8)
    else:
        cv2.putText(frame, "Bad!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 8)
        badRep[0] = True
    return badRep


def drawPerformanceTriceps(frame, x1, x2, badRep):
    # Draw performance
    if (abs(x1 - x2) < 50):
        cv2.putText(frame, "Good!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 10)
    elif (abs(x1 - x2) < 100):
        cv2.putText(frame, "Ok!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 10)
    else:
        cv2.putText(frame, "Bad!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 10)
        badRep[0] = True
    return badRep


def drawPerformanceLateralRaises(frame, per1, per2, badRep):
    # Draw performance
    if (abs(per1 - per2) < 5):
        cv2.putText(frame, "Good!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 6)
    elif (abs(per1 - per2) < 15):
        cv2.putText(frame, "Ok!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 6)
    else:
        cv2.putText(frame, "Bad!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 6)
        badRep[0] = True
    return badRep


def createBlackFrame():
    return np.zeros((1280, 1280, 3), dtype=np.uint8)
