import cv2
from util import *
import numpy as np
import time
import PoseModule as pm


def draw_bar(img, color, bar, per):
    # Draw Bar1
    cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
    cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
    cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_DUPLEX, 2, color, 2)


def draw_bar2(img, color, bar2, per2):
    # Draw Bar2
    cv2.rectangle(img, (900, 100), (975, 650), color, 3)
    cv2.rectangle(img, (900, int(bar2)), (975, 650), color, cv2.FILLED)
    cv2.putText(img, f'{int(per2)} %', (900, 75), cv2.FONT_HERSHEY_DUPLEX, 2, color, 2)


def drawCounter(img, count):
    # Draw Curl Count
    cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_DUPLEX, 5,
                (51, 255, 255), 25)


def crop_frame(frame, x12, y12, x14, y14, x16, y16):
    x_min = min(x12, x14, x16)
    x_max = max(x12, x14, x16)
    y_min = min(y12, y14, y16)
    y_max = max(y12, y14, y16)

    x_m = x_min - int(abs(x_max - x_min) * 0.25)
    x_M = x_max + int(abs(x_max - x_min) * 0.25)
    y_m = y_min - int(abs(y_max - y_min) * 0.2)
    y_M = y_max + int(abs(y_max - y_min) * 0.2)
    # Draw the bounding box
    cropped_image = frame[y_m:y_M, x_m:x_M]
    return cropped_image, x_m, x_M, y_m, y_M


def drawArmContours(frame, x12, y12, x14, y14, x16, y16):
    if (x12, y12, x14, y14, x16, y16):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([15, 165, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        masked_img = cv2.bitwise_and(frame, frame, mask=mask)
        arm_img, x_min, x_max, y_min, y_max = crop_frame(masked_img, x12, y12, x14, y14, x16, y16)

        # Extract arm contour
        if arm_img.size > 0:
            gray = cv2.cvtColor(arm_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # hsv = cv2.cvtColor(arm_img, cv2.COLOR_BGR2HSV)
            # h, s, v = cv2.split(hsv)
            # h = np.mod(h + 50, 180)
            # s = np.clip(s - 0, 0, 255)
            # v = np.clip(v + 20, 0, 255)
            # hsv = cv2.merge([h, s, v])
            # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # result = cv2.bitwise_or(cv2.bitwise_and(arm_img, arm_img, mask=inv_mask), cv2.bitwise_and(bgr, bgr, mask=mask))
            # alpha = 0.5
            # blended = cv2.addWeighted(arm_img, alpha, result, 1 - alpha, 0)
            # arm_img[y_min:y_max, x_min:x_max] = blended

            # Draw the contours on the original image
            cv2.drawContours(frame[y_min:y_max, x_min:x_max], contours, -1, (0, 0, 255), 2)
            # cv2.imshow("Image", arm_img)
            # cv2.waitKey(1)


def drawArmContours2(frame, x12, y12, x14, y14, x16, y16):
    if (x12, y12, x14, y14, x16, y16):
        imageYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        min_YCrCb = np.array([0, 133, 77], np.uint8)
        max_YCrCb = np.array([235, 173, 127], np.uint8)
        skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
        skinYCrCb = cv2.bitwise_and(frame, frame, mask=skinRegionYCrCb)
        arm_img, x_min, x_max, y_min, y_max = crop_frame(skinYCrCb, x12, y12, x14, y14, x16, y16)

        # Extract arm contour
        if arm_img.size > 0:
            gray = cv2.cvtColor(arm_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the contours on the original image
            cv2.drawContours(frame[y_min:y_max, x_min:x_max], contours, -1, (0, 0, 255), 2)
            cv2.imshow("Image", skinYCrCb)
            cv2.waitKey(1)


def drawArmsContours(frame, x12, y12, x14, y14, x16, y16, x13, y13, x15, y15, x17, y17):
    if (x12, y12, x14, y14, x16, y16):
        imageYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        min_YCrCb = np.array([0, 133, 77], np.uint8)
        max_YCrCb = np.array([235, 173, 127], np.uint8)
        skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
        skinYCrCb = cv2.bitwise_and(frame, frame, mask=skinRegionYCrCb)

        margin_x = 25
        margin_y = 20
        x1_min = min(x12, x14, x16) - margin_x
        x1_max = max(x12, x14, x16) + margin_x
        y1_min = min(y12, y14, y16) - margin_y
        y1_max = max(y12, y14, y16) + margin_y

        arm1_img = skinYCrCb[y1_min:y1_max, x1_min:x1_max]

        x2_min = min(x13, x15, x17) - margin_x
        x2_max = max(x13, x15, x17) + margin_x
        y2_min = min(y13, y15, y17) - margin_y
        y2_max = max(y13, y15, y17) + margin_y
        arm2_img = skinYCrCb[y2_min:y2_max, x2_min:x2_max]

        # Extract arm contour
        if arm1_img.size > 0:
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
            # cv2.imshow("Image", skinYCrCb)
            # cv2.waitKey(1)


def drawArmsContours2(frame, x12, y12, x14, y14, x16, y16, x13, y13, x15, y15, x17, y17):
    if (x12, y12, x14, y14, x16, y16):
        imageYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        min_YCrCb = np.array([0, 133, 77], np.uint8)
        max_YCrCb = np.array([235, 173, 127], np.uint8)
        skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
        skinYCrCb = cv2.bitwise_and(frame, frame, mask=skinRegionYCrCb)

        arm1_img, x1_min, x1_max, y1_min, y1_max = crop_frame(skinYCrCb, x12, y12, x14, y14, x16, y16)
        arm2_img, x2_min, x2_max, y2_min, y2_max = crop_frame(skinYCrCb, x13, y13, x15, y15, x17, y17)

        arm1_img = skinYCrCb[y1_min:y1_max, x1_min:x1_max]
        arm2_img = skinYCrCb[y2_min:y2_max, x2_min:x2_max]

        # Extract arm contour
        if arm1_img.size > 0:
            gray = cv2.cvtColor(arm1_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(gray, 50, 150)
            contours1, hierarchy1 = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if arm2_img.size > 0:
            gray = cv2.cvtColor(arm2_img, cv2.COLOR_BGR2GRAY)
            # blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(gray, 50, 100)
            contours2, hierarchy2 = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the contours on the original image
            cv2.drawContours(frame[y1_min:y1_max, x1_min:x1_max], contours1, -1, (0, 0, 255), 2)
            cv2.drawContours(frame[y2_min:y2_max, x2_min:x2_max], contours2, -1, (0, 0, 255), 2)
            # cv2.imshow("Image", frame)
            # cv2.waitKey(1)
