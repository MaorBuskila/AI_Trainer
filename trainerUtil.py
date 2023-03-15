import cv2
from util import *
import numpy as np
import time
import PoseModule as pm


def bicepsCurles(detector, frame, lmList, color, count, end, portrait_mode, black_frame, badRep):
    angle = detector.findAngle(frame, 12, 14, 16)
    x1, y1 = lmList[12][1:]
    x2, y2 = lmList[14][1:]
    x3, y3 = lmList[16][1:]

    per = np.interp(angle, (70, 160), (100, 0))
    bar = np.interp(angle, (70, 160), (100, 650))

    # draw Contours
    drawArmContours(frame, x1, y1, x2, y2, x3, y3)

    # draw Joints
    detector.drawJoints(frame, 12, 14, 16, angle)

    badRep = drawPerformanceBiceps(frame, x1, x2, badRep)

    # Check for the dumbbell curls
    incCounter(per, color, end, count, badRep)

    # Draw Curl Count
    drawCounter(frame, count)

    if portrait_mode:
        black_frame[:, 280:1000] = frame
        draw_bar(black_frame, color, bar, per)
        cv2.imshow("Image", black_frame)
        cv2.waitKey(1)
    else:
        draw_bar(frame, color, bar, per)
        cv2.imshow("Image", frame)
        cv2.waitKey(1)


def Triceps(detector, frame, lmList, color, count, end, portrait_mode, black_frame, badRep):


    # # Left Arm
    # angle = detector.findAngle(frame, 15, 13, 11)
    # x1, y1 = lmList[11][1:]
    # x2, y2 = lmList[13][1:]
    # x3, y3 = lmList[15][1:]
    # per = np.interp(angle, (90, 155), (0, 100))
    # bar = np.interp(angle, (90, 155), (650, 100))

    # Right Arm
    angle = detector.findAngle(frame, 12, 14, 16)
    x1, y1 = lmList[12][1:]
    x2, y2 = lmList[14][1:]
    x3, y3 = lmList[16][1:]

    per = np.interp(angle, (90, 155), (0, 100))
    bar = np.interp(angle, (90, 155), (650, 100))

    # draw Contours
    drawArmContours(frame, x1, y1, x2, y2, x3, y3)

    # draw Joints
    detector.drawJoints(frame, 12, 14, 16, angle)
    # detector.drawJoints(frame, 11, 13, 15, angle)

    badRep = drawPerformanceTriceps(frame, x1, x2, badRep)

    # Check for the Triceps
    incCounter(per, color, end, count, badRep)

    # Draw Bar
    draw_bar(frame, color, bar, per)

    # Draw Curl Count

    if portrait_mode:
        black_frame[:, 280:1000] = frame
        draw_bar(black_frame, color, bar, per)
        drawCounter(black_frame, count)
        cv2.imshow("Image", black_frame)
        cv2.waitKey(1)
    else:
        draw_bar(frame, color, bar, per)
        drawCounter(frame, count)
        cv2.imshow("Image", frame)
        cv2.waitKey(1)


def lateralRaises(detector, frame, lmList, color, count, end, portrait_mode, black_frame, badRep):
    angle1 = detector.findAngle(frame, 24, 12, 14)
    # Left arm
    angle2 = detector.findAngle(frame, 13, 11, 23)

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

    badRep = drawPerformanceLateralRaises(frame, per1, per2, badRep)
    incCounterLateralRaises(per1, per2, end, count, color, badRep)

    drawArmsContours(frame, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6)

    if portrait_mode:
        black_frame[:, 280:1000] = frame
        draw_bar(black_frame, color, bar1, per1)
        draw_bar2(black_frame, color, bar2, per2)
        drawCounter(black_frame, count)
        cv2.imshow("Image", black_frame)
        cv2.waitKey(1)
    else:
        draw_bar(frame, color, bar1, per1)
        draw_bar2(frame, color, bar2, per2)
        drawCounter(frame, count)
        cv2.imshow("Image", frame)
        cv2.waitKey(1)

    return count
