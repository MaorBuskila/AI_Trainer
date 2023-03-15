import sys

import cv2
from util import *
from trainerUtil import *
import numpy as np
import time
import PoseModule as pm



def startExercise(option):

    if option == 1:
        print("You selected Biceps Curls")
        cap = cv2.VideoCapture('Biceps/Hammer.mp4')
        # biceps_curls()
    elif option == 2:
        print("You selected Triceps Extensions")
        cap = cv2.VideoCapture('Triceps/Ronny_triceps.MOV')
        # triceps_extenstions()
    elif option == 3:
        print("You selected Lateral Raises")
        cap = cv2.VideoCapture('Lateral/Ronny_lateral.MOV')        # lateral_raises()
    elif option == 4:
        print("You selected Exit")
        sys.exit()
    else:
        print("Invalid option selected")

    detector = pm.poseDetector()
    count, end, angle = [0], [0], 0
    color = (255, 0, 255)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height, 3)
    black_frame = np.zeros(size)
    portrait_mode = frame_height >= frame_width

    while True:
        success, frame = cap.read()
        if not success:
            print("Video end or corrupted, returning to main")
            main()
        if portrait_mode:
            frame = cv2.resize(frame, (720, 1280))
            black_frame = createBlackFrame()

        frame = detector.findPose(frame, False)
        lmList = detector.findPosition(frame, False)

        badRep = [False]

        if len(lmList) != 0:
            if (option == 1):  # Biceps
                bicepsCurles(detector,frame,lmList,color,count,end,portrait_mode,black_frame,badRep)
            elif option == 2:
                Triceps(detector,frame,lmList,color,count,end,portrait_mode,black_frame,badRep)
            elif option == 3:
                lateralRaises(detector,frame,lmList,color,count,end,portrait_mode,black_frame,badRep)
            elif option == 4:
                sys.exit()

    # Check if there was a frame to read
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()








def main():
    print("Please select an exercise:")
    print("1. Biceps Curls")
    print("2. Triceps Extensions")
    print("3. Lateral Raises")
    print("4. Exit")

    option = int(input("Enter the number of the option you would like to select: "))

    # add the option for closing the video according to the number of counts that the user decied
    startExercise(option)


if __name__ == "__main__":
    main()
