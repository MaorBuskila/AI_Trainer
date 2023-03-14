import cv2
from AITrainerBiceps import *
from AITrainerTriceps import *
from AITrainerLateralRaisesColor import *


def main():
    print("Please select an exercise:")
    print("1. Biceps Curls")
    print("2. Triceps Extensions")
    print("3. Lateral Raises")

    option = int(input("Enter the number of the option you would like to select: "))
    # add the option for closing the video according to the number of counts that the user decied

    if option == 1:
        print("You selected Biceps Curls")
        biceps_curls()
    elif option == 2:
        print("You selected Triceps Extensions")
        triceps_extenstions()
    elif option == 3:
        print("You selected Lateral Raises")
        lateral_raises()
    else:
        print("Invalid option selected")


if __name__ == "__main__":
    main()
