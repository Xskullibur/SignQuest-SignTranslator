import sys
import cv2
import os
from sys import platform
import argparse

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGE_PATH = DIR_PATH + '/input/images/temp/'


def estimate_pose(image):

    # Import Openpose (Windows/Ubuntu/OSX)
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(DIR_PATH + '/pose_estimation/Release')
            os.environ['PATH'] = os.environ['PATH'] + ';' + DIR_PATH + '/pose_estimation/Release;' + DIR_PATH + '/pose_estimation/bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the
            # OpenPose/python module from there. This will install OpenPose and the python library at your desired
            # installation path. Ensure that this is in your python path in order to use it. sys.path.append(
            # '/usr/local/python')
            from openpose import pyopenpose as op

    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python '
            'script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default=IMAGE_PATH + image,
                        help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = DIR_PATH + "/pose_estimation/bin/models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1]) - 1:
            next_item = args[1][i + 1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:
                params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params:
                params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.execute()
    # opWrapper.start()

    # Get information
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    # Display Image
    print("Body keypoints: \n" + str(datum.handKeypoints))

    # cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
    # cv2.waitKey()


if __name__ == '__main__':
    estimate_pose('test1.jpg')