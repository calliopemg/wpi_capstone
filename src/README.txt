Repository Link: https://github.com/calliopemg/wpi_capstone

--------------
CAPTURE MOTION
--------------

to start:
"python3 capture_motion.py"

to stop:
CTRL+C

Notes:
mp4 files are output in the folder where the script is run

Configuration:
adjust motion sensitivity with the "mse" value in the while loop (lower number = less sensitive to movement between pixels)
by default the script does a single autofocus cycle, but it can also continously autofocus with a recent camera
the default framerate is set by the camera, but it can be set directly


-----------------
VIDEO RECOGNITION
-----------------

to start:
"python3 video_recognition.py"

it assumes there are two subfolders in the current directly:
"raw" - contains the original clips
"output" - contains the labeled clips

Configuration:
By default it attempts to use animal recognition with our onnx model
It can also be configured to do facial recognition with the haarcascade xml model