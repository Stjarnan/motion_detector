# import the necessary packages
from toolkit.keyclipwriter import KeyClipWriter
from toolkit.utils import Config
from imutils.video import VideoStream
import numpy as np
import argparse
import datetime
import imutils
import signal
import time
import sys
import cv2
import os

def signal_handler(sig, frame):
	print("\nYou pressed `ctrl + c`!")
	print("Your files are saved in the `{}` directory".format(
		conf["output_path"]))

	# check if we're recording and wrap up
	if kcw.recording:
		kcw.finish()

	sys.exit(0)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True,
	help="path to the JSON configuration file")
ap.add_argument("-v", "--video", type=str,
	help="path to optional input video file")
args = vars(ap.parse_args())

# load our configuration
conf = Config(args["config"])

# check if we are using a camera and start the stream
if not args.get("video", False):
	print("Starting video stream...")
	vs = VideoStream(usePiCamera=conf["picamera"]).start()
	time.sleep(3.0)

# start a video file stream
else:
	print("Opening video file `{}`".format(args["video"]))
	vs = cv2.VideoCapture(args["video"])

# background subtractors
OPENCV_BG_SUBTRACTORS = {
	"CNT": cv2.bgsegm.createBackgroundSubtractorCNT,
	"GMG": cv2.bgsegm.createBackgroundSubtractorGMG,
	"MOG": cv2.bgsegm.createBackgroundSubtractorMOG,
	"GSOC": cv2.bgsegm.createBackgroundSubtractorGSOC,
	"LSBP": cv2.bgsegm.createBackgroundSubtractorLSBP
}

# create our background subtractor
fgbg = OPENCV_BG_SUBTRACTORS[conf["bg_sub"]]()

# create kernels for erosion and dilation
eKernel = np.ones(tuple(conf["erode"]["kernel"]), "uint8")
dKernel = np.ones(tuple(conf["dilate"]["kernel"]), "uint8")

# init key clip writer
# consecutive number of frames without motion 
# frames since the last snapshot was written
kcw = KeyClipWriter(bufSize=conf["keyclipwriter_buffersize"])
framesWithoutMotion = 0
framesSinceSnap = 0

# capture "ctrl + c" signals
signal.signal(signal.SIGINT, signal_handler)
images = " and images.." if conf["write_snaps"] else ".."
print("Detecting motion and storing videos{}".format(images))

# loop through the frames
while True:
	# grab a frame from the stream
	fullFrame = vs.read()

	# No frame was read
	if fullFrame is None:
		break

	# handle the frame
	fullFrame = fullFrame[1] if args.get("video", False) \
		else fullFrame

	framesSinceSnap += 1

	# resize the frame
	# apply the background subtractor
	# generate motion mask
	frame = imutils.resize(fullFrame, width=500)
	mask = fgbg.apply(frame)

	# eliminate noise
	mask = cv2.erode(mask, eKernel,
		iterations=conf["erode"]["iterations"])
	# fill gaps
	mask = cv2.dilate(mask, dKernel,
		iterations=conf["dilate"]["iterations"])

	# find contours in the mask and reset the motion status
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	motionThisFrame = False

	# loop over the contours
	for c in cnts:
		# compute the bounding circle and rectangle for the contour
		((x, y), radius) = cv2.minEnclosingCircle(c)
		(rx, ry, rw, rh) = cv2.boundingRect(c)

		# convert floating point values to integers
		(x, y, radius) = [int(v) for v in (x, y, radius)]

		# only process motion contours above the specified size or move to next
		if radius < conf["min_radius"]:
			continue

		# grab the current timestamp
		timestamp = datetime.datetime.now()
		timestring = timestamp.strftime("%Y%m%d-%H%M%S")

		# set our motion flag to indicate we have found motion and
		# reset the motion counter
		motionThisFrame = True
		framesWithoutMotion = 0

		# check if we need to annotate the frame for display
		if conf["annotate"]:
			cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)
			cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh),
				(0, 255, 0), 2)

		# check to see if should write the frame to disk
		writeFrame = framesSinceSnap >= conf["frames_between_snaps"]
		if conf["write_snaps"] and writeFrame:
			# construct the path to the output photo and save it
			snapPath = os.path.sep.join([conf["output_path"],
				timestring])
			cv2.imwrite(snapPath + ".jpg", fullFrame)

			# reset the counter between snapshots
			framesSinceSnap = 0

		# start recording
		if not kcw.recording:
			# path to the video file
			videoPath = os.path.sep.join([conf["output_path"],
				timestring])

			# instantiate the video codec object and start the key
			# clip writer
			fourcc = cv2.VideoWriter_fourcc(*conf["codec"])
			kcw.start("{}.avi".format(videoPath), fourcc, conf["fps"])

	# check if no motion was detected in this frame and then increment
	# the number of consecutive frames without motion
	if not motionThisFrame:
		framesWithoutMotion += 1

	# update the key frame clip buffer
	kcw.update(frame)

	# check to see if the number of frames without motion is above our
	# defined threshold
	noMotion = framesWithoutMotion >= conf["keyclipwriter_buffersize"]

	# stop recording if there is no motion
	if kcw.recording and noMotion:
		kcw.finish()

	# displaying the frame to screen and grab keypress
	if conf["display"]:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

# check if we're recording and stop recording
if kcw.recording:
	kcw.finish()

# stop the video stream
vs.stop() if not args.get("video", False) else vs.release()