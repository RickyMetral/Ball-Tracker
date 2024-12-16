from collections import deque
import numpy as np
import argparse
import cv2
import time

def on_trackbar(val):
	pass

def resize(frame, width):
	"""resize the frame to desired size"""
	(height,width) = frame.shape[:2]
	ratio = 600/float(width)
	dim = (width, int((height * ratio)))
	return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def getContour(mask):
	"""Grabs the contour of a mask depending of version of OpenCV"""
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	if len(cnts) == 2:
		return cnts[0]
	elif len(cnts) == 3:
		return cnts[1]
	else:
		raise TypeError("Could not find contour of object")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-b", "--buffer", type=int, default=70,
		help="size of the line tracing object")

	ap.add_argument("-s", "--slider", action="store_true",
			help="Optional flag to include color slider" )

	args = vars(ap.parse_args())
	#Upper and lower boundries of green in HSV
	lowerBound = np.array([40, 80, 60])
	upperBound = np.array([70, 220, 220])
	pts = deque(maxlen=args["buffer"])
	vs = cv2.VideoCapture(0)
	# Check if camera was opened
	if not vs.isOpened():
		raise IOError("Cannot open Webcam")
	# allow the camera or video file to warm up
	time.sleep(2.0)

	#If slider flag was provided create a new tab that allows the Hue bounds to be changed
	if args["slider"]:
		cv2.namedWindow("Color Bounds")
		cv2.createTrackbar("lowerH", "Color Bounds", lowerBound[0], 180, on_trackbar)
		cv2.createTrackbar("upperH", "Color Bounds", upperBound[0], 180, on_trackbar)

	while True:
		#Get the values of the trackbar
		if args["slider"]:
			lowerBound[0] = cv2.getTrackbarPos("lowerH", "Color Bounds")
			upperBound[0] = cv2.getTrackbarPos("upperH", "Color Bounds")
		# grab the current frame
		ret, frame = vs.read()
		if not ret:
			break
		frame = resize(frame, 600)
		# blur it, and convert it to HSV
		blurred = cv2.GaussianBlur(frame, (11, 11), 0)
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
		#make bitmask of green and use erosion + dilation to remove blobs
		mask = cv2.inRange(hsv, lowerBound, upperBound)
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)
		# find contours in the mask 
		cnts = getContour(mask)
		center = None
		#make sure we have contours
		if len(cnts) > 0:
			# find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
			maxCtr = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(maxCtr)
			moments = cv2.moments(maxCtr)
			center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
			# only proceed if the radius meets a minimum size
			if radius > 10:
				# draw the circle and centroid on the frame, then update the list of tracked points
				cv2.circle(frame, (int(x), int(y)), int(radius),
					(0, 255, 255), 2)
				cv2.circle(frame, center, 5, (0, 0, 255), -1)
		# update the points queue
		pts.appendleft(center)
			# loop over the set of tracked points
		for i in range(1, len(pts)):
			# if either of the tracked points are None, ignore them
			if pts[i - 1] is None or pts[i] is None:
				continue
			# otherwise, compute the thickness of the line draw the connecting lines
			thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
			cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
		#Draw the frame to the screen
		cv2.imshow("Object Tracker", frame)
		key = cv2.waitKey(1)
		# if the 'q' key is pressed, stop the loop
		if key == ord("q"):
			break
	#release the camera
	vs.release()
	cv2.destroyAllWindows()

if __name__=="__main__":
	main()
	exit(0)