# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file") # --video è il path dove si trova il video, se non presente opencv proverà ad accedere alla webcam 
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size") #opzionale, serve per specificare la max size della struttura dati che raccoglie info sulle posizioni che cambia la palla
args = vars(ap.parse_args())
# define the lower and upper boundaries of the "green" ball in the HSV color space, then initialize the list of tracked points
greenLower = (29, 86, 6) # da modificare
greenUpper = (64, 255, 255) # da modificare
pts = deque(maxlen=args["buffer"]) # defaults to 64

# Inizializza il filtro di kalman
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.03
kalman.measurementNoiseCov = np.array([[1, 0],
                                       [0, 1]], np.float32) * 0.00003

# if a video path was not supplied, grab the reference to the webcam --> posso cancellare tutto e lasciare solo la riga 23
if not args.get("video", False):
	vs = VideoStream(src=0).start() # come VideoCapture(0) ma ottimizzato
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
# allow the camera or video file to warm up
time.sleep(2.0)
# keep looping
while True:
	# grab the current frame
	frame = vs.read()
	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break
	# Riduci la dimensione del frame se la palla è lontana dal centro dell'immagine 
	if len(pts) > 0:
		center = pts[0]
		img_center_x, img_center_y = frame.shape[1] // 2, frame.shape[0] // 2
		if center is not None:
			distance_to_center = np.sqrt((center[0] - img_center_x) ** 2 + (center[1] - img_center_y) ** 2)
			
			# Soglia per il ridimensionamento
			resize_threshold = 100  # Modifica questo valore a seconda delle tue esigenze
			if distance_to_center > resize_threshold:
				frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		# Calcola l'approssimazione del contorno
		epsilon = 0.04 * cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, epsilon, True)
		# Conta il numero dei vertici approssimati
		num_vertices = len(approx)
		# Imposta una soglia per la forma del contorno
		shape_trashold = 8  # Modifica questo valore a seconda della forma della palla
		if num_vertices <= shape_trashold:
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
			# only proceed if the radius meets a minimum size
			if radius > 10:
				# Aggiorna il filtro di kalman
				kalman.correct(np.array([center[0], center[1]], np.float32))
				# Ottieni la previsione della posizione dalla prossima iterazione del filtro
				prediction = kalman.predict()
				# draw the circle and centroid on the frame,
				# then update the list of tracked points
				cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
				cv2.circle(frame, center, 2, (0, 0, 255), -1)
				cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 5, (255, 255, 0), -1)
	# update the points queue
	pts.appendleft(center)
	# loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue
		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		# thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5) <- vecchio spessore linea
		thickness = 1
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()
# otherwise, release the camera
else:
	vs.release()
# close all windows
cv2.destroyAllWindows()