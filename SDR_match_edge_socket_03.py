#!/usr/bin/env python

'''
This sample demonstrates Canny edge detection.

Usage:
  *.py [<video source>]

  Trackbars control edge thresholds.

'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import numpy as np

import argparse
import imutils
import glob
import socket
import serial
import serial.threaded
import time


# relative module
import video

# built-in module
import sys


template = cv.imread("stop_sign_template_small_01.png")
#template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
template = cv.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
cv.imshow("Template", template)

#your tcp2serial server address and port (would be 127.0.0.1 if you are executing this py as the same RPi
host = "127.0.0.1"
port = "5555"
client_socket = socket.socket()


try:
    client_socket.connect((host, int(port)))
except socket.error as msg:
    sys.stderr.write('WARNING: {}\n'.format(msg))
    time.sleep(5)  # intentional delay on reconnection as client
    print("Socket Error... exiting")
    sys.exit(0)
    
    
sys.stderr.write('Connected\n')
client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    
if __name__ == '__main__':
    print(__doc__)

    try:
        fn = sys.argv[1]
    except:
        fn = 0

    def nothing(*arg):
        pass

    cv.namedWindow('edge')
    cv.createTrackbar('thrs1', 'edge', 2000, 5000, nothing)
    cv.createTrackbar('thrs2', 'edge', 4000, 5000, nothing)

    cap = video.create_capture(fn)
    while True:
        flag, img = cap.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        #to get a single pixel data at x = 100, y = 100
        #print (gray[100,100])
        thrs1 = cv.getTrackbarPos('thrs1', 'edge')
        thrs2 = cv.getTrackbarPos('thrs2', 'edge')
        edge = cv.Canny(gray, thrs1, thrs2, apertureSize=5)
        vis = img.copy()
        vis = np.uint8(vis/2.)
        vis[edge != 0] = (0, 255, 0)
#        cv.imshow('edge', vis)
        cv.imshow('edge', edge)
        found = None
	for scale in np.linspace(0.2, 1.0, 20)[::-1]:
		# resize the image according to the scale, and keep track
		# of the ratio of the resizing
		resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
		r = gray.shape[1] / float(resized.shape[1])

		# if the resized image is smaller than the template, then break
		# from the loop
		if resized.shape[0] < tH or resized.shape[1] < tW:
			break

		# detect edges in the resized, grayscale image and apply template
		# matching to find the template in the image
		edged = cv.Canny(resized, 50, 200)
		result = cv.matchTemplate(edged, template, cv.TM_CCOEFF)
		(_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, r)
                
	# unpack the bookkeeping varaible and compute the (x, y) coordinates
	# of the bounding box based on the resized ratio
	(_, maxLoc, r) = found
	(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

	# draw a bounding box around the detected result and display the image
	cv.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
	cv.imshow("Image", img)

	# you can send a control message based on the above processed data to your tcp2serial sever
        try:
            client_socket.send("found a sign")
        except socket.error as msg:
            print ("Sending error")
	print(found)
        
	found = None
        
        ch = cv.waitKey(5)
        if ch == 27:
            break
    cv.destroyAllWindows()
