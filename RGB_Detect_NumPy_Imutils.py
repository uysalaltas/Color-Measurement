# import the necessary packages
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100, help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1, help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

COLOR_ROWS = 80
COLOR_COLS = 250
colorArray = np.zeros((COLOR_ROWS, COLOR_COLS, 3), dtype=np.uint8)
colorArray2 = np.zeros((COLOR_ROWS, COLOR_COLS, 3), dtype=np.uint8)
cv2.imshow('RGB Pixel', colorArray)
pixelsR = np.array([], int)
pixelsG = np.array([], int)
pixelsB = np.array([], int)
sumR = 0
sumG = 0
sumB = 0
times = []


def on_mouse_click(event, x, y, flags, userParams):
    # Take RGB value of pixel with left click
    if event == cv2.EVENT_LBUTTONDOWN:
        colorArray[:] = and_img[y, x, :]
        rgb = and_img[y, x, [2, 1, 0]]

        luminance = 1 - (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
        if luminance < 0.5:
            textColor = [0, 0, 0]
        else:
            textColor = [255, 255, 255]

        cv2.putText(colorArray, str(rgb), (20, COLOR_ROWS - 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=textColor)
        cv2.imshow('RGB Pixel', colorArray)


def nothing():
    pass


red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)

# Create track bar and window name
winName = 'Target Detect'
cv2.namedWindow(winName)
cv2.createTrackbar('Switch Threshold', winName, 0, 2, nothing)
# cv2.createTrackbar('Threshold', winName, 100, 255, nothing)
cv2.createTrackbar('Contour', winName, 1, 10, nothing)

print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()

# loop over some frames...this time using the threaded stream
while True:
    # grab the frame from the threaded video stream and resize it
    start_time = time.time()
    frame = vs.read()
    frame = imutils.resize(frame, width=640, height=480)

    # trs = cv2.getTrackbarPos('Threshold', winName)                          # Get trackbar values
    swc = cv2.getTrackbarPos('Switch Threshold', winName)
    cnt = cv2.getTrackbarPos('Contour', winName)

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # BGR to grayscale
    ret, thresh_img = cv2.threshold(gray_img, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Threshold gray frame with OTSU
    rev_img = (255 - thresh_img)  # Reverse threshold img

    src = cv2.split(frame)  # Split RGB frame to R, G, B

    # -----Color frame and operation-----

    and_img1 = cv2.bitwise_and(src[0], rev_img)  # Take every color channel and multiply
    and_img2 = cv2.bitwise_and(src[1], rev_img)  # with reverse threshold image
    and_img3 = cv2.bitwise_and(src[2], rev_img)

    and_img = cv2.merge((and_img1, and_img2, and_img3))  # Merge all channel again

    cv2.setMouseCallback(winName, on_mouse_click)  # Call function with mouse callback

    # -----Gray frame and operation-----

    # and_img = cv2.bitwise_and(frame, rev_img)

    # -----Drawing Contour-----
    # if cnt != 0:
    #    im2, contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #    cv2.drawContours(and_img, contours, -1, red, cnt)

    # -----RGB pixel operations-----

    px = and_img[100, :]
    pixelsR = px[:, [0]]
    pixelsG = px[:, [1]]
    pixelsB = px[:, [2]]

    pixelsR = pixelsR.flatten()
    pixelsG = pixelsG.flatten()
    pixelsB = pixelsB.flatten()

    maskR = pixelsR > 0
    maskG = pixelsG > 0
    maskB = pixelsB > 0

    pixelsR = pixelsR[maskR]
    pixelsG = pixelsG[maskG]
    pixelsB = pixelsB[maskB]

    if len(pixelsR) != 0:
        sumR = int(pixelsR.sum() / len(pixelsR))
        sumG = int(pixelsG.sum() / len(pixelsG))
        sumB = int(pixelsB.sum() / len(pixelsB))

    sumAll = [sumR, sumG, sumB]
    sumRGB = [sumB, sumG, sumR]

    pixelsR = np.array([], int)
    pixelsG = np.array([], int)
    pixelsB = np.array([], int)

    colorArray2[:] = sumAll
    cv2.putText(colorArray2, str(sumRGB), (20, COLOR_ROWS - 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=red)
    cv2.imshow('RGB Average', colorArray2)

    # -----Show frames-----

    if swc == 0:
        cv2.imshow(winName, and_img)
    elif swc == 1:
        cv2.imshow(winName, frame)
    elif swc == 2:
        cv2.imshow(winName, thresh_img)

    key = cv2.waitKey(1) & 0xFF

    # update the FPS counter
    fps.update()

    if key == ord("q"):  # If user press q, break loop
        break

    times.append(time.time() - start_time)
    if len(times) % 30 == 0:
        print(sum(times))
        times.clear()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
