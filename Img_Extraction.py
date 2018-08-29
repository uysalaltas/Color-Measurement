import numpy as np
import cv2
import time

def nothing(jnk):
    pass

cap = cv2.VideoCapture(0)
img1 = cv2.imread('cam1.jpg')

winName = 'Target Detect'
cv2.namedWindow(winName)
cv2.createTrackbar('Switch Camera', winName, 0, 3, nothing)

if cap.isOpened() == False:
    print('Unable to access camera')
else:
    print('Start grabbing, press a key on live window to terminate')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while (cap.isOpened()):
        start_time = time.time()
        ret, frame = cap.read()
        if ret == False:
            print('Unable to grab from the picamera')
            break

        swc = cv2.getTrackbarPos('Switch Camera', winName)

        im_diff = cv2.absdiff(img1, frame)
        mask = cv2.cvtColor(im_diff, cv2.COLOR_BGR2GRAY)

        if swc == 0:
            cv2.imshow(winName, frame)
        elif swc == 1:
            cv2.imshow(winName, img1)
        elif swc == 2:
            cv2.imshow(winName, im_diff)
        elif swc == 3:
            cv2.imshow(winName, im_diff)

        key = cv2.waitKey(1) & 0xFF

        print("--- %s seconds ---" % (time.time() - start_time))

        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
quit()