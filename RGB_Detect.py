import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
COLOR_ROWS = 80
COLOR_COLS = 250
colorArray = np.zeros((COLOR_ROWS, COLOR_COLS, 3), dtype=np.uint8)
cv2.imshow('Color', colorArray)
pixels = []


def on_mouse_click(event, x, y, flags, userParams):
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
        cv2.imshow('Color', colorArray)


def nothing():
    pass


red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)

winName = 'Target Detect'
cv2.namedWindow(winName)
cv2.createTrackbar('Switch Threshold', winName, 0, 2, nothing)
cv2.createTrackbar('Threshold', winName, 100, 255, nothing)
cv2.createTrackbar('Contour', winName, 1, 10, nothing)

if not cap.isOpened():
    print('Unable to access camera')
else:
    print('Start grabbing, press a key on live window to terminate')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print('Unable to grab from the picamera')
            break

        trs = cv2.getTrackbarPos('Threshold', winName)
        swc = cv2.getTrackbarPos('Switch Threshold', winName)
        cnt = cv2.getTrackbarPos('Contour', winName)

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh_img = cv2.threshold(gray_img, trs, 255, cv2.THRESH_BINARY)
        rev_img = (255 - thresh_img)

        src = cv2.split(frame)

        # -----Color frame and operation-----

        and_img1 = cv2.bitwise_and(src[0], rev_img)
        and_img2 = cv2.bitwise_and(src[1], rev_img)
        and_img3 = cv2.bitwise_and(src[2], rev_img)

        and_img = cv2.merge((and_img1, and_img2, and_img3))

        cv2.setMouseCallback(winName, on_mouse_click)

        # -----Gray frame and operation-----

        # and_img = cv2.bitwise_and(frame, rev_img)

        # -----Drawing Contour-----

        im2, contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(and_img, contours, -1, red, cnt)

        # ----------

        for i in range(640):
            px = and_img[100, i]
            if px[0] > 0:
                pixels.append(px)
                print(px)

        if swc == 0:
            cv2.imshow(winName, and_img)
        elif swc == 1:
            cv2.imshow(winName, frame)
        elif swc == 2:
            cv2.imshow(winName, thresh_img)

        key = cv2.waitKey(1) & 0xFF
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        print("--- %s seconds ---" % (time.time() - start_time))

        if key == ord("q"):
            print(pixels)
            break
    print('Closing the camera')

cap.release()
cv2.destroyAllWindows()
quit()
