from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2
import time


print("[INFO] sampling THREADED frames from webcam...")


# -------------------------------------------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------------------------------------------
def calc_avg(tuple):
    for x in tuple:
        L = x[0]
        A = x[1]
        B = x[2]

    L = L.flatten()
    A = A.flatten()
    B = B.flatten()

    sumL = int(L.sum() / len(L))
    sumA = int(A.sum() / len(A))
    sumB = int(B.sum() / len(B))

    sumAll = [sumL, sumA, sumB]

    return sumAll


# -------------------------------------------------------------------------------------------------------------
# Definitions
# -------------------------------------------------------------------------------------------------------------
vs = WebcamVideoStream(src=0).start()
time.sleep(2)
fps = FPS().start()
times = []
pixel = []
pixel2 = []
COLOR_ROWS = 80
COLOR_COLS = 250
red = (0, 0, 255)
colorArray = np.zeros((COLOR_ROWS, COLOR_COLS, 3), dtype=np.uint8)
colorArray2 = np.zeros((COLOR_ROWS, COLOR_COLS, 3), dtype=np.uint8)

# -------------------------------------------------------------------------------------------------------------
# Frame Operations
# -------------------------------------------------------------------------------------------------------------
start_time = time.time()
frame = vs.read()
frame = imutils.resize(frame, width=640, height=480)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

plt.figure()
plt.axis("off")
plt.imshow(frame)

# -------------------------------------------------------------------------------------------------------------
# Color Conversion
# -------------------------------------------------------------------------------------------------------------
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

src = cv2.split(frame)

and_img1 = cv2.bitwise_and(src[0], thresh)
and_img2 = cv2.bitwise_and(src[1], thresh)
and_img3 = cv2.bitwise_and(src[2], thresh)
and_img = cv2.merge((and_img1, and_img2, and_img3))
lab_img = cv2.cvtColor(and_img, cv2.COLOR_RGB2Lab)

plt.figure()
plt.axis("off")
plt.imshow(and_img)

# -------------------------------------------------------------------------------------------------------------
# Calculations
# -------------------------------------------------------------------------------------------------------------
px = lab_img[100, :]
pix_f = px[1]

for x in px:
    if x[0] != 0 and x[1] != 0 and x[2] != 0:
        p1 = int(pix_f[1])
        p2 = int(pix_f[2])
        x1 = int(x[1])
        x2 = int(x[2])

        if (p1 - x1) >= 6 or (p1 - x1) <= -6 or (p2 - x2) >= 6 or (p2 - x2) <= -6:
            pixel2.append(x)
        else:
            pixel.append(x)

color1 = calc_avg(pixel)
color2 = calc_avg(pixel2)

# -------------------------------------------------------------------------------------------------------------
# Show On Screen
# -------------------------------------------------------------------------------------------------------------
colorArray[:] = color1
colorArray2[:] = color2

colorArray_rgb1 = cv2.cvtColor(colorArray, cv2.COLOR_Lab2RGB)
colorArray_rgb2 = cv2.cvtColor(colorArray2, cv2.COLOR_Lab2RGB)

result1 = colorArray_rgb1[1][1]
result2 = colorArray_rgb2[1][1]

cv2.putText(colorArray_rgb1, str(result1), (20, COLOR_ROWS - 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=red)
cv2.putText(colorArray_rgb2, str(result2), (20, COLOR_ROWS - 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=red)

plt.figure()
plt.axis("off")
plt.imshow(colorArray_rgb1)

plt.figure()
plt.axis("off")
plt.imshow(colorArray_rgb2)

# -------------------------------------------------------------------------------------------------------------
# Cleaning and Finishing Process
# -------------------------------------------------------------------------------------------------------------
times.append(time.time() - start_time)
print(sum(times))
times.clear()
plt.show()
cv2.destroyAllWindows()
vs.stop()
