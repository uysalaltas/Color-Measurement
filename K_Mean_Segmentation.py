from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2
import time


print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=0).start()
time.sleep(2)
fps = FPS().start()
times = []
pixel = []


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


start_time = time.time()
frame = vs.read()
frame = imutils.resize(frame, width=640, height=480)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

plt.figure()
plt.axis("off")
plt.imshow(frame)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

src = cv2.split(frame)

and_img1 = cv2.bitwise_and(src[0], thresh)
and_img2 = cv2.bitwise_and(src[1], thresh)
and_img3 = cv2.bitwise_and(src[2], thresh)
and_img = cv2.merge((and_img1, and_img2, and_img3))
px = and_img[100, :]

for x in px:
    if x[0] != 0 and x[1] != 0 and x[2] != 0:
        pixel.append(x)

# -------------------------------------------------------------------------------------
# image = and_img.reshape((and_img.shape[0] * and_img.shape[1], 3))

clt = KMeans(n_clusters=2)
clt.fit(pixel)
hist = centroid_histogram(clt)
bar = plot_colors(hist, clt.cluster_centers_)
# -------------------------------------------------------------------------------------

times.append(time.time() - start_time)
print(sum(times))
times.clear()

plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
