from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imutils
import cv2
import time


print("[INFO] sampling THREADED frames from webcam...")

# -------------------------------------------------------------------------------------------------------------
# Definitions
# -------------------------------------------------------------------------------------------------------------
vs = WebcamVideoStream(src=0).start()
time.sleep(1)
fps = FPS().start()
times = []
pixel = []

# -------------------------------------------------------------------------------------------------------------
# Frame Operations and Calculations
# -------------------------------------------------------------------------------------------------------------
start_time = time.time()
frame = vs.read()
frame = imutils.resize(frame, width=640, height=480)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

plt.figure()
plt.axis("off")
plt.imshow(frame)

gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
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

plt.figure()
plt.axis("off")
plt.imshow(lab_img)

p = lab_img[:, [50, 100, 150, 200]]
px = p.reshape(len(p) * len(p[0]), 3)

for x in px:
    if x[0] != 0 and x[1] != 0 and x[2] != 0:
        pixel.append(x)

pixel2 = np.delete(pixel, 0, axis=1)

dictionary = {"a": pixel2[:, 0], "b": pixel2[:, 1]}
data = pd.DataFrame(dictionary)

plt.figure()
plt.scatter(pixel2[:, 0], pixel2[:, 1], color="red", label="A and B")
plt.xlabel("A Channel")
plt.ylabel("B Channel")

# -------------------------------------------------------------------------------------------------------------
# K-Means Clustering
# -------------------------------------------------------------------------------------------------------------
wcss = []

for k in range(1, 5):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1,5),wcss)
plt.xlabel("number of k (cluster) value")
plt.ylabel("wcss")

kmeans2 = KMeans(n_clusters=2)
clusters = kmeans2.fit_predict(data)

data["label"] = clusters

plt.figure()
plt.scatter(data.a[data.label == 0], data.b[data.label == 0], color="red")
plt.scatter(data.a[data.label == 1], data.b[data.label == 1], color="green")
plt.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1], color="blue")
plt.xlabel("A Channel")
plt.ylabel("B Channel")

# -------------------------------------------------------------------------------------

plt.show()
times.append(time.time() - start_time)
print(sum(times))
times.clear()

cv2.destroyAllWindows()
vs.stop()
