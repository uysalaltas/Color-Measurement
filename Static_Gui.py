import sys
import cv2
import imutils
import numpy as np
import RALtoRGB
import time
import pyqtgraph as pg
import matplotlib.pyplot as plt
import pandas as pd

from imutils.video import WebcamVideoStream
from imutils.video import FPS
from sklearn.cluster import KMeans
from collections import Counter, OrderedDict
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import *


class Window(QWidget):

    def __init__(self):
        super().__init__()

        vbox1 = QVBoxLayout()
        vbox2 = QVBoxLayout()
        vbox3 = QVBoxLayout()
        vbox4 = QVBoxLayout()
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        vbox = QVBoxLayout()

        self.font1 = QFont("Times", 14)
        self.font2 = QFont("Times", 11)
        # -------------------------------------------------------------------------------------------------------------
        # Measuring Elements
        # -------------------------------------------------------------------------------------------------------------

        # Image Label
        self.imgLabel = QLabel()
        self.imgLabel.setFixedHeight(480)
        self.imgLabel.setFixedWidth(640)

        # Box1 Operations
        img_frame_box1 = QGroupBox("Kablo Görüntüsü")
        img_frame_box_layout = QGridLayout()
        img_frame_box1.setLayout(img_frame_box_layout)
        img_frame_box1.setFixedSize(660, 520)

        img_frame_box_layout.addWidget(self.imgLabel, 1, 0)

        # -------------------------------------------------------------------------------------------------------------
        # Button for capture image
        self.capPic = QPushButton("Ölçüm")
        self.capPic.setFixedSize(80, 40)
        self.capPic.clicked.connect(self.measure_color)

        # Box2 Operations
        img_frame_box2 = QGroupBox()
        img_frame_box_layout2 = QGridLayout()
        img_frame_box2.setLayout(img_frame_box_layout2)
        img_frame_box2.setFixedSize(100, 170)

        img_frame_box_layout2.addWidget(self.capPic, 1, 0)

        # -------------------------------------------------------------------------------------------------------------
        # Color1
        self.color1Title = QLabel('Renk 1')
        self.color1Title.setMinimumWidth(150)
        self.color1Title.setFont(self.font1)
        self.color1Title.setAlignment(Qt.AlignCenter)

        self.color1 = QLabel()
        self.color1.setFixedWidth(170)
        self.color1.setFixedHeight(30)

        self.rgbValue1 = QLabel("RGB DEĞER: [..., ..., ...]")
        self.rgbValue1.setFont(self.font2)
        self.rgbValue1.setAlignment(Qt.AlignLeft)

        self.labValue1 = QLabel("LAB DEĞER: [..., ..., ...]")
        self.labValue1.setFont(self.font2)
        self.labValue1.setAlignment(Qt.AlignLeft)

        # Color2
        self.color2Title = QLabel('Renk 2')
        self.color2Title.setMinimumWidth(150)
        self.color2Title.setFont(self.font1)
        self.color2Title.setAlignment(Qt.AlignCenter)

        self.color2 = QLabel()
        self.color2.setFixedWidth(170)
        self.color2.setFixedHeight(30)

        self.rgbValue2 = QLabel("RGB DEĞER: [..., ..., ...]")
        self.rgbValue2.setFont(self.font2)
        self.rgbValue2.setAlignment(Qt.AlignLeft)

        self.labValue2 = QLabel("LAB DEĞER: [..., ..., ...]")
        self.labValue2.setFont(self.font2)
        self.labValue2.setAlignment(Qt.AlignLeft)

        # Box3 Operations
        img_frame_box3 = QGroupBox()
        img_frame_box_layout3 = QGridLayout()
        img_frame_box3.setLayout(img_frame_box_layout3)
        img_frame_box3.setFixedSize(520, 170)

        img_frame_box_layout3.addWidget(self.color1Title, 1, 1)
        img_frame_box_layout3.addWidget(self.color1, 2, 1)
        img_frame_box_layout3.addWidget(self.rgbValue1, 3, 1)
        img_frame_box_layout3.addWidget(self.labValue1, 4, 1)

        img_frame_box_layout3.addWidget(self.color2Title, 1, 2)
        img_frame_box_layout3.addWidget(self.color2, 2, 2)
        img_frame_box_layout3.addWidget(self.rgbValue2, 3, 2)
        img_frame_box_layout3.addWidget(self.labValue2, 4, 2)

        # -------------------------------------------------------------------------------------------------------------
        # GUI Settings
        # -------------------------------------------------------------------------------------------------------------

        hbox2.addWidget(img_frame_box2)
        hbox2.addWidget(img_frame_box3)
        hbox2.setAlignment(Qt.AlignLeft)

        vbox1.addWidget(img_frame_box1)
        vbox1.setAlignment(Qt.AlignLeft)
        vbox1.addLayout(hbox2)
        vbox1.addStretch(1)

        hbox1.addLayout(vbox1)
        vbox.addLayout(hbox1)

        self.setLayout(vbox)
        self.setWindowTitle('FOTAM - NURSAN STATİK KABLO ÖLÇÜMÜ')
        self.setGeometry(0, 0, 1280, 720)
        # Function here
        self.show()

    def measure_color(self):

        # Definitions
        # -------------------------------------------------------------------------------------------------------------
        vs = WebcamVideoStream(src=0).start()
        time.sleep(1)
        pixel = []
        wcss = []

        # Frame Operations and Calculations
        # -------------------------------------------------------------------------------------------------------------
        frame = vs.read()
        frame = imutils.resize(frame, width=640, height=480)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        src = cv2.split(frame)

        and_img1 = cv2.bitwise_and(src[0], thresh)
        and_img2 = cv2.bitwise_and(src[1], thresh)
        and_img3 = cv2.bitwise_and(src[2], thresh)
        and_img = cv2.merge((and_img1, and_img2, and_img3))
        lab_img = cv2.cvtColor(and_img, cv2.COLOR_BGR2Lab)

        p = lab_img[:, [10, 20, 620, 630]]
        px = p.reshape(len(p) * len(p[0]), 3)

        for x in px:
            if x[0] != 0 and x[1] != 0 and x[2] != 0:
                pixel.append(x)

        L_mean = int(np.mean(pixel, axis=0)[0])
        print(L_mean)
        pixel2 = np.delete(pixel, 0, axis=1)

        dictionary = {"a": pixel2[:, 0], "b": pixel2[:, 1]}
        data = pd.DataFrame(dictionary)

        for k in range(1, 5):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)

        kmeans2 = KMeans(n_clusters=2)
        clusters = kmeans2.fit_predict(data)

        data["label"] = clusters

        # Show Frame
        # -------------------------------------------------------------------------------------------------------------
        plt.figure()
        plt.scatter(data.a[data.label == 0], data.b[data.label == 0], color="red")
        plt.scatter(data.a[data.label == 1], data.b[data.label == 1], color="green")
        plt.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1], color="blue")
        plt.xlabel("A Channel")
        plt.ylabel("B Channel")
        plt.show()

        print(kmeans2.cluster_centers_)
        c1 = kmeans2.cluster_centers_[0]
        c2 = kmeans2.cluster_centers_[1]

        c1_a = int(c1[0])
        c1_b = int(c1[1])
        c2_a = int(c2[0])
        c2_b = int(c2[1])

        print("1- A ", c1_a, "B: ", c1_b)
        print("2- A ", c2_a, "B: ", c2_b)

        qformat = QImage.Format_Indexed8
        if len(and_img.shape) == 3:
            if and_img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(and_img, and_img.shape[1], and_img.shape[0], and_img.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        self.imgLabel.setPixmap(QPixmap.fromImage(outImage))
        self.imgLabel.setScaledContents(True)

        lab1 = np.array([L_mean, c1_a, c1_b])
        lab_r1 = lab1.reshape(1, 1, 3)
        lab_r1 = lab_r1.astype('uint8')
        rgb1 = cv2.cvtColor(lab_r1, cv2.COLOR_Lab2RGB)
        rgb1 = rgb1.reshape(3)
        print(rgb1, " ", lab1)

        lab2 = np.array([L_mean, c2_a, c2_b])
        lab_r2 = lab2.reshape(1, 1, 3)
        lab_r2 = lab_r2.astype('uint8')
        rgb2 = cv2.cvtColor(lab_r2, cv2.COLOR_Lab2RGB)
        rgb2 = rgb2.reshape(3)
        print(rgb2, " ", lab2)

        self.color1.setStyleSheet('color: red; background-color: rgb' + str((rgb1[0], rgb1[1], rgb1[2])))
        self.color2.setStyleSheet('color: red; background-color: rgb' + str((rgb2[0], rgb2[1], rgb2[2])))
        self.rgbValue1.setText("RGB DEĞER: " + str(rgb1))
        self.labValue1.setText("LAB DEĞER: " + str(lab1))
        self.rgbValue2.setText("RGB DEĞER: " + str(rgb2))
        self.labValue2.setText("LAB DEĞER: " + str(lab2))


        cv2.destroyAllWindows()
        vs.stop()


app = QApplication(sys.argv)
window = Window()
sys.exit(app.exec_())
