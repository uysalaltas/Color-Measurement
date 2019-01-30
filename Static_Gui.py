import random
import sys
import cv2
import imutils
import numpy as np
import RALtoRGB
import time
import pyqtgraph as pg
import matplotlib.pyplot as plt
import pandas as pd

from colormath.color_objects import LabColor
from colormath.color_diff import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from imutils.video import WebcamVideoStream
from imutils.video import FPS
from sklearn.cluster import KMeans
from collections import Counter, OrderedDict
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import *

times = []
capture = WebcamVideoStream(src=0).start()


def color_calculate(frame):
    pixel = []
    wcss = []

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

    c1 = kmeans2.cluster_centers_[0]
    c2 = kmeans2.cluster_centers_[1]

    c1_a = int(c1[0])
    c1_b = int(c1[1])
    c2_a = int(c2[0])
    c2_b = int(c2[1])

    r_L_mean = int((L_mean*100)/255)
    r_c1_a = int(c1[0]-128)
    r_c1_b = int(c1[1]-128)
    r_c2_a = int(c2[0]-128)
    r_c2_b = int(c2[1]-128)

    lab1 = np.array([L_mean, c1_a, c1_b])
    lab_r1 = lab1.reshape(1, 1, 3)
    lab_r1 = lab_r1.astype('uint8')
    rgb1 = cv2.cvtColor(lab_r1, cv2.COLOR_Lab2RGB)
    rgb1 = rgb1.reshape(3)

    r_lab1 = np.array([r_L_mean, r_c1_a, r_c1_b])
    print(type(r_lab1[0]))
    print(rgb1, " ", r_lab1)

    lab2 = np.array([L_mean, c2_a, c2_b])
    lab_r2 = lab2.reshape(1, 1, 3)
    lab_r2 = lab_r2.astype('uint8')
    rgb2 = cv2.cvtColor(lab_r2, cv2.COLOR_Lab2RGB)
    rgb2 = rgb2.reshape(3)

    r_lab2 = np.array([r_L_mean, r_c2_a, r_c2_b])
    print(rgb2, " ", r_lab2)

    return data, kmeans2, and_img, r_lab1, rgb1, r_lab2, rgb2


class Window(QWidget):

    def __init__(self):
        super().__init__()

        self.timer = QTimer(self)
        self.frame = None

        vbox1 = QVBoxLayout()
        vbox2 = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox1 = QHBoxLayout()
        vbox = QVBoxLayout()

        self.font1 = QFont("Times", 14)
        self.font2 = QFont("Times", 11)

        # -------------------------------------------------------------------------------------------------------------
        # Toolbar
        # -------------------------------------------------------------------------------------------------------------

        # self.toolbar = QMenuBar()
        # exitMenu = self.toolbar.addMenu('File')
        # exitAction = QAction('Exit', self)
        # exitAction.triggered.connect(qApp.quit)
        # exitMenu.addAction(exitAction)

        # -------------------------------------------------------------------------------------------------------------
        # Measuring Elements
        # -------------------------------------------------------------------------------------------------------------

        # CameraTab Operations
        self.imgLabel1 = QLabel()
        self.imgLabel1.setFixedHeight(480)
        self.imgLabel1.setFixedWidth(640)

        tab_box1 = QGroupBox()
        tab_box1_layout = QGridLayout()
        tab_box1.setLayout(tab_box1_layout)
        tab_box1.setFixedSize(660, 520)

        tab_box1_layout.addWidget(self.imgLabel1, 1, 0)

        # SegmentationTab Operations
        self.imgLabel2 = QLabel()
        self.imgLabel2.setFixedHeight(480)
        self.imgLabel2.setFixedWidth(640)

        tab_box2 = QGroupBox()
        tab_box2_layout = QGridLayout()
        tab_box2.setLayout(tab_box2_layout)
        tab_box2.setFixedSize(660, 520)

        tab_box2_layout.addWidget(self.imgLabel2, 1, 0)

        # Box1 - Image Label
        self.tabWidget = QTabWidget()
        self.tabWidget.addTab(tab_box1, "Kamera")
        self.tabWidget.addTab(tab_box2, "Segmentasyon")

        # -------------------------------------------------------------------------------------------------------------
        # Button for capture image
        self.capPic = QPushButton("HESAPLA")
        self.capPic.setFixedSize(80, 40)
        self.capPic.clicked.connect(self.measure_color)

        # Box2 Operations
        img_frame_box2 = QGroupBox("Ölçüm")
        img_frame_box_layout2 = QGridLayout()
        img_frame_box2.setLayout(img_frame_box_layout2)
        img_frame_box2.setFixedSize(100, 170)

        img_frame_box_layout2.addWidget(self.capPic, 1, 0)
        img_frame_box_layout2.setAlignment(Qt.AlignTop)

        # -------------------------------------------------------------------------------------------------------------
        # Color1
        self.color1Title = QLabel('Renk 1')
        self.color1Title.setMinimumWidth(150)
        self.color1Title.setFont(self.font1)
        self.color1Title.setAlignment(Qt.AlignCenter)

        self.color1 = QLabel()
        self.color1.setStyleSheet("background-color: rgb(128, 128, 128)")
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
        self.color2.setStyleSheet("background-color: rgb(128, 128, 128)")
        self.color2.setFixedWidth(170)
        self.color2.setFixedHeight(30)

        self.rgbValue2 = QLabel("RGB DEĞER: [..., ..., ...]")
        self.rgbValue2.setFont(self.font2)
        self.rgbValue2.setAlignment(Qt.AlignLeft)

        self.labValue2 = QLabel("LAB DEĞER: [..., ..., ...]")
        self.labValue2.setFont(self.font2)
        self.labValue2.setAlignment(Qt.AlignLeft)

        # Box3 Operations
        img_frame_box3 = QGroupBox("Ölçülen Değerler")
        img_frame_box_layout3 = QGridLayout()
        img_frame_box3.setLayout(img_frame_box_layout3)
        img_frame_box3.setFixedSize(555, 170)

        img_frame_box_layout3.addWidget(self.color1Title, 1, 1)
        img_frame_box_layout3.addWidget(self.color1, 2, 1)
        img_frame_box_layout3.addWidget(self.rgbValue1, 3, 1)
        img_frame_box_layout3.addWidget(self.labValue1, 4, 1)

        img_frame_box_layout3.addWidget(self.color2Title, 1, 2)
        img_frame_box_layout3.addWidget(self.color2, 2, 2)
        img_frame_box_layout3.addWidget(self.rgbValue2, 3, 2)
        img_frame_box_layout3.addWidget(self.labValue2, 4, 2)

        # -------------------------------------------------------------------------------------------------------------
        # Plot
        # -------------------------------------------------------------------------------------------------------------

        self.figure = plt.figure()
        self.plot1 = FigureCanvas(self.figure)

        img_plot_box1 = QGroupBox("Grafikler")
        img_plot_box_layout1 = QGridLayout()
        img_plot_box1.setLayout(img_plot_box_layout1)
        img_plot_box1.setFixedHeight(500)

        img_plot_box_layout1.addWidget(self.plot1, 0, 0)

        # -------------------------------------------------------------------------------------------------------------
        # Delta Error
        # -------------------------------------------------------------------------------------------------------------

        self.errorCalibrate = QPushButton("KALİBRASYON")
        self.errorCalibrate.setFixedSize(80, 40)
        self.errorCalibrate.clicked.connect(self.error_calibrate)

        self.measureDeltaE = QPushButton("ÖLÇÜM")
        self.measureDeltaE.setFixedSize(80, 40)
        self.measureDeltaE.clicked.connect(self.measure_delta_e)

        self.deltaEColor1 = QLabel(self)
        self.deltaEColor1.setText("RENK 1 = ")

        self.deltaEResult1 = QLabel(self)
        self.deltaEResult1.setFixedSize(80, 40)

        self.deltaEColor2 = QLabel(self)
        self.deltaEColor2.setText("RENK 2 = ")

        self.deltaEResult2 = QLabel(self)
        self.deltaEResult2.setFixedSize(80, 40)

        img_error_box1 = QGroupBox("Delta E")
        img_error_box_layout1 = QGridLayout()
        img_error_box1.setLayout(img_error_box_layout1)

        img_error_box_layout1.addWidget(self.errorCalibrate, 0, 0)
        img_error_box_layout1.addWidget(self.measureDeltaE, 0, 1)
        img_error_box_layout1.addWidget(self.deltaEColor1, 1, 0)
        img_error_box_layout1.addWidget(self.deltaEResult1, 1, 1)
        img_error_box_layout1.addWidget(self.deltaEColor2, 2, 0)
        img_error_box_layout1.addWidget(self.deltaEResult2, 2, 1)

        # -------------------------------------------------------------------------------------------------------------
        # GUI Settings
        # -------------------------------------------------------------------------------------------------------------

        hbox1.addWidget(img_frame_box2)
        hbox1.addWidget(img_frame_box3)
        hbox1.setAlignment(Qt.AlignLeft)

        vbox1.addWidget(self.tabWidget)
        vbox1.setAlignment(Qt.AlignLeft)
        vbox1.addLayout(hbox1)

        vbox2.addWidget(img_plot_box1)
        vbox2.addWidget(img_error_box1)
        vbox2.setAlignment(Qt.AlignTop)

        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)
        vbox.addLayout(hbox)

        self.setLayout(vbox)
        self.setWindowTitle('FOTAM - NURSAN STATİK KABLO ÖLÇÜMÜ')
        self.setGeometry(0, 0, 1280, 720)
        # Function here
        self.start_webcam()
        self.show()

    def start_webcam(self):
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        # start_time = time.time()
        self.frame = capture.read()
        self.frame = imutils.resize(self.frame, width=640, height=480)

        self.display_image(self.frame)
        # times.append(time.time() - start_time)
        # if len(times) % 30 == 0:
        #     print(sum(times))
        #     times.clear()

    def display_image(self, img):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        self.imgLabel1.setPixmap(QPixmap.fromImage(outImage))
        self.imgLabel1.setScaledContents(True)

    def measure_color(self):
        # Definitions
        # -------------------------------------------------------------------------------------------------------------
        time.sleep(1)

        # Frame Operations and Calculations
        # -------------------------------------------------------------------------------------------------------------
        data, kmeans2, and_img, lab1, rgb1, lab2, rgb2 = color_calculate(self.frame)

        # Show Frame
        # -------------------------------------------------------------------------------------------------------------
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.scatter(data.a[data.label == 0], data.b[data.label == 0], color="red")
        ax.scatter(data.a[data.label == 1], data.b[data.label == 1], color="green")
        ax.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1], color="blue")
        ax.set_xlabel("A CHANNEL")
        ax.set_ylabel("B CHANNEL")
        self.plot1.draw()

        qformat = QImage.Format_Indexed8
        if len(and_img.shape) == 3:
            if and_img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(and_img, and_img.shape[1], and_img.shape[0], and_img.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        self.imgLabel2.setPixmap(QPixmap.fromImage(outImage))
        self.imgLabel2.setScaledContents(True)

        self.color1.setStyleSheet('color: red; background-color: rgb' + str((rgb1[0], rgb1[1], rgb1[2])))
        self.color2.setStyleSheet('color: red; background-color: rgb' + str((rgb2[0], rgb2[1], rgb2[2])))
        self.rgbValue1.setText("RGB DEĞER: " + str(rgb1))
        self.labValue1.setText("LAB DEĞER: " + str(lab1))
        self.rgbValue2.setText("RGB DEĞER: " + str(rgb2))
        self.labValue2.setText("LAB DEĞER: " + str(lab2))

        cv2.destroyAllWindows()

    def error_calibrate(self):
        data, kmeans, and_img, self.lab1c, rgb1, self.lab2c, rgb2 = color_calculate(self.frame)

    def measure_delta_e(self):
        data, kmeans, and_img, lab1e, rgb1, lab2e, rgb2 = color_calculate(self.frame)

        color1_c = LabColor(lab_l=self.lab1c[0], lab_a=self.lab1c[1], lab_b=self.lab1c[2])
        color1_e = LabColor(lab_l=lab1e[0], lab_a=lab1e[1], lab_b=lab1e[2])
        delta_e_c1 = delta_e_cie2000(color1_c, color1_e)
        self.deltaEResult1.setText(str(delta_e_c1))

        color2_c = LabColor(lab_l=self.lab2c[0], lab_a=self.lab2c[1], lab_b=self.lab2c[2])
        color2_e = LabColor(lab_l=lab2e[0], lab_a=lab2e[1], lab_b=lab2e[2])
        delta_e_c2 = delta_e_cie2000(color2_c, color2_e)
        self.deltaEResult2.setText(str(delta_e_c2))


app = QApplication(sys.argv)
window = Window()
sys.exit(app.exec_())
