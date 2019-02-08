import sys
import cv2
import imutils
import numpy as np
import RALtoRGB
import time
import pyqtgraph as pg
# import RPi.GPIO as GPIO

from collections import Counter, OrderedDict
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import *
from imutils.video import WebcamVideoStream

COLOR_ROWS = 80
COLOR_COLS = 250
times = []
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)

# GPIO.setmode(GPIO.BCM)
# GPIO.setup(24, GPIO.OUT)
# pwm24 = GPIO.PWM(24, 400)
# pwm24.start(0)


class Window(QWidget):

    def __init__(self):
        super().__init__()

        self.timer = QTimer(self)
        self.capture = WebcamVideoStream(src=1).start()

        self.image = None
        self.outImage = None
        self.countImg = 0
        self.frameNumber = 0
        self.font1 = QFont("Times", 14)
        self.font2 = QFont("Times", 11)

        vbox1 = QVBoxLayout()
        vbox2 = QVBoxLayout()
        vbox3 = QVBoxLayout()
        vbox4 = QVBoxLayout()
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        vbox = QVBoxLayout()

        # -------------------------------------------------------------------------------------------------------------
        # Producing Color Elements
        # -------------------------------------------------------------------------------------------------------------

        # Title
        self.rgbTitle = QLabel('Measured Color')
        self.rgbTitle.setFont(self.font1)
        self.rgbTitle.setMinimumWidth(150)
        self.rgbTitle.setAlignment(Qt.AlignCenter)

        # RGB Label
        self.rgbLabel = QLabel()
        self.rgbLabel.setFixedWidth(150)
        self.rgbLabel.setFixedHeight(150)
        self.rgbLabel.setAlignment(Qt.AlignCenter)

        # RBG Value
        self.rgbValue = QLabel()
        self.rgbValue.setFont(self.font2)
        self.rgbValue.setAlignment(Qt.AlignCenter)

        # RGB Value Label
        self.rgbValueLabel = QLabel('RGB Value: ')
        self.rgbValueLabel.setFont(self.font2)
        self.rgbValueLabel.setAlignment(Qt.AlignLeft)

        # LAB Value
        self.labValue = QLabel()
        self.labValue.setFont(self.font2)
        self.labValue.setAlignment(Qt.AlignCenter)

        # LAB Value Label
        self.labValueLabel = QLabel('LAB Value: ')
        self.labValueLabel.setFont(self.font2)
        self.labValueLabel.setAlignment(Qt.AlignLeft)

        # Box1 Operations
        rgb_frame_box = QGroupBox("Producing Color")
        rgb_frame_box_layout = QGridLayout()
        rgb_frame_box.setLayout(rgb_frame_box_layout)
        rgb_frame_box.setMaximumWidth(220)

        # Box2 Operations
        color_value_box = QGroupBox("Color Codes")
        color_value_box_layout = QGridLayout()
        color_value_box.setLayout(color_value_box_layout)
        color_value_box.setFixedWidth(220)

        # Adding Widgets
        rgb_frame_box_layout.addWidget(self.rgbTitle, 1, 0)
        rgb_frame_box_layout.addWidget(self.rgbLabel, 2, 0)
        color_value_box_layout.addWidget(self.rgbValueLabel, 1, 0)
        color_value_box_layout.addWidget(self.rgbValue, 1, 1)
        color_value_box_layout.addWidget(self.labValueLabel, 2, 0)
        color_value_box_layout.addWidget(self.labValue, 2, 1)

        # -------------------------------------------------------------------------------------------------------------
        # Desired Color
        # -------------------------------------------------------------------------------------------------------------

        # Title
        self.RALTitle = QLabel('RAL Codes')
        self.RALTitle.setFont(self.font1)
        self.RALTitle.setMinimumWidth(150)
        self.RALTitle.setAlignment(Qt.AlignLeft)

        # RAL Label
        self.RALLabel = QLabel()
        self.RALLabel.setFixedWidth(150)
        self.RALLabel.setFixedHeight(150)
        self.RALLabel.setAlignment(Qt.AlignCenter)

        # Combo Box
        self.RALCombo = QComboBox()
        self.RALCombo.addItems(RALtoRGB.RAL)
        self.RALCombo.currentIndexChanged.connect(self.apply)
        self.RALCombo.setFixedWidth(100)

        # RAL to RGB Label
        self.RALtoRGBValue = QLabel('RAL to RGB: ')
        self.RALtoRGBValue.setFont(self.font2)
        self.RALtoRGBValue.setAlignment(Qt.AlignLeft)

        # RAL to RGB Value
        self.RALtoRGB = QLabel("[...,...,...]")
        self.RALtoRGB.setFont(self.font2)
        self.RALtoRGB.setAlignment(Qt.AlignRight)
        self.RALtoRGB.setFixedWidth(100)

        # Box Operations
        desired_color_box = QGroupBox("Desired Color")
        desired_color_box_layout = QGridLayout()
        desired_color_box.setLayout(desired_color_box_layout)
        desired_color_box.setFixedWidth(220)

        # Adding Widgets
        desired_color_box_layout.addWidget(self.RALTitle, 1, 0)
        desired_color_box_layout.addWidget(self.RALLabel, 2, 0)
        desired_color_box_layout.addWidget(self.RALCombo, 3, 0)
        desired_color_box_layout.addWidget(self.RALtoRGBValue, 4, 0)
        desired_color_box_layout.addWidget(self.RALtoRGB, 4, 1)

        # -------------------------------------------------------------------------------------------------------------
        # Camera Elements
        # -------------------------------------------------------------------------------------------------------------

        # Image Label
        self.imgLabel = QLabel()
        self.imgLabel.setFixedHeight(480)
        self.imgLabel.setFixedWidth(640)

        # Button for capture image
        self.capPic = QPushButton("Capture")
        self.capPic.setFixedSize(80, 40)
        self.capPic.clicked.connect(self.capture_img)

        # Dialog Button for select folder
        self.dialogBtn = QToolButton()
        self.dialogBtn.setText("...")
        self.dialogBtn.clicked.connect(self.open_file_dialog)

        # Line for put file path in to it
        self.dialogLine = QLineEdit()
        self.dialogLine.setEnabled(False)

        # CheckBox
        self.autoCap = QCheckBox("Capture Automatically")

        # PWM text
        self.pwmTxt = QLineEdit()
        self.pwmTxt.setFixedSize(80, 40)

        # PWM Button
        self.pwmBtn = QPushButton("Set PWM")
        self.pwmBtn.setFixedSize(80, 40)
        self.pwmBtn.clicked.connect(self.set_pwm)

        # Box1 Operations
        img_frame_box1 = QGroupBox("Camera")
        img_frame_box_layout = QGridLayout()
        img_frame_box1.setLayout(img_frame_box_layout)
        img_frame_box1.setMinimumWidth(650)

        img_frame_box_layout.addWidget(self.imgLabel, 1, 0)

        # Box2 Operations
        img_frame_box2 = QGroupBox()
        img_frame_box_layout2 = QGridLayout()
        img_frame_box2.setLayout(img_frame_box_layout2)

        img_frame_box_layout2.addWidget(self.capPic, 1, 0)
        img_frame_box_layout2.addWidget(self.autoCap, 1, 1)
        img_frame_box_layout2.addWidget(self.dialogBtn, 1, 3)
        img_frame_box_layout2.addWidget(self.dialogLine, 1, 2)
        img_frame_box_layout2.addWidget(self.pwmTxt, 2, 1)
        img_frame_box_layout2.addWidget(self.pwmBtn, 2, 0)

        # -------------------------------------------------------------------------------------------------------------
        # Plot
        # -------------------------------------------------------------------------------------------------------------

        self.plot1 = pg.PlotWidget()
        self.plot1_curve1 = pg.PlotCurveItem(pen=(1, 1))
        self.plot1_curve2 = pg.PlotCurveItem(pen=(1, 2))
        self.plot1_curve3 = pg.PlotCurveItem(pen=(1, 3))
        self.plot1.addItem(self.plot1_curve1)
        self.plot1.addItem(self.plot1_curve2)
        self.plot1.addItem(self.plot1_curve3)

        self.plot2 = pg.PlotWidget()
        self.plot2_curve1 = pg.PlotCurveItem(pen=(1, 1))
        self.plot2.addItem(self.plot2_curve1)

        self.plot1.setXRange(0, 255)
        self.plot1.setYRange(0, 100)

        img_plot_box2 = QGroupBox()
        img_plot_box_layout2 = QGridLayout()
        img_plot_box2.setLayout(img_plot_box_layout2)
        img_plot_box_layout2.addWidget(self.plot1, 0, 0)
        img_plot_box_layout2.addWidget(self.plot2, 1, 0)

        # -------------------------------------------------------------------------------------------------------------
        # Box Operations
        # -------------------------------------------------------------------------------------------------------------

        vbox1.addWidget(rgb_frame_box)
        vbox1.addWidget(color_value_box)
        vbox1.addStretch(1)

        vbox2.addWidget(desired_color_box)
        vbox2.addStretch(1)

        vbox3.addWidget(img_frame_box1)
        vbox3.addWidget(img_frame_box2)

        vbox4.addWidget(img_plot_box2)

        hbox1.addLayout(vbox1)
        hbox1.addLayout(vbox2)
        hbox1.addLayout(vbox3)
        hbox2.addLayout(vbox4)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)

        self.setLayout(vbox)
        self.setWindowTitle('PyQt5')
        self.start_webcam()
        self.show()

    def start_webcam(self):
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        start_time = time.time()
        self.image = self.capture.read()
        self.image = imutils.resize(self.image, width=640, height=480)
        self.image = cv2.flip(self.image, 1)

        self.color_detect(self.image)
        self.display_image(self.image, 1)
        times.append(time.time() - start_time)
        if len(times) % 30 == 0:
            print(sum(times))
            times.clear()

    def display_image(self, img, windows):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if windows == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.imgLabel.setScaledContents(True)

    def color_detect(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        rev_img = (255 - thresh_img)

        src = cv2.split(img)

        and_img1 = cv2.bitwise_and(src[0], rev_img)
        and_img2 = cv2.bitwise_and(src[1], rev_img)
        and_img3 = cv2.bitwise_and(src[2], rev_img)

        self.and_img = cv2.merge((and_img1, and_img2, and_img3))
        self.lab_img = cv2.cvtColor(self.and_img, cv2.COLOR_BGR2Lab)

        # mask = cv2.inRange(self.lab_img, (0, -300, -300), (100, +300, +300))
        # imask = mask > 0
        # L = np.zeros_like(self.and_img, np.uint8)
        # L[imask] = self.and_img[imask]

        sumR, sumG, sumB, pixelsR, pixelsG, pixelsB = self.rgb_value(self.and_img, 100)
        sumRGB = [sumR, sumG, sumB]

        sumL, sumA, sumBB, pixelsL, pixelsA, pixelsBB = self.lab_value(self.lab_img, 100)
        sumLAB = [sumL, sumA, sumBB]
        x_range = list(range(0, len(pixelsL)))

        if len(pixelsR) & len(pixelsG) & len(pixelsB) != 0:
            x1, y1 = self.img_hist(pixelsR)
            x2, y2 = self.img_hist(pixelsG)
            x3, y3 = self.img_hist(pixelsB)

        # -------------------------------------------------------------------------------------------------------------
        # RGB, LAB Values / Setting text and background color / Plotting them
        # -------------------------------------------------------------------------------------------------------------
        self.rgbLabel.setStyleSheet('color: red; background-color: rgb' + str((sumR, sumG, sumB)))
        self.rgbValue.setText(str(sumRGB))
        self.labValue.setText(str(sumLAB))

        if len(pixelsR) & len(pixelsG) & len(pixelsB) != 0:
            if self.frameNumber % 100 == 0:
                self.plot1_curve1.setData(x=list(x1), y=list(y1))
                self.plot1_curve2.setData(x=list(x2), y=list(y2))
                self.plot1_curve3.setData(x=list(x3), y=list(y3))
                self.plot2_curve1.setData(x=x_range, y=pixelsL)

        self.auto_capture()
        self.frameNumber += 1

    # Calculate RAL to RGB with using RALtoRGB.py
    def apply(self):
        RALIndex = self.RALCombo.currentIndex()
        R_Ral = RALtoRGB.r_code[RALIndex]
        G_Ral = RALtoRGB.g_code[RALIndex]
        B_Ral = RALtoRGB.b_code[RALIndex]

        self.RALtoRGB.setText("[" + str(R_Ral) + "," + str(G_Ral)+"," + str(B_Ral) + "]")
        self.RALLabel.setStyleSheet('color: red; background-color: rgb' + str((R_Ral, G_Ral, B_Ral)))

    # Capture snapshot when function is used
    def capture_img(self):
        if self.dialogLine.text() is not "":
            cv2.imwrite(self.dialogLine.text() + '/Cable_' + str(self.countImg) + '.png', self.image)
            self.countImg += 1
        else:
            self.dialogLine.setText("Please spot folder")

    # Open file dialog and choose file path
    def open_file_dialog(self):
        directory = str(QFileDialog.getExistingDirectory())
        self.dialogLine.setText('{}'.format(directory))

    # If frameNumber is greater than 100 take a snapshot
    def auto_capture(self):
        if self.autoCap.isChecked():
            if self.frameNumber % 100 == 0:
                self.capture_img()

    # Set pwm Text
    def set_pwm(self):
        print("PWM")

    # Calculate RGB values and sum of this values of image from any row
    @staticmethod
    def rgb_value(rgb_img, row):
        sumR = 0
        sumG = 0
        sumB = 0
        px = rgb_img[row, :]
        pixelsB = px[:, [0]]
        pixelsG = px[:, [1]]
        pixelsR = px[:, [2]]

        pixelsR = pixelsR.flatten()
        pixelsG = pixelsG.flatten()
        pixelsB = pixelsB.flatten()

        maskR = pixelsR > 0
        maskG = pixelsG > 0
        maskB = pixelsB > 0

        pixelsR = pixelsR[maskR]
        pixelsG = pixelsG[maskG]
        pixelsB = pixelsB[maskB]

        if len(pixelsR) & len(pixelsG) & len(pixelsB) != 0:
            sumR = int(pixelsR.sum() / len(pixelsR))
            sumG = int(pixelsG.sum() / len(pixelsG))
            sumB = int(pixelsB.sum() / len(pixelsB))

        return sumR, sumG, sumB, pixelsR, pixelsG, pixelsB

    # Calculate LAB values and sum of this values of image from any row
    @staticmethod
    def lab_value(lab_img, row):
        sumL = 0
        sumA = 0
        sumB = 0
        pix = lab_img[row, :]
        pixelsL = pix[:, [0]]
        pixelsA = pix[:, [1]]
        pixelsB = pix[:, [2]]

        pixelsL = pixelsL.flatten()
        pixelsA = pixelsA.flatten()
        pixelsB = pixelsB.flatten()

        maskL = pixelsL > 0
        maskA = pixelsA > 0
        maskB = pixelsB > 0

        pixelsL = pixelsL[maskL]
        pixelsA = pixelsA[maskA]
        pixelsB = pixelsB[maskB]

        if len(pixelsL) != 0:
            sumL = int(pixelsL.sum() / len(pixelsL))
            sumA = int(pixelsA.sum() / len(pixelsA))
            sumB = int(pixelsB.sum() / len(pixelsB))

        return sumL, sumA, sumB, pixelsL, pixelsA, pixelsB

    # Calculate histogram from pixel row or column
    @staticmethod
    def img_hist(value):
        c = Counter(value)
        s_c = sorted(c.items(), key=lambda t: t[0])
        x, y = zip(*s_c)

        return x, y

    # @staticmethod
    # def pwm_control(frq):
    #     pwm24.ChangeDutyCycle(frq)


# pwm24.stop()
# GPIO.cleanup()
app = QApplication(sys.argv)
window = Window()
sys.exit(app.exec_())
