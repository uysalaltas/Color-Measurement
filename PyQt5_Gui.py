import sys
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import *

COLOR_ROWS = 80
COLOR_COLS = 250
pixels = []
r = []
g = []
b = []

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)


class Window(QWidget):

    def __init__(self):
        super().__init__()

        self.timer = QTimer(self)
        self.capture = cv2.VideoCapture(0)

        self.image = None
        self.outImage = None
        self.font1 = QFont("Times", 14)
        self.font2 = QFont("Times", 11)

        vbox1 = QVBoxLayout()
        vbox2 = QVBoxLayout()
        hbox = QHBoxLayout()

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
        self.rgbValueLabel.setAlignment(Qt.AlignCenter)

        # Box1 Operations
        rgb_frame_box = QGroupBox("Producing Color")
        rgb_frame_box_layout = QGridLayout()
        rgb_frame_box.setLayout(rgb_frame_box_layout)
        rgb_frame_box.setMaximumWidth(220)

        # Box2 Operations
        rgb_value_box = QGroupBox("RGB Values")
        rgb_value_box_layout = QGridLayout()
        rgb_value_box.setLayout(rgb_value_box_layout)
        rgb_value_box.setMaximumWidth(220)

        # Adding Widgets
        rgb_frame_box_layout.addWidget(self.rgbTitle, 1, 0)
        rgb_frame_box_layout.addWidget(self.rgbLabel, 2, 0)
        rgb_value_box_layout.addWidget(self.rgbValueLabel, 1, 0)
        rgb_value_box_layout.addWidget(self.rgbValue, 1, 1)

        # -------------------------------------------------------------------------------------------------------------
        # Camera Elements
        # -------------------------------------------------------------------------------------------------------------

        # Image Label
        self.imgLabel = QLabel()
        self.imgLabel.setFixedHeight(480)
        self.imgLabel.setFixedWidth(640)

        # Slider for threshold
        self.trsSlider = QSlider(Qt.Horizontal)
        self.trsSlider.setMinimum(0)
        self.trsSlider.setMaximum(255)
        self.trsSlider.setValue(100)
        self.trsSlider.setTickInterval(10)
        self.trsSlider.setTickPosition(QSlider.TicksBelow)

        # Box Operations
        img_frame_box = QGroupBox("Camera")
        img_frame_box_layout = QGridLayout()
        img_frame_box.setLayout(img_frame_box_layout)
        img_frame_box.setMinimumWidth(650)

        img_frame_box_layout.addWidget(self.imgLabel, 1, 0)
        img_frame_box_layout.addWidget(self.trsSlider, 2, 0)

        # -------------------------------------------------------------------------------------------------------------
        # Box Operations
        # -------------------------------------------------------------------------------------------------------------

        vbox1.addWidget(rgb_frame_box)
        vbox1.addWidget(rgb_value_box)
        vbox1.addStretch(1)

        vbox2.addWidget(img_frame_box)

        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)

        self.setLayout(hbox)
        self.setWindowTitle('PyQt5')
        self.start_webcam()
        self.show()

    def start_webcam(self):
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)

        self.color_detect(self.image)

        self.display_image(self.and_img, 1)

    def display_image(self, img, window):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.imgLabel.setScaledContents(True)

    def color_detect(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.trs = self.trsSlider.value()
        ret, thresh_img = cv2.threshold(gray_img, self.trs, 255, cv2.THRESH_BINARY)
        rev_img = (255 - thresh_img)

        src = cv2.split(img)

        and_img1 = cv2.bitwise_and(src[0], rev_img)
        and_img2 = cv2.bitwise_and(src[1], rev_img)
        and_img3 = cv2.bitwise_and(src[2], rev_img)

        self.and_img = cv2.merge((and_img1, and_img2, and_img3))

        for i in range(640):
            px = self.and_img[100, i]
            if px[0] > 0:
                pixels.append(px)

        for i in pixels:
            r.append(i[0])
            g.append(i[1])
            b.append(i[2])

        sumR = 0
        sumG = 0
        sumB = 0

        if len(r) != 0:
            sumR = int(sum(r) / len(r))
            sumG = int(sum(g) / len(g))
            sumB = int(sum(b) / len(b))

        sumAll = [sumB, sumG, sumR]

        # -------------------------------------------------------------------------------------------------------------
        # RGB Values / Setting text and background color
        # -------------------------------------------------------------------------------------------------------------
        self.rgbLabel.setStyleSheet('color: red; background-color: rgb' + str((sumB, sumG, sumR)))
        self.rgbValue.setText(str(sumAll))

        pixels.clear()  # Reset arrays
        r.clear()
        g.clear()
        b.clear()


app = QApplication(sys.argv)
window = Window()
sys.exit(app.exec_())
