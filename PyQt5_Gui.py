import sys
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import *
import RALtoRGB
import time

COLOR_ROWS = 80
COLOR_COLS = 250
pixels = []
r = []
g = []
b = []
times = []

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
        self.countImg = 0
        self.frameNumber = 0
        self.font1 = QFont("Times", 14)
        self.font2 = QFont("Times", 11)

        vbox1 = QVBoxLayout()
        vbox2 = QVBoxLayout()
        vbox3 = QVBoxLayout()
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
        self.rgbValueLabel.setAlignment(Qt.AlignLeft)

        # Box1 Operations
        rgb_frame_box = QGroupBox("Producing Color")
        rgb_frame_box_layout = QGridLayout()
        rgb_frame_box.setLayout(rgb_frame_box_layout)
        rgb_frame_box.setMaximumWidth(220)

        # Box2 Operations
        rgb_value_box = QGroupBox("Color Codes")
        rgb_value_box_layout = QGridLayout()
        rgb_value_box.setLayout(rgb_value_box_layout)
        rgb_value_box.setFixedWidth(220)

        # Adding Widgets
        rgb_frame_box_layout.addWidget(self.rgbTitle, 1, 0)
        rgb_frame_box_layout.addWidget(self.rgbLabel, 2, 0)
        rgb_value_box_layout.addWidget(self.rgbValueLabel, 1, 0)
        rgb_value_box_layout.addWidget(self.rgbValue, 1, 1)

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

        # -------------------------------------------------------------------------------------------------------------
        # Box Operations
        # -------------------------------------------------------------------------------------------------------------

        vbox1.addWidget(rgb_frame_box)
        vbox1.addWidget(rgb_value_box)
        vbox1.addStretch(1)

        vbox2.addWidget(desired_color_box)
        vbox2.addStretch(1)

        vbox3.addWidget(img_frame_box1)
        vbox3.addWidget(img_frame_box2)

        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)
        hbox.addLayout(vbox3)

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
        start_time = time.time()
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)

        self.color_detect(self.image)
        self.display_image(self.and_img, 1)
        times.append(time.time() - start_time)
        if len(times) % 30 == 0:
            print(sum(times))
            times.clear()

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
        ret, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
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

        self.auto_capture()
        self.frameNumber += 1

    def apply(self):
        RALIndex = self.RALCombo.currentIndex()
        R_Ral = RALtoRGB.r_code[RALIndex]
        G_Ral = RALtoRGB.g_code[RALIndex]
        B_Ral = RALtoRGB.b_code[RALIndex]

        self.RALtoRGB.setText("[" + str(R_Ral) + "," + str(G_Ral)+"," + str(B_Ral) + "]")
        self.RALLabel.setStyleSheet('color: red; background-color: rgb' + str((R_Ral, G_Ral, B_Ral)))

    def capture_img(self):
        if self.dialogLine.text() is not "":
            cv2.imwrite(self.dialogLine.text() + '/Cable_' + str(self.countImg) + '.png', self.and_img)
            self.countImg += 1
        else:
            self.dialogLine.setText("Please spot folder")

    def open_file_dialog(self):
        directory = str(QFileDialog.getExistingDirectory())
        self.dialogLine.setText('{}'.format(directory))

    def auto_capture(self):
        if self.autoCap.isChecked():
            if self.frameNumber % 100 == 0:
                self.capture_img()


app = QApplication(sys.argv)
window = Window()
sys.exit(app.exec_())
