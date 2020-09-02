from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QDesktopWidget, QFileDialog, \
    QMessageBox
from PyQt5.QtGui import QIcon, QFont, QPixmap
import sys
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

path_rest = 'Files/xray-vs-rest-model-v1'
path_classification = 'Files/xray-model-v1.2'

model_rest = load_model(path_rest)
model_classification = load_model(path_classification)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.title = 'Chest X-ray Classifier'
        self.left = 0
        self.right = 0
        self.width = 1000
        self.height = 800
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setStyleSheet('background-color: #b9d3e9')
        self.setWindowIcon(QIcon('Files/lung.png'))
        self.setFixedSize(self.width, self.height)
        self.center()

        self.image_background = QtWidgets.QLabel(self)
        self.image_background.setStyleSheet('background-color: #609bcd;'
                                            'border-radius: 10px')
        self.image_background.setFixedSize(540, 440)
        self.image_background.move(130, 155)

        self.diagnosis_background = QtWidgets.QLabel(self)
        self.diagnosis_background.setStyleSheet('background-color: #609bcd;'
                                            'border-radius: 10px')
        self.diagnosis_background.setFixedSize(540, 100)
        self.diagnosis_background.move(130, 600)

        self.photo = QtWidgets.QLabel(self)
        self.photo.move(150, 175)
        self.photo.setScaledContents(True)
        self.photo.resize(500, 400)
        self.photo.setPixmap(QPixmap('Files/xray-flower.jpg'))

        self.class_text = QtWidgets.QLabel(self)
        self.class_text.setText(' ')
        self.class_text.setFont(QFont('Arial', 20))
        self.class_text.move(170, 620)
        self.class_text.setStyleSheet('background-color: #609bcd;')

        self.bar = QtWidgets.QLabel(self)
        self.bar.setText('          CHEST X-RAY CLASSIFIER')
        self.bar.setFont(QFont('Arial', 15))
        self.bar.setStyleSheet('background-color: #609bcd;')
        self.bar.setFixedSize(self.width, 50)
        self.bar_icon = QtWidgets.QLabel(self)
        self.bar_icon.setPixmap(QPixmap('Files/lung.png'))
        self.bar_icon.setScaledContents(True)
        self.bar_icon.resize(40,40)
        self.bar_icon.move(10,5)
        self.bar_icon.setStyleSheet('background-color: #609bcd;')

        self.signature = QtWidgets.QLabel(self)
        self.signature.setText('Ivo John \n version 0.1')
        self.signature.setFont(QFont('Times', 10))
        self.signature.move(10, self.height - 40)

        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText('Upload an image')
        self.b1.move(700, 250)
        self.b1.resize(200, 64)
        self.b1.setFont(QFont('Times', 13))
        self.b1.setStyleSheet('background-color: #609bcd;'
                              'border-radius: 20px')
        self.b1.setIcon(QIcon('Files/mental-health.png'))
        self.b1.setIconSize(QtCore.QSize(40, 40))
        self.b1.setToolTip("Click to upload an image to classify")
        self.b1.clicked.connect(self.upload_and_classify)

        self.b2 = QtWidgets.QPushButton(self)
        self.b2.setText('Documentation')
        self.b2.move(700, 350)
        self.b2.resize(200, 64)
        self.b2.setFont(QFont('Times', 13))
        self.b2.setStyleSheet('background-color: #609bcd;'
                              'border-radius: 20px')
        self.b2.setIcon(QIcon('Files/psychology.png'))
        self.b2.setIconSize(QtCore.QSize(40, 40))
        self.b2.setToolTip("Click to access the documentation")
        self.b2.clicked.connect(self.documentation_window)

        self.b3 = QtWidgets.QPushButton(self)
        self.b3.setText('General information')
        self.b3.move(700, 450)
        self.b3.resize(200, 64)
        self.b3.setFont(QFont('Times', 13))
        self.b3.setStyleSheet('background-color: #609bcd;'
                              'border-radius: 20px;'
                              'border-style: outset;')
        self.b3.setIcon(QIcon('Files/medical-report.png'))
        self.b3.setIconSize(QtCore.QSize(40, 40))
        self.b3.setToolTip("Click to access general information")
        self.b3.clicked.connect(self.information_window)

    def documentation_window(self):
        self.documentation_window = DocumentationWindow()
        self.documentation_window.show()

    def information_window(self):
        self.information_window = InformationWindow()
        self.information_window.show()

    def xray_vs_rest(self, path):
        try:
            self.path = path
            self.image = Image.open(self.path)
            self.image = self.image.resize((250,250))
            self.image = np.array(self.image)
            if len(self.image.shape) == 2:
                self.image = np.dstack([self.image, self.image, self.image])
            elif self.image.shape[2] >= 4:
                self.image = self.image[:,:,:3]
            self.image = self.image/255.
            self.image = np.expand_dims(self.image, axis=0)
            self.pred = (model_rest.predict(self.image)[0, 0] > 0.8)
            return self.pred

        except:
            self.msg = QMessageBox()
            self.msg.setWindowTitle('File error')
            self.msg.setText('During upload process error occurred. \n'
                             'Do you want to try again?')
            self.msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            self.msg.setIcon(QMessageBox.Critical)
            self.msg.setWindowIcon(QIcon('Files/lung.png'))
            x = self.msg.exec_()
            if x == QMessageBox.Yes:
                self.upload_and_classify()
            else:
                pass

    def upload_and_classify(self):
        self.full_path = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image files( *.jpg *.jpeg *.gif *.png)")
        self.image_path = self.full_path[0]
        self.returned_prediction = self.xray_vs_rest(self.image_path)
        if self.returned_prediction:
            pixmap = QPixmap(self.image_path)
            self.photo.setPixmap(QPixmap(pixmap))
            self.photo.resize(500, 400)
            self.returned_class = self.classify_rtg(self.image_path)
            self.class_text.setStyleSheet('color: green')
            self.probability = abs(2* (0.5-self.returned_class)) * 100
            if self.returned_class > 0.5:
                self.class_text.setText('The x-ray shows a pneumonia case \n'
                                        'with probability of: {:.2f} %'.format(self.probability))
                self.class_text.setStyleSheet('background-color: #900000;'
                                              'color: black')
                self.diagnosis_background.setStyleSheet('background-color: #900000;'
                                                        'border-radius: 10px')

            else:
                self.class_text.setText('There is no pneumonia on the x-ray \n'
                                        'with probability of {:.2f} %'.format(self.probability))
                self.class_text.setStyleSheet('background-color: #007000;'
                                              'color: black')
                self.diagnosis_background.setStyleSheet('background-color: #007000;'
                                                        'border-radius: 10px')
            self.class_text.adjustSize()

        else:
            self.msg = QMessageBox()
            self.msg.setWindowTitle('Wrong image')
            self.msg.setText('Uploaded image does not show a chest x-ray! \n'
                             'Please, upload an image of a chest x-ray')
            self.msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            self.msg.setIcon(QMessageBox.Critical)
            self.msg.setWindowIcon(QIcon('Files/lung.png'))
            x = self.msg.exec_()
            if x == QMessageBox.Ok:
                self.upload_and_classify()
            else:
                pass


    def classify_rtg(self, path):
        self.path = path
        self.image = Image.open(self.path)
        self.image = self.image.resize((250, 250))
        self.image = np.array(self.image)
        if len(self.image.shape) == 2:
            self.image = np.dstack([self.image, self.image, self.image])
        elif self.image.shape[2] >= 4:
            self.image = self.image[:, :, :3]
        self.image = self.image / 255.
        self.image = np.expand_dims(self.image, axis=0)
        self.pred = model_classification.predict(self.image)[0, 0]
        return self.pred

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class DocumentationWindow(QMainWindow):
    def __init__(self):
        super(DocumentationWindow, self).__init__()
        self.title = 'Chest RTG Classifier Documentation'
        self.left = 100
        self.right = 100
        self.width = 600
        self.height = 700

        self.setWindowTitle(self.title)
        self.setStyleSheet('background-color: #b9d3e9')
        self.setWindowIcon(QIcon('Files/lung.png'))
        self.setGeometry(self.left, self.right, self.width, self.height)

        self.bar = QtWidgets.QLabel(self)
        self.bar.setText('          CHEST X-RAY CLASSIFIER DOCUMENTATION')
        self.bar.setFont(QFont('Arial', 15))
        self.bar.setStyleSheet('background-color: #609bcd;')
        self.bar.setFixedSize(self.width, 50)
        self.bar_icon = QtWidgets.QLabel(self)
        self.bar_icon.setPixmap(QPixmap('Files/lung.png'))
        self.bar_icon.setScaledContents(True)
        self.bar_icon.resize(40,40)
        self.bar_icon.move(10,5)
        self.bar_icon.setStyleSheet('background-color: #609bcd;')

        self.model_architecture = QtWidgets.QLabel(self)
        self.model_architecture.setScaledContents(True)
        self.model_architecture.resize(500, 300)
        self.model_architecture.move(50, 75)
        self.model_architecture.setPixmap(QPixmap('Files/documentation.png'))

        self.documentation_background = QtWidgets.QLabel(self)
        self.documentation_background.setFixedSize(500, 200)
        self.documentation_background.move(50, 400)
        self.documentation_background.setStyleSheet('background-color: #609bcd;'
                                                    'border-radius: 10px')

        self.documentation_text = QtWidgets.QLabel(self)
        self.documentation_text.setFont(QFont('Arial', 13))
        self.documentation_text.setText(' This program was build with the usage of Tensorflow with Keras \n'
                                        'and PyQt5 to develop a graphical user interface. The backend is \n'
                                        'run by two Convolutional Neural Networks with architecture\n'
                                        'presented above. As visible both are based on pretrained VGG16.\n\n'
                                        'The first network classify if the input image presents the\n'
                                        'chest x-ray and the second one classify if there is a pneumonia \n'
                                        'on the provided image.')
        self.documentation_text.adjustSize()
        self.documentation_text.setStyleSheet('background-color: #609bcd;')
        self.documentation_text.move(60, 420)




class InformationWindow(QMainWindow):
    def __init__(self):
        super(InformationWindow, self).__init__()
        self.title = 'General information about Chest RTG Classifier'
        self.left = 400
        self.right = 100
        self.width = 600
        self.height = 200

        self.setWindowTitle(self.title)
        self.setStyleSheet('background-color: #b9d3e9')
        self.setWindowIcon(QIcon('Files/lung.png'))
        self.setGeometry(self.left, self.right, self.width, self.height)

        self.information_background = QtWidgets.QLabel(self)
        self.information_background.setFixedSize(550, 150)
        self.information_background.move(25, 25)
        self.information_background.setStyleSheet('background-color: #609bcd;'
                                                    'border-radius: 10px')

        self.information_text = QtWidgets.QLabel(self)
        self.information_text.setFont(QFont('Arial', 13))
        self.information_text.setText('Please remember that this program is not designed to clinical usage \n'
                                      'The dataset used in the developing process consisted of limited \n'
                                      'amount of cases and no broad testing process was done.')
        self.information_text.adjustSize()
        self.information_text.setStyleSheet('background-color: #609bcd;')
        self.information_text.move(50, 65)




def window():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


window()

"""
Icons from Freepik"""
