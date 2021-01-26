import sys
import time

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QMainWindow, QWidget, QFrame, QLabel, QFileDialog, QDialog
from PyQt5.QtGui import QPixmap
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QBuffer

from utils import readNetTypes, getClientsNumber, getTesterPort, getServerPort
from ClientConnection import ClientConnection
from ServerConnection import ServerConnection
from comunicationCodes import ComCodes
from convNet1 import convModel
from PIL import Image
import io
import numpy as np


class MainApp(QMainWindow):
    def __init__(self, nets=[]):
        super().__init__()

        self.imgToPredict = None
        self.imgToPredictClass = None
        self.imgPredictedClass = None
        self.imgTrueClassLabel = QLabel()
        self.imgPredictedClassLabel = QLabel()

        self.model = convModel()
        self.nets = nets

        host = 'localhost'
        serverPort = getServerPort()
        self.clientsInNet = getClientsNumber()
        self.firstPort = getTesterPort()

        self.serverConnection = ServerConnection(host, serverPort, serverPort + 1)

        self.clients = []
        self.accBtns = []
        for i in range(self.clientsInNet):
            self.clients.append(ClientConnection('client-' + str(i), self.firstPort + (i * 2),
                                                 self.firstPort + (i * 2) + 1, self))

        for client in self.clients:
            client.start()

        self.setWindowTitle('Federated learning controller')

        self.mainWidget = QWidget()

        self.setCentralWidget(self.mainWidget)

        qss_file = open('style.qss').read()
        self.centralWidget().setStyleSheet(qss_file)

        self.layout = QGridLayout()
        self.layout.setAlignment(Qt.AlignTop)

        self.mainWidget.setLayout(self.layout)

        self.__addDropdown(nets)
        self.addClientInfo()
        self.__addImageFrame()
        self.addTrainButton()
        self.addPredictImage()

        self.serverConnection.addQtControls(self.preTrainVal, self.postTrainVal, self.trainBtn, self.netBtn,
                                                self.accBtns, self.downloadBtn, self.currentModel, self.modelDownloadedAcc)
        self.serverConnection.addModelRef(self.model)
        self.serverConnection.setCallbacks(self.imageIsSet, lambda: self.predictChangeState(True))
        self.serverConnection.start()

        self.disableButtons()

        self.show()

    def addClientInfo(self):
        clientWidget = QWidget()
        clientLayout = QVBoxLayout()
        clientWidget.setLayout(clientLayout)
        # clientLayout.setAlignment(Qt.AlignTop)

        clientsInNet = QWidget()
        clientNumberLayout = QHBoxLayout()
        clientNumberLayout.setContentsMargins(0, 0, 0, 0)
        clientsInNet.setLayout(clientNumberLayout)
        clientNumberLayout.addWidget(QLabel("Clients in symulation: " + str(self.clientsInNet)))

        clientLayout.addWidget(clientsInNet)

        for client in self.clients:
            clientLayout.addWidget(self.__addClientPanel(client))

        self.layout.addWidget(clientWidget, 1, 0)

    def addTrainButton(self):
        trainWidget = QWidget()
        trainLayout = QVBoxLayout()
        trainWidget.setLayout(trainLayout)

        trainingDetailsWidget = QWidget()
        trainDetLayout = QHBoxLayout()
        trainingDetailsWidget.setLayout(trainDetLayout)

        preLabel = QLabel('Pre train:')
        self.preTrainVal = QLabel()
        postLabel = QLabel('Post train:')
        self.postTrainVal = QLabel()

        trainDetLayout.addWidget(preLabel)
        trainDetLayout.addWidget(self.preTrainVal)
        trainDetLayout.addWidget(postLabel)
        trainDetLayout.addWidget(self.postTrainVal)

        def handleTrain():
            self.disableButtons()
            for client in self.clients:
                client.setServerStatusText('-')
                client.setAccuracyText('-')
                client.send([ComCodes.RETRAIN_MODEL])

        ########################################################### Add click event! Done
        btnWidget = QWidget()
        btnLayout = QHBoxLayout()
        btnLayout.setContentsMargins(0, 0, 0, 0)
        btnWidget.setLayout(btnLayout)
        self.trainBtn = QPushButton('Train')
        self.trainBtn.setFixedWidth(100)
        self.trainBtn.clicked.connect(handleTrain)
        btnLayout.addWidget(self.trainBtn)

        trainLayout.addWidget(QLabel('Accuracy:'))
        trainLayout.addWidget(trainingDetailsWidget)
        trainLayout.addWidget(btnWidget)

        self.layout.addWidget(trainWidget, 3, 0)

    def addPredictImage(self):
        predictWidget = QWidget()
        predictLayout = QVBoxLayout()
        predictLayout.setAlignment(Qt.AlignTop)

        predictWidget.setLayout(predictLayout)

        resultWidget = QWidget()
        resultLayout = QHBoxLayout()
        resultWidget.setLayout(resultLayout)

        trueClass = QLabel('True: ')
        predictedClass = QLabel('Predicted: ')
        resultLayout.addWidget(trueClass)
        resultLayout.addWidget(self.imgTrueClassLabel)
        resultLayout.addWidget(predictedClass)
        resultLayout.addWidget(self.imgPredictedClassLabel)

        def predict():
            im = self.pixmapToPIL()
            prediction = self.model.predict(im)
            self.imgPredictedClassLabel.setText(prediction)

        btnWidget = QWidget()
        btnWidget.setFixedWidth(255)
        btnLayout = QHBoxLayout()
        btnLayout.setContentsMargins(0, 0, 0, 0)
        btnWidget.setLayout(btnLayout)
        self.predictBtn = QPushButton('Predict')
        self.predictBtn.clicked.connect(predict)
        self.predictBtn.setFixedWidth(100)
        self.predictBtn.setEnabled(False)
        btnLayout.addWidget(self.predictBtn)

        predictLayout.addWidget(QLabel('Class:'))
        predictLayout.addWidget(resultWidget)
        predictLayout.addWidget(btnWidget)

        self.layout.addWidget(predictWidget, 3, 1)

    def __addDownloadModel(self):
        downloadWidget = QWidget()
        downloadLayout = QVBoxLayout()
        downloadLayout.setAlignment(Qt.AlignTop)
        downloadWidget.setLayout(downloadLayout)
        downloadWidget.setFixedWidth(255)
        downloadLayout.setContentsMargins(0, 0, 0, 0)

        downloadLayout.addWidget(QLabel('Model:'))

        modelDetailsWidget = QWidget()
        modelDetailsLayout = QHBoxLayout()
        modelDetailsWidget.setLayout(modelDetailsLayout)

        modelDetailsLayout.addWidget(QLabel('Current: '))
        self.currentModel = QLabel('-')
        modelDetailsLayout.addWidget(self.currentModel)

        modelDetailsLayout.addWidget(QLabel('Accuracy:'))
        self.modelDownloadedAcc = QLabel('-')
        modelDetailsLayout.addWidget(self.modelDownloadedAcc)

        def download():
            self.serverConnection.send([ComCodes.GET_STRUCTURE])

        ################################################################ Add event!
        self.downloadBtn = QPushButton('Download')
        self.downloadBtn.setFixedWidth(100)
        self.downloadBtn.clicked.connect(download)
        downloadBtnWidget = QWidget()
        downloadBtnLayout = QHBoxLayout()
        downloadBtnWidget.setLayout(downloadBtnLayout)
        downloadBtnLayout.addWidget(self.downloadBtn)

        downloadLayout.addWidget(modelDetailsWidget)
        downloadLayout.addWidget(downloadBtnWidget)

        return downloadWidget

    def __addClientPanel(self, conn):

        panel = QWidget()
        panelLayout = QVBoxLayout()
        panelLayout.setContentsMargins(0, 5, 0, 0)
        panelLayout.setAlignment(Qt.AlignTop)
        panel.setLayout(panelLayout)

        nameWidget = QWidget()
        nameLayout = QHBoxLayout()
        nameWidget.setLayout(nameLayout)
        nameLayout.setContentsMargins(0, 0, 0, 0)
        nameLayout.setAlignment(Qt.AlignTop)

        name = QLabel(conn.getName())
        server = QLabel('Server:')
        self.serverStatus = QLabel('-')
        nameLayout.addWidget(name)
        nameLayout.addWidget(server)
        nameLayout.addWidget(self.serverStatus)

        connection = QLabel('Connection: ' + conn.getConnectionDetails())

        accWidget = QWidget()
        accLayout = QHBoxLayout()
        accLayout.setContentsMargins(0, 0, 0, 0)
        accWidget.setLayout(accLayout)

        accuracy = QLabel('Accuracy: -')
        accLayout.addWidget(accuracy)

        conn.setQLabels(accuracy, self.serverStatus)

        getAccBtn = QPushButton('Update model')
        getAccBtn.setFixedWidth(100)
        getAccBtn.clicked.connect(lambda: conn.send([ComCodes.GET_ACCURACY]))
        accLayout.addWidget(getAccBtn)
        self.accBtns.append(getAccBtn)

        bottomLine = QFrame()
        bottomLine.setFixedHeight(1)
        bottomLine.setFrameStyle(1)

        panelLayout.addWidget(nameWidget)
        panelLayout.addWidget(connection)
        panelLayout.addWidget(accWidget)
        panelLayout.addWidget(bottomLine)

        return panel

    def __addDropdown(self, options):
        netSelect = QWidget()
        dropdownLayout = QHBoxLayout()
        # dropdownLayout.setAlignment(Qt.AlignTop)
        dropdownLayout.setContentsMargins(0, 0, 0, 0)
        netSelect.setFixedHeight(30)
        netSelect.setLayout(dropdownLayout)

        netLabel = QLabel("Selected network: ")
        netLabel.setFixedWidth(100)
        dropdownLayout.addWidget(netLabel)

        netsDropdown = QComboBox()
        netsDropdown.addItems(options)
        netsDropdown.setFixedWidth(50)
        dropdownLayout.addWidget(netsDropdown)

        def handleSetNet(net):
            self.disableButtons()
            print('sending')
            self.serverConnection.send([ComCodes.LOAD_MODEL, net])

        ############################################################# Add click event!
        self.netBtn = QPushButton("Set/ Reset Net")
        self.netBtn.setFixedWidth(100)
        self.netBtn.clicked.connect(lambda: handleSetNet(netsDropdown.currentText()))
        dropdownLayout.addWidget(self.netBtn)

        self.layout.addWidget(netSelect, 0, 0)

    def __addImageFrame(self):
        imageWidget = QWidget()
        widgetLayout = QVBoxLayout()
        widgetLayout.setAlignment(Qt.AlignTop)
        imageWidget.setLayout(widgetLayout)
        widgetLayout.addWidget(QLabel('Image to predict:'))

        self.imgFrame = QFrame()
        self.imgFrame.setFrameStyle(1)
        self.imgFrame.setFixedWidth(255)
        self.imgFrame.setFixedHeight(255)
        imgFrameLayout = QGridLayout()
        imgFrameLayout.setContentsMargins(0, 0, 0, 0)
        self.imgFrame.setLayout(imgFrameLayout)
        widgetLayout.addWidget(self.imgFrame)

        self.image = QLabel()
        self.__showImage()
        imgFrameLayout.addWidget(self.image)

        imageBtnWidget = QWidget()
        imageBtnWidget.setFixedWidth(255)
        imageBtnLayout = QHBoxLayout()
        imageBtnWidget.setLayout(imageBtnLayout)

        loadImageBtn = QPushButton("Open Image")
        loadImageBtn.setFixedWidth(100)
        loadImageBtn.clicked.connect(self.loadImg)
        imageBtnLayout.addWidget(loadImageBtn)

        clearImageBtn = QPushButton("Clear Image")
        clearImageBtn.setFixedWidth(100)
        clearImageBtn.clicked.connect(self.__clearImage)
        imageBtnLayout.addWidget(clearImageBtn)

        download = self.__addDownloadModel()

        widgetLayout.addWidget(imageBtnWidget)
        widgetLayout.addWidget(download)
        self.layout.addWidget(imageWidget, 0, 1, 2, 2)

    def setImgClass(self, classLabel):
        self.imgToPredictClass = classLabel
        if classLabel is None:
            self.imgTrueClassLabel.setText('-')
        else:
            self.imgTrueClassLabel.setText(classLabel)
        self.imgPredictedClassLabel.setText('-')

    def __showImage(self):
        if self.imgToPredict is not None and self.imgToPredict != '':
            self.image.setPixmap(QPixmap(self.imgToPredict).scaled(255, 255))
            iClass = self.imgToPredict.split('/')
            iClass = iClass[len(iClass) - 2]
            self.setImgClass(iClass)
        else:
            self.setImgClass(None)
            self.image.setText('No image selected.')
            self.image.setAlignment(Qt.AlignCenter)

    def pixmapToPIL(self):
        pmap = self.image.pixmap()
        imageBuffer = QBuffer()
        imageBuffer.open(QBuffer.ReadWrite)
        pmap.save(imageBuffer, "PNG")
        img = Image.open(io.BytesIO(imageBuffer.data()))
        return img

    def loadImg(self):
        img = QFileDialog.getOpenFileName(self, 'Open File', r'C:\Users\Pawel\Studia\inz\COVID-19 Radiography Database',
                                          'Image files (*.png *.jpg)')
        self.imgToPredict = img[0]
        if self.currentModel != '-':
            self.predictChangeState(True)
        self.__showImage()

    def __clearImage(self):
        self.imgToPredict = None
        self.predictChangeState(False)
        self.__showImage()

    def imageIsSet(self):
        return self.imgToPredict is not None

    def disableUpdateBtns(self):
        for btn in self.accBtns:
            btn.setEnabled(False)

    def disableButtons(self):
        self.disableUpdateBtns()
        self.trainBtn.setEnabled(False)
        self.netBtn.setEnabled(False)
        self.downloadBtn.setEnabled(False)

    def predictChangeState(self, state):
        self.predictBtn.setEnabled(state)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    nets = readNetTypes()

    window = MainApp(nets)

    sys.exit(app.exec_())
