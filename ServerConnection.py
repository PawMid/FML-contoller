import threading
import socket
import zlib
import pickle
import sys
import time
from comunicationCodes import ComCodes


class ServerConnection(threading.Thread):

    def __init__(self, host, sendPort, listenPort):
        super().__init__()

        self.__host = host
        self.__sendPort = sendPort
        self.__listenPort = listenPort
        self.__listenConnection = socket.socket()
        self.__sendConnection = socket.socket()
        self.__bufferSize = 1024

        self.__preAccuracy = None
        self.__postAccuracy = None

    def setCallbacks(self, imageIsSet, enablePredictBtn):
        self.enablePredictBtn = enablePredictBtn
        self.imageIsSet = imageIsSet

    def addQtControls(self, preLabel, postLabel, trainBtn, netBtn, modelUpdateBtns, downloadBtn, downloadedModelLabel, downloadedModelAccuracy):
        self.netBtn = netBtn
        self.trainBtn = trainBtn
        self.preLabel = preLabel
        self.postLabel = postLabel
        self.downloadedModelLabel = downloadedModelLabel
        self.downloadedModelAccuracyLabel = downloadedModelAccuracy
        self.modelUpdateBtns = modelUpdateBtns
        self.downloadBtn = downloadBtn

    def addModelRef(self, model):
        self.model = model

    def setAccuracyText(self):
        self.enableButtons()
        self.preLabel.setText(str(self.__preAccuracy)[:5])
        self.postLabel.setText(str(self.__postAccuracy)[:5])

    def enableButtons(self):
        for btn in self.modelUpdateBtns:
            btn.setEnabled(True)
        self.trainBtn.setEnabled(True)
        self.netBtn.setEnabled(True)
        self.downloadBtn.setEnabled(True)

    def send(self, message):
        print('sending', self.__sendPort, message)
        resp = zlib.compress(pickle.dumps(message), 4)
        size = sys.getsizeof(resp)

        self.__sendConnection.sendall(pickle.dumps(size))
        time.sleep(1)
        self.__sendConnection.sendall(resp)

    def __listenerThread(self):
        while True:

            try:
                received_data = b''
                size = pickle.loads(self.__listenConnection.recv(self.__bufferSize))
                # print('Device', self.getName(),'receiving data of size', size)
                while sys.getsizeof(received_data) < size:
                    data = self.__listenConnection.recv(self.__bufferSize)
                    received_data += data
                    # progressBar(size, sys.getsizeof(received_data))
                # progressBar(size, sys.getsizeof(received_data), True)
                # self.__listenerMutex.acquire()
                if received_data != b'':
                    response = (pickle.loads(zlib.decompress(received_data)))
                    if response[0] == ComCodes.POST_ACCURACY:
                        self.__preAccuracy = response[1][0]
                        self.__postAccuracy = response[1][1]
                        self.setAccuracyText()
                        self.enableButtons()
                    elif response[0] == ComCodes.LOAD_MODEL and response[1] is True:
                        pass
                    elif response[0] == ComCodes.GET_STRUCTURE:
                        self.model.setNet(response[1], False)
                        self.send([ComCodes.GET_WEIGHTS])
                        self.downloadedModelLabel.setText(self.model.getModelType())
                    elif response[0] == ComCodes.GET_WEIGHTS:
                        self.model.setTrainableWeights(response[1])
                        self.downloadedModelAccuracyLabel.setText(str(response[2])[:5])
                        if self.imageIsSet():
                            self.enablePredictBtn()

                # self.__listenerMutex.release()
                # print('Device', self.getName(), 'received all data of size', size)
            finally:
                time.sleep(1)

    def run(self):
        print('Trying to connect on ports', self.__listenPort, self.__sendPort)
        self.__listenConnection.connect((self.__host, self.__listenPort))
        self.__sendConnection.connect((self.__host, self.__sendPort))

        listener = threading.Thread(target=self.__listenerThread)
        listener.start()
