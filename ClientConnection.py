import socket
import threading
import zlib
import pickle
import sys
import time
from comunicationCodes import ComCodes

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QLabel


class ClientConnection(threading.Thread):

    def __init__(self, name, sendPort, listenPort, parent):
        super().__init__()

        self.parent = parent

        self.setName(name)
        self.modelAccuracy = None

        self.__host = 'localhost'
        self.__listenPort = listenPort
        self.__sendPort = sendPort
        self.__listenConnection = socket.socket()
        self.__sendConnection = socket.socket()

        self.__bufferSize = 1024
        self.__listenerBuffer = []
        self.__listenerMutex = threading.Lock()
        self.__sendBuffer = []
        self.__sendMutex = threading.Lock()

    def send(self, mesage):
        print('sending')
        resp = zlib.compress(pickle.dumps(mesage), 4)
        size = sys.getsizeof(resp)

        self.__sendConnection.sendall(pickle.dumps(size))
        time.sleep(1)
        self.__sendConnection.sendall(resp)

    def getConnectionDetails(self):
        return self.__host + ':' + str(self.__sendPort) + ', ' +str(self.__listenPort)

    def setQLabels(self, accuracy, serverStatus):
        self.accuracyLabel = accuracy
        self.serverStatus = serverStatus

    def getAccuracy(self):
        return self.__modelAccuracy

    def setAccuracyText(self, accuracy):
        self.accuracyLabel.setText('Accuracy: ' + accuracy)

    def setServerStatusText(self, text):
        status = 'refused'
        style = '''
            color: #c00000
        '''
        if text:
            status = 'accepted'
            style = '''
                color: #00c000
            '''
        if text == '-':
            status = '-'
            style = '''
                            color: #ffffff
                        '''
        self.serverStatus.setStyleSheet(style)
        self.serverStatus.setText(status)

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
                self.__listenerMutex.acquire()
                if received_data != b'':
                    response = (pickle.loads(zlib.decompress(received_data)))
                    if response[0] == ComCodes.POST_ACCURACY:
                        self.__modelAccuracy = response[1]
                        self.setAccuracyText(str(self.__modelAccuracy)[:5])
                    if response[0] == ComCodes.IS_PARTICIPANT:
                        self.setServerStatusText(response[1])
                self.__listenerMutex.release()
                # print('Device', self.getName(), 'received all data of size', size)
            finally:
                time.sleep(1)

    def run(self):
        print('Trying to connect on ports', self.__listenPort, self.__sendPort)
        self.__listenConnection.connect((self.__host, self.__listenPort))
        self.__sendConnection.connect((self.__host, self.__sendPort))

        listener = threading.Thread(target=self.__listenerThread)
        listener.start()