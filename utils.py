def readNetTypes(path='..\\config.txt'):
    return __readProperty('nets', path).split(',')


def getClientsNumber(path='..\\config.txt'):
    return int(__readProperty('nDevices', path))


def getTesterPort(path='..\\config.txt'):
    return int(__readProperty('testerPort', path))


def getServerPort(path='..\\config.txt'):
    return int(__readProperty('serverPort', path))


def __readProperty(prop, path='..\\config.txt'):
    with open(path, 'r') as config:
        lines = config.readlines()
        for line in lines:
            line = line.strip()
            line = line.replace(' ', '')
            line = line.split('=')
            if line[0] == prop:
                return line[1]
