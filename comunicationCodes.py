from enum import Enum


class ComCodes(Enum):
    CAN_TRAIN = 0
    POST_WEIGHTS = 1
    GET_STRUCTURE = 2
    GET_WEIGHTS = 3
    RETRAIN_MODEL = 4
    LOAD_MODEL = 5
    PREDICT = 6
    POST_ACCURACY = 7
    GET_ACCURACY = 8
    IS_PARTICIPANT = 9