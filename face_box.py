import dlib
from blink_detector import BlinkDetector
from paramaters import *
import time

class FaceBox(object):
    def __init__(self, box, frame, shape_predictor, rect=None):
        self.frame = frame
        if rect is None:
            self.rect = dlib.rectangle(box[0], box[1], box[2], box[3])
        else:
            self.rect = rect
        self.shape_predictor = shape_predictor
        self.counter = 0
        self.id = BlinkDetector.registerBox(self)
        self.left_open, self.right_open = self.__getEyesStatus()
        self.is_previos_eye_closed = not (self.left_open and self.right_open)
        self.open_counter = 0
    
    def __getEyesStatus(self):
        return BlinkDetector.getEyesStatus(self.id, self.frame)

    
    def checkFrame(self):
        blink_detector_response = BlinkDetector.detect(self.id, self.frame)
        if blink_detector_response == 2: # Eyes are closed
            print("Closed")
            if not self.is_previos_eye_closed:
                self.is_previos_eye_closed = True
            self.counter += 1
            self.open_counter = 0
        elif blink_detector_response == 1 : # Eyes are opened
            print("Opened")
            self.open_counter += 1
            if self.counter >= EYE_AR_THRESH and self.open_counter >= EYE_AR_THRESH: # Eyes are completely opened after blink
                print("Real")
                return True
            self.counter = 0   
        else:
            self.counter = 0
            print("Can not decided")
        return False
    
    def updateRect(self, box, rect=None):
        if rect is None:
            self.rect = dlib.rectangle(box[0], box[1], box[2], box[3])
        else:
            self.rect = rect
    
    def updateFrame(self, frame):
        self.frame = frame
