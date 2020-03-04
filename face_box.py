import dlib
from blink_detector import BlinkDetector
from paramaters import *
import time

class FaceBox(object):
    def __init__(self, box=None, frame=None, shape_predictor=None, rect=None):
        self.frame = frame
        if rect is None and box is not None:
            self.rect = dlib.rectangle(box[0], box[1], box[2], box[3])
        elif rect is not None:
            self.rect = rect
        self.shape_predictor = shape_predictor
        self.__loadDefault()
    
    def __loadDefault(self):
        self.counter = 0
        self.blink_detector = BlinkDetector.getInstance()
        self.id = self.blink_detector.registerBox(self)
        self.registered = True
        self.is_previos_eye_closed = False
        self.blink_detected = False
        self.open_counter = 0
        self.close_detected = False

    
    def deRegister(self):
        self.registered = False
        return self.blink_detector.deRegisterBox(self.id)
    
    def register(self):
        if self.registered:
            return self.id
        else:
            self.__loadDefault()
            return self.id
    
    def checkFrame(self):
        # To prevent not registered boxes to check
        if not self.registered :
            raise "Face box is not registered"
            return False
        if self.blink_detected: # If this box verified
            return True
        else:
            blink_detector_response = self.blink_detector.detect(self.id, self.frame)
            if blink_detector_response == 2: # Eyes are closed
                if not self.is_previos_eye_closed:
                    self.is_previos_eye_closed = True
                    self.counter = 1
                else:
                    self.counter += 1
                if self.counter >= EYE_AR_CONSEC_FRAMES:
                    self.close_detected = True
                self.open_counter = 0
            elif blink_detector_response == 1 : # Eyes are opened
                if self.is_previos_eye_closed:
                    self.is_previos_eye_closed = False
                    self.open_counter = 1
                    self.counter = 0
                    self.is_previos_eye_closed = False   
                else:
                    self.open_counter += 1
                    if self.close_detected and self.open_counter >= EYE_AR_CONSEC_FRAMES: 
                        # Eyes are completely opened for N frames after blink
                        self.blink_detected = True
                        return True
            else:
                self.counter = 0
                self.open_counter = 0
                self.is_previos_eye_closed = False
            #print("({},{})".format(self.counter, self.open_counter))
            return False
    
    def updateRect(self, box, rect=None):
        if rect is None:
            self.rect = dlib.rectangle(box[0], box[1], box[2], box[3])
        else:
            self.rect = rect
    
    def updateFrame(self, frame):
        self.frame = frame
