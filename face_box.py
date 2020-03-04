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
        self.counter = 0
        self.id = BlinkDetector.registerBox(self)
        self.is_previos_eye_closed = False
        self.blink_detected = False
        self.open_counter = 0
        self.registered = True
    
    def deRegister(self):
        self.registered = False
        return BlinkDetector.deRegisterBox(self.id)
    
    def register(self):
        if self.registered:
            return self.id
        else:
            self.id = BlinkDetector.registerBox(self)
            return self.id
    
    def __getEyesStatus(self):
        return BlinkDetector.getEyesStatus(self.id, self.frame)

    
    def checkFrame(self):
        # To prevent not registered boxes to check
        if not self.registered :
            raise "Face box is not registered"
            return False
        blink_detector_response = BlinkDetector.detect(self.id, self.frame)
        if blink_detector_response == 2: # Eyes are closed
            #print("Closed")
            if not self.is_previos_eye_closed:
                self.is_previos_eye_closed = True
            self.counter += 1
            if self.counter >= EYE_AR_CONSEC_FRAMES:
                self.blink_detected = True
            self.open_counter = 0
        elif blink_detector_response == 1 : # Eyes are opened
            #print("Opened")
            self.open_counter += 1
            if self.blink_detected and self.open_counter >= EYE_AR_CONSEC_FRAMES and not self.is_previos_eye_closed: 
                # Eyes are completely opened for N frames after blink
                return True
            self.is_previos_eye_closed = False
            self.counter = 0   
        else:
            self.counter = 0
            self.open_counter = 0
            self.is_previos_eye_closed = False
            #print("Can not decided")
        print("({},{})".format(self.counter, self.open_counter))
        return False
    
    def updateRect(self, box, rect=None):
        if rect is None:
            self.rect = dlib.rectangle(box[0], box[1], box[2], box[3])
        else:
            self.rect = rect
    
    def updateFrame(self, frame):
        self.frame = frame
