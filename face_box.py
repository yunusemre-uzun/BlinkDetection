import dlib
from blink_detector import BlinkDetector
from paramaters import *

class FaceBox(object):
    def __init__(self, box, frame, shape_predictor, rect=None):
        self.frame = frame
        if rect is None:
            self.rect = dlib.rectangle(box[0], box[1], box[2], box[3])
        else:
            self.rect = rect
        self.shape_predictor = dlib.shape_predictor(shape_predictor)
        self.counter = 0
        self.is_previos_eye_closed = False
        self.id = BlinkDetector.registerBox(self)

    
    def checkFrame(self):
        if BlinkDetector.detect(self.id):
            print(self.is_previos_eye_closed, ":", self.counter)
            # If eye is closed
            if not self.is_previos_eye_closed:
                # If eye is not closed in previous frame
                self.is_previos_eye_closed = True
            self.counter += 1
        elif not self.is_previos_eye_closed :
            self.counter = 0   
        else:
            self.is_previos_eye_closed = False
        if self.counter >= EYE_AR_CONSEC_FRAMES:
            return True
        return False
    
    def updateRect(self, box, rect=None):
        if rect is None:
            self.rect = dlib.rectangle(box[0], box[1], box[2], box[3])
        else:
            self.rect = rect
    
    def updateFrame(self, frame):
        self.frame = frame
