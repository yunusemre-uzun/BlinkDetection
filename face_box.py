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
        self.liveness_score = 0
    
    def __getEyesStatus(self):
        return BlinkDetector.getEyesStatus(self.id, self.frame)

    
    def checkFrame(self, frame):
        '''
        current_left_eye_open, current_right_eye_open = BlinkDetector.getEyesStatus(self.id, self.frame)
        if current_left_eye_open != self.left_open:
            self.liveness_score += 1
            self.left_open = current_left_eye_open
        if current_right_eye_open != self.right_open:
            self.liveness_score += 1
            self.right_open = current_right_eye_open
        print (self.liveness_score)
        #time.sleep(1.0)
        if self.liveness_score > LIVENESS_THRESH:
            return True
        else:
            return False

        '''
        if BlinkDetector.detect(self.id, frame):
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
