from imutils import face_utils
from paramaters import *
import math
import cv2
import random

class BlinkDetector(object):
    __instance = None
    __registered_face_boxes = {}

    @staticmethod
    def getInstance():
        if BlinkDetector.__instance == None:
            BlinkDetector()
        return BlinkDetector.__instance

    def __init__(self):
        if BlinkDetector.__instance != None:
            raise Exception("Singleton class")
        else:
            BlinkDetector.__instance = self

    def registerBox(self, face_box):
        facebox_id = random.getrandbits(8)
        while facebox_id in BlinkDetector.__registered_face_boxes:
            facebox_id = random.getrandbits(8)
        self.__registered_face_boxes[facebox_id] = face_box
        return facebox_id
    
    def deRegisterBox(self, id):
        del self.__registered_face_boxes[id]
        return None

    def detect(self, id, frame):
        shape = self.getFaceShape(id)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        left_ear, right_ear = self.getEyeAspectRatio(shape, frame)
        # If eye is closed return True
        if left_ear < EYE_AR_THRESH and right_ear < EYE_AR_THRESH:   
           return 2
        elif left_ear > EYE_AR_THRESH and right_ear > EYE_AR_THRESH:
            return 1
        return 0

    def getFaceShape(self, id):
        face_box = self.getFaceBox(id)
        frame_rgb = face_box.frame
        rect = face_box.rect
        predictor = face_box.shape_predictor
        shape = predictor(frame_rgb, rect)
        shape = face_utils.shape_to_np(shape)
        return shape

    
    def getFaceBox(self, id):
        return BlinkDetector.__registered_face_boxes[id]
    
    def getEyeAspectRatio(self, shape, frame):
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        leftEyeHull = cv2.convexHull(left_eye)
        rightEyeHull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 255), 1)
        leftEAR = self.calculateEyeAspectRatio(left_eye)
        rightEAR = self.calculateEyeAspectRatio(right_eye)
        return leftEAR, rightEAR
    
    def calculateEyeAspectRatio(self, eye):
        vertical_distance_1 = self.calculateEuclidianDistance2DPoints(eye[1], eye[5])
        vertical_distance_2 = self.calculateEuclidianDistance2DPoints(eye[2], eye[4])
        horizontal_distance = self.calculateEuclidianDistance2DPoints(eye[0], eye[3])
        # compute the eye aspect ratio
        ear = (vertical_distance_1 + vertical_distance_2) / (2.0 * horizontal_distance)
        return ear
    
    def calculateEuclidianDistance2DPoints(self, point_1, point_2):
        tmp_1 = point_1[0] - point_2[0]
        tmp_2 = point_1[1] - point_2[1]
        return math.sqrt(math.pow(tmp_1,2)+math.pow(tmp_2,2))


