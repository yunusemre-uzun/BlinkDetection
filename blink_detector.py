from imutils import face_utils
from paramaters import *
import math
import cv2

class BlinkDetector(object):
    __instance = None
    __registered_face_boxes = []

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

    @staticmethod
    def registerBox(face_box):
        BlinkDetector.__registered_face_boxes.append([face_box])
        return len(BlinkDetector.__registered_face_boxes) - 1
    
    @staticmethod
    def deRegisterBox(id):
        BlinkDetector.__registered_face_boxes.pop(id)
        return None

    @staticmethod
    def getEyesStatus(id, frame):
        shape = BlinkDetector.getFaceShape(id)
        return BlinkDetector.calculateEyesStatus(shape, frame)
    
    @staticmethod
    def calculateEyesStatus(shape, frame):
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        leftEAR = BlinkDetector.calculateEyeAspectRatio(left_eye)
        rightEAR = BlinkDetector.calculateEyeAspectRatio(right_eye)
        #print("Left: ", leftEAR, "-Right: ", rightEAR)
        if leftEAR > EYE_AR_THRESH:
            left_eye_open = True
        else:
            left_eye_open = False
        if rightEAR > EYE_AR_THRESH:
            right_eye_open = True
        else:
            right_eye_open = False
        return left_eye_open, right_eye_open

    @staticmethod
    def detect(id, frame):
        shape = BlinkDetector.getFaceShape(id)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        left_ear, right_ear = BlinkDetector.getEyeAspectRatio(shape, frame)
        # If eye is closed return True
        if left_ear < EYE_AR_THRESH and right_ear < EYE_AR_THRESH:   
           return 2
        elif left_ear > EYE_AR_THRESH and right_ear > EYE_AR_THRESH:
            return 1
        return 0

    @staticmethod
    def getFaceShape(id):
        face_box = BlinkDetector.get_face_box(id)
        frame_rgb = face_box[0].frame
        rect = face_box[0].rect
        predictor = face_box[0].shape_predictor
        shape = predictor(frame_rgb, rect)
        shape = face_utils.shape_to_np(shape)
        return shape

    
    @staticmethod
    def get_face_box(id):
        return BlinkDetector.__registered_face_boxes[id]
    
    @staticmethod
    def getEyeAspectRatio(shape, frame):
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        leftEyeHull = cv2.convexHull(left_eye)
        rightEyeHull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 255), 1)
        leftEAR = BlinkDetector.calculateEyeAspectRatio(left_eye)
        rightEAR = BlinkDetector.calculateEyeAspectRatio(right_eye)
        return leftEAR, rightEAR
    
    @staticmethod
    def calculateEyeAspectRatio(eye):
        vertical_distance_1 = BlinkDetector.calculateEuclidianDistance2DPoints(eye[1], eye[5])
        vertical_distance_2 = BlinkDetector.calculateEuclidianDistance2DPoints(eye[2], eye[4])
        horizontal_distance = BlinkDetector.calculateEuclidianDistance2DPoints(eye[0], eye[3])
        # compute the eye aspect ratio
        ear = (vertical_distance_1 + vertical_distance_2) / (2.0 * horizontal_distance)
        return ear
    
    def calculateEuclidianDistance2DPoints(point_1, point_2):
        tmp_1 = point_1[0] - point_2[0]
        tmp_2 = point_1[1] - point_2[1]
        return math.sqrt(math.pow(tmp_1,2)+math.pow(tmp_2,2))


