import argparse
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from face_box import FaceBox
import cv2
import time
import dlib
import imutils

class Driver(object):
    __environment = None
    __file_stream = None
    __detector = None
    __shape_predictor = None
    __vs = None
    __face_box = None
    __is_real = False

    def __init__(self, shape_predictor, video_path="", environment="pc"):
        print("[INFO] loading facial landmark predictor...")
        self.__shape_predictor = dlib.shape_predictor(args["shape_predictor"])
        self.__face_box = FaceBox(shape_predictor=self.__shape_predictor)
        self.__environment = environment
        self.__video_path = video_path

    def getVideoStream(self):
        '''
            Creates VideoStream object, and returns it. 
        '''
        if self.__video_path=="":
            if Driver.__environment == "pi":
                # Choose raspberry pi cam
                vs = VideoStream(usePiCamera=True, framerate=20, resolution=(480,480)).start()
            else:
                # Choose webcam
                vs = VideoStream(0, resolution=(1920,1080), framerate=60).start()
            self.__file_tream = False
            print("[INFO] starting camera capturing")
            # Sleep to let camera to warm up
            time.sleep(2.0)
        else:
            # Choose the given video file
            vs = cv2.VideoCapture(self.__video_path)
            self.__file_stream = True
            print("[INFO] starting video stream thread...")
        return vs
    
    def getFrame(self):
        ''' 
            Read the next frame from the camera, returns the gray scale image with width 250 pixels
            :param vs: VideoStream object from imutils, camera_stream: True if the camera is being used
        '''
        if self.__file_stream :
            success, frame = self.__vs.read()
            if not success:
                return None
        else:
            frame = self.__vs.read()
        if Driver.__environment == "pi":
            frame = imutils.resize(frame, width=300)
        else:
            frame = imutils.resize(frame, width=600)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame_gray

    def startDetection(self):
        start = time.time()
        frame_counter = self.detectionLoop()
        end = time.time()
        print("FPS: ", frame_counter/(end-start))
        print("Time elapsed: ", end-start)
        #print("Frames processed: ", frame_counter)
        cv2.destroyAllWindows()

    def detectionLoop(self):
        frame = self.getFrame()
        frame_counter = 0
        while frame is not None:
            print("Frame read: ", frame_counter)
            self.processFrame(frame, rects=None)
            cv2.imshow("Frame", frame)
            frame_counter += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break 
            if self.__is_real:
                print("Real")
                self.__face_box.deRegister()
                #time.sleep(0.5)
                self.__is_real = False
            frame = self.getFrame()
        return frame_counter
    
    def processFrame(self, frame_gray, rects=None):
        '''
            This function searches for liveness clues in the frame. It can be used standalone with gray scale image.
            If :param: rects is given, the function does not search for faces again.
            Returns true when the face box regarded as real else return false.
                Warning: The Driver can only track 1 face at the same time. 
                    The behaviour can be undeterministic when more faces exist.
                Warning: After this function returned True, the face box must be deRegistered.
        '''
        if rects is None:
            rects = self.__detector(frame_gray, 0)
        # When a face detected, but the face box is not registered in blink detection engine
        # Happens in when a face is detected after a successful detection.
        if len(rects) and not self.__face_box.registered:
            self.__face_box.register()
        for rect in rects:
            self.__face_box.updateFrame(frame_gray)
            self.__face_box.updateRect(None, rect)
            check_liveness = self.__face_box.checkFrame()
            if check_liveness :
                self.__is_real = True 
        return False

    def run(self):
        print("[INFO] loading face detector...")
        self.__detector = dlib.get_frontal_face_detector()
        self.__vs = self.getVideoStream()
        self.startDetection()
        

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
    ap.add_argument("-v", "--video", type=str, default="",
        help="path to input video file")
    ap.add_argument("-e", "--env", type=str, default="pc",
        help="environment type pi or pc")
    args = vars(ap.parse_args())
    driver = Driver(args["shape_predictor"], args["video"], args["env"])
    driver.run()
    
