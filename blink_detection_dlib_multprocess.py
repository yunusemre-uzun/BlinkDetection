from multiprocessing import Process
import argparse
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from face_box import FaceBox
import cv2
import time
import dlib
import imutils

FACE_BOX = None
IS_REAL = False

def getVideoStream(args):
    if args["video"]=="":
        # Choose the camera as default
        vs = VideoStream(usePiCamera=True).start()
        fileStream = False
        print("[INFO] starting camera capturing")
        #vs = None
    else:
        # Choose the given video file
        vs = cv2.VideoCapture(args["video"])
        fileStream = True
        print("[INFO] starting video stream thread...")
    return vs, fileStream

def startVideoStream(vs, detector):
    success, frame = vs.read()
    face_box = None
    i = 0
    runtime_array = []
    face_detection_runtime_array = []
    is_real = False
    while success and not is_real:
        print("Frame: ", i)
        i+=1
        frame = imutils.resize(frame, width=450)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # detect faces in the rgb frame
        start = time.time()
        rects = detector(frame_rgb, 0)
        stop = time.time()
        face_detection_runtime_array.append(stop-start)
        frame_draw = frame.copy()
        for rect in rects:
            if face_box is None:
                face_box = FaceBox(None, frame, args["shape_predictor"], rect)
            else:
                face_box.updateFrame(frame)
                face_box.updateRect(None, rect)
            start = time.time()
            check_liveness = face_box.checkFrame() 
            stop = time.time()
            runtime_array.append(stop-start)
            if check_liveness :
                print("Real")
                is_real = True
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break  
        success, frame = vs.read()

    cv2.destroyAllWindows()
    avgCalculations(runtime_array, face_detection_runtime_array)


def startCameraSteam(vs, detector):
    i = 0
    runtime_array = []
    face_detection_runtime_array = []
    is_real = False
    time.sleep(2.0)
    while True:
        start = time.time()
        frame = getFrame(vs)
        p1 = Process(target=processFrame, args=(frame))
        time.sleep(0.05)
        frame = getFrame(vs)
        p2 = Process(target=processFrame, args=(frame))
        time.sleep(0.05)
        frame = getFrame(vs)
        p3 = Process(target=processFrame, args=(frame))
        p1.start()
        p2.start()
        p3.start()
        p1.join()
        p2.join()
        p3.join()
        end = time.time()
        print("Time elapsed: ", stop-start)
        cv2.imshow("Frame", frame_gray)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q") or is_real:
            break  

    cv2.destroyAllWindows()
    vs.stop()
    avgCalculations(runtime_array, face_detection_runtime_array)

def getFrame(vs):
    frame = vs.read()
    frame = imutils.resize(frame, width=250)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame_gray

def processFrame(frame):
    rects = detector(frame, 0)
    for rect in rects:
        if FACE_BOX is None:
            FACE_BOX = FaceBox(None, frame_gray, args["shape_predictor"], rect)
        else:
            FACE_BOX.updateFrame(frame_gray)
            FACE_BOX.updateRect(None, rect)
        check_liveness = FACE_BOX.checkFrame() 
        if check_liveness :
            print("Real")
            IS_REAL = True
    return None

def avgCalculations(runtime_array, face_detection_runtime_array):
    sum = 0
    for runtime in runtime_array:
        sum += runtime
    avg = sum / len(runtime_array)
    print ("Avg blink detection time:" , avg)
    sum = 0
    for runtime in face_detection_runtime_array:
        sum += runtime
    avg = sum / len(face_detection_runtime_array)
    print ("Avg face detection time:" , avg)


def main(args):
    print("[INFO] loading facial landmark predictor...")
    # Load mtcnn detector from facenet
    detector = dlib.get_frontal_face_detector()
    vs, file_stream = getVideoStream(args)
    # Read the first frame
    if(file_stream):
        startVideoStream(vs, detector)
    else:
        startCameraSteam(vs, detector)
    

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
    ap.add_argument("-v", "--video", type=str, default="",
        help="path to input video file")
    args = vars(ap.parse_args())
    main(args)
    
