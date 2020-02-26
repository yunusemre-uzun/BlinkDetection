from multiprocessing import Process, Value, Manager
from multiprocessing.managers import BaseManager
import argparse
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from face_box import FaceBox
import cv2
import time
import dlib
import imutils

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

def getFrame(vs, camera_stream):
    if not camera_stream:
        success, frame = vs.read()
        if not success:
            return None
    else:
        frame = vs.read()
    frame = imutils.resize(frame, width=300)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame_gray

def startVideoStream(vs, detector):
    frame_counter = 0
    # Read the first frame
    frame = getFrame(vs, False)
    is_real = Value('c', 0)
    manager = BaseManager()
    manager.start()
    face_box = manager.FaceBox(dummy=True)
    process_array = []
    #Initialize dlib's shape predictor to decide landmarks on face
    shape_predictor = dlib.shape_predictor(args["shape_predictor"])
    start = time.time()
    while frame is not None:
        # We have 4 core system, so create max 3 processes
        frame = getFrame(vs, False)
        frame_counter += 1
        # Create a new os process to process frame
        if len(process_array) == 3:
            process_array[0].join()
            del process_array[0]
        new_process = Process(target=processFrame, args=(frame, detector, shape_predictor, frame_counter, is_real,face_box,))
        new_process.start()
        process_array.append(new_process)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break 
        with is_real.get_lock():
            if is_real == 1:
                break
    end = time.time()
    print("Time elapsed: ", end-start)
    print("Frames processed: ", frame_counter)

    cv2.destroyAllWindows()
    #avgCalculations(runtime_array, face_detection_runtime_array)

def processFrame(frame_gray, detector, shape_predictor, frame_counter, is_real, face_box):
    rects = detector(frame_gray, 0)
    for rect in rects:
        if face_box.isDummy():
            face_box.changeState(None, frame_gray, shape_predictor, rect)
        else:
            face_box.updateFrame(frame_gray)
            face_box.updateRect(None, rect)
        check_liveness = face_box.checkFrame() 
        if check_liveness :
            print("Real")
        with is_real.get_lock():
            is_real = 1
    cv2.imshow("Frame", frame_gray)
    return False

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
    BaseManager.register('FaceBox', FaceBox)
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
    ap.add_argument("-v", "--video", type=str, default="",
        help="path to input video file")
    args = vars(ap.parse_args())
    main(args)
    
