import argparse
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from face_detection import FaceDetector
from face_box import FaceBox
import cv2
import time

def getVideoStream(args):
    if args["video"] is None:
        # Choose the camera as default
        vs = VideoStream(0)
        fileStream = False
        print("[INFO] starting video stream thread...")
    else:
        # Choose the given video file
        vs = cv2.VideoCapture(args["video"])
        fileStream = True
        print("[INFO] startign camera capturing")
    return vs, fileStream

def main(args):
    print("[INFO] loading facial landmark predictor...")
    # Load mtcnn detector from facenet
    face_detect = FaceDetector(True, None)
    vs, file_stream = getVideoStream(args)
    # Read the first frame
    success, frame = vs.read()
    face_box = None
    i = 0
    runtime_array = []
    face_detection_runtime_array = []
    while success:
        print("Frame: ", i)
        i+=1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # detect faces in the rgb frame
        start = time.time()
        boxes, _ = face_detect.detectFace(frame_rgb)
        stop = time.time()
        face_detection_runtime_array.append(stop-start)
        frame_draw = frame.copy()
        if boxes is not None:
            if face_box is None:
                face_box = FaceBox(boxes[0], frame, args["shape_predictor"])
            else:
                face_box.updateFrame(frame)
                face_box.updateRect(boxes[0])
            start = time.time()
            check_liveness = face_box.checkFrame() 
            stop = time.time()
            runtime_array.append(stop-start)
            if check_liveness :
                print("Real")
                break
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break  
        success, frame = vs.read()
    cv2.destroyAllWindows()
    if not file_stream:
        vs.stop()
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
    ap.add_argument("-v", "--video", type=str, default="",
        help="path to input video file")
    args = vars(ap.parse_args())
    main(args)
    
