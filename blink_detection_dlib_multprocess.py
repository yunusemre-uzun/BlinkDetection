from multiprocessing import Process, Value, Manager, Queue
from multiprocessing.managers import BaseManager
import argparse
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from face_box import FaceBox
import cv2
import time
import dlib
import imutils

class Driver(object):
    __number_of_processes = None
    __frame_queue = None

    @staticmethod
    def getVideoStream(args):
        '''
            Creates VideoStream object, returns object and stream flag as True if file stream enabled. 
            :param args: args of the main program
        '''
        if args["video"]=="":
            if args["env"] == "pi":
                # Choose raspberry pi cam
                vs = VideoStream(usePiCamera=True).start()
            else:
                vs = VideoStream(0).start()
            fileStream = False
            print("[INFO] starting camera capturing")
        else:
            # Choose the given video file
            vs = cv2.VideoCapture(args["video"])
            fileStream = True
            print("[INFO] starting video stream thread...")
        return vs, fileStream
    
    @staticmethod
    def getFrame(vs, camera_stream):
        ''' 
            Read the next frame from the camera, returns the gray scale image with width 250 pixels
            :param vs: VideoStream object from imutils, camera_stream: True if the camera is being used
        '''
        if not camera_stream:
            success, frame = vs.read()
            if not success:
                return None
        else:
            frame = vs.read()
        frame = imutils.resize(frame, width=250)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame_gray

    @staticmethod
    def startVideoDetection(vs, detector):
        # Create a shared object between processes
        is_real = Value('i', False)
        # Create manager to share FaceBox class between processes
        manager = BaseManager()
        manager.start()
        # Read the first frame and push it to the shared queue
        frame = getFrame(vs, False)
        frame_queue = Queue()
        frame_queue.put(frame)
        #Initialize dlib's shape predictor to decide landmarks on face
        shape_predictor = dlib.shape_predictor(args["shape_predictor"])
        #Initilize FaceBox object with manager
        face_box = manager.FaceBox(shape_predictor=shape_predictor, frame=frame)
        #Create n number of processes
        proc_list = Driver.createProcesses(detector, shape_predictor, is_real, face_box)
        start = time.time()
        frame_number = Driver.detectionLoop(frame, is_real, frame_queue, vs, False)
        end = time.time()
        print("FPS: ", frame_counter/(end-start))
        Driver.waitForProcesses(proc_list)
        print("Time elapsed: ", end-start)
        #print("Frames processed: ", frame_counter)
        cv2.destroyAllWindows()

    @staticmethod
    def startCameraDetection(vs, detector):
        # Create a shared object between processes
        is_real = Value('i', False)
        # Create manager to share FaceBox class between processes
        manager = BaseManager()
        manager.start()
        time.sleep(2.0)
        # Read the first frame and push it to the shared queue
        frame = Driver.getFrame(vs, True)
        Driver.__frame_queue = Queue()
        Driver.__frame_queue.put(frame)
        #Initialize dlib's shape predictor to decide landmarks on face
        shape_predictor = dlib.shape_predictor(args["shape_predictor"])
        #Initilize FaceBox object with manager
        face_box = manager.FaceBox(shape_predictor=shape_predictor, frame=frame)
        #Create n number of processes
        proc_list = Driver.createProcesses(detector, shape_predictor, is_real, face_box)
        start = time.time()
        frame_number = Driver.detectionLoop(frame, is_real, vs, True)
        end = time.time()
        print("FPS: ", frame_number/(end-start))
        Driver.waitForProcesses(proc_list)
        print("Time elapsed: ", end-start)
        #print("Frames processed: ", frame_counter)
        cv2.destroyAllWindows()

    @staticmethod
    def detectionLoop(frame, is_real, vs, camera_stream):
        frame_counter = 0
        while frame is not None:
            print("Frame read: ", frame_counter)
            cv2.imshow("Frame", frame)
            Driver.__frame_queue.put(frame)
            while Driver.__frame_queue.qsize() > 3:
                print("Bottleneck detected")
                time.sleep(0.05)
            frame_counter += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break 
            with is_real.get_lock():
                if is_real.value:
                    print("Real")
                    break
            frame = Driver.getFrame(vs, camera_stream)
            #time.sleep(0.05)
        return frame_counter
    
    @staticmethod
    def waitForFrame(detector, shape_predictor, is_real, face_box, frame_queue,process_id):
        while True:
            try:
                frame = frame_queue.get(block=True, timeout=1)
                print("Process{} got frame".format(process_id))
            except:
                return None
            Driver.processFrame(frame, detector, shape_predictor, is_real, face_box, process_id)

    @staticmethod
    def processFrame(frame_gray, detector, shape_predictor, is_real, face_box, id):
        rects = detector(frame_gray, 0)
        for rect in rects:
            face_box.updateFrame(frame_gray)
            face_box.updateRect(None, rect)
            check_liveness = face_box.checkFrame()
            if check_liveness :
                with is_real.get_lock():
                    is_real.value = True 
        return False

    @staticmethod
    def createProcesses(detector, shape_predictor, is_real, face_box):
        ''' 
        Creates processes to process the frames, returns list of processes
        '''
        proc_list = []
        for i in range(Driver.__number_of_processes):
            temp_proc = Process(target=Driver.waitForFrame, args=(detector, shape_predictor, is_real,face_box, 
                Driver.__frame_queue,i,))
            temp_proc.start()
            proc_list.append(temp_proc)
        return proc_list

    @staticmethod
    def waitForProcesses(proc_list):
        for proc in proc_list:
            proc.join()
        return True    

    @staticmethod
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

    @staticmethod
    def run(args):
        print("[INFO] loading facial landmark predictor...")
        # Load mtcnn detector from facenet
        detector = dlib.get_frontal_face_detector()
        vs, file_stream = Driver.getVideoStream(args)
        Driver.__number_of_processes = int(args["proc"])
        # Read the first frame
        if(file_stream):
            Driver.startVideoDetection(vs, detector)
        else:
            Driver.startCameraDetection(vs, detector)
        

if __name__ == "__main__":
    BaseManager.register('FaceBox', FaceBox)
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
    ap.add_argument("-v", "--video", type=str, default="",
        help="path to input video file")
    ap.add_argument("-e", "--env", type=str, default="pc",
        help="path to input video file")
    ap.add_argument("-p", "--proc", type=str, default="1",
        help="path to input video file")
    args = vars(ap.parse_args())
    Driver.run(args)
    
