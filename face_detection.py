from facenetSource import MTCNN
import torch


class FaceDetector(object):
    def __init__(self, keep_all=True, device= None):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        elif device == "cuda:0" and not torch.cuda.is_available() :
            self.device = "cpu"
        else:
            self.device = device
        self.keep_all = keep_all
        self.detector = MTCNN(keep_all=self.keep_all, device=self.device)
    
    def detectFace(self, frame):
        return self.detector.detect(frame)
