from celery import shared_task
import time
import subprocess
import cv2
import json
import time
from brain.track import detect
from brain.own_work.detection_manager import DetectionManager
import argparse
import torch

class Options:
    def __init__(self):
        self.yolo_weights = './brain/yolov5l.pt'
        self.deep_sort_weights = './brain/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
        self.source="./brain/hiv00177.mp4"
        self.imgsz = [640, 640]
        self.conf_thres = 0.4
        self.iou_thres = 0.5
        self.half= False
        self.classes = [1, 2, 3, 5, 7]
        self.device = ''
        self.agnostic_nms = True
        self.augment = False
        self.evaluate = False
        self.config_deepsort = './brain/deep_sort_pytorch/configs/deep_sort.yaml'
        self.max_det = 1000
        self.dnn = False

@shared_task
def adding_task(x, y):
  
    print("?")
    detection_manager = DetectionManager()
    with torch.no_grad():
        detect(Options(), detection_manager)
    return "finish"