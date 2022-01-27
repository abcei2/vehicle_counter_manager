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
from manager.models import Video, DetectionManager, ZonePoly

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
      
    video=Video(owner_name="santi",video_link="./brain/hiv00177.mp4")
    video.save()
    detection_manager = DetectionManager(video=video)
    detection_manager.save()
    ZonePoly(detection_manager=detection_manager,name="north",poly=[[717, 1001], [670, 496], [1576, 438], [1924, 875]]).save()
    ZonePoly(detection_manager=detection_manager,name="south",poly=[[681, 23],[684, 222], [1362, 176], [1073, 11]]).save()
    ZonePoly(detection_manager=detection_manager,name="west",poly=[[1407, 223], [1751, 555], [1905, 505], [1893, 246]]).save()
    ZonePoly(detection_manager=detection_manager,name="east",poly=[[14, 227], [26, 688],  [614, 565],[602, 160]]).save()
    ZonePoly(detection_manager=detection_manager,name="center",poly=[[598, 244],[596, 521], [1600, 460],[1337, 180]]).save()
    with torch.no_grad():
        detect(Options(), detection_manager)
    return "finish"