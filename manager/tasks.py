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
from manager.models import Video

import websocket
import _thread

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



class Websockets:

    def __init__(self):
        websocket.enableTrace(True)
        ws = websocket.WebSocketApp("ws://192.168.1.1:8000/ws/chat/python/",
                            on_open=self.on_open,
                            on_message=self.on_message,
                            on_error=self.on_error,
                            on_close=self.on_close)        
        ws.run_forever()

    def on_message(self, ws, message):
        pass

    def on_error(self, ws, error):
        print(error)

    def on_close(self, ws, close_status_code, close_msg):
        print("### closed ###")

    def on_open(self, ws):
        def run(*args):
            video=Video(owner_name="santi",video_link="./brain/hiv00177.mp4")
            video.save()
            polys={
                "north":[[717, 1001], [670, 496], [1576, 438], [1924, 875]],
                "south":[[681, 23],[684, 222], [1362, 176], [1073, 11]],
                "west":[[1407, 223], [1751, 555], [1905, 505], [1893, 246]],
                "east":[[14, 227], [26, 688],  [614, 565],[602, 160]],
                "center":[[598, 244],[596, 521], [1600, 460],[1337, 180]]
            }
            detection_manager = DetectionManager(video,polys,ws)
            with torch.no_grad():
                detect(Options(), detection_manager)
            time.sleep(1)
            ws.close()
        _thread.start_new_thread(run, ())

@shared_task
def adding_task(x, y):
  
    Websockets()
    return "finish"