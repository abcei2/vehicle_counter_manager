from celery import shared_task
import time
import subprocess
import json
import time
import argparse
from manager.models import Video

import websocket
import _thread


DETECT = True
if DETECT:
    import cv2
    from brain.track import detect
    from brain.own_work.detection_manager import DetectionManager
    import torch

class Options:
    def __init__(self, source):
        self.yolo_weights = './brain/yolov5l.pt'
        self.deep_sort_weights = './brain/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
        self.source= source
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

    def __init__(self,username,video,polys):
        websocket.enableTrace(False)
        ws = websocket.WebSocketApp(f"ws://localhost:8000/ws/chat/video/",
                            on_open=self.on_open,
                            on_message=self.on_message,
                            on_error=self.on_error,
                            on_close=self.on_close)     
        self.video = video
        self.polys = polys
        self.max = 100
        self.counter = 0
        ws.run_forever()

    def on_message(self, ws, message):
        pass

    def on_error(self, ws, error):
        print(error)

    def on_close(self, ws, close_status_code, close_msg):
        print("### closed ###")

    def on_open(self, ws):
        def run(*args):

            self.video.status = self.video.PROCESSING
            self.video.save()

            if DETECT:
                detection_manager = DetectionManager(self.video,self.polys,ws)
                with torch.no_grad():
                    detect(Options(self.video.video_link.path), detection_manager)
            else:
                while(self.counter<self.max):
                    self.counter+=1
                    to_ws = {
                        "type":"proccess",
                        "message":(self.counter/self.max)*100,
                        "username":"detector"
                    }
                    ws.send(json.dumps(to_ws))
                    time.sleep(0.05)
            ws.close()
        _thread.start_new_thread(run, ())

@shared_task
def video_to_queue(video_pk):
    video = Video.objects.get(pk=video_pk)
    polys = video.zone_set.all()

    Websockets(video.owner.username,video,polys)
    video.status = video.FINISHED
    video.save()
    return "finish"