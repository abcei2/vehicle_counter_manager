import subprocess
import cv2
import json
import time
from brain.track import detect
from manager.models import Video
from django.core.management.base import BaseCommand



class Command(BaseCommand):
    help = 'Displays current time'

    def handle(self, *args, **kwargs):
        video = Video.objects.get(pk=12)
        
        if video:
            video_url = video.video_link.path
            cap = cv2.VideoCapture(video_url)           
            
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) )  # float `width`
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) ) # float `height` 
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*'XVID')


            out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))
            
            frame_idx = 0
            while cap.isOpened():         
                res, frame = cap.read()   
                if res:
                    frame_detections = video.frames.filter(frame_idx=frame_idx) 
                
                    for detection in frame_detections:
                        bbox = detection.bbox
                        p1, p2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])

                        center = (int((bbox[0]+bbox[2])/2),int((bbox[1] + bbox[3])/2))
                        
                        cv2.rectangle(frame,  p1, p2, (100,0,0), 2, cv2.LINE_AA)  
                        cv2.putText(
                            frame,
                            f"class: {detection.detection.class_id}, obj:{detection.detection.pk}", 
                            center,0, 1, (0,0,255),thickness=2, 
                            lineType=cv2.LINE_AA
                        )
                    frame_idx = frame_idx + 1
                else:
                    break
                out.write(frame) 
        
        response  = Response("-")
        response.status_code = 200
        response.content = json.dumps("s")
        return response