import pandas as pd
import datetime
import copy
import numpy as np
import cv2
import torch
from brain.yolov5.utils.general import scale_coords
import os
from itertools import permutations

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from manager.models import AfarmentDataDB, DetectionDB, Video
import json
import time

TEMP_FINISH_TIMER_MINUTES = 10
TEMP_FINISH_TIMER_SECONDS = 0
MIN_FRAME_DETECTION_AMOUNT = 10


classes ={
    0:{
        "name":"person",
    },
    1:{
        "name":"bicycle",
    },
    3:{
        "name":"motorcylcle",
    },
    2:{
        "name":"car",
    },
    5:{
        "name":"bus",
    },
    7:{
        "name":"truck",
    }           
}


     
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Detection:
    def __init__(self,
        id, track_id, class_id, first_bbox,
        image, input_zone
    ):
    
        self.id  = id
        self.track_id = track_id
        self.class_id = class_id
        
        self.first_bbox = first_bbox
        self.last_bbox = first_bbox
        self.dist_btw_bbox = 0
        
        self.first_image = image.copy()        
        self.last_image = image.copy()    
        self.input_zone = input_zone
        
        self.output_zone = -1
        self.frames_counter = 0        
        self.last_frame_detection_id = 0
        self.is_lost = False
        self.frames_counter_class={
            0:{
                "name":"person",
                "frames_detected":0
            },
            1:{
                "name":"bicycle",
                "frames_detected":0
            },
            3:{
                "name":"motorcylcle",
                "frames_detected":0
            },
            2:{
                "name":"car",
                "frames_detected":0
            },
            5:{
                "name":"bus",
                "frames_detected":0
            },
            7:{
                "name":"truck",
                "frames_detected":0
            }           
        }
    
        self.detection_time = datetime.datetime.now()
        self.last_detection_time = None     


class DetectionManager:
    def __init__(self,video, polys,ws):
        self.frame_ammount = None
        self.fps = None
        self.video = video
        self.image_path ="static/"+ str(video.owner)+"/"+str(video.id) +"/"       
        os.makedirs(self.image_path, exist_ok=False)
        self.ws = ws
        self.zoneconfig = polys
        self.global_counter=0
        self.global_frame = -1
        self.detections = []
        self.detections_dataframe = None
        self.count_timer = datetime.datetime.now()
        self.max_age = 6
        self.ref_frame = None
        self.orientation = None
        self.data_path = "data/"

    def calculate_afarment(self, video):
   
        zoneconfig = video.zone_set.all()
        zones=[poly.name for poly in zoneconfig]
        perm_zones = list(permutations(zones, 2))
        all_data = [

        ]
      
        for orientation in perm_zones:
            aux_data = {
                "orientation":orientation[0]+"-"+orientation[1],
                "detections":[ ]
            }
            for class_id in classes:
                ammount = len(video.detectiondb_set.filter(input_zone=orientation[0], output_zone=orientation[1], class_id=class_id))
                aux_data["detections"].append(
                    {
                        "class":classes[class_id]["name"],
                        "amount": ammount
                    }
                )
            all_data.append(aux_data)
        return all_data

     
    def obj_new(self,track_id,class_id, bbox,img,frame_idx):
        aux_detection = Detection(self.global_counter,track_id, class_id, bbox,img,self.object_detectable(bbox)["zone"])
        aux_detection.last_frame_detection_id = frame_idx
        self.global_counter += 1
        self.detections.append(aux_detection)
        
    def update_detection(self, detection, class_id, bbox, img, frame_idx):
        detection.frames_counter+=1
        detection.last_bbox=bbox                
        detection.last_image = img.copy()
        detection.last_detection_time = datetime.datetime.now()
        detection.last_frame_detection_id = frame_idx
        detection.frames_counter_class[class_id]["frames_detected"]+=1

    def ask_obj_exists(self, track_id):
        for detection in self.detections:
            if detection.track_id == track_id and not detection.is_lost:
                return True, detection                
        return False, None
        
    def bboxs_distance(self,detection):        
        bbox1 = detection.first_bbox        
        bbox2 = detection.last_bbox
        c1 = [(bbox1[0]+bbox1[2])/2, (bbox1[3]+bbox1[1])/2]
        c2 = [(bbox2[0]+bbox2[2])/2, (bbox2[1]+bbox2[3])/2]
        detection.dist_btw_bbox  = ((c1[0]-c2[0])**2+(c1[1]-c2[1])**2)**0.5

    def same_bbox_by_distance(self, detection):
        self.bboxs_distance( detection)       
        max_dist=((self.ref_frame.shape[0]*0.1)**2+(self.ref_frame.shape[1]*0.1)**2)**0.5

        if detection.dist_btw_bbox >max_dist:
            return False
        else:
            return True

    def set_obj_lost(self,detection):    

        max_frame_class = -1
        max_frame_class_idx = -1

        # Takes the predominant class detection as class id of the object
        for idx in detection.frames_counter_class.keys():
            if max_frame_class<detection.frames_counter_class[idx]["frames_detected"]:
                max_frame_class = detection.frames_counter_class[idx]["frames_detected"]
                max_frame_class_idx = idx

        if max_frame_class != -1:
            detection.class_id = max_frame_class_idx       
                
        detection.is_lost = True
        
        detection.output_zone = self.object_detectable(detection.last_bbox)["zone"]
        # SET ORIENTATION
        detection.orientation = detection.input_zone+detection.output_zone           
        if self.valid_detection_to_save(detection):
            self.save_detection_to_db(detection)
    
    def valid_detection_to_save(self,detection):     
        return (
            not self.same_bbox_by_distance(detection) 
            and detection.frames_counter > MIN_FRAME_DETECTION_AMOUNT
        )

    def save_detection_to_db(self,detection):
        self.save_detection(detection)
        DetectionDB(
            video = self.video,
            class_id = detection.class_id,
            last_bbox= list(detection.last_bbox),
            first_bbox = list(detection.first_bbox),
            input_zone = detection.input_zone ,
            output_zone = detection.output_zone,
            dist_btw_bbox = detection.dist_btw_bbox ,
            frames_counter = detection.frames_counter,
            last_frame_detection_id =  detection.last_frame_detection_id,
            detection_time = detection.detection_time,
            last_detection_time = detection.last_detection_time 
        ).save()

    def save_detection(self,detection):
        first_img_zones = self.draw_zones_on_image(detection.first_image.copy())
        class_id = detection.class_id
        p1, p2 = (int(detection.first_bbox[0]), int(detection.first_bbox[1])), (int(detection.first_bbox[2]), int(detection.first_bbox[3]))
        cv2.rectangle(first_img_zones, p1, p2, (0,255,0), 2, cv2.LINE_AA)  
        cv2.putText(
            first_img_zones,
            detection.frames_counter_class[class_id]["name"], 
            p1,0, 2, (0,0,255),thickness=3, 
            lineType=cv2.LINE_AA
        )
        first_img_zones = cv2.circle(first_img_zones,(int((p2[0]+p1[0])/2), int((p2[1]+p1[1])/2)), radius=5, color=(0, 0, 255), thickness=-1)
        
        path_file = f"{self.image_path}{detection.orientation}/"+classes[class_id]["name"]     

        os.makedirs(path_file, exist_ok=True)

        
        cv2.imwrite(f"{path_file}/{str(detection.id)}_in.jpg",first_img_zones)
        
        last_img_zones = self.draw_zones_on_image(detection.last_image)
        p1, p2 = (int(detection.last_bbox[0]), int(detection.last_bbox[1])), (int(detection.last_bbox[2]), int(detection.last_bbox[3]))
        cv2.rectangle(last_img_zones, p1, p2, (0,0,255), 2, cv2.LINE_AA) 
        cv2.putText(
            last_img_zones,
            detection.frames_counter_class[class_id]["name"], 
            p1,0, 2, (0,0,255),thickness=3, 
            lineType=cv2.LINE_AA
        )
        last_img_zones = cv2.circle(last_img_zones,(int((p2[0]+p1[0])/2), int((p2[1]+p1[1])/2)), radius=5, color=(0, 0, 255), thickness=-1)

        cv2.imwrite(f"{path_file}/{str(detection.id)}_out.jpg",last_img_zones)

        
    def draw_zones_on_image(self,img):
        for zone in self.zoneconfig:
            img= self.draw_poly(img,zone.poly)
        return img

    def draw_poly(self,img,actual_poly):
        # Initialize black image of same dimensions for drawing the rectangles
        blk = np.zeros(img.shape, np.uint8)
        # Draw rectangles
        cv2.fillPoly(blk,self.to_polyline(actual_poly),(0,255,255))     
        img = cv2.addWeighted(img, 1.0, blk, 0.25, 1)
        cv2.polylines( img,self.to_polyline(actual_poly),True,(0,255,255),2)
        return img

    def to_polyline(self, actual_poly):
        pts = np.array([ [k_points['x'],k_points['y']] for k_points in actual_poly], np.int32)
        pts = pts.reshape((-1,1,2))
        return [pts]        
     
    def filter_obj_lost(self, track_id, frame_idx, track_is_lost):
        '''
            filtering detections who are lost. 
        '''
        new_det = []
        for detection in self.detections:
            if (
                not detection.is_lost 
                and
                (
                    (
                        track_is_lost and                        
                        detection.track_id == track_id                  
                    )
                    or frame_idx-detection.last_frame_detection_id > self.max_age
                )
            ):
                self.set_obj_lost(detection)
            else:

                self.bboxs_distance( detection)
                new_det.append(detection)

        self.detections = new_det

    def filter_bbox_by_zones(self,preds,img,im0):
        filtered_preds=None
        bboxs=copy.deepcopy( preds[0][:,0:4])
            
        bboxs = scale_coords(img.shape[2:], bboxs, im0.shape).round()
        for i in range(len(bboxs)):
            bbox = bboxs[i]
            if self.object_detectable(bbox)["detectable"]: 
                if filtered_preds is None:
                    #numpy_filtered_bboxs = bbox
                    filtered_preds = torch.reshape(preds[0][i], (1, 6))
                else:                                    
                    pred = torch.reshape(preds[0][i], (1, 6))
                    filtered_preds = torch.cat((filtered_preds, pred),0)
 
        return [filtered_preds]
        
    def object_detectable(self, bbox):        
        point = [int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2)]      
        for zone_obj in self.zoneconfig:           
            zone = Polygon( [[poly_point["x"],poly_point["y"]] for poly_point in zone_obj.poly] )
            if zone.contains( Point( point)  ):                    
                return {"zone":zone_obj.name, "detectable":True}
       
        return {"zone":"NO ZONE", "detectable":False}

    def update(self, bbox, track_id, class_id, img, track_is_lost, frame_idx):  
        before = time.time()
        self.filter_obj_lost(track_id, frame_idx, track_is_lost)
        if track_is_lost:
            return

        obj_exists, detection = self.ask_obj_exists(track_id)
        if obj_exists:
            self.update_detection(detection, class_id, bbox, img, frame_idx)     
        else:
            self.obj_new(track_id, class_id, bbox, img, frame_idx)   
        print(f"takes {time.time()-before}")
   
    def send_status(
        self,
        sync_det=[]
    ):   
        
        self.global_frame+=1
        to_ws = {
            "type":"proccess",
            "message":{
                "progress":(self.global_frame/self.frame_ammount)*100,
                "global_frame":self.global_frame,
                "detections":list(sync_det),
                "data":self.calculate_afarment(self.video)
            },
            "username":"detector"
        }
        if self.video.status != self.video.PROCESSING:
            to_ws["type"] = "end"
        self.ws.send(json.dumps(to_ws, cls=NumpyEncoder))
        

