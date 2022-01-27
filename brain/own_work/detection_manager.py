import pandas as pd
import datetime
import cv2
import copy
import numpy as np
import torch
from .zone_manager import ZoneConfig
from brain.yolov5.utils.general import scale_coords
import os
from itertools import permutations

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from manager.models import AfarmentDataDB, DetectionDB

#YOLOV5s

# STRUCT_DATA =   {
#     'track_id': None, 
#     'class_id': None, 
#     'score': None, 
#     'first_bbox': None, 
#     'last_bbox': None,
#     'input_zone': None,
#     'output_zone':None,
#     'frames_counter':None,
#     'is_lost':None,
#     'person':0,
#     'bicycle':0,
#     'car':0,
#     'motorcycle':0,
#     'bus':0,
#     'truck':0,    
#     'detection_time':None,
    
#     'last_detection_time':None
#     }

TEMP_FINISH_TIMER_MINUTES = 15
TEMP_FINISH_TIMER_SECONDS = 0

zones =[        
    "north_poly",
    "south_poly",
    "west_poly",
    "east_poly",      
    "center_poly"
]  

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

class AfarmentData: 
    def __init__(self, class_id, name, maneuver, ammount):
        self.class_id = class_id
        self.name = name
        self.maneuver = maneuver
        self.ammount = ammount
        
class AfarmentDataManager: 
       
    def __init__(self, detections,count_timer, video, zoneconfig):


        dets_dict = []
        for detection in detections:    
        
            det_dict = copy.deepcopy(detection.__dict__)
         
            det_dict.pop("first_image")
            det_dict.pop("last_image")
            dets_dict.append(det_dict)
        zones=[i for i in zoneconfig.polys.keys()]
        detections_df = pd.DataFrame(dets_dict)
        perm_zones = list(permutations(zones, 2))

        afarment_data_list=[]
        for orientation in perm_zones:     
            for idx in classes.keys():
                ammount = len(detections_df.loc[(detections_df['class_id'] == idx) & (detections_df['input_zone'] == orientation) & (detections_df['output_zone'] == orientation)])
                afarment_data_list.append(AfarmentData(idx,classes[idx]["name"],orientation+orientation,ammount))
                   
        for afarment_data in afarment_data_list:            
            AfarmentDataDB(
                video = video,
                class_id = afarment_data.class_id,
                ammount = afarment_data.ammount, 
                maneuver = afarment_data.maneuver,
                class_name = afarment_data.name
            ).save()
            
        for detection in detections:    
            DetectionDB(
            video = video,
            class_id = detection.class_id,
            last_bbox= list(detection.last_bbox),
            first_bbox = list(detection.first_bbox),
            input_zone = detection.input_zone ,
            output_zone = detection.output_zone,
            dist_btw_bbox = detection.dist_btw_bbox ,
            frames_counter = detection.frames_counter,
            last_frame_detection_id =  detection.last_frame_detection_id,
            detection_time = detection.detection_time,
            last_detection_time = detection.last_detection_time ).save()


class DetectionManager:
    def __init__(self,video, polys):
        self.video = video
        self.zoneconfig = ZoneConfig(polys)    
        self.global_counter=0
        self.zones = []
        self.detections = []
        self.detections_dataframe = None
        self.count_timer = datetime.datetime.now()
        self.max_age = 6
        self.ref_frame = None
        self.orientation = None
        self.data_path = "data/"
        # perm_zones = list(permutations(zones, 2))
        # for idx in classes.keys():
        #     for orientation in perm_zones:
                
        #         os.makedirs(f"{self.data_path}{orientation[0][0]}{orientation[1][0]}/"+classes[idx]["name"], exist_ok=True)
            

    def obj_new(self,track_id,class_id, bbox,img,frame_idx):
        self.object_detectable(bbox)
        aux_detection = Detection(self.global_counter,track_id, class_id, bbox,img,self.object_detectable(bbox)["zone"])
        aux_detection.last_frame_detection_id = frame_idx
        self.global_counter += 1
        self.detections.append(aux_detection)
        

    def ask_obj_exists(self,track_id,class_id, bbox, img,frame_idx):
        for detection in self.detections:
            if detection.track_id == track_id and not detection.is_lost:
                detection.frames_counter+=1
                detection.last_bbox=bbox                
                detection.last_image = img.copy()
                detection.last_detection_time = datetime.datetime.now()
                detection.last_frame_detection_id = frame_idx
                detection.frames_counter_class[class_id]["frames_detected"]+=1
                return True                
        return False
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
        if detection.is_lost:
            return
        max_frame_class = -1
        max_frame_class_idx = -1
        for idx in detection.frames_counter_class.keys():
            if max_frame_class<detection.frames_counter_class[idx]["frames_detected"]:
                max_frame_class = detection.frames_counter_class[idx]["frames_detected"]
                max_frame_class_idx = idx

        if max_frame_class != -1:
            detection.class_id = max_frame_class_idx
        

       
            
        detection.is_lost=True
        
        detection.output_zone = self.object_detectable(detection.last_bbox)["zone"]
        # SET ORIENTATION
        detection.orientation = detection.input_zone+detection.output_zone
        print(detection.orientation)
        # # IN    
        # first_img_zones = self.zoneconfig.draw_zones(detection.first_image)
        # p1, p2 = (int(first_bbox[0]), int(first_bbox[1])), (int(first_bbox[2]), int(first_bbox[3]))
        # cv2.rectangle(first_img_zones, p1, p2, (0,255,0), 2, cv2.LINE_AA)                 
        # cv2.putText(
        #     first_img_zones,
        #     detection.frames_counter_class[class_id]["name"], 
        #     p1,0, 2, (0,0,255),thickness=3, 
        #     lineType=cv2.LINE_AA
        # )
        # first_img_zones = cv2.circle(first_img_zones,(int((p2[0]+p1[0])/2), int((p2[1]+p1[1])/2)), radius=5, color=(0, 0, 255), thickness=-1)
        # cv2.imwrite(f"{self.data_path}{detection.orientation}/"+classes[class_id]["name"]+"/"+str(detection.id)+"_in.jpg",first_img_zones)
        # # OUT     
        # last_img_zones = self.zoneconfig.draw_zones(detection.last_image)
        # p1, p2 = (int(last_bbox[0]), int(last_bbox[1])), (int(last_bbox[2]), int(last_bbox[3]))
        # cv2.rectangle(last_img_zones, p1, p2, (0,0,255), 2, cv2.LINE_AA) 
        # cv2.putText(
        #     last_img_zones,
        #     detection.frames_counter_class[detection.class_id]["name"], 
        #     p1,0, 2, (0,0,255),thickness=3, 
        #     lineType=cv2.LINE_AA
        # )
        # last_img_zones = cv2.circle(last_img_zones,(int((p2[0]+p1[0])/2), int((p2[1]+p1[1])/2)), radius=5, color=(0, 0, 255), thickness=-1)
      
        # cv2.imwrite(f"{self.data_path}{detection.orientation}/"+classes[class_id]["name"]+"/"+str(detection.id)+"_out.jpg",last_img_zones)

    def obj_lost(self,track_id,class_id):
        new_det = []
        for detection in self.detections:
            if detection.track_id == track_id and not detection.is_lost:  
         
               
                if self.same_bbox_by_distance(detection):                    
                    continue 
                new_det.append(detection)
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
        for key_poly in self.zoneconfig.polys:           
            zone = Polygon( [self.zoneconfig.polys[key_poly][0], self.zoneconfig.polys[key_poly][1], self.zoneconfig.polys[key_poly][2], self.zoneconfig.polys[key_poly][3]] )
            if zone.contains( Point( point)  ):                    
                return {"zone":key_poly, "detectable":True}
       
        return {"zone":"NO ZONE", "detectable":False}

    def update(self,bbox,track_id,class_id,img,is_lost,frame_idx):    

        if len(self.detections)>0 and self.count_timer + datetime.timedelta( minutes = TEMP_FINISH_TIMER_MINUTES, seconds=TEMP_FINISH_TIMER_SECONDS) < datetime.datetime.now():
            self.count_timer = datetime.datetime.now()
            new_det = []
            for detection in self.detections:
                if not detection.is_lost and frame_idx-detection.last_frame_detection_id > self.max_age:
                    self.set_obj_lost(detection)
                if self.same_bbox_by_distance(detection):                    
                    continue 
                new_det.append(detection)
         
            AfarmentDataManager(new_det,self.count_timer,self.video,self.zoneconfig)
            self.detections=[]

        if is_lost:
            self.obj_lost(track_id,class_id)
            return     

        if self.ask_obj_exists(track_id,class_id,bbox,img,frame_idx):
            return

        self.obj_new(track_id,class_id,bbox,img,frame_idx)
   

