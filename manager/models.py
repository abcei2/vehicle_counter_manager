from django.db import models
from django.contrib.postgres.fields import ArrayField

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from brain.yolov5.utils.general import scale_coords
import datetime


import copy
import torch
import time
# Create your models here.
        
TEMP_FINISH_TIMER_MINUTES = 15
TEMP_FINISH_TIMER_SECONDS = 0
DEFAULT_POLY = [
    [50 ,50],
    [50 ,100],            
    [100 ,100],
    [100 ,50]
]
DEFAULT_FRAMES_COUNTER_CLASS={
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

def return_default_fields():
    return DEFAULT_FRAMES_COUNTER_CLASS
class Video(models.Model):

    owner_name = models.CharField(max_length=64)
    video_link = models.FileField(db_index=True, upload_to='not_used')
    
  
        
class DetectionManager(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    count_timer = models.TimeField(auto_now=True)
    max_age = models.IntegerField(default=6)
    video_shape = ArrayField(
        models.IntegerField(),
        size = 4,
        null= True
    )

    def object_detectable(self, bbox): 
        '''
            If object is detectable in some zone
        '''   
        point=([int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2)])        
        for zone_poly in self.zonepoly_set.all():
            zone = Polygon( [points for points in zone_poly.poly] )
            if zone.contains( Point(point)  ):                
                return {"zone":zone_poly, "detectable":True}

        return {"zone":None, "detectable":False}

    def obj_new(self,track_id,class_id, bbox,img,frame_idx):
        '''
            Add new object if is detectable in some zone.
        '''  
        detectable = self.object_detectable(bbox)
        if detectable["detectable"]:
            Detection(
                detection_manager=self,
                input_zone=detectable["zone"],
                track_id=track_id,
                class_id=class_id,
                first_bbox=list(bbox),
                last_bbox=list(bbox),
                last_frame_detection_id=frame_idx
            ).save()

    def filter_bbox_by_zones(self,preds,img,im0):
        '''
            Filter predictions of main model who not are on some zone
        '''

        filtered_preds = None
        bboxs = copy.deepcopy( preds[0][:,0:4])
            
        bboxs = scale_coords(img.shape[2:], bboxs, im0.shape).round()
        
        for i in range(len(bboxs)):
            bbox = bboxs[i]
            if self.object_detectable(bbox)["detectable"]: 
                if filtered_preds is None:
                    filtered_preds = torch.reshape(preds[0][i], (1, 6))
                else:                                    
                    pred = torch.reshape(preds[0][i], (1, 6))
                    filtered_preds = torch.cat((filtered_preds, pred),0)
        return [filtered_preds]

        
    def ask_obj_exists(self,track_id,class_id, bbox, img,frame_idx):
        detections=self.detection_set.filter(track_id=track_id,is_lost=False)
        if len(detections)>0:
            detection = detections[0]
            detection.frames_counter+=1
            detection.last_bbox=list(bbox)
            detection.last_image = img.copy()
            detection.last_detection_time = datetime.datetime.now()
            detection.last_frame_detection_id = frame_idx
            detection.frames_counter_class[str(class_id)]["frames_detected"]+=1
            #detection.save()
            return True                
        return False

    def update(self,bbox,track_id,class_id,img,is_lost,frame_idx):    
        before = time.time()
        '''
            Main function to update frame.
        '''
        delta = datetime.timedelta( minutes = TEMP_FINISH_TIMER_MINUTES, seconds=TEMP_FINISH_TIMER_SECONDS)
        aux_timer = (datetime.datetime.combine(datetime.date(1,1,1),self.count_timer) + delta)
        # if len(self.detection_set.all())>0 and  aux_timer < datetime.datetime.now():
        #     self.count_timer = datetime.datetime.now()
        #     self.save()
        #     for detection in self.detection_set.all():
        #         if not detection.is_lost and frame_idx-detection.last_frame_detection_id > self.max_age:
        #             detection.set_obj_lost()
        #         if detection.same_bbox_by_distance(self.video_shape):                    
        #             continue 
        #         # new_det.append(detection)
            
            # AfarmentDataManager(new_det,self.count_timer)
            # self.detections=[]

        if is_lost:
            self.obj_lost(track_id,class_id)
            print("obj_lost ",time.time()-before)
            return     
        if self.ask_obj_exists(track_id,class_id,bbox,img,frame_idx):
            # print("ask_obj_exists ",time.time()-before)

            return

        self.obj_new(track_id,class_id,bbox,img,frame_idx)

        print("obj_new",time.time()-before)
 
    def obj_lost(self,track_id,class_id):
        for detection in self.detection_set.all():
            if detection.track_id == track_id and not detection.is_lost:           
               
                if detection.same_bbox_by_distance(self.video_shape):                    
                    continue 
                detection.set_obj_lost() 
            # else:           
            #     #Update distance between first and las box
            #     detection.bbox_distance( )
       



class ZonePoly(models.Model):
    detection_manager = models.ForeignKey(DetectionManager, on_delete=models.CASCADE)
    name = models.CharField(max_length=64)
    poly = ArrayField(
        ArrayField(
            models.IntegerField(),
            size=2            
        )
    ) 
    
    
class Detection(models.Model):

    detection_manager = models.ForeignKey(DetectionManager, on_delete=models.CASCADE, null=True)

    input_zone = models.ForeignKey(ZonePoly, on_delete=models.SET_NULL, related_name="input_zone",null = True)
    output_zone =  models.ForeignKey(ZonePoly, on_delete=models.SET_NULL, related_name="output_zone",null = True)

    frames_counter_class= models.JSONField(default = return_default_fields)     

    track_id = models.IntegerField(default = -1)
    class_id = models.IntegerField(default = -1)
    dist_btw_bbox = models.IntegerField(default = -1)  
    frames_counter = models.IntegerField(default = 0)        
    last_frame_detection_id = models.IntegerField(default = 0)  

    last_bbox = ArrayField(models.IntegerField(default = 0), null=True)
    first_bbox = ArrayField(models.IntegerField(default = 0), null=True)     
    
    is_lost = models.BooleanField(default=False)
    valid_detection = models.BooleanField(default=False)

    detection_time = models.TimeField(auto_now=True)
    last_detection_time = models.TimeField(null=True)  


    def bbox_distance(self):
        
        bbox1 = self.first_bbox        
        bbox2 = self.last_bbox
        c1 = [(bbox1[0]+bbox1[2])/2, (bbox1[3]+bbox1[1])/2]
        c2 = [(bbox2[0]+bbox2[2])/2, (bbox2[1]+bbox2[3])/2]
        self.dist_btw_bbox  = ((c1[0]-c2[0])**2+(c1[1]-c2[1])**2)**0.5
        # self.save()
 
        
    def same_bbox_by_distance(self, video_shape):
        
        bbox1 = self.first_bbox        
        bbox2 = self.last_bbox
        c1 = [(bbox1[0]+bbox1[2])/2, (bbox1[3]+bbox1[1])/2]
        c2 = [(bbox2[0]+bbox2[2])/2, (bbox2[1]+bbox2[3])/2]
        self.dist_btw_bbox  = ((c1[0]-c2[0])**2+(c1[1]-c2[1])**2)**0.5
 
        max_dist=((video_shape[0]*0.1)**2+(video_shape[1]*0.1)**2)**0.5

        if self.dist_btw_bbox <max_dist:
            self.valid_detection = True
        # self.save()

    def set_obj_lost(self):        
        if self.is_lost:
            return
        # Find the class who have been detected in more frames
        max_frame_class = -1
        max_frame_class_idx = -1
        for idx in self.frames_counter_class.keys():
            if max_frame_class<self.frames_counter_class[idx]["frames_detected"]:
                max_frame_class = self.frames_counter_class[idx]["frames_detected"]
                max_frame_class_idx = idx

        if max_frame_class != -1:
            self.class_id = max_frame_class_idx
            
        self.is_lost=True        
        self.output_zone = self.detection_manager.object_detectable(self.last_bbox)["zone"]
        # SET ORIENTATION
        self.orientation = self.input_zone.name+"-"+self.output_zone.name
        self.save()

    

class AfarmentData(models.Model):
    ammount = models.IntegerField()
    maneuver = models.CharField(max_length=64)
    class_name = models.CharField(max_length=64)
    class_id = models.IntegerField()
