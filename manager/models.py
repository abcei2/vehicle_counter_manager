from django.db import models
from django.contrib.postgres.fields import ArrayField, JSONField
from django.contrib.auth.models import User
# Create your models here.
        
DEFAULT_POLY = {
    "p1":{
        "point":[50 ,50],
        "pressed":False
    },            
    "p2":{
        "point":[50 ,100],
        "pressed":False
    },            
    "p3":{
        "point":[100 ,100],
        "pressed":False
    },            
    "p4":{
        "point":[100 ,50],
        "pressed":False
    }
}
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


class Video(models.Model):
    QUEUED = "Queued"
    PROCESSING = "Processing"
    FINISHED = "Finished"

    VIDEO_STATUS = (
        (QUEUED, "Queued"),
        (PROCESSING, "Processing"),
        (FINISHED, "Finished"),
    )

    owner = models.ForeignKey(User, on_delete=models.CASCADE ) 
    video_link = models.FileField(db_index=True, upload_to='not_used')
    frame_ammount = models.IntegerField(default=-1)
    frame_processed = models.IntegerField(default=0)
    fps = models.IntegerField(default=-1)
    status = models.CharField(max_length=64,
                    choices=VIDEO_STATUS,
                    default=QUEUED)
    task_id = models.CharField(max_length=128, null = True)
        

class Zone(models.Model):
    
    video = models.ForeignKey(Video, on_delete=models.CASCADE, null = True)
    name = models.CharField(max_length=64)
    poly = ArrayField(    
        models.JSONField()   
    ) 

class DetectionDB(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE ) 
    class_id = models.IntegerField()

    last_bbox = ArrayField(models.IntegerField(default = 0), null=True)
    first_bbox = ArrayField(models.IntegerField(default = 0), null=True)         
      
   
    input_zone = models.CharField(max_length=64)
    output_zone =  models.CharField(max_length=64)

    dist_btw_bbox = models.IntegerField(default = -1)    
    frames_counter = models.IntegerField(default = 0)        
    
    first_frame_detection_id = models.IntegerField(null = True)   
    last_frame_detection_id = models.IntegerField(null = True)   

    detection_time = models.TimeField(auto_now=True)
    last_detection_time = models.TimeField(null= True)  

    

class FrameDetection(models.Model):
    
    detection = models.ForeignKey(DetectionDB, on_delete=models.CASCADE, related_name="frames" ) 
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name="frames" ) 
    frame_idx = models.IntegerField() 
    bbox = ArrayField(models.IntegerField(default = 0))
