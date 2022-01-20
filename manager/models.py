from django.db import models
from django.contrib.postgres.fields import ArrayField
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
    owner_name = models.CharField(max_length=64)
    video_link = models.FileField(db_index=True, upload_to='not_used')
    
class ZoneConfig(models.Model):
    
    poly = models.JSONField() 
 
 
    def point_inside_area(self, point):      
        
        """
            DEFINE FUNCT
        """
        return to_return


    
class Detection(models.Model):

    frames_counter_class= models.JSONField() 
    
    class_id = models.IntegerField()

    last_bbox = ArrayField(models.IntegerField())
    first_bbox = ArrayField(models.IntegerField()) 
    dist_btw_bbox = models.IntegerField()    
    
    input_zone = models.ForeignKey(ZoneConfig, on_delete=models.CASCADE)
    input_zone = models.IntegerField(default = -1)
    output_zone = models.IntegerField(default = -1)
    frames_counter = models.IntegerField(default = 0)        
    last_frame_detection_id = models.IntegerField(default = 0)   
    is_lost = models.BooleanField()

    detection_time = models.TimeField(auto_now=True)
    last_detection_time = models.TimeField()  


class AfarmentData(models.Model):
    ammount = models.IntegerField()
    maneuver = models.CharField(max_length=64)
    class_name = models.CharField(max_length=64)
    class_id = models.IntegerField()
