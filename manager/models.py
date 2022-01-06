from django.db import models
from django.contrib.postgres.fields import ArrayField
# Create your models here.


class Detection(models.Model):

    pieces = ArrayField(ArrayField(models.IntegerField()))
    track_id = track_id
    class_id = models.IntegerField()
    first_bbox = models.ArrayField() first_bbox
    last_bbox = ArrayField(ArrayField(models.IntegerField()))
    dist_btw_bbox = models.IntegerField()
    first_image = image.copy()        
    last_image = image.copy()    
    input_zone = models.IntegerField(default = -1)
    output_zone = models.IntegerField(default = -1)
    frames_counter = models.IntegerField(default = 0)        
    last_frame_detection_id = models.IntegerField(default = 0)   
    is_lost = models.BooleanField()
    frames_counter_class= models.JSONField() 

    detection_time = models.TimeField(auto_now=True)
    last_detection_time = models.TimeField()  


class AfarmentData(models.Model):
    ammount = models.IntegerField()
    maneuver = models.CharField()
    class_name = models.CharField()
    class_id = models.IntegerField()
        
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
class ZoneConfig(models.Model):
    
    frames_counter_class= models.JSONField() 
    zones = ArrayField(models.JSONField(default=DEFAULT_POLY) )
 