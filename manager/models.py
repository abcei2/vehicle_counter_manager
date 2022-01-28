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
    frame_ammount = models.IntegerField(default=-1)
    fps = models.IntegerField(default=-1)
    
class ZoneConfigDB(models.Model):
    
    video = models.ForeignKey(Video, on_delete=models.CASCADE ) 
 
 
    def object_detectable(self, bbox): 
        '''
            If object is detectable in some zone
        '''   
        point=([int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2)])        
        for zone_poly in self.zone_set.all():
            zone = Polygon( [points for points in zone_set.poly] )
            if zone.contains( Point(point)  ):                
                return {"zone":zone_poly, "detectable":True}

        return {"zone":None, "detectable":False}
        

class Zone(models.Model):
    
    zone_config = models.ForeignKey(ZoneConfigDB, on_delete=models.CASCADE)
    name = models.CharField(max_length=64)
    poly = ArrayField(
        ArrayField(
            models.IntegerField(),
            size=2            
        )
    ) 



class AfarmentDataDB(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE ) 
    ammount = models.IntegerField()
    maneuver = models.CharField(max_length=64)
    class_name = models.CharField(max_length=64)
    class_id = models.IntegerField()
    
class DetectionDB(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE ) 
    class_id = models.IntegerField()

    last_bbox = ArrayField(models.IntegerField(default = 0), null=True)
    first_bbox = ArrayField(models.IntegerField(default = 0), null=True)         
   
    input_zone = models.CharField(max_length=64)
    output_zone =  models.CharField(max_length=64)

    dist_btw_bbox = models.IntegerField(default = -1)    
    frames_counter = models.IntegerField(default = 0)        
    last_frame_detection_id = models.IntegerField(default = 0)   

    detection_time = models.TimeField(auto_now=True)
    last_detection_time = models.TimeField(null= True)  