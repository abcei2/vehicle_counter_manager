from django.db import models

# Create your models here.


class Detection(models.Model):
    id  = id
    track_id = track_id
    class_id = class_id
    first_bbox = first_bbox
    last_bbox = first_bbox
    dist_btw_bbox = 0
    first_image = image.copy()        
    last_image = image.copy()    
    input_zone = input_zone
    output_zone = -1
    frames_counter = 0        
    last_frame_detection_id = 0
    is_lost = False
    frames_counter_class={
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
    x_1=models.FloatField()
    y_1=models.FloatField()
    x_2=models.FloatField()
    y_2=models.FloatField()
