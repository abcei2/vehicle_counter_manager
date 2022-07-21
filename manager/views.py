from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Video, Zone
import json
from .tasks import video_to_queue
# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import HttpResponse, Http404
from rest_framework.permissions import IsAuthenticated  # <-- Here
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
import shutil

SECONDS_IN_MINUTES = 60
class UploadVideo(APIView):
    
    
    permission_classes = (IsAuthenticated,)
    def post(self, request):
        
        data = json.loads(request.POST.get('json'))
        response  = Response("-")

        if "zones" in data:
            if "zones" in data:
                for zone in data["zones"]:
                    if "poly" in zone:
                        if len(zone["poly"])<3:
                            response.status_code = 400
                            response.content = "each poly must have length more than 3 points to go"
                            return response
                    else:
                        response.status_code = 400
                        response.content = "each zone must have a 'poly' element with more than 3 points to go" 
                        return response
                                
                    if not "name" in zone:
                        response.status_code = 400
                        response.content = "each zone must have a different name"
                        return response
        else:            
            response.status_code = 400
            response.content = "need to tell 'zones'"   
            return response

      
        owner = request.user

        
        
        video = Video.objects.filter(owner=owner).exclude( status = Video.FINISHED)
        

        if len(video)>0:           
            response.status_code = 200
            response.content = f"User already has 1 video, please connect to socket room "   
            
        
            return response
        else:
            video = Video(owner=owner, status = Video.QUEUED,video_link=request.FILES.get('myfile'))
            video.save()

            for zone in data["zones"]:
                Zone(video = video, name = zone["name"], poly = zone["poly"]).save()

            response.status_code = 200
            response.content = json.dumps({"video_pk": video.pk,"message":"Your video is on queued please connect to the socket."})
         
            task_on_queue = video_to_queue.delay(video.pk)
            video.task_id = task_on_queue.id
            video.save()
            return response




class DownloadVideo(APIView):
    permission_classes = (IsAuthenticated,)
    
    def get(self, request):        
        owner = request.user
        video_set = owner.video_set.all()
        response  = Response("-")
        response.status_code = 200
        json_response = [
            {
                "id":video.pk,
                "status":video.status              
            }               
            for video in video_set   
        ]
        response.content = json.dumps(json_response)
        return response

    def post(self, request, video_pk):
        owner = request.user
        video = owner.video_set.get(pk=video_pk)
        frame_amount = video.frame_ammount
        if video:
            video_url = video.video_link.path
            cap = cv2.VideoCapture(video_url)           
            
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height` 
            
            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            
            out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))
            frame_idx = 0
            detection_counters = {}
            while cap.isOpened():         
                res, frame = cap.read()   
                if res:
                    frame_detections = video.frames.filter(frame_idx=frame_idx) 
              
                    for detection in frame_detections:
                        if detection.detection.class_id not in detection_counters.keys():
                            detection_counters[detection.detection.class_id]={
                                "amount":0,
                                "det_list":{}
                            }
                        if detection.detection.pk not in detection_counters[detection.detection.class_id]["det_list"].keys():
                            detection_counters[detection.detection.class_id]["amount"]+=1
                            detection_counters[detection.detection.class_id]["det_list"][detection.detection.pk]=detection_counters[detection.detection.class_id]["amount"]

               
                        bbox = detection.bbox
                        p1, p2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
                        
                        cv2.rectangle(frame,  p1, p2, (100,0,0), 2, cv2.LINE_AA)  
                        cv2.putText(
                            frame,
                            f"class: {detection.detection.class_id}, obj:{detection_counters[detection.detection.class_id]['det_list'][detection.detection.pk]}", 
                            p1,0, 1, (0,0,255),thickness=2, 
                            lineType=cv2.LINE_AA
                        )
                    frame_idx = frame_idx + 1
                else:
                    break
                out.write(frame) 
        cap.release()
        out.release()
        with open('output.avi', 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="video/avi")
            response['Content-Disposition'] = 'inline; filename=' + f'{video_pk}.avi'
            return response
       
        raise Http404
        
        

    

class VideoStatus(APIView):
    permission_classes = (IsAuthenticated,)
    
    def post(self, request):
        
        owner = request.user
        response_content = {
            "status":"NO_VID",
            "message":"user can request to upload video"
        }        
        
        video = owner.video_set.all().exclude( status = Video.FINISHED)
        if len(video)>0:  
            response_content["status"] = video[0].status
            if video[0].status == video[0].PROCESSING:
                response_content["message"]  =  f"User already has 1 video proccesing, please connect to socket room "   
        
            elif video[0].status == video[0].QUEUED:
                response_content["message"]  =  f"User already has 1 video on queue."   
        else:
            
            print("not video")

        response  = Response("-")
        response.status_code = 200
        response.content = json.dumps(response_content)
        return response
    
def testview(request):
    return render(request, 'testview.html')