from django.shortcuts import render
from django.core.exceptions import MultipleObjectsReturned
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Video, VideoOwner, ZoneConfigDB, Zone
import json

# Create your views here.
@csrf_exempt 
def app_save(request):
    if request.method == 'POST':

        data = json.loads(request.POST.get('json'))
        response  = HttpResponse("-")
        if not "username" in data:
            response.status_code = 400
            response.content = "need to tell 'username'"
            return response

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

      
        owner, created = VideoOwner.objects.get_or_create(name=data["username"])

        
        
        video = Video.objects.filter(owner=owner).exclude( status = Video.FINISHED)

        if len(video)>0:           
            response.status_code = 200
            response.content = f"User already has 1 video, please connect to socket room {data['username']}"   
            return response
        else:
            video = Video(owner=owner, status = Video.QUEUED,video_link=request.FILES.get('myfile'))
            video.save()

      
            zone_config = ZoneConfigDB(video = video )
            zone_config.save()
            for zone in data["zones"]:
                Zone(zone_config = zone_config, name = zone["name"], poly = zone["poly"]).save()

            response.status_code = 200
            response.content = "Your video is on queued please connect to the socket."
            return response

        # app/front/views.py

@csrf_exempt 
def user_status(request):
    if request.method == 'POST':
        
        username = request.POST.get('username')
        owner, created = VideoOwner.objects.get_or_create(name=username)
        response_content = {
            "queued_video":False,
            "message":"user can request to upload video"
        }
        
        
        if not created:
            
            video = Video.objects.filter(owner=owner).exclude( status = Video.FINISHED)

            if len(video)>0:  
                response_content["queued_video"] = True   
                response_content["message"]  =  f"User already has 1 video, please connect to socket room {username}"   
        

        response  = HttpResponse("-")
        response.status_code = 200
        response.content = json.dumps(response_content)
        return response
    
def chat(request):
    return render(request, 'chat.html')