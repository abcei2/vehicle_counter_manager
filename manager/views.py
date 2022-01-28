from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Video

# Create your views here.
@csrf_exempt 
def app_save(request):
    if request.method == 'POST':
        print(request.FILES.get('myfile'))
        newdoc = Video(owner_name="santi",video_link=request.FILES.get('myfile'))
        newdoc.save()
        return HttpResponse("ok")

        # app/front/views.py

def chat(request):
    return render(request, 'chat.html')