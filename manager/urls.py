from django.urls import path
from django.conf.urls import url
from .views import  chat, VideoStatus, UploadVideo
from rest_framework.authtoken.views import obtain_auth_token  # <-- Here

app_name = "manager"
urlpatterns = [
    path("", UploadVideo.as_view(), name="upload_video"),
    path("login",obtain_auth_token, name="login"),
    path("chat", chat, name="chat"),
    path("estado", VideoStatus.as_view(), name="video_status")
]
