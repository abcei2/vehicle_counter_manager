from django.urls import path
from django.conf.urls import url
from .views import app_save

app_name = "manager"
urlpatterns = [
    path("", app_save, name="upload_video")
]
