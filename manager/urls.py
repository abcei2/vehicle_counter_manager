from django.urls import path
from django.conf.urls import url
from .views import app_save, chat, user_status

app_name = "manager"
urlpatterns = [
    path("", app_save, name="upload_video"),
    path("chat", chat, name="chat"),
    path("estado", user_status, name="user_status")
]
