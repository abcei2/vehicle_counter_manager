from django.contrib import admin

# Register your models here.
from manager.models import DetectionDB, Video, Zone, FrameDetection

admin.site.register(DetectionDB)
admin.site.register(Video)
admin.site.register(Zone)
admin.site.register(FrameDetection)