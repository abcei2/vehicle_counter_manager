from django.contrib import admin

# Register your models here.
from manager.models import DetectionDB, AfarmentDataDB, VideoOwner, Video

admin.site.register(DetectionDB)
admin.site.register(VideoOwner)
admin.site.register(Video)
@admin.register(AfarmentDataDB)
class AfarmentDataDB(admin.ModelAdmin):
    list_display = ('video', 'ammount', 'maneuver', 'class_name', 'class_id')