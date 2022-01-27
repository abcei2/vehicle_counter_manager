from django.contrib import admin

# Register your models here.
from manager.models import DetectionDB, AfarmentDataDB

admin.site.register(DetectionDB)
@admin.register(AfarmentDataDB)
class AfarmentDataDB(admin.ModelAdmin):
    list_display = ('video', 'ammount', 'maneuver', 'class_name', 'class_id')