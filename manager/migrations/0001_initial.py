# Generated by Django 3.2.13 on 2022-05-12 11:38

from django.conf import settings
import django.contrib.postgres.fields
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='DetectionDB',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('class_id', models.IntegerField()),
                ('last_bbox', django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(default=0), null=True, size=None)),
                ('first_bbox', django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(default=0), null=True, size=None)),
                ('input_zone', models.CharField(max_length=64)),
                ('output_zone', models.CharField(max_length=64)),
                ('dist_btw_bbox', models.IntegerField(default=-1)),
                ('frames_counter', models.IntegerField(default=0)),
                ('first_frame_detection_id', models.IntegerField(null=True)),
                ('last_frame_detection_id', models.IntegerField(null=True)),
                ('detection_time', models.TimeField(auto_now=True)),
                ('last_detection_time', models.TimeField(null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Video',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('video_link', models.FileField(db_index=True, upload_to='not_used')),
                ('frame_ammount', models.IntegerField(default=-1)),
                ('frame_processed', models.IntegerField(default=0)),
                ('fps', models.IntegerField(default=-1)),
                ('status', models.CharField(choices=[('Queued', 'Queued'), ('Processing', 'Processing'), ('Finished', 'Finished')], default='Queued', max_length=64)),
                ('task_id', models.CharField(max_length=128, null=True)),
                ('owner', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Zone',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=64)),
                ('poly', django.contrib.postgres.fields.ArrayField(base_field=models.JSONField(), size=None)),
                ('video', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='manager.video')),
            ],
        ),
        migrations.CreateModel(
            name='FrameDetection',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('frame_idx', models.IntegerField()),
                ('bbox', django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(default=0), size=None)),
                ('detection', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='frames', to='manager.detectiondb')),
                ('video', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='frames', to='manager.video')),
            ],
        ),
        migrations.AddField(
            model_name='detectiondb',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='manager.video'),
        ),
    ]
