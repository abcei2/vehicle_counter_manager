# Generated by Django 3.2.9 on 2022-02-02 16:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('manager', '0003_video_task_id'),
    ]

    operations = [
        migrations.AlterField(
            model_name='video',
            name='status',
            field=models.CharField(choices=[('Queued', 'Queued'), ('Processing', 'Processing'), ('Finished', 'Finished')], default='Queued', max_length=64),
        ),
    ]