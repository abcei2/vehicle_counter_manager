# Generated by Django 3.2.9 on 2022-01-31 20:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('manager', '0002_auto_20220131_1449'),
    ]

    operations = [
        migrations.AddField(
            model_name='video',
            name='task_id',
            field=models.CharField(max_length=128, null=True),
        ),
    ]