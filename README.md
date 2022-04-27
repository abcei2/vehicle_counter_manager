
## excecute task
from app.tasks import task
task_object = task.delay(1,2)
## Get task by id
from celery.result import AsyncResult
res = AsyncResult("your-task-id")
res.ready()
## End task by id
from celery.task.control import revoke
revoke(task_id, terminate=True)

## Start Worker
docker exec -it django-web python manage.py makemigrations
docker exec -it django-web python manage.py migrate
docker exec -it django-web python manage.py createsuperuser

## Start Worker
docker exec -it django-web python -m celery -A core worker -l INFO -P gevent
