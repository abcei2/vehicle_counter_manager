

## Compile and run docker-compose
```
git clone https://github.com/abcei2/vehicle_counter_manage
cd vehicle_counter_manager  
docker-compose up --build -d //-d para q haga todo en background
```
toma unos minutos y casi 16GB de espacio en el disco duro.  
## Migrar y crear super usuario (se hace una vez)
```
docker exec -it django-web python manage.py makemigrations
docker exec -it django-web python manage.py migrate
docker exec -it django-web python manage.py createsuperuser
```
## Iniciar worker
```
docker exec -it django-web python -m celery -A core worker -l INFO -P gevent
```

## En python

### excecute task
```python
from app.tasks import task  
task_object = task.delay(1,2)  
```
### Get task by id
```python
from celery.result import AsyncResult
res = AsyncResult("your-task-id")
res.ready()
```
### End task by id
```python
from celery.task.control import revoke
revoke(task_id, terminate=True)
```
