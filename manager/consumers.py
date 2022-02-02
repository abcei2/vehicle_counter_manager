# app/chat/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from manager.models import Video, VideoOwner

@sync_to_async
def get_video(username):
    owner, create = VideoOwner.objects.get_or_create(name=username)

    video = Video.objects.filter(owner=owner).exclude( status = Video.FINISHED)
    if len(video)>0:  
       return video[0]
    return None
class ChatConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        ''' Cliente se conecta '''

        # Recoge el nombre de la sala
        self.room_name = self.scope["url_route"]["kwargs"]["room_name"]
        self.room_group_name =  self.room_name

        # Se une a la sala
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        # Informa al cliente del éxito
        await self.accept()

    async def disconnect(self, close_code):
        ''' Cliente se desconecta '''
        # Leave room group
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    async def receive(self, text_data):
        ''' Cliente envía información y nosotros la recibimos '''
        print(text_data)
        text_data_json = json.loads(text_data)
    
    
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                "type":text_data_json["type"],
                "message": text_data_json["message"],
                "username": text_data_json["username"],
            },
        )
    async def info(self, data):
        '''
        Recibimos información de la sala 
        
        '''
        print(data)
        
        video = await get_video(data["username"])
        print(video)
        if video is None:
            data["message"]= "NOVID"
        else:
            data["message"]= video.status
        # Send message to WebSocket
        await self.send(
            text_data=json.dumps(data)
        )




    async def proccess(self, data):
        ''' Recibimos información de la sala '''
        # print("evente",event)

        # Send message to WebSocket
        await self.send(
            text_data=json.dumps(data)
        )