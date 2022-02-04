# app/chat/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from manager.models import Video
class ChatConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        ''' Cliente se conecta '''
        self.room_name = "detection_room"
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
        text_data_json = json.loads(text_data)    
        
        await self.channel_layer.group_send(
            self.room_name,
            {
                "type":text_data_json["type"],
                "message": text_data_json["message"],
                "username": text_data_json["username"]
            },
        )


    async def proccess(self, data):
        ''' Recibimos información de la sala '''
        # print("evente",event)

        # Send message to WebSocket
        await self.send(
            text_data=json.dumps(data)
        )
        
    async def end(self, data):
        ''' Recibimos información de la sala '''
        # print("evente",event)

        # Send message to WebSocket
        await self.send(
            text_data=json.dumps(data)
        )