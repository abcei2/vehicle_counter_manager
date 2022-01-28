# app/chat/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async

class ChatConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        ''' Cliente se conecta '''

        # Recoge el nombre de la sala
        self.room_name = self.scope["url_route"]["kwargs"]["room_name"]
        self.room_group_name = "chat_%s" % self.room_name

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
        print(text_data)
    

        # # Enviamos el mensaje a la sala
        # await self.channel_layer.group_send(
        #     self.room_group_name,
        #     {
        #         "type": "chat_message",
        #         "name": name,
        #         "text": text,
        #     },
        # )
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                "type": "message",
                "text": text_data,
            },
        )

    async def message(self, event):
        ''' Recibimos información de la sala '''
        # print(event)

        # Send message to WebSocket
        await self.send(
            text_data=event["text"]
        )