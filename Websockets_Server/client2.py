import asyncio
import websockets
import json

dados = {'Fruits': {'Red_Apple': [], 'Green_Apple': [], 'Orange': [{'x': 1.985, 'y': 0.164, 'z': -1.983, 'width': 0.072, 'height': 0.101, 'weeks': 2}, {'x': 4.385, 'y': 1.531, 'z': -0.948, 'width': 0.179, 'height': 0.234, 'weeks': 4}, {'x': 4.345, 'y': 1.65, 'z': -1.696, 'width': 0.151, 'height': 0.214, 'weeks': 3}]}}

async def hello():
    async with websockets.connect("ws://127.0.0.1:3306") as websocket:
        await websocket.send(json.dumps({"topic": "digital_twin", "data": dados}))
        msg = await websocket.recv()
        msg_dict = msg
        print(msg_dict[0]['object'])
        print("Received:" + str(type(msg_dict)))

asyncio.run(hello())