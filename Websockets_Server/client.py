import asyncio
import websockets
import json

dados = {'Fruits': 
         {
    'Red_Apple': [], 
    'Green_Apple': [], 
    'Orange': [
    {'x': 1.985, 'y': 0.164, 'z': 1.983, 'width': 0.072, 'height': 0.101, 'weeks': 2}, 
    {'x': 4.385, 'y': 1.531, 'z': 0.948, 'width': 0.179, 'height': 0.234, 'weeks': 4}, 
    {'x': 4.345, 'y': 1.65, 'z': -1.696, 'width': 0.151, 'height': 0.214, 'weeks': 3}]},
    'Vase': {'x': 1.685, 'y': -0.181, 'z': -0.546, 'width': 0.376, 'height': 0.101, 'weeks': 2}}

dados2 = {'Fruits': {'Red_Apple': [], 'Green_Apple': [], 'Orange': [{'x': 1.585, 'y': 0.158, 'z': 0.171, 'width': 0.05, 'height': 0.067, 'weeks': 4}, {'x': 1.685, 'y': 0.073, 'z': 0.071, 'width': 0.046, 'height': 0.059, 'weeks': 1}, {'x': 1.645, 'y': 0.058, 'z': 0.113, 'width': 0.054, 'height': 0.056, 'weeks': 6}, {'x': 1.585, 'y': 0.196, 'z': 0.105, 'width': 0.054, 'height': 0.072, 'weeks': 4}]}, 'Vase': {'x': 1.585, 'y': 0.028, 'z': 0.068, 'width': 0.0, 'height': 0.0, 'perc': '99%'}}

async def hello():
    async with websockets.connect("ws://172.22.21.135:3306") as websocket:
        await websocket.send(json.dumps({"topic": "digital_twin", "data": dados2}))
        # msg = await websocket.recv()
        # msg_dict = eval(msg)
        # print("Received:" + str(msg_dict["ruben"]))
        # print("Received:" + str(type(msg_dict)))

asyncio.run(hello())