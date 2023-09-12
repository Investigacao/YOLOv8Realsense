import websocket
import cv2
import numpy as np
import base64

def on_message(ws, message):
    # Decode and display the frame
    buffer = base64.b64decode(message)
    frame = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), -1)
    cv2.imshow("Received Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

# Set up the WebSocket connection for receiving frames
ws_url = "ws://127.0.0.1:8765"
ws = websocket.WebSocketApp(ws_url, on_message=on_message)

if __name__ == "__main__":
    ws.run_forever()