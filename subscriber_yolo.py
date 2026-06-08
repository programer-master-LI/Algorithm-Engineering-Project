import cv2
import numpy as np
import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT = 1883
TOPIC = "yolo/video"

def on_message(client, userdata, msg):

    frame_bytes = msg.payload

    np_arr = np.frombuffer(
        frame_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        np_arr,
        cv2.IMREAD_COLOR
    )

    if frame is not None:

        cv2.imshow(
            "YOLO Subscriber",
            frame
        )

        cv2.waitKey(1)

client = mqtt.Client()

client.on_message = on_message

client.connect(
    BROKER,
    PORT,
    60
)

client.subscribe(TOPIC)

print("Waiting YOLO Stream...")

client.loop_forever()