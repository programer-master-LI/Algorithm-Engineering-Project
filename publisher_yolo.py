from ultralytics import YOLO
import cv2
import paho.mqtt.client as mqtt

# 加载模型
model = YOLO(r"E:\ultralytics-main\runs\detect\train5\weights\best.onnx")

# MQTT配置
BROKER = "localhost"
PORT = 1883
TOPIC = "yolo/video"

client = mqtt.Client()
client.connect(BROKER, PORT, 60)

# 摄像头
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO推理
    results = model(frame, conf=0.1)

    # 绘制检测框、类别、置信度
    annotated = results[0].plot()

    # 压缩JPEG
    success, buffer = cv2.imencode(
        ".jpg",
        annotated,
        [int(cv2.IMWRITE_JPEG_QUALITY), 60]
    )

    if success:
        client.publish(
            TOPIC,
            buffer.tobytes()
        )

    # 本地显示
    cv2.imshow(
        "YOLO Publisher",
        annotated
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client.disconnect()