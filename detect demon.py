from ultralytics import YOLO
yolo = YOLO(r'E:\ultralytics-main\runs\detect\train4\weights\best.onnx',task="detect")  # load a pretrained model (recommended for training)
results = yolo(source=r'E:\ultralytics-main\datasets\VisDrone-DET-test-challenge\images',save=True,conf=0.05)  # perform inference on an image