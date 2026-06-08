# 加在所有 import 之前！
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
# 👇 必须加这一行！！！Windows 训练 YOLOv8 时会开多进程加载图片，不加这句保护，Python 会无限递归启动进程，直接崩溃。
#Linux/Mac 不用加，但 Windows 必须加

if __name__ == '__main__':
    # 你的训练代码全部放在这里面
# Load model
    model = YOLO("yolov8s.pt")  # nano版本适合RasPi n是指nano版本 0,8n-pose是         ,8n-obb是        ，8n-cls是      ，8n-seg是
# Train 
    model.train(
    data=r"E:\ultralytics-main\args.yaml",
    epochs=100,
    imgsz=960,             #训练图像大小，默认640，设置大点可以提升精度，但会增加训练时间和显存占用,让小目标更清晰
    batch=2,
    device=0,  
    workers=0,               # 加载线程，这个必须加，不加就不生成train.txt和val.txt文件，默认多进程 → Windows 卡死
                               #不训练、不输出、不生成文件夹
    cache=True,              # ✅ 缓存数据集，解决越训越慢、磁盘IO卡顿 
    augment=True,
    patience=20,              # 20轮没提升就停止训练，默认300轮，设置小点可以节省时间
    lr0=0.002,                #降低初始学习率，避免学崩溃，默认0.01，设置小点可以稳定训练
    mosaic=1.0,                #马赛克增强，默认1.0，设置小点可以稳定训练
    close_mosaic=10,             #close_mosaic=10（最后 10 轮关闭，避免影响精度），默认0，设置小点可以稳定训练
                 #复制粘贴增强，默认0.5，设置小点可以稳定训练,对密集小目标提升明显
    hsv_h=0.015,              #色调增强，默认0.015，设置小点可以稳定训练,增强颜色鲁棒性
    hsv_s=0.7,                     #饱和度增强，默认0.7，设置小点可以稳定训练
    hsv_v=0.4,                  #亮度增强，默认0.4，设置小点可以稳定训练
    resume=True            #继续上次训练，默认False，设置True可以接着上次训练继续训

)

# Save model
    model.export(format="onnx")
