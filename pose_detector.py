"""
MoveNet 姿态检测引擎
====================
封装 MoveNet Lightning 模型的加载、推理和关键点绘制逻辑。
支持 TF Hub SavedModel 和 TFLite 两种推理方式，默认使用 TFLite（更轻量快速）。
"""

import numpy as np
import cv2
import tensorflow as tf

# ========== 17个关键点定义 ==========
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# 骨骼连接关系（关节对 → 颜色 BGR 格式）
# 左侧用紫色 (255,0,255)，右侧用青色 (255,255,0)，躯干用黄色 (0,255,255)
SKELETON_CONNECTIONS = [
    # 面部
    (0, 1, (255, 0, 255)),   # nose → left_eye
    (0, 2, (255, 255, 0)),   # nose → right_eye
    (1, 3, (255, 0, 255)),   # left_eye → left_ear
    (2, 4, (255, 255, 0)),   # right_eye → right_ear
    # 躯干上部
    (0, 5, (255, 0, 255)),   # nose → left_shoulder
    (0, 6, (255, 255, 0)),   # nose → right_shoulder
    (5, 6, (0, 255, 255)),   # left_shoulder ↔ right_shoulder
    # 左臂
    (5, 7, (255, 0, 255)),   # left_shoulder → left_elbow
    (7, 9, (255, 0, 255)),   # left_elbow → left_wrist
    # 右臂
    (6, 8, (255, 255, 0)),   # right_shoulder → right_elbow
    (8, 10, (255, 255, 0)),  # right_elbow → right_wrist
    # 躯干下部
    (5, 11, (255, 0, 255)),  # left_shoulder → left_hip
    (6, 12, (255, 255, 0)),  # right_shoulder → right_hip
    (11, 12, (0, 255, 255)), # left_hip ↔ right_hip
    # 左腿
    (11, 13, (255, 0, 255)), # left_hip → left_knee
    (13, 15, (255, 0, 255)), # left_knee → left_ankle
    # 右腿
    (12, 14, (255, 255, 0)), # right_hip → right_knee
    (14, 16, (255, 255, 0)), # right_knee → right_ankle
]

# 关键点中文名称（用于界面展示）
KEYPOINT_CN_NAMES = {
    0: '鼻子', 1: '左眼', 2: '右眼', 3: '左耳', 4: '右耳',
    5: '左肩', 6: '右肩', 7: '左肘', 8: '右肘',
    9: '左手腕', 10: '右手腕', 11: '左髋', 12: '右髋',
    13: '左膝', 14: '右膝', 15: '左脚踝', 16: '右脚踝'
}


class PoseDetector:
    """MoveNet 姿态检测引擎"""

    # Lightning: 输入 192x192, 速度优先
    # Thunder:  输入 256x256, 精度优先
    MODEL_CONFIGS = {
        'lightning': {
            'input_size': 192,
            'tflite_url': 'https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite',
            'savedmodel_url': 'https://tfhub.dev/google/movenet/singlepose/lightning/4',
        },
        'thunder': {
            'input_size': 256,
            'tflite_url': 'https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite',
            'savedmodel_url': 'https://tfhub.dev/google/movenet/singlepose/thunder/4',
        }
    }

    def __init__(self, model_name='lightning', backend='tflite', model_path=None):
        """
        初始化姿态检测引擎。

        Args:
            model_name: 模型名称，'lightning'（速度快）或 'thunder'（精度高）
            backend: 推理后端，'tflite'（推荐，轻量快速）或 'savedmodel'（TF Hub 模型）
            model_path: 本地模型文件路径（如果已有下载好的模型，直接指定路径）
        """
        self.model_name = model_name
        self.backend = backend
        self.config = self.MODEL_CONFIGS[model_name]
        self.input_size = self.config['input_size']
        self.interpreter = None
        self.model = None
        self._load_model(model_path)

    def _load_model(self, model_path=None):
        """加载模型"""
        if self.backend == 'tflite':
            self._load_tflite(model_path)
        else:
            self._load_savedmodel(model_path)

    def _load_tflite(self, model_path=None):
        """加载 TFLite 模型"""
        if model_path is None:
            import os
            model_path = f'movenet_{self.model_name}.tflite'
            if not os.path.exists(model_path):
                print(f'[PoseDetector] 正在下载 MoveNet {self.model_name} TFLite 模型...')
                url = self.config['tflite_url']
                import urllib.request
                urllib.request.urlretrieve(url, model_path)
                print(f'[PoseDetector] 模型已保存到 {model_path}')

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f'[PoseDetector] TFLite 模型加载完成 (input_size={self.input_size})')

    def _load_savedmodel(self, model_path=None):
        """加载 TF Hub SavedModel"""
        import tensorflow_hub as hub
        if model_path is None:
            model_path = self.config['savedmodel_url']
        self.module = hub.load(model_path)
        self.model = self.module.signatures['serving_default']
        print(f'[PoseDetector] SavedModel 加载完成 (input_size={self.input_size})')

    def detect(self, image):
        """
        对单帧图像进行姿态检测。

        Args:
            image: numpy 数组，形状为 [height, width, 3]，BGR 格式（OpenCV 读取的格式）

        Returns:
            keypoints_with_scores: numpy 数组，形状为 [1, 1, 17, 3]
                最后一维: [y, x, score]，y/x 为归一化坐标 (0~1)
        """
        # BGR → RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 转为 TensorFlow 张量并调整大小（保持宽高比 + 填充）
        input_tensor = tf.expand_dims(tf.cast(rgb_image, dtype=tf.float32), axis=0)
        input_tensor = tf.image.resize_with_pad(input_tensor, self.input_size, self.input_size)

        if self.backend == 'tflite':
            input_tensor = tf.cast(input_tensor, dtype=tf.uint8)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor.numpy())
            self.interpreter.invoke()
            keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            input_tensor = tf.cast(input_tensor, dtype=tf.int32)
            outputs = self.model(input_tensor)
            keypoints_with_scores = outputs['output_0'].numpy()

        return keypoints_with_scores

    def draw_keypoints(self, image, keypoints_with_scores, confidence_threshold=0.3):
        """
        在图像上绘制关键点和骨骼连线。

        Args:
            image: 原始图像 (numpy, BGR)
            keypoints_with_scores: detect() 的返回值
            confidence_threshold: 置信度阈值，低于此值的关键点不绘制

        Returns:
            标注后的图像 (numpy, BGR)
        """
        height, width, _ = image.shape
        kpts = keypoints_with_scores[0, 0]  # shape: (17, 3)

        # ---- 绘制骨骼连线 ----
        for idx_a, idx_b, color in SKELETON_CONNECTIONS:
            score_a = kpts[idx_a, 2]
            score_b = kpts[idx_b, 2]
            if score_a > confidence_threshold and score_b > confidence_threshold:
                ya, xa = int(kpts[idx_a, 0] * height), int(kpts[idx_a, 1] * width)
                yb, xb = int(kpts[idx_b, 0] * height), int(kpts[idx_b, 1] * width)
                cv2.line(image, (xa, ya), (xb, yb), color, thickness=3, lineType=cv2.LINE_AA)

        # ---- 绘制关键点 ----
        for i in range(17):
            score = kpts[i, 2]
            if score > confidence_threshold:
                y = int(kpts[i, 0] * height)
                x = int(kpts[i, 1] * width)
                # 置信度越高，点越大
                radius = max(4, int(8 * score))
                # 高置信度用亮绿色，低置信度用黄色
                color = (0, 255, 0) if score > 0.6 else (0, 255, 255)
                cv2.circle(image, (x, y), radius, color, thickness=-1, lineType=cv2.LINE_AA)
                # 在关键点旁边标注名称（可选，默认关闭以避免画面杂乱）
                # name = KEYPOINT_CN_NAMES.get(i, '')
                # cv2.putText(image, name, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        return image

    def get_keypoints_dict(self, keypoints_with_scores, image_shape, confidence_threshold=0.3):
        """
        将检测结果转为可读的字典格式。

        Args:
            keypoints_with_scores: detect() 的返回值
            image_shape: (height, width) 用于将归一化坐标转为像素坐标
            confidence_threshold: 置信度阈值

        Returns:
            dict: {关键点名: {'x': 像素x, 'y': 像素y, 'score': 置信度}}
        """
        height, width = image_shape
        kpts = keypoints_with_scores[0, 0]
        result = {}
        for name, idx in KEYPOINT_DICT.items():
            score = kpts[idx, 2]
            if score > confidence_threshold:
                result[name] = {
                    'x': int(kpts[idx, 1] * width),
                    'y': int(kpts[idx, 0] * height),
                    'score': round(float(score), 3)
                }
        return result


# ========== 智能裁剪算法（用于视频连续帧，提升检测精度） ==========

MIN_CROP_KEYPOINT_SCORE = 0.2

def init_crop_region(image_height, image_width):
    """初始化裁剪区域（将整张图填充为正方形）"""
    if image_width > image_height:
        box_height = image_width / image_height
        box_width = 1.0
        y_min = (image_height / 2 - image_width / 2) / image_height
        x_min = 0.0
    else:
        box_height = 1.0
        box_width = image_height / image_width
        y_min = 0.0
        x_min = (image_width / 2 - image_height / 2) / image_width
    return {
        'y_min': y_min, 'x_min': x_min,
        'y_max': y_min + box_height, 'x_max': x_min + box_width,
        'height': box_height, 'width': box_width
    }

def torso_visible(keypoints):
    """检查躯干关键点是否可见"""
    return ((keypoints[0, 0, KEYPOINT_DICT['left_hip'], 2] > MIN_CROP_KEYPOINT_SCORE or
             keypoints[0, 0, KEYPOINT_DICT['right_hip'], 2] > MIN_CROP_KEYPOINT_SCORE) and
            (keypoints[0, 0, KEYPOINT_DICT['left_shoulder'], 2] > MIN_CROP_KEYPOINT_SCORE or
             keypoints[0, 0, KEYPOINT_DICT['right_shoulder'], 2] > MIN_CROP_KEYPOINT_SCORE))

def determine_crop_region(keypoints, image_height, image_width):
    """根据上一帧检测结果确定当前帧的裁剪区域"""
    target_keypoints = {}
    for joint in KEYPOINT_DICT.keys():
        target_keypoints[joint] = [
            keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
            keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width
        ]

    if torso_visible(keypoints):
        center_y = (target_keypoints['left_hip'][0] + target_keypoints['right_hip'][0]) / 2
        center_x = (target_keypoints['left_hip'][1] + target_keypoints['right_hip'][1]) / 2

        torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        max_torso_yrange = max(abs(center_y - target_keypoints[j][0]) for j in torso_joints)
        max_torso_xrange = max(abs(center_x - target_keypoints[j][1]) for j in torso_joints)

        max_body_yrange = max(
            abs(center_y - target_keypoints[j][0])
            for j in KEYPOINT_DICT.keys()
            if keypoints[0, 0, KEYPOINT_DICT[j], 2] >= MIN_CROP_KEYPOINT_SCORE
        )
        max_body_xrange = max(
            abs(center_x - target_keypoints[j][1])
            for j in KEYPOINT_DICT.keys()
            if keypoints[0, 0, KEYPOINT_DICT[j], 2] >= MIN_CROP_KEYPOINT_SCORE
        )

        crop_length_half = np.amax([
            max_torso_xrange * 1.9, max_torso_yrange * 1.9,
            max_body_yrange * 1.2, max_body_xrange * 1.2
        ])
        crop_length_half = np.amin([
            crop_length_half,
            np.amax([center_x, image_width - center_x, center_y, image_height - center_y])
        ])
        crop_corner = [center_y - crop_length_half, center_x - crop_length_half]

        if crop_length_half > max(image_width, image_height) / 2:
            return init_crop_region(image_height, image_width)

        crop_length = crop_length_half * 2
        return {
            'y_min': crop_corner[0] / image_height,
            'x_min': crop_corner[1] / image_width,
            'y_max': (crop_corner[0] + crop_length) / image_height,
            'x_max': (crop_corner[1] + crop_length) / image_width,
            'height': (crop_corner[0] + crop_length) / image_height - crop_corner[0] / image_height,
            'width': (crop_corner[1] + crop_length) / image_width - crop_corner[1] / image_width
        }
    else:
        return init_crop_region(image_height, image_width)

def crop_and_resize(image, crop_region, crop_size):
    """裁剪并调整图像大小"""
    boxes = [[crop_region['y_min'], crop_region['x_min'],
              crop_region['y_max'], crop_region['x_max']]]
    output_image = tf.image.crop_and_resize(
        tf.expand_dims(image, axis=0), box_indices=[0], boxes=boxes, crop_size=crop_size)
    return output_image

def run_inference_with_crop(detector, image, crop_region, crop_size):
    """在裁剪区域上运行推理，并将坐标映射回原图"""
    image_height, image_width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = crop_and_resize(
        tf.cast(rgb_image, dtype=tf.float32), crop_region, crop_size=crop_size)

    if detector.backend == 'tflite':
        input_image = tf.cast(input_image, dtype=tf.uint8)
        detector.interpreter.set_tensor(
            detector.input_details[0]['index'], input_image.numpy())
        detector.interpreter.invoke()
        keypoints_with_scores = detector.interpreter.get_tensor(
            detector.output_details[0]['index'])
    else:
        input_image = tf.cast(input_image, dtype=tf.int32)
        outputs = detector.model(input_image)
        keypoints_with_scores = outputs['output_0'].numpy()

    # 将裁剪区域的坐标映射回原图
    for idx in range(17):
        keypoints_with_scores[0, 0, idx, 0] = (
            crop_region['y_min'] * image_height +
            crop_region['height'] * image_height * keypoints_with_scores[0, 0, idx, 0]
        ) / image_height
        keypoints_with_scores[0, 0, idx, 1] = (
            crop_region['x_min'] * image_width +
            crop_region['width'] * image_width * keypoints_with_scores[0, 0, idx, 1]
        ) / image_width

    return keypoints_with_scores
