"""
SMART TRAINER - 实时人体姿态检测应用
=====================================
基于 MoveNet + Kivy 的实时姿态检测与健身辅助应用。
功能：实时摄像头姿态检测、骨骼可视化、健身动作计数、角度显示。

运行方式:
    python main.py          # Kivy GUI 模式
    python main.py --opencv # OpenCV 轻量模式

快捷键:
    S - 深蹲模式
    P - 俯卧撑模式
    C - 弯举模式
    R - 重置计数
    Q - 退出
"""

import os
import sys
import time
import threading
import numpy as np
import cv2

# 配置 Kivy 环境变量（需在 import kivy 之前设置）
os.environ['KIVY_NO_ARGS'] = '1'

import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.slider import Slider
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle, Color
from kivy.core.window import Window
from kivy.core.text import LabelBase

from pose_detector import PoseDetector, init_crop_region, determine_crop_region, run_inference_with_crop
from exercise_counter import ExerciseCounter
from text_renderer import put_chinese_text, draw_rounded_rect, get_font_status

# ========== 查找并注册中文字体给 Kivy ==========
def _find_kivy_font():
    """查找系统中可用的中文字体文件，供 Kivy 注册使用"""
    import platform
    system = platform.system()
    candidates = []

    if system == 'Windows':
        font_dir = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
        candidates = [
            os.path.join(font_dir, 'msyh.ttc'),
            os.path.join(font_dir, 'msyhbd.ttc'),
            os.path.join(font_dir, 'simhei.ttf'),
        ]
    elif system == 'Darwin':
        candidates = [
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/Hiragino Sans GB.ttc',
        ]
    else:
        candidates = [
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        ]

    for path in candidates:
        if os.path.exists(path):
            return path
    return None

# 注册 Kivy 中文字体
_KIVY_FONT_PATH = _find_kivy_font()
_KIVY_FONT_NAME = 'ChineseDefault'

if _KIVY_FONT_PATH:
    try:
        LabelBase.register(name=_KIVY_FONT_NAME, fn_regular=_KIVY_FONT_PATH)
        os.environ['KIVY_FONT_NAME'] = _KIVY_FONT_NAME
    except Exception as e:
        print(f'[字体] 注册中文字体失败: {e}')
        _KIVY_FONT_PATH = None


def _make_label(**kwargs):
    """创建支持中文的 Label"""
    if _KIVY_FONT_PATH:
        kwargs.setdefault('font_name', _KIVY_FONT_NAME)
    return Label(**kwargs)


def _make_button(**kwargs):
    """创建支持中文的 Button"""
    if _KIVY_FONT_PATH:
        kwargs.setdefault('font_name', _KIVY_FONT_NAME)
    return Button(**kwargs)


# ========== 自定义样式组件 ==========

class StyledButton(Button):
    """统一风格的主按钮"""
    def __init__(self, **kwargs):
        if _KIVY_FONT_PATH:
            kwargs.setdefault('font_name', _KIVY_FONT_NAME)
        super().__init__(**kwargs)
        self.background_normal = ''
        self.background_color = (0.12, 0.53, 0.90, 1)   # 科技蓝
        self.font_size = '20sp'
        self.bold = True
        self.color = (1, 1, 1, 1)


class ExerciseButton(Button):
    """运动模式选择按钮"""
    def __init__(self, exercise_type='', **kwargs):
        if _KIVY_FONT_PATH:
            kwargs.setdefault('font_name', _KIVY_FONT_NAME)
        super().__init__(**kwargs)
        self.exercise_type = exercise_type
        self.background_normal = ''
        self.background_color = (0.12, 0.15, 0.22, 0.92)
        self.font_size = '15sp'
        self.color = (0.6, 0.85, 1, 1)
        self.size_hint = (None, None)
        self.size = (130, 44)


# ========== 主界面 ==========

class MainScreen(Screen):
    """应用主界面"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()

       # 1. 背景图片 (Widget 1: Image as Background)
        # 请确保目录下有一张名为 'gym_bg.jpg' 的高清健身图 
        self.bg_image = Image(source='gym_bg.jpg', allow_stretch=True, keep_ratio=False, size_hint=(1, 1))
        layout.add_widget(self.bg_image)

        # 添加半透明黑色遮罩层，提升 UI 质感
        with layout.canvas.before:
            Color(0, 0, 0, 0.4)
            self.bg_rect = Rectangle(size=self.size, pos=self.pos)
        layout.bind(size=self._update_rect, pos=self._update_rect)

        # 顶部装饰线
        with layout.canvas.after:
            Color(0.12, 0.53, 0.90, 0.35)
            self.line_rect = Rectangle(
                size=(Window.width * 0.5, 2),
                pos=(Window.width * 0.25, Window.height * 0.50)
            )

        # 标题
        title = _make_label(
            text='SMART TRAINER',
            font_size='44sp',
            pos_hint={'center_x': 0.5, 'top': 0.88},
            color=(1, 1, 1, 1)
        )
        layout.add_widget(title)

        # 副标题
        subtitle = _make_label(
            text='AI 姿态检测 · 实时计数 · 动作纠错',
            font_size='16sp',
            pos_hint={'center_x': 0.5, 'top': 0.78},
            color=(0.45, 0.70, 0.95, 0.85)
        )
        layout.add_widget(subtitle)

        # 模型信息
        model_info = _make_label(
            text=f'Powered by MoveNet Lightning · 17 关键点实时检测',
            font_size='13sp',
            pos_hint={'center_x': 0.5, 'center_y': 0.53},
            color=(0.35, 0.42, 0.52, 0.7)
        )
        layout.add_widget(model_info)

        # 字体状态提示
        font_status = get_font_status()
        if '未找到' in font_status:
            font_warn = _make_label(
                text=font_status,
                font_size='11sp',
                pos_hint={'center_x': 0.5, 'center_y': 0.48},
                color=(1, 0.7, 0, 0.6)
            )
            layout.add_widget(font_warn)

        # 强度选择
        intensity_label = _make_label(
            text='训练强度',
            font_size='16sp',
            pos_hint={'center_x': 0.5, 'center_y': 0.42},
            color=(0.75, 0.75, 0.80, 0.9)
        )
        layout.add_widget(intensity_label)

        self.intensity = Slider(
            min=1, max=10, value=5,
            size_hint=(0.45, 0.08),
            pos_hint={'center_x': 0.5, 'center_y': 0.36}
        )
        layout.add_widget(self.intensity)

        self.intensity_value = _make_label(
            text='5',
            font_size='22sp',
            pos_hint={'center_x': 0.5, 'center_y': 0.29},
            color=(0.12, 0.53, 0.90, 1)
        )
        layout.add_widget(self.intensity_value)
        self.intensity.bind(value=self._on_intensity_change)

        # 开始按钮
        start_btn = StyledButton(
            text='开始训练',
            size_hint=(0.45, 0.10),
            pos_hint={'center_x': 0.5, 'center_y': 0.17}
        )
        start_btn.bind(on_press=self.go_to_workout)
        layout.add_widget(start_btn)

        # 底部提示
        hint = _make_label(
            text='请确保摄像头可用 · 建议全身入镜',
            font_size='12sp',
            pos_hint={'center_x': 0.5, 'y': 0.02},
            color=(0.35, 0.38, 0.45, 0.55)
        )
        layout.add_widget(hint)

        self.add_widget(layout)

    def _update_rect(self, instance, value):
        self.bg_rect.pos = instance.pos
        self.bg_rect.size = instance.size

    def _on_intensity_change(self, instance, value):
        self.intensity_value.text = str(int(value))

    def go_to_workout(self, instance):
        self.manager.current = 'workout'


# ========== 检测界面 ==========

class WorkoutScreen(Screen):
    """姿态检测界面"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.detector = None
        self.counter = None
        self.crop_region = None
        self.use_crop = True
        self.fps = 0
        self.frame_times = []
        self._stop_event = threading.Event()
        self._detection_thread = None
        self._annotated_frame = None
        self._lock = threading.Lock()
        self._model_loaded = False

        layout = FloatLayout()

        # 摄像头画面
        self.camera_feed = Image(size_hint=(1, 1), allow_stretch=True, keep_ratio=False)
        layout.add_widget(self.camera_feed)

        # ---- 顶部信息栏 ----
        top_bar = BoxLayout(
            size_hint=(1, None), height=46,
            pos_hint={'top': 1},
            orientation='horizontal', padding=[12, 4]
        )
        with top_bar.canvas.before:
            Color(0.04, 0.04, 0.08, 0.75)
            top_bar.rect = Rectangle(size=top_bar.size, pos=top_bar.pos)
        top_bar.bind(
            size=lambda i, v: setattr(top_bar.rect, 'size', v),
            pos=lambda i, v: setattr(top_bar.rect, 'pos', v)
        )

        self.mode_label = _make_label(
            text='深蹲模式', font_size='17sp',
            color=(0.2, 1, 0.5, 1), size_hint_x=0.3
        )
        top_bar.add_widget(self.mode_label)

        self.fps_label = _make_label(
            text='FPS: --', font_size='14sp',
            color=(0.7, 0.7, 0.7, 0.8), size_hint_x=0.2
        )
        top_bar.add_widget(self.fps_label)

        self.status_label = _make_label(
            text='正在加载模型...', font_size='14sp',
            color=(1, 0.85, 0.2, 0.9), size_hint_x=0.5
        )
        top_bar.add_widget(self.status_label)

        layout.add_widget(top_bar)

        # ---- 左侧计数面板 ----
        counter_panel = FloatLayout(
            size_hint=(None, None), width=190, height=170,
            pos_hint={'x': 0.02, 'top': 0.89}
        )
        with counter_panel.canvas.before:
            Color(0.04, 0.04, 0.08, 0.72)
            counter_panel.rect = Rectangle(size=counter_panel.size, pos=counter_panel.pos)
        counter_panel.bind(
            size=lambda i, v: setattr(counter_panel.rect, 'size', v),
            pos=lambda i, v: setattr(counter_panel.rect, 'pos', v)
        )

        self.count_label = _make_label(
            text='0', font_size='60sp',
            color=(0.2, 1, 0.5, 1), bold=True,
            pos_hint={'center_x': 0.5, 'top': 0.85}
        )
        counter_panel.add_widget(self.count_label)

        self.angle_label = _make_label(
            text='角度: --', font_size='16sp',
            color=(0.85, 0.85, 0.9, 0.9),
            pos_hint={'center_x': 0.5, 'top': 0.52}
        )
        counter_panel.add_widget(self.angle_label)

        self.feedback_label = _make_label(
            text='等待开始...', font_size='14sp',
            color=(1, 0.85, 0.2, 0.95),
            pos_hint={'center_x': 0.5, 'top': 0.30}
        )
        counter_panel.add_widget(self.feedback_label)

        layout.add_widget(counter_panel)

        # ---- 底部控制栏 ----
        bottom_bar = BoxLayout(
            size_hint=(1, None), height=56,
            pos_hint={'y': 0},
            orientation='horizontal',
            padding=[8, 6], spacing=8
        )
        with bottom_bar.canvas.before:
            Color(0.04, 0.04, 0.08, 0.80)
            bottom_bar.rect = Rectangle(size=bottom_bar.size, pos=bottom_bar.pos)
        bottom_bar.bind(
            size=lambda i, v: setattr(bottom_bar.rect, 'size', v),
            pos=lambda i, v: setattr(bottom_bar.rect, 'pos', v)
        )

        btn_squat = ExerciseButton(text='深蹲', exercise_type='squat')
        btn_squat.bind(on_press=self.change_exercise)
        bottom_bar.add_widget(btn_squat)

        btn_pushup = ExerciseButton(text='俯卧撑', exercise_type='pushup')
        btn_pushup.bind(on_press=self.change_exercise)
        bottom_bar.add_widget(btn_pushup)

        btn_curl = ExerciseButton(text='弯举', exercise_type='curl')
        btn_curl.bind(on_press=self.change_exercise)
        bottom_bar.add_widget(btn_curl)

        btn_reset = ExerciseButton(text='重置')
        btn_reset.bind(on_press=self.reset_counter)
        bottom_bar.add_widget(btn_reset)

        stop_btn = _make_button(
            text='停止',
            background_normal='',
            background_color=(0.85, 0.12, 0.12, 0.95),
            font_size='15sp', color=(1, 1, 1, 1),
            size_hint=(None, None), size=(100, 44)
        )
        stop_btn.bind(on_press=self.stop_workout)
        bottom_bar.add_widget(stop_btn)

        layout.add_widget(bottom_bar)

        self.add_widget(layout)

    def on_enter(self):
        """进入检测界面时初始化"""
        self._stop_event.clear()
        self.counter = ExerciseCounter(exercise_type='squat')

        self._detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._detection_thread.start()

        Clock.schedule_interval(self._update_ui, 1.0 / 30.0)

    def on_leave(self):
        """离开检测界面时清理"""
        self._stop_event.set()
        Clock.unschedule(self._update_ui)
        if self.capture and self.capture.isOpened():
            self.capture.release()
            self.capture = None
        self._model_loaded = False

    def _detection_loop(self):
        """检测主循环（运行在后台线程）"""
        try:
            self.status_label.text = '正在加载模型...'
            self.detector = PoseDetector(model_name='lightning', backend='tflite')
            self._model_loaded = True
            self.status_label.text = '模型就绪'

            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                self.status_label.text = '摄像头打开失败'
                return

            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.status_label.text = '检测中...'
            self.crop_region = None

            while not self._stop_event.is_set():
                ret, frame = self.capture.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)

                if self._model_loaded:
                    try:
                        h, w, _ = frame.shape

                        if self.use_crop and self.crop_region is not None:
                            keypoints_with_scores = run_inference_with_crop(
                                self.detector, frame, self.crop_region,
                                crop_size=(self.detector.input_size, self.detector.input_size)
                            )
                        else:
                            keypoints_with_scores = self.detector.detect(frame)

                        self.crop_region = determine_crop_region(keypoints_with_scores, h, w)

                        annotated = self.detector.draw_keypoints(
                            frame, keypoints_with_scores, confidence_threshold=0.3
                        )

                        if self.counter:
                            count, angle, feedback = self.counter.update(
                                keypoints_with_scores, (h, w)
                            )

                        # 使用 PIL 渲染中文 HUD
                        self._draw_hud(annotated)

                        with self._lock:
                            self._annotated_frame = annotated

                    except Exception as e:
                        with self._lock:
                            self._annotated_frame = frame
                else:
                    with self._lock:
                        self._annotated_frame = frame

                now = time.time()
                self.frame_times.append(now)
                self.frame_times = [t for t in self.frame_times if now - t < 1.0]
                self.fps = len(self.frame_times)

        except Exception as e:
            print(f'[WorkoutScreen] 检测线程异常: {e}')
            self.status_label.text = f'错误: {str(e)[:30]}'

    def _draw_hud(self, frame):
        """使用 PIL 在帧上绘制中文 HUD"""
        h, w, _ = frame.shape

        # ---- 左上角信息面板背景 ----
        draw_rounded_rect(frame, (8, 8), (210, 155), (10, 10, 18, 200), thickness=-1, radius=12)

        # 计数
        if self.counter:
            count_text = str(self.counter.count)
            frame = put_chinese_text(frame, count_text, (18, 12), font_size=48,
                                     color=(50, 255, 120))

            angle_text = f'角度: {int(self.counter.angle)}°'
            frame = put_chinese_text(frame, angle_text, (18, 68), font_size=20,
                                     color=(220, 220, 230))

            if self.counter.feedback:
                frame = put_chinese_text(frame, self.counter.feedback, (18, 98), font_size=18,
                                         color=(50, 220, 255))

        # 模式
        mode_names = {'squat': '深蹲', 'pushup': '俯卧撑', 'curl': '弯举'}
        if self.counter:
            mode_text = f'模式: {mode_names.get(self.counter.exercise_type, self.counter.exercise_type)}'
            frame = put_chinese_text(frame, mode_text, (18, 128), font_size=16,
                                     color=(160, 200, 255))

        # FPS（右上角）
        fps_text = f'FPS: {self.fps}'
        frame = put_chinese_text(frame, fps_text, (w - 130, 14), font_size=18,
                                 color=(180, 180, 180))

    def _update_ui(self, dt):
        """定时刷新 Kivy UI"""
        with self._lock:
            frame = self._annotated_frame

        if frame is None:
            return

        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.camera_feed.texture = texture

        if self.counter:
            self.count_label.text = str(self.counter.count)
            self.angle_label.text = f'角度: {int(self.counter.angle)}°'
            self.feedback_label.text = self.counter.feedback or '检测中...'

        self.fps_label.text = f'FPS: {self.fps}'

    def change_exercise(self, instance):
        """切换运动模式"""
        exercise = instance.exercise_type
        if exercise and self.counter:
            self.counter = ExerciseCounter(exercise_type=exercise)
            names = {'squat': '深蹲', 'pushup': '俯卧撑', 'curl': '弯举'}
            self.mode_label.text = f'{names.get(exercise, exercise)}模式'

    def reset_counter(self, instance):
        """重置计数器"""
        if self.counter:
            self.counter.reset()
            self.count_label.text = '0'
            self.angle_label.text = '角度: --'
            self.feedback_label.text = '已重置'

    def stop_workout(self, instance):
        """停止训练"""
        self._stop_event.set()
        Clock.unschedule(self._update_ui)
        if self.capture and self.capture.isOpened():
            self.capture.release()
            self.capture = None
        self._model_loaded = False
        self.manager.current = 'main'


# ========== 键盘快捷键 ==========

class PoseApp(App):
    """主应用"""

    def build(self):
        self.title = 'Smart Trainer - AI 姿态检测'
        Window.size = (960, 720)

        sm = ScreenManager(transition=FadeTransition())
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(WorkoutScreen(name='workout'))
        self.sm = sm

        Window.bind(on_keyboard=self._on_keyboard)
        return sm

    def _on_keyboard(self, instance, key, scancode, codepoint, modifier):
        current = self.sm.current_screen
        if not isinstance(current, WorkoutScreen):
            return False

        if codepoint in ('s', 'S'):
            current.change_exercise(type('obj', (), {'exercise_type': 'squat'})())
        elif codepoint in ('p', 'P'):
            current.change_exercise(type('obj', (), {'exercise_type': 'pushup'})())
        elif codepoint in ('c', 'C'):
            current.change_exercise(type('obj', (), {'exercise_type': 'curl'})())
        elif codepoint in ('r', 'R'):
            current.reset_counter(None)
        elif codepoint in ('q', 'Q'):
            current.stop_workout(None)
        return True


# ========== OpenCV 独立运行模式 ==========

def run_opencv_mode():
    """使用纯 OpenCV 窗口运行姿态检测（无需 Kivy）"""
    print('=' * 50)
    print('  Smart Trainer - OpenCV 模式')
    print('  快捷键: Q-退出  S-深蹲  P-俯卧撑  C-弯举  R-重置')
    print('=' * 50)

    font_info = get_font_status()
    print(f'\n  {font_info}')

    print('\n正在加载 MoveNet 模型...')
    detector = PoseDetector(model_name='lightning', backend='tflite')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('无法打开摄像头')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    counter = ExerciseCounter(exercise_type='squat')
    crop_region = None
    frame_times = []

    print('检测开始！')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if crop_region is not None:
            keypoints_with_scores = run_inference_with_crop(
                detector, frame, crop_region,
                crop_size=(detector.input_size, detector.input_size)
            )
        else:
            keypoints_with_scores = detector.detect(frame)

        crop_region = determine_crop_region(keypoints_with_scores, h, w)

        annotated = detector.draw_keypoints(frame, keypoints_with_scores, confidence_threshold=0.3)
        count, angle, feedback = counter.update(keypoints_with_scores, (h, w))

        # ---- HUD 绘制（全部使用 PIL 渲染中文） ----

        # 左上角面板背景
        draw_rounded_rect(annotated, (8, 8), (230, 168), (10, 10, 18, 200), thickness=-1, radius=12)

        # 计数（大字）
        annotated = put_chinese_text(annotated, str(count), (18, 12), font_size=52,
                                     color=(50, 255, 120))

        # 角度
        annotated = put_chinese_text(annotated, f'角度: {int(angle)}°', (18, 72), font_size=22,
                                     color=(220, 220, 230))

        # 反馈
        annotated = put_chinese_text(annotated, feedback, (18, 102), font_size=18,
                                     color=(50, 220, 255))

        # 当前模式
        mode_names = {'squat': '深蹲', 'pushup': '俯卧撑', 'curl': '弯举'}
        mode_text = f'模式: {mode_names.get(counter.exercise_type, counter.exercise_type)}'
        annotated = put_chinese_text(annotated, mode_text, (18, 132), font_size=16,
                                     color=(160, 200, 255))

        # FPS（右上角）
        now = time.time()
        frame_times.append(now)
        frame_times = [t for t in frame_times if now - t < 1.0]
        fps = len(frame_times)
        annotated = put_chinese_text(annotated, f'FPS: {fps}', (w - 130, 14), font_size=18,
                                     color=(180, 180, 180))

        # 底部提示栏
        draw_rounded_rect(annotated, (8, h - 42), (w - 8, h - 8), (10, 10, 18, 160), thickness=-1, radius=8)
        annotated = put_chinese_text(annotated, 'S-深蹲  P-俯卧撑  C-弯举  R-重置  Q-退出',
                                     (16, h - 38), font_size=16, color=(140, 140, 150))

        cv2.imshow('Smart Trainer - Pose Detection', annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('s'), ord('S')):
            counter = ExerciseCounter(exercise_type='squat')
            print('切换到: 深蹲模式')
        elif key in (ord('p'), ord('P')):
            counter = ExerciseCounter(exercise_type='pushup')
            print('切换到: 俯卧撑模式')
        elif key in (ord('c'), ord('C')):
            counter = ExerciseCounter(exercise_type='curl')
            print('切换到: 弯举模式')
        elif key in (ord('r'), ord('R')):
            counter.reset()
            print('计数已重置')

    cap.release()
    cv2.destroyAllWindows()
    print('已退出。')


# ========== 入口 ==========

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--opencv':
        run_opencv_mode()
    else:
        PoseApp().run()
