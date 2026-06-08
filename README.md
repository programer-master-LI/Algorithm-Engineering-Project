# SMART TRAINER - 实时人体姿态检测应用

基于 **MoveNet Lightning** + **Kivy** 的实时人体姿态检测与健身辅助应用。

## ✨ 功能特性

- 🎯 **实时姿态检测**：基于 MoveNet 模型，17 个关键点实时追踪
- 🦴 **骨骼可视化**：彩色骨骼连线 + 关键点标注，左右侧颜色区分
- 🏋️ **健身动作计数**：支持深蹲、俯卧撑、二头弯举，自动计数
- 📐 **角度显示**：实时显示关节角度，辅助动作纠错
- 🧠 **智能裁剪**：基于上一帧检测结果自动裁剪，提升精度
- 🖥️ **双模式运行**：Kivy GUI 模式 + OpenCV 轻量模式
- 🔤 **中文支持**：自动检测系统字体，PIL 渲染中文，告别乱码

## 🛠️ 环境要求

- Python 3.8+
- 摄像头（USB 或内置）

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

> 首次运行会自动下载 MoveNet Lightning TFLite 模型（约 4.5MB）

## 🚀 运行方式

### 方式一：Kivy GUI 模式（推荐）

```bash
python main.py
```

### 方式二：OpenCV 轻量模式

```bash
python main.py --opencv
```

## 🎮 操作指南

### Kivy 模式

| 操作 | 方式 |
|------|------|
| 切换运动模式 | 点击底部按钮 |
| 重置计数 | 点击「重置」按钮 |
| 停止训练 | 点击「停止」按钮 |
| 快捷键 S/P/C/R/Q | 深蹲/俯卧撑/弯举/重置/退出 |

### OpenCV 模式

| 快捷键 | 功能 |
|--------|------|
| S | 深蹲模式 |
| P | 俯卧撑模式 |
| C | 弯举模式 |
| R | 重置计数 |
| Q | 退出 |

## 📁 项目结构

```
姿态检测应用/
├── main.py                # 主入口（Kivy + OpenCV 双模式）
├── pose_detector.py       # MoveNet 姿态检测引擎
├── exercise_counter.py    # 健身动作计数器
├── text_renderer.py       # 中文文字渲染（PIL）+ 圆角矩形绘制
├── requirements.txt       # 依赖列表
├── README.md              # 本文件
└── movenet_lightning.tflite  # 模型文件（首次运行自动下载）
```

## 🔧 模块说明

### `main.py` — 主应用
- `MainScreen`：主页，训练强度选择 + 开始按钮
- `WorkoutScreen`：检测界面，后台线程检测 + Kivy UI 刷新
- `run_opencv_mode()`：纯 OpenCV 窗口模式
- 自动查找系统中文字体，注册给 Kivy 使用

### `pose_detector.py` — 姿态检测引擎
- `PoseDetector` 类：模型加载 + 推理 + 关键点绘制
- 智能裁剪算法：`determine_crop_region()`

### `exercise_counter.py` — 动作计数器
- `ExerciseCounter` 类：基于关节角度的动作计数
- 支持：深蹲（膝盖角度）、俯卧撑（肘部角度）、弯举（肘部角度）

### `text_renderer.py` — 中文文字渲染
- `put_chinese_text()`：PIL 渲染中文，替代 cv2.putText
- `draw_rounded_rect()`：圆角矩形绘制
- `_find_chinese_font()`：跨平台自动查找中文字体

## 🎨 界面色彩编码

| 颜色 | 含义 |
|------|------|
| 🟣 紫色 | 左侧肢体 |
| 🔵 青色 | 右侧肢体 |
| 🟡 黄色 | 躯干横连 |
| 🟢 亮绿 | 高置信度关键点 / 计数 |
| 🔵 浅蓝 | 动作反馈 |

## 📝 中文乱码修复说明

如果你仍然看到乱码：
1. 确保系统安装了中文字体（Windows: 微软雅黑, macOS: 苹方, Linux: Noto Sans CJK）
2. Linux 用户可执行: `sudo apt install fonts-noto-cjk`
3. 程序启动时会在控制台打印字体状态，留意提示

## 📄 License

- 应用代码：MIT License
- MoveNet 模型：Apache 2.0 License (Google)
