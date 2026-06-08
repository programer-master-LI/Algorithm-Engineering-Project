"""
文字渲染工具
============
解决 OpenCV cv2.putText 不支持中文的问题，使用 PIL 渲染中文文字。
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import platform


def _find_chinese_font():
    """自动查找系统中可用的中文字体"""
    system = platform.system()
    candidates = []

    if system == 'Windows':
        font_dir = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
        candidates = [
            os.path.join(font_dir, 'msyh.ttc'),      # 微软雅黑
            os.path.join(font_dir, 'msyhbd.ttc'),     # 微软雅黑粗体
            os.path.join(font_dir, 'simhei.ttf'),     # 黑体
            os.path.join(font_dir, 'simsun.ttc'),     # 宋体
        ]
    elif system == 'Darwin':  # macOS
        candidates = [
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/Hiragino Sans GB.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
        ]
    else:  # Linux
        candidates = [
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        ]

    for path in candidates:
        if os.path.exists(path):
            return path

    # 尝试 fc-match 命令查找（Linux）
    try:
        import subprocess
        result = subprocess.run(
            ['fc-match', '-f', '%{file}', 'Noto Sans CJK SC'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and os.path.exists(result.stdout.strip()):
            return result.stdout.strip()
    except Exception:
        pass

    return None


# 缓存字体
_font_cache = {}
_default_font_path = _find_chinese_font()


def _get_font(size=20):
    """获取字体对象（带缓存）"""
    if size in _font_cache:
        return _font_cache[size]

    try:
        if _default_font_path:
            font = ImageFont.truetype(_default_font_path, size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    _font_cache[size] = font
    return font


def put_chinese_text(img, text, position, font_size=20, color=(255, 255, 255),
                     stroke_color=None, stroke_width=0):
    """
    在 OpenCV 图像上绘制中文文字。

    Args:
        img: numpy 数组 (BGR 格式)
        text: 要绘制的文字（支持中英文混合）
        position: 左上角坐标 (x, y)
        font_size: 字体大小
        color: 文字颜色 (B, G, R)
        stroke_color: 描边颜色 (B, G, R)，None 表示无描边
        stroke_width: 描边宽度

    Returns:
        标注后的图像
    """
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)

    # PIL 使用 RGB 格式颜色
    color_rgb = (color[2], color[1], color[0])
    stroke_rgb = None
    if stroke_color is not None:
        stroke_rgb = (stroke_color[2], stroke_color[1], stroke_color[0])

    draw.text(
        position, text,
        fill=color_rgb,
        font=font,
        stroke_width=stroke_width,
        stroke_fill=stroke_rgb
    )

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=10):
    """
    在 OpenCV 图像上绘制圆角矩形。

    Args:
        img: numpy 数组 (BGR)
        pt1: 左上角 (x, y)
        pt2: 右下角 (x, y)
        color: 颜色 (B, G, R)
        thickness: -1 为填充
        radius: 圆角半径
    """
    x1, y1 = pt1
    x2, y2 = pt2

    # 确保坐标合法
    if radius > (x2 - x1) // 2 or radius > (y2 - y1) // 2:
        radius = min((x2 - x1) // 2, (y2 - y1) // 2)

    # 四个角的圆弧
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)

    # 四条边
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)


def get_font_status():
    """返回当前字体状态信息"""
    if _default_font_path:
        return f'中文字体: {_default_font_path}'
    else:
        return '⚠ 未找到中文字体，中文可能显示为方块'
