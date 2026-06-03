"""漫画翻译测试脚本的共享工具函数。"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_translator(lang_pair: str):
    """加载 OPUS-MT 翻译模型。"""
    from transformers import MarianMTModel, MarianTokenizer
    model_name = f"Helsinki-NLP/opus-mt-{lang_pair}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


def translate(text: str, tokenizer, model) -> str:
    """翻译单条文本。"""
    if not text or not text.strip():
        return ""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def inpaint_text(img: np.ndarray, boxes: list, padding: int = 4) -> np.ndarray:
    """用 OpenCV 修复去除文字区域。

    boxes 可以是 numpy 多边形数组列表或 (x, y, w, h) 元组列表。
    """
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for box in boxes:
        if isinstance(box, np.ndarray):
            pts = box.astype(np.int32)
            x1 = max(0, int(pts[:, 0].min()) - padding)
            y1 = max(0, int(pts[:, 1].min()) - padding)
            x2 = min(img.shape[1], int(pts[:, 0].max()) + padding)
            y2 = min(img.shape[0], int(pts[:, 1].max()) + padding)
        else:
            x, y, w, h = box[:4]
            x1 = max(0, int(x) - padding)
            y1 = max(0, int(y) - padding)
            x2 = min(img.shape[1], int(x + w) + padding)
            y2 = min(img.shape[0], int(y + h) + padding)
        mask[y1:y2, x1:x2] = 255
    return cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def render_text(
    img: np.ndarray,
    text: str,
    box,
    font_path: str | None = None,
) -> np.ndarray:
    """在修复后的区域渲染译文。

    box 可以是 numpy 多边形数组或 (x, y, w, h) 元组。
    """
    if isinstance(box, np.ndarray):
        pts = box.astype(np.int32)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        w, h = pts[:, 0].max() - x1, pts[:, 1].max() - y1
    else:
        x, y, w, h = box[:4]
        x1, y1 = int(x), int(y)

    font_size = min(max(10, int(h * 0.6)), 36)
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    if tw > w * 1.3:
        lines = []
        words = text.split()
        line = ""
        for word in words:
            test = f"{line} {word}".strip()
            tb = draw.textbbox((0, 0), test, font=font)
            if tb[2] - tb[0] <= w * 1.3:
                line = test
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
        if not lines:
            lines = [text]
    else:
        lines = [text]

    line_h = font_size + 4
    start_y = y1 + max(0, (h - len(lines) * line_h) // 2)

    for i, line in enumerate(lines):
        tb = draw.textbbox((0, 0), line, font=font)
        lw = tb[2] - tb[0]
        lx = x1 + max(0, int((w - lw) // 2))
        ly = start_y + i * line_h
        draw.text(
            (lx, ly), line, fill=(0, 0, 0), font=font,
            stroke_width=1, stroke_fill=(255, 255, 255),
        )

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
