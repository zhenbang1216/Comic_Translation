"""M7.4：绘制器 — 在修复后的图像上分层渲染译文。"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.utils.schemas import TypesetResult, RenderStyle


class Painter:
    """在修复后的漫画图像上渲染翻译文字。

    按四层顺序绘制以确保正确的视觉效果：
    1. 阴影（如原文有）
    2. 描边（如检测到）
    3. 主体文字
    4. 合成到目标图像
    """

    def render(
        self,
        image: np.ndarray,
        typeset: TypesetResult,
        style: RenderStyle,
        position: tuple[int, int],
    ) -> np.ndarray:
        """在图像指定位置渲染文字。"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        try:
            font = ImageFont.truetype("arial.ttf", typeset.font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

        x, y = position
        for i, line in enumerate(typeset.lines):
            line_y = y + i * typeset.line_height
            line_x = self._align_x(line, x, typeset, font, draw)

            if style.shadow_offset:
                sx = line_x + style.shadow_offset[0]
                sy = line_y + style.shadow_offset[1]
                draw.text((sx, sy), line, fill=(80, 80, 80), font=font)

            if style.stroke_width and style.stroke_width > 0:
                stroke_fill = style.stroke_color or (255, 255, 255)
                draw.text(
                    (line_x, line_y), line,
                    fill=style.text_color, font=font,
                    stroke_width=style.stroke_width,
                    stroke_fill=stroke_fill,
                )
            else:
                draw.text((line_x, line_y), line, fill=style.text_color, font=font)

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _align_x(
        line: str, base_x: int, typeset: TypesetResult, font, draw
    ) -> int:
        """根据对齐方式计算x坐标。"""
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        if typeset.alignment == "center":
            return base_x - text_width // 2
        elif typeset.alignment == "right":
            return base_x - text_width
        return base_x
