"""M6：图像修复器 — 基于LaMa的文字去除与背景重建。"""
import cv2
import numpy as np


class Inpainter:
    """去除漫画图像中的原文文字并重建背景。

    使用LaMa（基于FFT的全局感受野修复）实现高质量背景重建，
    特别适合漫画中的网点纹理和渐变背景。
    """

    def __init__(self) -> None:
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from src.core.model_manager import ModelManager
            self._model = ModelManager().get_model("inpainting", "lama")
        return self._model

    def inpaint(
        self, image: np.ndarray, boxes: list[dict], padding: int = 5
    ) -> np.ndarray:
        """去除所有指定文本框中的文字并重建背景。

        Args:
            image: BGR或RGB格式的numpy数组。
            boxes: 包含x, y, width, height键的字典列表。
            padding: 每个框额外扩展的像素数（确保完全覆盖文字）。

        Returns:
            文字区域被修复后的图像。
        """
        if not boxes:
            return image

        mask = self.generate_mask(boxes, image.shape[:2], padding)

        try:
            if self._model is not None:
                result = self._inpaint_lama(image, mask)
            else:
                result = self._inpaint_fallback(image, mask)
        except Exception:
            result = self._inpaint_fallback(image, mask)

        return result

    def generate_mask(
        self, boxes: list[dict], img_shape: tuple[int, int], padding: int = 5
    ) -> np.ndarray:
        """生成覆盖所有文本区域的二值蒙版。"""
        mask = np.zeros(img_shape, dtype=np.uint8)
        for box in boxes:
            expanded = self._expand_box(box, padding)
            x1 = max(0, int(expanded["x"]))
            y1 = max(0, int(expanded["y"]))
            x2 = min(img_shape[1], int(expanded["x"] + expanded["width"]))
            y2 = min(img_shape[0], int(expanded["y"] + expanded["height"]))
            mask[y1:y2, x1:x2] = 255
        return mask

    def _inpaint_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """运行LaMa ONNX模型进行修复。"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[-1] == 3 else image
        img_norm = img_rgb.astype(np.float32) / 255.0
        mask_norm = mask.astype(np.float32) / 255.0
        img_input = np.expand_dims(np.transpose(img_norm, (2, 0, 1)), axis=0)
        mask_input = np.expand_dims(np.expand_dims(mask_norm, axis=0), axis=0)
        ort_inputs = {"image": img_input, "mask": mask_input}
        ort_outputs = self._model.run(None, ort_inputs)
        result = ort_outputs[0][0]
        result = np.transpose(result, (1, 2, 0))
        result = (result * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    def _inpaint_fallback(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """回退方案：LaMa不可用时使用OpenCV Telea修复。"""
        return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    @staticmethod
    def _expand_box(box: dict, padding: int = 5) -> dict:
        """将边界框向四周扩展指定像素。"""
        return {
            "x": box["x"] - padding,
            "y": box["y"] - padding,
            "width": box["width"] + 2 * padding,
            "height": box["height"] + 2 * padding,
        }
