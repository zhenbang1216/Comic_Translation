"""M7.1：风格检测器 — 为SAAT渲染提取原文字视觉风格。"""
import cv2
import numpy as np
from sklearn.cluster import KMeans
from src.utils.schemas import RenderStyle


class StyleDetector:
    """从原文字区域检测视觉风格参数。

    全部使用传统CV（无深度学习），每个区域<5ms。
    提取：文字颜色、描边、阴影、字号估计、字重、对齐方式。
    """

    def analyze(self, text_region: np.ndarray) -> RenderStyle:
        """对文本区域图像进行完整的风格分析。"""
        stroke = self.detect_stroke(text_region)
        return RenderStyle(
            text_color=self.detect_text_color(text_region),
            stroke_color=stroke.get("stroke_color") if stroke else None,
            stroke_width=stroke.get("stroke_width", 0) if stroke else 0,
            shadow_offset=self.detect_shadow(text_region),
            shadow_color=None,
            estimated_font_size=self.estimate_font_size(text_region),
            font_weight=self.detect_font_weight(text_region),
            alignment=self.detect_alignment(text_region),
        )

    def detect_text_color(self, image: np.ndarray) -> tuple[int, int, int]:
        """使用K-Means聚类检测主要文字颜色。"""
        pixels = image.reshape(-1, 3).astype(np.float32)
        if len(pixels) < 3:
            return (0, 0, 0)
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        kmeans.fit(pixels)
        centers = kmeans.labels_
        unique, counts = np.unique(centers, return_counts=True)
        bg_cluster = unique[counts.argmax()]
        text_clusters = [i for i in range(3) if i != bg_cluster]
        if text_clusters:
            darkest = min(text_clusters, key=lambda i: np.sum(kmeans.cluster_centers_[i]))
            color = kmeans.cluster_centers_[darkest]
            return tuple(int(c) for c in np.clip(color, 0, 255))
        return (0, 0, 0)

    def detect_stroke(self, image: np.ndarray) -> dict | None:
        """使用Canny边缘检测判断文字是否有描边。"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[-1] == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        binary = (gray < 128).astype(np.uint8) * 255
        dilated = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)
        stroke_region = dilated - binary
        stroke_pixels = np.sum(stroke_region > 0)
        total_pixels = np.sum(binary > 0)
        if total_pixels > 0 and stroke_pixels / total_pixels > 0.1:
            return {"stroke_width": 2, "stroke_color": (255, 255, 255)}
        return None

    def detect_shadow(self, image: np.ndarray) -> tuple[int, int] | None:
        """通过亮度梯度分析检测投影阴影偏移。"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[-1] == 3 else image
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag_x = np.mean(np.abs(sobel_x))
        mag_y = np.mean(np.abs(sobel_y))
        if mag_x > 10 or mag_y > 10:
            dx = 2 if mag_x > mag_y else 1
            dy = 2 if mag_y > mag_x else 1
            return (dx, dy)
        return None

    def estimate_font_size(self, image: np.ndarray) -> int:
        """从文本区域高度估计字号。"""
        height = image.shape[0]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[-1] == 3 else image
        projection = np.mean(gray, axis=1)
        threshold = np.mean(projection) - 0.2 * np.std(projection)
        text_rows = np.sum(projection < threshold)
        if text_rows == 0:
            text_rows = height * 0.6
        estimated = int(height * 0.75)
        return max(8, min(48, estimated))

    def detect_font_weight(self, image: np.ndarray) -> str:
        """通过黑色像素占比检测字重（粗体/常规/细体）。"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[-1] == 3 else image
        binary = (gray < 128).astype(np.uint8)
        text_pixel_ratio = np.sum(binary) / binary.size
        if text_pixel_ratio > 0.25:
            return "bold"
        elif text_pixel_ratio < 0.08:
            return "light"
        return "regular"

    def detect_alignment(self, image: np.ndarray) -> str:
        """检测文本在图像内的水平对齐方式。"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[-1] == 3 else image
        binary = (gray < 128).astype(np.uint8)
        cols = np.sum(binary, axis=0)
        if cols.sum() == 0:
            return "center"
        center_of_mass = np.average(np.arange(len(cols)), weights=cols + 1)
        img_center = image.shape[1] / 2
        offset = (center_of_mass - img_center) / image.shape[1]
        if offset < -0.1:
            return "left"
        elif offset > 0.1:
            return "right"
        return "center"
