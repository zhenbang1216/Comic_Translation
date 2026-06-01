"""M2：文本检测器 — YOLOv8-OBB + DBNet 双路检测。"""
import uuid
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """转换为TextBox之前的原始检测结果。"""
    x: float
    y: float
    width: float
    height: float
    angle: float
    confidence: float
    character_id: str | None = None


class Detector:
    """使用YOLOv8-OBB作为主力检测漫画图像中的文本区域。

    DBNet作为第二通路处理不规则文本形状。
    两路结果通过NMS合并去重。
    """

    def __init__(self) -> None:
        from src.core.model_manager import ModelManager
        from src.utils.config import config

        self._mgr = ModelManager()
        self._conf_threshold = config.get("pipeline.detection.confidence_threshold", 0.5)
        self._nms_iou = config.get("pipeline.detection.nms_iou_threshold", 0.45)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._mgr.get_model("detection", "yolo_obb")
        return self._model

    def detect(self, image_path: str) -> list:
        """在图像上运行检测，返回TextBox对象列表。"""
        from src.utils.schemas import TextBox

        results_obb = self._run_yolo_obb(image_path)
        textboxes = []
        for det in results_obb:
            tb = TextBox(
                id=str(uuid.uuid4())[:8],
                x=det.x,
                y=det.y,
                width=det.width,
                height=det.height,
                angle=det.angle,
                confidence=det.confidence,
                character_id=det.character_id,
            )
            textboxes.append(tb)
        return textboxes

    def _run_yolo_obb(self, image_path: str) -> list[DetectionResult]:
        """运行YOLOv8-OBB推理并解析结果。"""
        preds = self.model.predict(
            source=image_path,
            conf=self._conf_threshold,
            verbose=False,
        )

        results = []
        for pred in preds:
            if pred.boxes is None or len(pred.boxes) == 0:
                continue
            for box_data, conf_data in zip(pred.boxes.xyxy, pred.boxes.conf):
                if len(box_data) < 4:
                    continue
                x1, y1, x2, y2 = box_data[:4].tolist()
                conf = float(conf_data)
                if conf < self._conf_threshold:
                    continue
                angle = 0.0
                results.append(DetectionResult(
                    x=float(x1),
                    y=float(y1),
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                    angle=angle,
                    confidence=conf,
                ))
        return results

    def _merge_results(
        self,
        primary: list[DetectionResult],
        secondary: list[DetectionResult] | None = None,
        iou_threshold: float = 0.5,
    ) -> list[DetectionResult]:
        """合并主辅检测结果，按置信度排序后用NMS去重。"""
        all_dets = list(primary)
        if secondary:
            all_dets.extend(secondary)
        if len(all_dets) <= 1:
            return all_dets

        all_dets.sort(key=lambda d: d.confidence, reverse=True)
        kept = []
        for det in all_dets:
            if not any(self._compute_iou(det, k) > iou_threshold for k in kept):
                kept.append(det)
        return kept

    @staticmethod
    def _compute_iou(a: DetectionResult, b: DetectionResult) -> float:
        """计算两个检测框的IoU（交并比）。"""
        ax1, ay1 = a.x, a.y
        ax2, ay2 = a.x + a.width, a.y + a.height
        bx1, by1 = b.x, b.y
        bx2, by2 = b.x + b.width, b.y + b.height

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

        area_a = a.width * a.height
        area_b = b.width * b.height
        union_area = area_a + area_b - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
