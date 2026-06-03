"""后台工作线程，运行翻译管线而不阻塞GUI界面。"""
import os
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import QThread, Signal


class PipelineWorker(QThread):
    """在后台线程中运行完整的漫画翻译管线。

    发射进度信号让GUI更新进度条和状态标签，界面不会冻结。
    """

    progress = Signal(int, str)
    finished = Signal(str, str)
    error = Signal(str)

    def __init__(self, image_path: str, source_lang: str, target_lang: str) -> None:
        super().__init__()
        self._image_path = image_path
        self._source_lang = source_lang
        self._target_lang = target_lang

    def run(self) -> None:
        """按顺序执行完整管线。"""
        try:
            self.progress.emit(5, "加载模型...")
            img = cv2.imread(self._image_path)
            if img is None:
                self.error.emit("无法读取图像")
                return

            self.progress.emit(15, "检测文本区域(YOLO)...")
            from src.core.detector import Detector
            detector = Detector()
            boxes = detector.detect(self._image_path)
            self.progress.emit(25, f"检测到 {len(boxes)} 个文本区域")

            if not boxes:
                self.error.emit("未检测到文字区域")
                return

            # YOLO检测可视化
            det_img = img.copy()
            for box in boxes:
                x1 = max(0, int(box.x))
                y1 = max(0, int(box.y))
                x2 = min(img.shape[1], int(box.x + box.width))
                y2 = min(img.shape[0], int(box.y + box.height))
                cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            self.progress.emit(30, "识别文本内容(OCR)...")
            from src.core.recognizer import Recognizer
            recognizer = Recognizer()
            for box in boxes:
                x1 = max(0, int(box.x))
                y1 = max(0, int(box.y))
                x2 = min(img.shape[1], int(box.x + box.width))
                y2 = min(img.shape[0], int(box.y + box.height))
                if x2 > x1 and y2 > y1:
                    region = img[y1:y2, x1:x2]
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                        cv2.imwrite(f.name, region)
                        rec_results = recognizer.recognize(f.name)
                        os.unlink(f.name)
                    if rec_results:
                        box.text = rec_results[0].text

            self.progress.emit(50, "构建对话图...")
            from src.core.translator.page_analyzer import PageAnalyzer
            from src.core.translator.dialogue_graph import DialogueGraph
            analyzer = PageAnalyzer()
            ordered = analyzer.determine_reading_order(boxes)
            graph = DialogueGraph()
            nodes, edges = graph.build(ordered)

            self.progress.emit(60, "提取角色画像...")
            from src.core.translator.character_profile import CharacterProfileExtractor
            extractor = CharacterProfileExtractor()
            char_groups = {}
            for node in nodes:
                cid = node.character_id or "unknown"
                if cid not in char_groups:
                    char_groups[cid] = []
                char_groups[cid].append(node.text)
            profiles = {
                cid: extractor.extract_profile(cid, lines)
                for cid, lines in char_groups.items()
            }

            self.progress.emit(70, "翻译中...")
            from src.core.translator.translation_engine import TranslationEngine
            from src.utils.schemas import Language
            engine = TranslationEngine()
            target = Language.ZH if self._target_lang == "zh" else Language.EN
            translations = engine.translate_all(nodes, profiles, "unknown", target)

            self.progress.emit(80, "修复图像...")
            from src.core.inpainter import Inpainter
            inpainter = Inpainter()
            box_dicts = [
                {"x": b.x, "y": b.y, "width": b.width, "height": b.height}
                for b in boxes
            ]
            cleaned = inpainter.inpaint(img, box_dicts)

            self.progress.emit(90, "渲染译文...")
            from src.core.renderer.style_detector import StyleDetector
            from src.core.renderer.typesetter import Typesetter
            from src.core.renderer.painter import Painter
            from src.utils.config import config

            style_detector = StyleDetector()
            typesetter = Typesetter()
            painter = Painter()
            output_dir = config.get("paths.output_dir", "output/gui")
            os.makedirs(output_dir, exist_ok=True)

            stem = Path(self._image_path).stem
            for box, trans in zip(boxes, translations):
                x1 = max(0, int(box.x))
                y1 = max(0, int(box.y))
                x2 = min(cleaned.shape[1], int(box.x + box.width))
                y2 = min(cleaned.shape[0], int(box.y + box.height))
                if x2 <= x1 or y2 <= y1:
                    continue
                region = cleaned[y1:y2, x1:x2]
                style = style_detector.analyze(region)
                typeset_ = typesetter.compute(
                    trans.translated_text,
                    int(box.width), int(box.height),
                    trans.source_lang.value,
                    style,
                )
                cleaned = painter.render(cleaned, typeset_, style, (int(box.x), int(box.y)))

            translated_path = os.path.join(output_dir, f"{stem}_translated.png")
            cv2.imwrite(translated_path, cleaned)

            det_path = os.path.join(output_dir, f"{stem}_yolo_detect.png")
            cv2.imwrite(det_path, det_img)

            # 翻译对照日志
            log_path = os.path.join(output_dir, f"{stem}_log.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                for box, trans in zip(boxes, translations):
                    f.write(f"{box.text} → {trans.translated_text}\n")

            self.progress.emit(100, "完成!")
            self.finished.emit(translated_path, det_path)

        except Exception as e:
            self.error.emit(str(e))
