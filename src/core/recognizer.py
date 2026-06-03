"""M4：文本识别器 — PP-OCRv5 + SVTR骨干网络。"""
import re
from dataclasses import dataclass


@dataclass
class RecognitionResult:
    """单条OCR识别结果。"""
    text: str
    confidence: float
    language: str


class Recognizer:
    """使用PP-OCRv5 + SVTR进行多语种漫画文本OCR识别。

    自动检测文本方向（竖排/横排），运行SVTR识别，
    当置信度低于阈值时触发TACE三级增强并重试。
    """

    ISO_TO_PADDLE_LANG = {"ja": "japan", "zh": "ch", "en": "en"}

    def __init__(self, language: str = "japan") -> None:
        from src.core.model_manager import ModelManager
        from src.utils.config import config

        self._mgr = ModelManager()
        self._ocr = None
        self._language = self.ISO_TO_PADDLE_LANG.get(language, language)
        self._conf_threshold = config.get("pipeline.ocr.confidence_threshold", 0.7)
        self._fallback_enabled = config.get("pipeline.ocr.fallback_on_low_conf", True)

    @property
    def ocr(self):
        if self._ocr is None:
            self._ocr = self._mgr.get_model("ocr", "recognition", lang=self._language)
        return self._ocr

    def recognize(self, image_path: str) -> list[RecognitionResult]:
        """识别图像区域中的文本。"""
        results = self._run_ocr(image_path)
        if self._fallback_enabled:
            results = self._apply_fallback(image_path, results)
        return results

    def _run_ocr(self, image_path: str) -> list[RecognitionResult]:
        """执行PP-OCRv5 + SVTR识别。"""
        ocr_results = self.ocr.ocr(image_path, cls=True)

        results = []
        if ocr_results is None or ocr_results[0] is None:
            return results

        for line in ocr_results[0]:
            text = line[1][0]
            conf = line[1][1]
            lang = self._detect_language(text)
            results.append(RecognitionResult(
                text=text,
                confidence=conf,
                language=lang,
            ))
        return results

    def _apply_fallback(
        self, image_path: str, results: list[RecognitionResult]
    ) -> list[RecognitionResult]:
        """对低置信度结果触发TACE三级增强并重试。"""
        from src.core.enhancer import Enhancer

        fixed = []
        for r in results:
            if r.confidence < self._conf_threshold:
                import cv2
                enhancer = Enhancer()
                img = cv2.imread(image_path)
                if img is not None:
                    enhanced = enhancer.enhance(img, tier=3)
                    temp_path = image_path + ".enhanced.png"
                    cv2.imwrite(temp_path, enhanced)
                    retry_results = self._run_ocr(temp_path)
                    if retry_results:
                        fixed.extend(retry_results)
                        continue
            fixed.append(r)
        return fixed

    @staticmethod
    def _detect_language(text: str) -> str:
        """基于Unicode范围的启发式语言检测。"""
        if not text:
            return "unknown"
        has_hiragana = bool(re.search(r'[぀-ゟ]', text))
        has_katakana = bool(re.search(r'[゠-ヿ]', text))
        has_cjk = bool(re.search(r'[一-鿿]', text))
        has_latin = bool(re.search(r'[a-zA-Z]', text))

        if has_hiragana or has_katakana:
            return "ja"
        elif has_cjk:
            return "zh"
        elif has_latin:
            return "en"
        return "unknown"
