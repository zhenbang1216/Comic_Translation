"""端到端集成测试 — 验证全管线连通性。"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestIntegration:
    """使用模拟模型测试全管线端到端流程。"""

    def test_pipeline_flow_with_mocks(self, tmp_path):
        """验证所有模块正确连接。"""
        import cv2

        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Hello test", (250, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        img_path = str(tmp_path / "test_comic.png")
        cv2.imwrite(img_path, img)

        with patch("src.core.detector.Detector.detect") as mock_detect, \
             patch("src.core.recognizer.Recognizer.recognize") as mock_ocr, \
             patch("src.core.translator.translation_engine.TranslationEngine.translate_all") as mock_trans, \
             patch("src.core.inpainter.Inpainter.inpaint") as mock_inpaint:

            from src.utils.schemas import TextBox, Language, TranslationResult
            mock_detect.return_value = [
                TextBox(id="t1", x=240, y=170, width=120, height=50,
                        text="hello", language=Language.EN, confidence=0.9),
            ]
            from src.core.recognizer import RecognitionResult
            mock_ocr.return_value = [RecognitionResult(text="hello", confidence=0.92, language="en")]
            mock_trans.return_value = [
                TranslationResult(textbox_id="t1", original_text="hello",
                                  translated_text="你好", source_lang=Language.EN,
                                  target_lang=Language.ZH, translation_method="opus_mt",
                                  character_id=None, confidence=0.85),
            ]
            mock_inpaint.return_value = img

            from src.core.detector import Detector
            from src.core.translator.page_analyzer import PageAnalyzer
            from src.core.translator.dialogue_graph import DialogueGraph
            from src.core.translator.character_profile import CharacterProfileExtractor
            from src.core.translator.translation_engine import TranslationEngine
            from src.core.translator.consistency_checker import ConsistencyChecker
            from src.core.inpainter import Inpainter

            detector = Detector()
            boxes = detector.detect(img_path)
            assert len(boxes) == 1

            analyzer = PageAnalyzer()
            ordered = analyzer.determine_reading_order(boxes)
            assert len(ordered) == 1

            graph = DialogueGraph()
            nodes, edges = graph.build(ordered)
            assert len(nodes) == 1

            extractor = CharacterProfileExtractor()
            profile = extractor.extract_profile("char_1", ["hello"])
            assert profile.speech_style == "neutral"

            engine = TranslationEngine()
            translations = engine.translate_all(
                nodes, {"char_1": profile}, "daily", Language.ZH
            )
            assert len(translations) == 1
            assert translations[0].translated_text == "你好"

            checker = ConsistencyChecker()
            passed, failed = checker.check(translations)
            assert len(passed) == 1

            inpainter = Inpainter()
            box_dicts = [{"x": b.x, "y": b.y, "width": b.width, "height": b.height} for b in boxes]
            cleaned = inpainter.inpaint(img, box_dicts)
            assert cleaned.shape == img.shape
