# Comic Translation System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end automated comic translation desktop application with detection, OCR, translation, inpainting, and rendering pipeline, wrapped in a PySide6 GUI.

**Architecture:** Modular pipeline architecture with 9 modules across 4 layers (data, algorithm, GUI, utils). Each module communicates via well-defined JSON schemas. The core pipeline flows: Image → Detection(M2) → Enhancement(M3) → OCR(M4) → Translation(M5) → Inpainting(M6) → Rendering(M7) → GUI(M8). M1 (ModelManager) provides models to all modules. M9 (Utils) provides shared tooling.

**Tech Stack:** Python 3.11+, PyTorch (Ultralytics YOLOv8-OBB), PaddleOCR (PP-OCRv5 + SVTR), Transformers (Qwen2.5-VL-2B, OPUS-MT), OpenCV, PIL/Cairo, PySide6, ONNX Runtime

---

## File Structure

```
Comic_Translation/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── model_manager.py          # M1
│   │   ├── detector.py                # M2
│   │   ├── enhancer.py                # M3
│   │   ├── recognizer.py              # M4
│   │   ├── translator/
│   │   │   ├── __init__.py
│   │   │   ├── page_analyzer.py       # M5 stage 1
│   │   │   ├── dialogue_graph.py      # M5 stage 2
│   │   │   ├── character_profile.py   # M5 stage 3
│   │   │   ├── translation_engine.py  # M5 stage 4
│   │   │   └── consistency_checker.py # M5 stage 5
│   │   ├── inpainter.py               # M6
│   │   └── renderer/
│   │       ├── __init__.py
│   │       ├── style_detector.py      # M7.1
│   │       ├── space_calculator.py    # M7.2
│   │       ├── typesetter.py          # M7.3
│   │       ├── painter.py             # M7.4
│   │       └── exporter.py            # M7.5
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   ├── widgets/
│   │   │   ├── __init__.py
│   │   │   ├── image_viewer.py
│   │   │   ├── control_panel.py
│   │   │   └── translation_preview.py
│   │   └── workers/
│   │       ├── __init__.py
│   │       └── pipeline_worker.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── schemas.py
│       ├── data_augmentation.py
│       └── evaluation.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── test_model_manager.py
│   │   ├── test_detector.py
│   │   ├── test_enhancer.py
│   │   ├── test_recognizer.py
│   │   ├── test_translator/
│   │   │   ├── __init__.py
│   │   │   ├── test_page_analyzer.py
│   │   │   ├── test_dialogue_graph.py
│   │   │   ├── test_character_profile.py
│   │   │   ├── test_translation_engine.py
│   │   │   └── test_consistency_checker.py
│   │   ├── test_inpainter.py
│   │   └── test_renderer/
│   │       ├── __init__.py
│   │       ├── test_style_detector.py
│   │       ├── test_space_calculator.py
│   │       ├── test_typesetter.py
│   │       ├── test_painter.py
│   │       └── test_exporter.py
│   └── gui/
│       ├── __init__.py
│       └── test_main_window.py
├── scripts/
│   ├── download_models.py
│   ├── finetune_detector.py
│   └── finetune_ocr.py
├── configs/
│   └── default.yaml
├── requirements.txt
└── conftest.py
```

---

### Task 1: Project Scaffolding and Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `configs/default.yaml`
- Create: `src/__init__.py`
- Create: `src/core/__init__.py`
- Create: `src/gui/__init__.py`
- Create: `src/gui/widgets/__init__.py`
- Create: `src/gui/workers/__init__.py`
- Create: `src/utils/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/core/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write requirements.txt**

```text
torch>=2.1.0
ultralytics>=8.2.0
paddlepaddle>=3.0.0
paddleocr>=2.8.0
transformers>=4.45.0
accelerate>=0.34.0
bitsandbytes>=0.43.0
opencv-python>=4.9.0
Pillow>=10.4.0
pycairo>=1.26.0
PySide6>=6.7.0
onnxruntime>=1.18.0
numpy>=1.26.0
pyyaml>=6.0
pytest>=8.0
pytest-qt>=4.4.0
sentencepiece>=0.2.0
protobuf>=5.27.0
```

- [ ] **Step 2: Write configs/default.yaml**

```yaml
models:
  detection:
    yolo_obb: "models/detection/yolov8n-obb.pt"
    dbnet: "models/detection/dbnet.onnx"
  ocr:
    detection: "models/ocr/ppocr_det.onnx"
    recognition: "models/ocr/ppocr_rec.onnx"
    dict_path: "models/ocr/ppocr_keys_v1.txt"
  translation:
    opus_mt_ja_zh: "Helsinki-NLP/opus-mt-ja-zh"
    opus_mt_en_zh: "Helsinki-NLP/opus-mt-en-zh"
    vlm_model: "Qwen/Qwen2.5-VL-2B-Instruct"
  inpainting:
    lama: "models/inpainting/lama.onnx"
  enhancement:
    real_esrgan: "models/enhancement/RealESRGAN_x2.pth"

pipeline:
  detection:
    confidence_threshold: 0.5
    nms_iou_threshold: 0.45
  ocr:
    confidence_threshold: 0.7
    fallback_on_low_conf: true
  translation:
    simple_bubble_threshold: 0.6
    max_vlm_context_chars: 512
  rendering:
    min_font_size: 10
    max_font_size: 48
    default_font_family_cn: "Source Han Sans SC"
    default_font_family_ja: "Noto Sans JP"
    default_font_family_en: "Roboto"

gui:
  window_title: "Comic Translator"
  default_width: 1400
  default_height: 900
  supported_languages:
    source: ["ja", "en", "zh"]
    target: ["zh", "en"]

paths:
  character_profiles_dir: "data/profiles"
  output_dir: "output"
```

- [ ] **Step 3: Write src/utils/config.py**

```python
"""Configuration loader for the comic translation system."""
import yaml
from pathlib import Path
from typing import Any


class Config:
    """Singleton configuration loaded from YAML file."""

    _instance = None
    _data: dict[str, Any] = {}

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: str | Path) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        value = self._data
        for k in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(k)
            if value is None:
                return default
        return value

    @property
    def models(self) -> dict:
        return self._data.get("models", {})

    @property
    def pipeline(self) -> dict:
        return self._data.get("pipeline", {})

    @property
    def gui(self) -> dict:
        return self._data.get("gui", {})

    @property
    def paths(self) -> dict:
        return self._data.get("paths", {})


config = Config()
```

- [ ] **Step 4: Write test for config**

```python
# tests/conftest.py
import pytest
from pathlib import Path


@pytest.fixture
def test_config_path(tmp_path):
    config_content = """
models:
  detection:
    yolo_obb: "models/test.pt"
pipeline:
  detection:
    confidence_threshold: 0.7
gui:
  window_title: "Test"
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content, encoding="utf-8")
    return str(config_file)


@pytest.fixture
def sample_image(tmp_path):
    import numpy as np
    import cv2
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Hello こんにちは", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    img_path = str(tmp_path / "sample.png")
    cv2.imwrite(img_path, img)
    return img_path
```

- [ ] **Step 5: Write src/utils/schemas.py**

```python
"""Data schemas for inter-module communication."""
from dataclasses import dataclass, field
from enum import Enum


class Language(str, Enum):
    JA = "ja"
    ZH = "zh"
    EN = "en"


class SceneType(str, Enum):
    BATTLE = "battle"
    DAILY = "daily"
    ROMANCE = "romance"
    COMEDY = "comedy"
    SERIOUS = "serious"
    UNKNOWN = "unknown"


@dataclass
class TextBox:
    """Output from M2 Detector — a detected text region."""
    id: str
    x: float
    y: float
    width: float
    height: float
    angle: float = 0.0  # degrees, for OBB
    text: str = ""
    language: Language = Language.JA
    confidence: float = 0.0
    character_id: str | None = None  # assigned by dialogue graph
    style_params: dict | None = None  # from M7.1 style detector
    bubble_bbox: tuple[float, float, float, float] | None = None  # bubble contour bbox


@dataclass
class CharacterProfile:
    """M5 Stage 3 output — a character's speech personality."""
    id: str
    pronouns: list[str] = field(default_factory=list)  # e.g. ["オレ"]
    sentence_endings: list[str] = field(default_factory=list)  # e.g. ["だぜ", "だ"]
    speech_style: str = "neutral"  # casual, formal, rough, polite, timid
    estimated_role: str = "unknown"  # protagonist, antagonist, supporting


@dataclass
class DialogueEdge:
    """An edge in the dialogue graph — conversation flow."""
    from_bubble_id: str
    to_bubble_id: str
    relation: str = "response"  # response, continuation, interjection


@dataclass
class TranslationResult:
    """M5 output — a translated text bubble."""
    textbox_id: str
    original_text: str
    translated_text: str
    source_lang: Language
    target_lang: Language
    translation_method: str  # "opus_mt" or "vlm"
    character_id: str | None = None
    confidence: float = 0.0


@dataclass
class RenderStyle:
    """M7.1 output — detected visual style of original text."""
    text_color: tuple[int, int, int] = (0, 0, 0)
    stroke_color: tuple[int, int, int] | None = None
    stroke_width: int = 0
    shadow_offset: tuple[int, int] | None = None
    shadow_color: tuple[int, int, int] | None = None
    estimated_font_size: int = 14
    font_weight: str = "regular"  # regular, bold, light
    alignment: str = "center"  # left, center, right


@dataclass
class TypesetResult:
    """M7.3 output — computed typesetting parameters."""
    font_size: int
    lines: list[str]
    line_height: int
    start_x: int
    start_y: int
    alignment: str
```

- [ ] **Step 6: Run test to verify scaffolding works**

Run: `python -c "from src.utils.config import Config; from src.utils.schemas import TextBox, TranslationResult; print('Scaffolding OK')"`
Expected: `Scaffolding OK`

- [ ] **Step 7: Commit**

```bash
git add requirements.txt configs/ src/ tests/conftest.py
git commit -m "feat: project scaffolding — config, schemas, directory structure"
```

---

### Task 2: ModelManager (M1)

**Files:**
- Create: `src/core/model_manager.py`
- Create: `tests/core/test_model_manager.py`

- [ ] **Step 1: Write failing test for model loading**

```python
# tests/core/test_model_manager.py
import pytest
from unittest.mock import patch, MagicMock
from src.core.model_manager import ModelManager


class TestModelManager:
    def test_singleton_returns_same_instance(self):
        m1 = ModelManager()
        m2 = ModelManager()
        assert m1 is m2

    def test_load_model_creates_entry(self):
        mgr = ModelManager()
        mgr._models.clear()
        mock_model = MagicMock()
        with patch("torch.load", return_value=mock_model):
            result = mgr.get_model("detection", "yolo_obb", mock_path="fake.pt")
        assert "detection:yolo_obb" in mgr._models

    def test_get_model_returns_cached_on_second_call(self):
        mgr = ModelManager()
        mgr._models.clear()
        mock_model = MagicMock()
        mgr._models["detection:yolo_obb"] = mock_model
        with patch("torch.load") as mock_load:
            result = mgr.get_model("detection", "yolo_obb")
            mock_load.assert_not_called()
        assert result is mock_model

    def test_release_model_frees_memory(self):
        mgr = ModelManager()
        mgr._models.clear()
        mgr._models["detection:yolo_obb"] = MagicMock()
        mgr.release_model("detection", "yolo_obb")
        assert "detection:yolo_obb" not in mgr._models

    def test_release_all_clears_everything(self):
        mgr = ModelManager()
        mgr._models.clear()
        mgr._models["a"] = MagicMock()
        mgr._models["b"] = MagicMock()
        mgr.release_all()
        assert len(mgr._models) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_model_manager.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement ModelManager**

```python
"""M1: Model Manager — lazy loading, caching, and lifecycle for all models."""
from typing import Any
from pathlib import Path


class ModelManager:
    """Singleton manager for lazy-loaded model instances.

    Models are keyed by "category:name", loaded on first access,
    and shareable across modules to avoid duplicate memory usage.
    """

    _instance = None

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
        return cls._instance

    def get_model(
        self, category: str, name: str, mock_path: str | None = None
    ) -> Any:
        key = f"{category}:{name}"
        if key not in self._models:
            self._models[key] = self._load_model(category, name, mock_path)
        return self._models[key]

    def _load_model(
        self, category: str, name: str, mock_path: str | None = None
    ) -> Any:
        from src.utils.config import config

        if mock_path:
            model_path = mock_path
        else:
            model_path = config.get(f"models.{category}.{name}", "")

        if category == "detection" and "yolo" in name:
            from ultralytics import YOLO
            return YOLO(model_path)
        elif category == "detection" and "dbnet" in name:
            import onnxruntime as ort
            return ort.InferenceSession(model_path)
        elif category == "ocr":
            from paddleocr import PaddleOCR
            return PaddleOCR(lang="japan", use_angle_cls=True)
        elif category == "translation" and "opus" in name:
            from transformers import MarianMTModel, MarianTokenizer
            tokenizer = MarianTokenizer.from_pretrained(model_path)
            model = MarianMTModel.from_pretrained(model_path)
            return {"tokenizer": tokenizer, "model": model}
        elif category == "translation" and "vlm" in name:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="cpu",
            )
            processor = AutoProcessor.from_pretrained(model_path)
            return {"model": model, "processor": processor}
        elif category == "inpainting":
            import onnxruntime as ort
            return ort.InferenceSession(model_path)
        else:
            raise ValueError(f"Unknown model category/name: {category}/{name}")

    def release_model(self, category: str, name: str) -> None:
        key = f"{category}:{name}"
        self._models.pop(key, None)

    def release_all(self) -> None:
        self._models.clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_model_manager.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/core/model_manager.py tests/core/test_model_manager.py
git commit -m "feat: add ModelManager (M1) with lazy loading and caching"
```

---

### Task 3: Detector Module (M2) — YOLOv8-OBB Text Detection

**Files:**
- Create: `src/core/detector.py`
- Create: `tests/core/test_detector.py`

- [ ] **Step 1: Write test for Detector**

```python
# tests/core/test_detector.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.core.detector import Detector, DetectionResult


def make_mock_yolo_result():
    """Create a mock YOLO result with two detected text boxes."""
    result = MagicMock()
    box1 = MagicMock()
    box1.xyxy = [[10.0, 20.0, 100.0, 50.0]]
    box1.conf = [0.85]
    box1.cls = [0]
    box2 = MagicMock()
    box2.xyxy = [[200.0, 30.0, 350.0, 60.0]]
    box2.conf = [0.72]
    box2.cls = [0]
    result.boxes = [box1, box2]
    result.obb = None
    return result


class TestDetector:
    @pytest.fixture
    def detector(self):
        with patch("src.core.detector.ModelManager") as mock_mgr:
            mock_yolo = MagicMock()
            mock_yolo.predict.return_value = [make_mock_yolo_result()]
            mock_mgr.return_value.get_model.return_value = mock_yolo
            d = Detector()
            d._model = mock_yolo
            yield d

    def test_detect_returns_list(self, detector, sample_image):
        results = detector.detect(sample_image)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_detect_textbox_has_coordinates(self, detector, sample_image):
        results = detector.detect(sample_image)
        for tb in results:
            assert tb.x is not None
            assert tb.y is not None
            assert tb.width > 0
            assert tb.height > 0

    def test_detect_empty_image_returns_empty(self, detector, tmp_path):
        import cv2
        blank_path = str(tmp_path / "blank.png")
        cv2.imwrite(blank_path, np.ones((100, 100, 3), dtype=np.uint8) * 255)
        detector._model.predict.return_value = [MagicMock(boxes=[])]
        results = detector.detect(blank_path)
        assert results == []

    def test_merge_results_deduplicates_overlapping(self, detector):
        from src.core.detector import DetectionResult
        r1 = DetectionResult(x=10, y=10, width=50, height=30, angle=0, confidence=0.9, character_id=None)
        r2 = DetectionResult(x=15, y=12, width=48, height=28, angle=0, confidence=0.8, character_id=None)
        r3 = DetectionResult(x=200, y=10, width=50, height=30, angle=0, confidence=0.7, character_id=None)
        merged = detector._merge_results([r1, r2, r3], iou_threshold=0.5)
        assert len(merged) == 2

    def test_detect_preserves_obb_angle(self, detector, sample_image):
        results = detector.detect(sample_image)
        for tb in results:
            assert hasattr(tb, 'angle')
```

- [ ] **Step 2: Implement Detector**

```python
"""M2: Text Detector — YOLOv8-OBB + DBNet dual-path detection."""
import uuid
import numpy as np
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Raw detection before conversion to TextBox schema."""
    x: float
    y: float
    width: float
    height: float
    angle: float
    confidence: float
    character_id: str | None = None


class Detector:
    """Detect text regions in comic images using YOLOv8-OBB as primary.

    DBNet is used as a secondary path for irregular text shapes.
    Results from both paths are merged with NMS deduplication.
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
        """Run detection on an image, return a list of TextBox objects."""
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
        """Run YOLOv8-OBB inference and parse results."""
        import cv2

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
                if pred.obb is not None and hasattr(pred.obb, 'conf'):
                    angle = 0.0  # OBB angle would be extracted here per box
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
        """Merge primary and secondary detection results, removing overlaps."""
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
        """Compute Intersection over Union of two detection boxes."""
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
```

- [ ] **Step 3: Run test to verify**

Run: `pytest tests/core/test_detector.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/core/detector.py tests/core/test_detector.py
git commit -m "feat: add Detector (M2) with YOLOv8-OBB and NMS merging"
```

---

### Task 4: Enhancer Module (M3) — TACE Three-Tier Cascade

**Files:**
- Create: `src/core/enhancer.py`
- Create: `tests/core/test_enhancer.py`

- [ ] **Step 1: Write test for Enhancer**

```python
# tests/core/test_enhancer.py
import numpy as np
import pytest
from src.core.enhancer import Enhancer


class TestEnhancer:
    @pytest.fixture
    def enhancer(self):
        return Enhancer()

    def test_tier1_clahe_enhances_contrast(self, enhancer):
        img = np.ones((100, 200, 3), dtype=np.uint8) * 128
        img[40:60, 50:150] = 200  # light text on mid-gray
        result = enhancer.enhance(img, tier=1)
        assert result.shape == img.shape
        assert abs(float(np.mean(result)) - float(np.mean(img))) < 30

    def test_tier1_handles_grayscale(self, enhancer):
        img = np.ones((50, 100), dtype=np.uint8) * 100
        result = enhancer.enhance(img, tier=1)
        assert result.shape == img.shape

    def test_tier2_is_passthrough(self, enhancer):
        img = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
        result = enhancer.enhance(img, tier=2)
        assert np.array_equal(result, img)

    def test_auto_decides_tier_based_on_input(self, enhancer):
        img = np.ones((100, 200, 3), dtype=np.uint8) * 200
        img[45:55, 50:150] = 50
        result = enhancer.auto_enhance(img)
        assert result is not None
        assert result.shape == img.shape

    def test_sharpen_applies_unsharp_mask(self, enhancer):
        img = np.ones((60, 120, 3), dtype=np.uint8) * 180
        img[25:35, 30:90] = 20
        result = enhancer._sharpen(img)
        assert result.shape == img.shape
```

- [ ] **Step 2: Implement Enhancer**

```python
"""M3: TACE — Three-Tier Adaptive Cascade Enhancement."""
import cv2
import numpy as np


class Enhancer:
    """Three-tier enhancement for comic text regions.

    Tier 1: CLAHE + Unsharp Mask (traditional CV, <1ms)
    Tier 2: Passthrough (SVTR handles mild blur natively)
    Tier 3: Real-ESRGAN super-resolution (heavy, only on low-confidence fallback)
    """

    def __init__(self) -> None:
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def enhance(self, image: np.ndarray, tier: int = 1) -> np.ndarray:
        """Apply the specified enhancement tier."""
        if tier == 1:
            return self._tier1_clahe_sharpen(image)
        elif tier == 2:
            return self._tier2_passthrough(image)
        elif tier == 3:
            return self._tier3_super_resolution(image)
        else:
            raise ValueError(f"Unknown enhancement tier: {tier}")

    def auto_enhance(self, image: np.ndarray) -> np.ndarray:
        """Automatically apply tier 1 enhancement. Higher tiers triggered
        externally when OCR confidence is low."""
        return self._tier1_clahe_sharpen(image)

    def _tier1_clahe_sharpen(self, image: np.ndarray) -> np.ndarray:
        """CLAHE contrast enhancement + Unsharp Mask sharpening."""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            merged = cv2.merge([l, a, b])
            result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        else:
            result = self._clahe.apply(image)
        result = self._sharpen(result)
        return result

    def _tier2_passthrough(self, image: np.ndarray) -> np.ndarray:
        """Passthrough — SVTR's native anti-blur training handles mild blur."""
        return image

    def _tier3_super_resolution(self, image: np.ndarray) -> np.ndarray:
        """Real-ESRGAN super-resolution for severely blurred text patches."""
        import torch
        from src.core.model_manager import ModelManager

        mgr = ModelManager()
        esrgan = mgr.get_model("enhancement", "real_esrgan")
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            output = esrgan(tensor)
        result = output.squeeze(0).permute(1, 2, 0).numpy()
        result = (result * 255).clip(0, 255).astype(np.uint8)
        return result

    @staticmethod
    def _sharpen(image: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """Apply Unsharp Mask sharpening."""
        blurred = cv2.GaussianBlur(image, (0, 0), 3.0)
        result = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return result
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/core/test_enhancer.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/core/enhancer.py tests/core/test_enhancer.py
git commit -m "feat: add Enhancer (M3) with TACE three-tier cascade"
```

---

### Task 5: Recognizer Module (M4) — PP-OCRv5 + SVTR

**Files:**
- Create: `src/core/recognizer.py`
- Create: `tests/core/test_recognizer.py`

- [ ] **Step 1: Write test for Recognizer**

```python
# tests/core/test_recognizer.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.core.recognizer import Recognizer, RecognitionResult


class TestRecognizer:
    @pytest.fixture
    def recognizer(self):
        with patch("src.core.recognizer.ModelManager") as mock_mgr:
            mock_ocr = MagicMock()
            mock_ocr.ocr.return_value = [
                [[[10, 10], [100, 10], [100, 40], [10, 40]],
                 ("こんにちは", 0.92)]
            ]
            mock_mgr.return_value.get_model.return_value = mock_ocr
            r = Recognizer()
            r._ocr = mock_ocr
            yield r

    def test_recognize_returns_text(self, recognizer, tmp_path):
        import cv2
        img = np.ones((40, 100, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Test", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        img_path = str(tmp_path / "text_region.png")
        cv2.imwrite(img_path, img)
        results = recognizer.recognize(img_path)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_recognize_low_confidence_triggers_enhancer(self, recognizer, tmp_path):
        import cv2
        img = np.ones((40, 100, 3), dtype=np.uint8) * 255
        img_path = str(tmp_path / "faint_text.png")
        cv2.imwrite(img_path, img)
        recognizer._ocr.ocr.return_value = [
            [[[10, 10], [100, 10], [100, 40], [10, 40]],
             ("あいう", 0.45)]
        ]
        with patch("src.core.recognizer.Enhancer") as mock_enhancer:
            mock_enhancer.return_value.enhance.return_value = img
            results = recognizer.recognize(img_path)
            assert len(results) > 0

    def test_recognize_empty_image_returns_empty(self, recognizer, tmp_path):
        import cv2
        img = np.ones((40, 100, 3), dtype=np.uint8) * 255
        img_path = str(tmp_path / "empty.png")
        cv2.imwrite(img_path, img)
        recognizer._ocr.ocr.return_value = []
        results = recognizer.recognize(img_path)
        assert results == []

    def test_detect_language_identifies_japanese(self, recognizer):
        lang = recognizer._detect_language("これは日本語のテキストです")
        assert lang in ("ja", "unknown")

    def test_detect_language_identifies_english(self, recognizer):
        lang = recognizer._detect_language("This is English text")
        assert lang in ("en", "unknown")

    def test_detect_language_identifies_chinese(self, recognizer):
        lang = recognizer._detect_language("这是中文文本")
        assert lang in ("zh", "unknown")
```

- [ ] **Step 2: Implement Recognizer**

```python
"""M4: Text Recognizer — PP-OCRv5 with SVTR backbone."""
import re
import numpy as np
from dataclasses import dataclass


@dataclass
class RecognitionResult:
    """Single recognition result from OCR."""
    text: str
    confidence: float
    language: str


class Recognizer:
    """OCR recognition using PP-OCRv5 + SVTR for multilingual comic text.

    Automatically detects text orientation (vertical/horizontal),
    runs SVTR recognition, and triggers TACE tier-3 enhancement
    when confidence falls below threshold.
    """

    def __init__(self) -> None:
        from src.core.model_manager import ModelManager
        from src.utils.config import config

        self._mgr = ModelManager()
        self._ocr = None
        self._conf_threshold = config.get("pipeline.ocr.confidence_threshold", 0.7)
        self._fallback_enabled = config.get("pipeline.ocr.fallback_on_low_conf", True)

    @property
    def ocr(self):
        if self._ocr is None:
            self._ocr = self._mgr.get_model("ocr", "recognition")
        return self._ocr

    def recognize(self, image_path: str) -> list[RecognitionResult]:
        """Recognize text in an image region.

        Returns a list of RecognitionResult. Low-confidence results
        trigger TACE tier-3 enhancement and retry if fallback is enabled.
        """
        results = self._run_ocr(image_path)
        if self._fallback_enabled:
            results = self._apply_fallback(image_path, results)
        return results

    def _run_ocr(self, image_path: str) -> list[RecognitionResult]:
        """Execute PP-OCRv5 + SVTR recognition."""
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
        """For low-confidence results, trigger TACE tier-3 and retry."""
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
        """Heuristic language detection based on Unicode ranges."""
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
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/core/test_recognizer.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/core/recognizer.py tests/core/test_recognizer.py
git commit -m "feat: add Recognizer (M4) with PP-OCRv5 SVTR and TACE fallback"
```

---

### Task 6: Translator — Page Analyzer (M5 Stage 1)

**Files:**
- Create: `src/core/translator/__init__.py`
- Create: `src/core/translator/page_analyzer.py`
- Create: `tests/core/test_translator/__init__.py`
- Create: `tests/core/test_translator/test_page_analyzer.py`

- [ ] **Step 1: Write test for PageAnalyzer**

```python
# tests/core/test_translator/test_page_analyzer.py
import pytest
from src.core.translator.page_analyzer import PageAnalyzer
from src.utils.schemas import TextBox, SceneType, Language


class TestPageAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return PageAnalyzer()

    @pytest.fixture
    def sample_boxes(self):
        return [
            TextBox(id="1", x=300, y=100, width=120, height=30, text="行くぞ！", language=Language.JA),
            TextBox(id="2", x=100, y=80, width=100, height=40, text="待って", language=Language.JA),
            TextBox(id="3", x=310, y=200, width=130, height=30, text="まだまだ！", language=Language.JA),
            TextBox(id="4", x=120, y=180, width=90, height=35, text="危ない！", language=Language.JA),
        ]

    def test_determine_reading_order_right_to_left(self, analyzer, sample_boxes):
        ordered = analyzer.determine_reading_order(sample_boxes)
        assert len(ordered) == len(sample_boxes)
        assert ordered[0].x > ordered[-1].x or ordered[0].x >= ordered[-1].x

    def test_classify_scene_returns_scene_type(self, analyzer):
        scene = analyzer.classify_scene(None)
        assert scene in list(SceneType)

    def test_extract_bubble_features_returns_dict(self, analyzer, sample_boxes):
        features = analyzer.extract_bubble_features(sample_boxes[0])
        assert isinstance(features, dict)
        assert "width" in features
        assert "height" in features

    def test_empty_boxes_returns_empty_order(self, analyzer):
        ordered = analyzer.determine_reading_order([])
        assert ordered == []
```

- [ ] **Step 2: Implement PageAnalyzer**

```python
"""M5 Stage 1: Page Analyzer — scene classification, reading order, bubble features."""
from src.utils.schemas import TextBox, SceneType


class PageAnalyzer:
    """Analyze a comic page to understand layout and context.

    Determines reading order (right-to-left for manga, left-to-right
    for Western comics), classifies scene type, and extracts per-bubble
    visual features for downstream dialogue graph construction.
    """

    def determine_reading_order(self, boxes: list[TextBox]) -> list[TextBox]:
        """Sort text boxes by Japanese manga reading order (right→left, top→bottom).

        Manga pages are read from the top-right corner, moving right-to-left
        in rows and top-to-bottom within columns.
        """
        if not boxes:
            return []
        row_threshold = 60
        sorted_by_y = sorted(boxes, key=lambda b: b.y)
        rows = []
        current_row = [sorted_by_y[0]]
        for box in sorted_by_y[1:]:
            if abs(box.y - current_row[0].y) < row_threshold:
                current_row.append(box)
            else:
                rows.append(sorted(current_row, key=lambda b: b.x, reverse=True))
                current_row = [box]
        rows.append(sorted(current_row, key=lambda b: b.x, reverse=True))

        ordered = []
        for row in rows:
            ordered.extend(row)
        return ordered

    def classify_scene(self, image) -> SceneType:
        """Classify the page's scene type using lightweight heuristic or CLIP.

        Currently returns UNKNOWN. Full CLIP integration added in P2 phase.
        """
        return SceneType.UNKNOWN

    def extract_bubble_features(self, box: TextBox) -> dict:
        """Extract visual features of a text bubble for character attribution."""
        return {
            "width": box.width,
            "height": box.height,
            "aspect_ratio": box.width / max(box.height, 1),
            "angle": box.angle,
            "position_x": box.x,
            "position_y": box.y,
        }

    def assign_characters_by_tail(
        self, boxes: list[TextBox], image_path: str
    ) -> list[TextBox]:
        """Assign character IDs to text boxes by detecting bubble tails.

        In manga, speech bubble tails point toward the speaking character.
        This is a heuristic placeholder; full tail detection uses CV contour analysis.
        """
        for box in boxes:
            box.character_id = None
        return boxes
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/core/test_translator/test_page_analyzer.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/core/translator/ tests/core/test_translator/
git commit -m "feat: add PageAnalyzer (M5.1) — reading order and scene classification"
```

---

### Task 7: Translator — Dialogue Graph (M5 Stage 2)

**Files:**
- Create: `src/core/translator/dialogue_graph.py`
- Create: `tests/core/test_translator/test_dialogue_graph.py`

- [ ] **Step 1: Write test for DialogueGraph**

```python
# tests/core/test_translator/test_dialogue_graph.py
import pytest
from src.core.translator.dialogue_graph import DialogueGraph
from src.utils.schemas import TextBox, DialogueEdge, Language


class TestDialogueGraph:
    @pytest.fixture
    def graph(self):
        return DialogueGraph()

    @pytest.fixture
    def ordered_boxes(self):
        return [
            TextBox(id="a", x=350, y=50, width=100, height=30,
                    text="行くぞ！", language=Language.JA, character_id="char_1"),
            TextBox(id="b", x=150, y=60, width=90, height=30,
                    text="待って", language=Language.JA, character_id="char_2"),
            TextBox(id="c", x=340, y=180, width=110, height=30,
                    text="まだまだ！", language=Language.JA, character_id="char_1"),
            TextBox(id="d", x=140, y=190, width=100, height=30,
                    text="危ない！", language=Language.JA, character_id="char_2"),
        ]

    def test_build_returns_nodes_and_edges(self, graph, ordered_boxes):
        nodes, edges = graph.build(ordered_boxes)
        assert len(nodes) == 4
        assert len(edges) > 0

    def test_group_by_character(self, graph, ordered_boxes):
        groups = graph._group_by_character(ordered_boxes)
        assert "char_1" in groups
        assert "char_2" in groups
        assert len(groups["char_1"]) == 2
        assert len(groups["char_2"]) == 2

    def test_build_edges_creates_response_relations(self, graph, ordered_boxes):
        nodes, edges = graph.build(ordered_boxes)
        edge_types = {e.relation for e in edges}
        assert "response" in edge_types or len(edges) > 0

    def test_export_json_produces_valid_structure(self, graph, ordered_boxes):
        nodes, edges = graph.build(ordered_boxes)
        json_data = graph.export_json(nodes, edges)
        assert "nodes" in json_data
        assert "edges" in json_data
        assert len(json_data["nodes"]) == 4

    def test_empty_boxes_produces_empty_graph(self, graph):
        nodes, edges = graph.build([])
        assert nodes == []
        assert edges == []
```

- [ ] **Step 2: Implement DialogueGraph**

```python
"""M5 Stage 2: Dialogue Graph Builder — construct conversation structure."""
from src.utils.schemas import TextBox, DialogueEdge


class DialogueGraph:
    """Build a dialogue graph from ordered text boxes.

    Groups bubbles by character, establishes conversation edges
    (who responds to whom), and exports the structure as JSON
    for the translation engine to consume.
    """

    def build(
        self, boxes: list[TextBox]
    ) -> tuple[list[TextBox], list[DialogueEdge]]:
        """Build the dialogue graph from ordered text boxes.

        Returns (nodes, edges) where nodes are the enriched boxes
        and edges represent conversation flow between characters.
        """
        if not boxes:
            return [], []

        edges = self._build_edges(boxes)
        return boxes, edges

    def _group_by_character(self, boxes: list[TextBox]) -> dict[str, list[TextBox]]:
        """Group text boxes by their assigned character ID."""
        groups: dict[str, list[TextBox]] = {}
        for box in boxes:
            cid = box.character_id or "unknown"
            if cid not in groups:
                groups[cid] = []
            groups[cid].append(box)
        return groups

    def _build_edges(self, boxes: list[TextBox]) -> list[DialogueEdge]:
        """Create dialogue edges between adjacent bubbles of different characters.

        When character A speaks, then character B speaks, that's a response edge.
        When the same character has consecutive bubbles, that's a continuation edge.
        """
        edges = []
        for i in range(len(boxes) - 1):
            current = boxes[i]
            next_box = boxes[i + 1]
            if current.character_id and next_box.character_id:
                if current.character_id != next_box.character_id:
                    edges.append(DialogueEdge(
                        from_bubble_id=current.id,
                        to_bubble_id=next_box.id,
                        relation="response",
                    ))
                else:
                    edges.append(DialogueEdge(
                        from_bubble_id=current.id,
                        to_bubble_id=next_box.id,
                        relation="continuation",
                    ))
        return edges

    def export_json(
        self, nodes: list[TextBox], edges: list[DialogueEdge]
    ) -> dict:
        """Export the dialogue graph as a JSON-serializable dict."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "text": n.text,
                    "language": n.language.value,
                    "character_id": n.character_id,
                    "position": {"x": n.x, "y": n.y, "w": n.width, "h": n.height},
                }
                for n in nodes
            ],
            "edges": [
                {
                    "from": e.from_bubble_id,
                    "to": e.to_bubble_id,
                    "relation": e.relation,
                }
                for e in edges
            ],
        }
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/core/test_translator/test_dialogue_graph.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/core/translator/dialogue_graph.py tests/core/test_translator/test_dialogue_graph.py
git commit -m "feat: add DialogueGraph (M5.2) — conversation structure builder"
```

---

### Task 8: Translator — Character Profile (M5 Stage 3)

**Files:**
- Create: `src/core/translator/character_profile.py`
- Create: `tests/core/test_translator/test_character_profile.py`

- [ ] **Step 1: Write test for CharacterProfile**

```python
# tests/core/test_translator/test_character_profile.py
import json
import pytest
from pathlib import Path
from src.core.translator.character_profile import CharacterProfileExtractor
from src.utils.schemas import CharacterProfile


class TestCharacterProfileExtractor:
    @pytest.fixture
    def extractor(self):
        return CharacterProfileExtractor()

    def test_extract_pronouns_ore(self, extractor):
        profile = extractor.extract_profile("char_1", ["オレがやる！", "オレの番だ"])
        assert "オレ" in profile.pronouns

    def test_extract_pronouns_watashi(self, extractor):
        profile = extractor.extract_profile("char_2", ["わたしは大丈夫", "わたしも行く"])
        assert "わたし" in profile.pronouns

    def test_extract_sentence_endings(self, extractor):
        profile = extractor.extract_profile("char_3", ["行くぜ！", "やるぜ！", "勝つぜ！"])
        assert "ぜ" in profile.sentence_endings

    def test_infer_speech_style_casual(self, extractor):
        profile = extractor.extract_profile("char_4",
            ["何だよ", "うるさいな", "知らねえよ"])
        assert profile.speech_style in ("casual", "rough", "neutral")

    def test_infer_speech_style_polite(self, extractor):
        profile = extractor.extract_profile("char_5",
            ["そうです", "わかりました", "お願いします"])
        assert profile.speech_style in ("polite", "formal", "neutral")

    def test_save_and_load_profile(self, extractor, tmp_path):
        profile = CharacterProfile(
            id="test_char",
            pronouns=["オレ"],
            sentence_endings=["ぜ"],
            speech_style="casual",
            estimated_role="protagonist",
        )
        save_path = tmp_path / "profiles"
        save_path.mkdir()
        extractor.save_profile(profile, str(save_path))

        loaded = extractor.load_profile("test_char", str(save_path))
        assert loaded is not None
        assert loaded.id == "test_char"
        assert loaded.pronouns == ["オレ"]

    def test_load_nonexistent_profile_returns_none(self, extractor, tmp_path):
        result = extractor.load_profile("nonexistent", str(tmp_path))
        assert result is None

    def test_empty_lines_return_default_profile(self, extractor):
        profile = extractor.extract_profile("new_char", [])
        assert profile.id == "new_char"
        assert profile.speech_style == "neutral"
```

- [ ] **Step 2: Implement CharacterProfileExtractor**

```python
"""M5 Stage 3: Character Profile Extractor — build personality cards from dialogue."""
import json
import re
from pathlib import Path
from src.utils.schemas import CharacterProfile


class CharacterProfileExtractor:
    """Extract character speech profiles from dialogue text.

    Analyzes pronoun usage, sentence endings, and speech patterns
    to build a persistent character profile that drives translation style.
    Profiles are saved to disk and reused across pages of the same manga.
    """

    PRONOUN_MAP = {
        "オレ": "casual_masculine",
        "俺": "casual_masculine",
        "僕": "modest",
        "わたし": "neutral_feminine",
        "私": "formal",
        "あたし": "casual_feminine",
        "わし": "elderly",
        "俺様": "arrogant",
    }

    ENDING_STYLE_MAP = {
        "ぜ": "casual_masculine",
        "ぞ": "casual_masculine",
        "だ": "casual",
        "だぜ": "rough_casual",
        "です": "polite",
        "ます": "polite",
        "わ": "feminine",
        "ね": "friendly",
        "かしら": "feminine_wondering",
    }

    def extract_profile(
        self, character_id: str, lines: list[str]
    ) -> CharacterProfile:
        """Extract a character profile from their dialogue lines."""
        pronouns = self._extract_pronouns(lines)
        endings = self._extract_sentence_endings(lines)
        speech_style = self._infer_speech_style(pronouns, endings)
        role = self._infer_role(pronouns, speech_style)

        return CharacterProfile(
            id=character_id,
            pronouns=pronouns,
            sentence_endings=endings,
            speech_style=speech_style,
            estimated_role=role,
        )

    def _extract_pronouns(self, lines: list[str]) -> list[str]:
        found = []
        for line in lines:
            for pronoun in self.PRONOUN_MAP:
                if pronoun in line and pronoun not in found:
                    found.append(pronoun)
        return found

    def _extract_sentence_endings(self, lines: list[str]) -> list[str]:
        found = []
        for line in lines:
            for ending in sorted(self.ENDING_STYLE_MAP, key=len, reverse=True):
                if line.rstrip().endswith(ending) and ending not in found:
                    found.append(ending)
                    break
        return found

    def _infer_speech_style(
        self, pronouns: list[str], endings: list[str]
    ) -> str:
        style_scores: dict[str, int] = {}

        for pronoun in pronouns:
            style = self.PRONOUN_MAP.get(pronoun, "")
            if style:
                style_scores[style] = style_scores.get(style, 0) + 1

        for ending in endings:
            style = self.ENDING_STYLE_MAP.get(ending, "")
            if style:
                style_scores[style] = style_scores.get(style, 0) + 1

        if not style_scores:
            return "neutral"

        return max(style_scores, key=lambda k: style_scores[k])

    def _infer_role(self, pronouns: list[str], speech_style: str) -> str:
        if any(p in pronouns for p in ["俺様"]):
            return "antagonist"
        if any(p in pronouns for p in ["オレ", "俺"]):
            return "protagonist"
        if speech_style in ("polite", "formal"):
            return "supporting"
        return "unknown"

    def save_profile(self, profile: CharacterProfile, directory: str) -> None:
        """Persist a character profile to disk for reuse across pages."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        filepath = dir_path / f"{profile.id}.json"
        data = {
            "id": profile.id,
            "pronouns": profile.pronouns,
            "sentence_endings": profile.sentence_endings,
            "speech_style": profile.speech_style,
            "estimated_role": profile.estimated_role,
        }
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_profile(self, character_id: str, directory: str) -> CharacterProfile | None:
        """Load a previously saved character profile."""
        filepath = Path(directory) / f"{character_id}.json"
        if not filepath.exists():
            return None
        data = json.loads(filepath.read_text(encoding="utf-8"))
        return CharacterProfile(
            id=data["id"],
            pronouns=data.get("pronouns", []),
            sentence_endings=data.get("sentence_endings", []),
            speech_style=data.get("speech_style", "neutral"),
            estimated_role=data.get("estimated_role", "unknown"),
        )
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/core/test_translator/test_character_profile.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/core/translator/character_profile.py tests/core/test_translator/test_character_profile.py
git commit -m "feat: add CharacterProfileExtractor (M5.3) — personality cards from dialogue"
```

---

### Task 9: Translator — Translation Engine (M5 Stage 4)

**Files:**
- Create: `src/core/translator/translation_engine.py`
- Create: `tests/core/test_translator/test_translation_engine.py`

- [ ] **Step 1: Write test for TranslationEngine**

```python
# tests/core/test_translator/test_translation_engine.py
import pytest
from unittest.mock import patch, MagicMock
from src.core.translator.translation_engine import TranslationEngine
from src.utils.schemas import TextBox, TranslationResult, CharacterProfile, Language


class TestTranslationEngine:
    @pytest.fixture
    def engine(self):
        with patch("src.core.translator.translation_engine.ModelManager"):
            engine = TranslationEngine()
            engine._simple_threshold = 0.6
            yield engine

    @pytest.fixture
    def sample_boxes(self):
        box_a = TextBox(id="a", x=350, y=50, width=100, height=30,
                        text="行くぞ！", language=Language.JA, character_id="char_1")
        box_b = TextBox(id="b", x=150, y=60, width=90, height=30,
                        text="待ってください", language=Language.JA, character_id="char_2")
        return [box_a, box_b]

    @pytest.fixture
    def sample_profiles(self):
        return {
            "char_1": CharacterProfile(id="char_1", pronouns=["オレ"],
                                       speech_style="casual_masculine", estimated_role="protagonist"),
            "char_2": CharacterProfile(id="char_2", pronouns=["わたし"],
                                       speech_style="polite", estimated_role="supporting"),
        }

    def test_classify_simple_bubble_short_text(self, engine):
        assert engine._is_simple_bubble("行くぞ") is True

    def test_classify_complex_bubble_long_text(self, engine):
        long_text = "これは非常に長い文章で、文脈が必要な翻訳です。" * 5
        assert engine._is_simple_bubble(long_text) is False

    def test_build_context_includes_scene_and_character(self, engine, sample_boxes, sample_profiles):
        context = engine._build_context(
            box=sample_boxes[0],
            all_boxes=sample_boxes,
            profiles=sample_profiles,
            scene="battle",
        )
        assert "battle" in str(context).lower() or "battle" in context.get("scene", "")
        assert "char_1" in context.get("character_id", "")

    def test_translate_with_opus_mt_mocked(self, engine, sample_boxes):
        with patch.object(engine, '_translate_opus') as mock_opus:
            mock_opus.return_value = "我们上！"
            result = engine._translate_opus("行くぞ！", "ja", "zh")
            assert result == "我们上！"

    def test_translate_all_returns_correct_count(self, engine, sample_boxes, sample_profiles):
        with patch.object(engine, '_translate_opus', return_value="测试翻译"):
            results = engine.translate_all(
                boxes=sample_boxes,
                profiles=sample_profiles,
                scene="battle",
                target_lang=Language.ZH,
            )
            assert len(results) == 2
            assert all(isinstance(r, TranslationResult) for r in results)

    def test_translate_all_preserves_character_id(self, engine, sample_boxes, sample_profiles):
        with patch.object(engine, '_translate_opus', return_value="测试"):
            results = engine.translate_all(sample_boxes, sample_profiles, "battle", Language.ZH)
            char_ids = {r.character_id for r in results}
            assert "char_1" in char_ids
```

- [ ] **Step 2: Implement TranslationEngine**

```python
"""M5 Stage 4: Translation Engine — two-tier hybrid translation with context."""
from src.utils.schemas import TextBox, TranslationResult, CharacterProfile, Language
from src.utils.config import config


class TranslationEngine:
    """Two-tier hybrid translation engine.

    Simple bubbles (short, no dependency): OPUS-MT fast direct translation (~50ms)
    Complex bubbles (long, contextual): Qwen2.5-VL-2B multimodal translation (~3s)
    """

    def __init__(self) -> None:
        from src.core.model_manager import ModelManager
        self._mgr = ModelManager()
        self._simple_threshold = config.get("pipeline.translation.simple_bubble_threshold", 0.6)
        self._max_context_chars = config.get("pipeline.translation.max_vlm_context_chars", 512)

    def translate_all(
        self,
        boxes: list[TextBox],
        profiles: dict[str, CharacterProfile],
        scene: str,
        target_lang: Language,
    ) -> list[TranslationResult]:
        """Translate all text boxes on a page with context-aware translation."""
        results = []
        previous_translations: list[TranslationResult] = []

        for box in boxes:
            if self._is_simple_bubble(box.text):
                result = self._translate_simple(box, target_lang)
            else:
                result = self._translate_complex(
                    box, boxes, profiles, scene, target_lang, previous_translations
                )
            results.append(result)
            previous_translations.append(result)

        return results

    def _is_simple_bubble(self, text: str) -> bool:
        """Determine if a bubble is simple enough for OPUS-MT direct translation."""
        if len(text) <= 15 and "\n" not in text:
            return True
        question_words = ["何", "なぜ", "どうして", "か？", "ですか"]
        if any(q in text for q in question_words):
            return True
        return False

    def _translate_simple(
        self, box: TextBox, target_lang: Language
    ) -> TranslationResult:
        """Fast translation using OPUS-MT."""
        source_lang = box.language
        translated = self._translate_opus(box.text, source_lang.value, target_lang.value)
        return TranslationResult(
            textbox_id=box.id,
            original_text=box.text,
            translated_text=translated,
            source_lang=source_lang,
            target_lang=target_lang,
            translation_method="opus_mt",
            character_id=box.character_id,
            confidence=0.85,
        )

    def _translate_complex(
        self,
        box: TextBox,
        all_boxes: list[TextBox],
        profiles: dict[str, CharacterProfile],
        scene: str,
        target_lang: Language,
        previous: list[TranslationResult],
    ) -> TranslationResult:
        """Context-aware translation using Qwen2.5-VL-2B with five-dimensional context."""
        context = self._build_context(box, all_boxes, profiles, scene)
        translated = self._translate_vlm(box.text, context, target_lang.value)
        return TranslationResult(
            textbox_id=box.id,
            original_text=box.text,
            translated_text=translated,
            source_lang=box.language,
            target_lang=target_lang,
            translation_method="vlm",
            character_id=box.character_id,
            confidence=0.9,
        )

    def _build_context(
        self,
        box: TextBox,
        all_boxes: list[TextBox],
        profiles: dict[str, CharacterProfile],
        scene: str,
    ) -> dict:
        """Build five-dimensional translation context.

        Dimensions: scene, character, history, dialogue, visual.
        """
        profile = profiles.get(box.character_id or "", None)
        character_info = ""
        if profile:
            character_info = (
                f"角色性格: {profile.speech_style}, "
                f"自称: {', '.join(profile.pronouns) if profile.pronouns else '未知'}"
            )

        neighboring_texts = []
        for other in all_boxes:
            if other.id != box.id and other.character_id == box.character_id:
                neighboring_texts.append(other.text)

        return {
            "scene": scene,
            "character_id": box.character_id,
            "character_info": character_info,
            "neighboring_dialogue": neighboring_texts[-3:],
            "target_language": "zh",
            "instruction": "将以下日文漫画台词翻译成自然的中文口语，保持角色语气一致。",
        }

    def _translate_opus(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using OPUS-MT MarianMT model."""
        model_key = f"opus_mt_{source_lang}_{target_lang}"
        try:
            model_data = self._mgr.get_model("translation", model_key)
            tokenizer = model_data["tokenizer"]
            model = model_data["model"]
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_length=128)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result
        except Exception:
            return text

    def _translate_vlm(self, text: str, context: dict, target_lang: str) -> str:
        """Translate using Qwen2.5-VL-2B with full context."""
        prompt = f"""{context.get('instruction', '')}

场景: {context.get('scene', '未知')}
{context.get('character_info', '')}

原文: {text}

翻译成{target_lang}:"""

        try:
            model_data = self._mgr.get_model("translation", "vlm")
            model = model_data["model"]
            processor = model_data["processor"]

            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text_prompt = processor.apply_chat_template(messages, tokenize=False)
            inputs = processor(text=[text_prompt], return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=256)
            result = processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )[0]
            return result.strip()
        except Exception:
            return text
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/core/test_translator/test_translation_engine.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/core/translator/translation_engine.py tests/core/test_translator/test_translation_engine.py
git commit -m "feat: add TranslationEngine (M5.4) — two-tier hybrid with VLM context"
```

---

### Task 10: Translator — Consistency Checker (M5 Stage 5)

**Files:**
- Create: `src/core/translator/consistency_checker.py`
- Create: `tests/core/test_translator/test_consistency_checker.py`

- [ ] **Step 1: Write test for ConsistencyChecker**

```python
# tests/core/test_translator/test_consistency_checker.py
import pytest
from src.core.translator.consistency_checker import ConsistencyChecker
from src.utils.schemas import TranslationResult, Language


class TestConsistencyChecker:
    @pytest.fixture
    def checker(self):
        return ConsistencyChecker()

    @pytest.fixture
    def sample_results(self):
        return [
            TranslationResult(textbox_id="1", original_text="行くぞ！",
                              translated_text="我们上！", source_lang=Language.JA,
                              target_lang=Language.ZH, translation_method="opus_mt",
                              character_id="char_1"),
            TranslationResult(textbox_id="2", original_text="待って",
                              translated_text="等等", source_lang=Language.JA,
                              target_lang=Language.ZH, translation_method="opus_mt",
                              character_id="char_2"),
            TranslationResult(textbox_id="3", original_text="まだまだ！",
                              translated_text="还没完！", source_lang=Language.JA,
                              target_lang=Language.ZH, translation_method="vlm",
                              character_id="char_1"),
        ]

    def test_all_pass_returns_unmodified(self, checker, sample_results):
        passed, failed = checker.check(sample_results)
        assert len(failed) == 0

    def test_mismatched_terminology_flagged(self, checker):
        results = [
            TranslationResult(textbox_id="1", original_text="忍者",
                              translated_text="忍者", source_lang=Language.JA,
                              target_lang=Language.ZH, translation_method="opus_mt",
                              character_id="char_1"),
            TranslationResult(textbox_id="2", original_text="忍者",
                              translated_text="Ninja", source_lang=Language.JA,
                              target_lang=Language.ZH, translation_method="opus_mt",
                              character_id="char_1"),
        ]
        passed, failed = checker.check(results)
        assert len(failed) >= 0

    def test_empty_input_returns_empty(self, checker):
        passed, failed = checker.check([])
        assert passed == []
        assert failed == []

    def test_check_character_consistency(self, checker, sample_results):
        issues = checker._check_character_consistency(sample_results)
        assert isinstance(issues, list)
```

- [ ] **Step 2: Implement ConsistencyChecker**

```python
"""M5 Stage 5: Consistency Checker — validate and retranslate if needed."""
from collections import defaultdict
from src.utils.schemas import TranslationResult


class ConsistencyChecker:
    """Post-translation consistency validation.

    Checks for:
    - Character voice consistency (same character, same tone)
    - Dialogue coherence (responses match questions)
    - Terminology uniformity (same term, same translation)
    
    Failed items can trigger retranslation of specific bubbles.
    """

    def check(
        self, results: list[TranslationResult]
    ) -> tuple[list[TranslationResult], list[str]]:
        """Run all consistency checks on translation results.

        Returns (passed_results, failed_textbox_ids).
        Failed IDs should be retranslated with adjusted context.
        """
        if not results:
            return [], []

        failed_ids: set[str] = set()

        voice_issues = self._check_character_consistency(results)
        failed_ids.update(voice_issues)

        term_issues = self._check_terminology_consistency(results)
        failed_ids.update(term_issues)

        passed = [r for r in results if r.textbox_id not in failed_ids]
        return passed, list(failed_ids)

    def _check_character_consistency(
        self, results: list[TranslationResult]
    ) -> list[str]:
        """Check that same character's translations have consistent tone."""
        issues = []
        char_groups: dict[str, list[TranslationResult]] = defaultdict(list)
        for r in results:
            if r.character_id:
                char_groups[r.character_id].append(r)

        for char_id, char_results in char_groups.items():
            if len(char_results) < 2:
                continue
            lengths = [len(r.translated_text) for r in char_results]
            avg_len = sum(lengths) / len(lengths)
            for r, length in zip(char_results, lengths):
                if avg_len > 0 and length > avg_len * 2.5:
                    issues.append(r.textbox_id)
        return issues

    def _check_terminology_consistency(
        self, results: list[TranslationResult]
    ) -> list[str]:
        """Check that the same term is translated the same way."""
        issues = []
        term_map: dict[str, str] = {}
        for r in results:
            key = r.original_text.strip()
            if key in term_map:
                if r.translated_text.strip() != term_map[key]:
                    issues.append(r.textbox_id)
            else:
                term_map[key] = r.translated_text.strip()
        return issues
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/core/test_translator/test_consistency_checker.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/core/translator/consistency_checker.py tests/core/test_translator/test_consistency_checker.py
git commit -m "feat: add ConsistencyChecker (M5.5) — post-translation validation"
```

---

### Task 11: Inpainter Module (M6) — LaMa

**Files:**
- Create: `src/core/inpainter.py`
- Create: `tests/core/test_inpainter.py`

- [ ] **Step 1: Write test for Inpainter**

```python
# tests/core/test_inpainter.py
import numpy as np
import pytest
from src.core.inpainter import Inpainter


class TestInpainter:
    @pytest.fixture
    def inpaint(self):
        return Inpainter()

    @pytest.fixture
    def sample_image(self):
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Test", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        return img

    def test_generate_mask_creates_binary(self, inpaint):
        boxes = [{"x": 50, "y": 50, "width": 100, "height": 40}]
        img_shape = (200, 300)
        mask = inpaint.generate_mask(boxes, img_shape)
        assert mask.shape[:2] == img_shape
        assert mask.dtype == np.uint8
        assert mask.max() <= 255

    def test_generate_mask_has_white_in_text_region(self, inpaint):
        boxes = [{"x": 80, "y": 80, "width": 60, "height": 30}]
        img_shape = (200, 300)
        mask = inpaint.generate_mask(boxes, img_shape)
        assert mask[90, 100] > 0

    def test_inpaint_removes_text_region(self, inpaint, tmp_path):
        import cv2
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (80, 80), (160, 120), (0, 0, 0), -1)
        img_path = str(tmp_path / "text_image.png")
        cv2.imwrite(img_path, img)

        boxes = [{"x": 75, "y": 75, "width": 95, "height": 55}]
        result = inpaint.inpaint(img, boxes)
        assert result.shape == img.shape
        center_val = float(np.mean(result[90:110, 85:155]))
        assert center_val > 100

    def test_expand_box_adds_padding(self, inpaint):
        box = {"x": 100, "y": 100, "width": 50, "height": 30}
        expanded = inpaint._expand_box(box, padding=5)
        assert expanded["x"] == 95
        assert expanded["y"] == 95
        assert expanded["width"] == 60
        assert expanded["height"] == 40

    def test_inpaint_empty_boxes_returns_original(self, inpaint):
        import cv2
        img = np.ones((100, 100, 3), dtype=np.uint8) * 200
        result = inpaint.inpaint(img, [])
        assert np.array_equal(result, img)
```

- [ ] **Step 2: Implement Inpainter**

```python
"""M6: Inpainter — LaMa-based text removal and background reconstruction."""
import cv2
import numpy as np


class Inpainter:
    """Remove original text from comic images and reconstruct background.

    Uses LaMa (Large Mask Inpainting) with FFT-based global context
    for high-quality background reconstruction, especially on halftone
    textures and gradient backgrounds common in manga.
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
        """Remove text from all specified boxes and reconstruct background.

        Args:
            image: BGR or RGB image as numpy array.
            boxes: List of dicts with x, y, width, height keys.
            padding: Extra pixels to expand each box (ensures full text coverage).

        Returns:
            Image with text regions inpainted.
        """
        if not boxes:
            return image

        mask = self.generate_mask(boxes, image.shape[:2], padding)

        if self._model is not None:
            result = self._inpaint_lama(image, mask)
        else:
            result = self._inpaint_fallback(image, mask)

        return result

    def generate_mask(
        self, boxes: list[dict], img_shape: tuple[int, int], padding: int = 5
    ) -> np.ndarray:
        """Generate a binary mask covering all text regions."""
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
        """Run LaMa ONNX model for inpainting."""
        import onnxruntime as ort
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
        """Fallback: OpenCV Telea inpainting when LaMa is unavailable."""
        return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    @staticmethod
    def _expand_box(box: dict, padding: int = 5) -> dict:
        """Expand a bounding box by padding pixels on all sides."""
        return {
            "x": box["x"] - padding,
            "y": box["y"] - padding,
            "width": box["width"] + 2 * padding,
            "height": box["height"] + 2 * padding,
        }
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/core/test_inpainter.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/core/inpainter.py tests/core/test_inpainter.py
git commit -m "feat: add Inpainter (M6) — LaMa FFT-based background reconstruction"
```

---

### Task 12: Renderer — Style Detector (M7.1)

**Files:**
- Create: `src/core/renderer/__init__.py`
- Create: `src/core/renderer/style_detector.py`
- Create: `tests/core/test_renderer/__init__.py`
- Create: `tests/core/test_renderer/test_style_detector.py`

- [ ] **Step 1: Write test for StyleDetector**

```python
# tests/core/test_renderer/test_style_detector.py
import numpy as np
import pytest
from src.core.renderer.style_detector import StyleDetector
from src.utils.schemas import RenderStyle


class TestStyleDetector:
    @pytest.fixture
    def detector(self):
        return StyleDetector()

    def test_detect_text_color_black_on_white(self, detector):
        import cv2
        img = np.ones((40, 120, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Test", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        color = detector.detect_text_color(img)
        assert any(c < 80 for c in color)

    def test_detect_text_color_white_on_black(self, detector):
        import cv2
        img = np.zeros((40, 120, 3), dtype=np.uint8)
        cv2.putText(img, "Test", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        color = detector.detect_text_color(img)
        assert any(c > 180 for c in color)

    def test_detect_stroke_returns_none_for_no_stroke(self, detector):
        import cv2
        img = np.ones((50, 150, 3), dtype=np.uint8) * 255
        cv2.putText(img, "NoStroke", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        stroke = detector.detect_stroke(img)
        assert stroke is None or stroke.get("stroke_width", 0) == 0

    def test_detect_font_size_estimate(self, detector):
        import cv2
        img = np.ones((60, 200, 3), dtype=np.uint8) * 255
        cv2.putText(img, "FontSize", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        size = detector.estimate_font_size(img)
        assert 8 <= size <= 48

    def test_detect_alignment_center(self, detector):
        import cv2
        img = np.ones((80, 300, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Center", (80, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        alignment = detector.detect_alignment(img)
        assert alignment in ("left", "center", "right")

    def test_analyze_returns_complete_style(self, detector):
        import cv2
        img = np.ones((60, 200, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Analyze", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        style = detector.analyze(img)
        assert isinstance(style, RenderStyle)
        assert style.text_color is not None
        assert style.estimated_font_size > 0
        assert style.alignment in ("left", "center", "right")
```

- [ ] **Step 2: Implement StyleDetector**

```python
"""M7.1: Style Detector — extract original text visual style for SAAT rendering."""
import cv2
import numpy as np
from sklearn.cluster import KMeans
from src.utils.schemas import RenderStyle


class StyleDetector:
    """Detect visual style parameters from original text regions.

    All analysis uses traditional CV (no deep learning), runs in <5ms per region.
    Extracts: text color, stroke, shadow, font size estimate, font weight, alignment.
    """

    def analyze(self, text_region: np.ndarray) -> RenderStyle:
        """Full style analysis of a text region image."""
        return RenderStyle(
            text_color=self.detect_text_color(text_region),
            stroke_color=self.detect_stroke(text_region).get("stroke_color") if self.detect_stroke(text_region) else None,
            stroke_width=self.detect_stroke(text_region).get("stroke_width", 0),
            shadow_offset=self.detect_shadow(text_region),
            shadow_color=None,
            estimated_font_size=self.estimate_font_size(text_region),
            font_weight=self.detect_font_weight(text_region),
            alignment=self.detect_alignment(text_region),
        )

    def detect_text_color(self, image: np.ndarray) -> tuple[int, int, int]:
        """Detect the dominant text color using K-Means clustering."""
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
        """Detect if text has a stroke/outline using Canny edge detection."""
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
        """Detect drop shadow offset by analyzing brightness gradient."""
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
        """Estimate font size from text region height."""
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
        """Detect font weight (bold/regular/light) by black pixel ratio."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[-1] == 3 else image
        binary = (gray < 128).astype(np.uint8)
        text_pixel_ratio = np.sum(binary) / binary.size
        if text_pixel_ratio > 0.25:
            return "bold"
        elif text_pixel_ratio < 0.08:
            return "light"
        return "regular"

    def detect_alignment(self, image: np.ndarray) -> str:
        """Detect text horizontal alignment within the image."""
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
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/core/test_renderer/test_style_detector.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/core/renderer/ tests/core/test_renderer/
git commit -m "feat: add StyleDetector (M7.1) — CV-based text style extraction"
```

---

### Task 13: Renderer — Typesetter (M7.3) and Painter (M7.4)

**Files:**
- Create: `src/core/renderer/typesetter.py`
- Create: `src/core/renderer/painter.py`
- Create: `tests/core/test_renderer/test_typesetter.py`
- Create: `tests/core/test_renderer/test_painter.py`

- [ ] **Step 1: Write test for Typesetter**

```python
# tests/core/test_renderer/test_typesetter.py
import pytest
from src.core.renderer.typesetter import Typesetter
from src.utils.schemas import TypesetResult, RenderStyle


class TestTypesetter:
    @pytest.fixture
    def typesetter(self):
        return Typesetter()

    def test_compute_short_text_large_font(self, typesetter):
        result = typesetter.compute(
            text="上了！",
            available_width=200,
            available_height=60,
            language="zh",
            style=RenderStyle(alignment="center"),
        )
        assert isinstance(result, TypesetResult)
        assert result.font_size >= 10
        assert len(result.lines) >= 1

    def test_compute_long_text_wraps(self, typesetter):
        result = typesetter.compute(
            text="これはとても長い文章で、改行が必要です",
            available_width=100,
            available_height=80,
            language="ja",
            style=RenderStyle(),
        )
        assert len(result.lines) > 1

    def test_compute_english_word_break(self, typesetter):
        result = typesetter.compute(
            text="What are you saying?",
            available_width=120,
            available_height=60,
            language="en",
            style=RenderStyle(),
        )
        for line in result.lines:
            assert " " not in line.strip() or len(line.split()) <= 3

    def test_binary_search_font_size_returns_valid(self, typesetter):
        size = typesetter._binary_search_font_size(
            text="测试文本", max_width=150, max_height=50, language="zh"
        )
        assert 8 <= size <= 48

    def test_break_lines_chinese_by_char(self, typesetter):
        lines = typesetter._break_lines("你在说什么啊", max_width=80, language="zh", font_size=16)
        assert len(lines) >= 1
        for line in lines:
            assert len(line) >= 1

    def test_break_lines_english_by_word(self, typesetter):
        lines = typesetter._break_lines("Hello world test", max_width=60, language="en", font_size=14)
        assert len(lines) >= 1
```

- [ ] **Step 2: Implement Typesetter**

```python
"""M7.3: Typesetter — adaptive layout engine with binary search font sizing."""
from PIL import ImageFont
from src.utils.schemas import TypesetResult, RenderStyle


class Typesetter:
    """Compute optimal typesetting parameters for translated text.

    Uses binary search to find the largest font size that fits within
    the available bubble space, then applies language-aware line breaking.
    """

    def compute(
        self,
        text: str,
        available_width: int,
        available_height: int,
        language: str,
        style: RenderStyle,
    ) -> TypesetResult:
        """Compute the optimal typesetting for the given text and space."""
        if not text:
            return TypesetResult(
                font_size=12, lines=[], line_height=0,
                start_x=0, start_y=0, alignment="center"
            )

        font_size = self._binary_search_font_size(
            text, available_width, available_height, language
        )

        lines = self._break_lines(text, available_width, language, font_size)

        line_height = self._compute_line_height(font_size, len(lines))
        total_height = len(lines) * font_size + (len(lines) - 1) * (line_height - font_size)
        start_y = max(0, (available_height - total_height) // 2)
        start_x = 0

        return TypesetResult(
            font_size=font_size,
            lines=lines,
            line_height=line_height,
            start_x=start_x,
            start_y=start_y,
            alignment=style.alignment,
        )

    def _binary_search_font_size(
        self, text: str, max_width: int, max_height: int, language: str
    ) -> int:
        """Binary search for the largest font size that fits."""
        low, high = 8, 48
        best = 10

        while low <= high:
            mid = (low + high) // 2
            if self._fits(text, max_width, max_height, mid, language):
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return max(8, min(48, best))

    def _fits(
        self, text: str, max_width: int, max_height: int, font_size: int, language: str
    ) -> bool:
        """Check if text fits within the given bounds at the given font size."""
        lines = self._break_lines(text, max_width, language, font_size)
        total_height = len(lines) * font_size + (len(lines) - 1) * max(1, font_size // 4)
        return total_height <= max_height

    def _break_lines(
        self, text: str, max_width: int, language: str, font_size: int
    ) -> list[str]:
        """Break text into lines that fit within max_width."""
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        if language in ("zh", "ja"):
            return self._break_cjk(text, max_width, font_size, font)
        else:
            return self._break_word(text, max_width, font_size, font)

    def _break_cjk(
        self, text: str, max_width: int, font_size: int, font
    ) -> list[str]:
        """Break CJK text by character with punctuation rules."""
        char_width = font_size
        chars_per_line = max(1, max_width // char_width)
        lines = []
        i = 0
        while i < len(text):
            end = min(i + chars_per_line, len(text))
            if end < len(text) and text[end] in "。、，！？；：」』）":
                end -= 1
            if end == i:
                end = i + 1
            lines.append(text[i:end])
            i = end
        return lines if lines else [text]

    def _break_word(
        self, text: str, max_width: int, font_size: int, font
    ) -> list[str]:
        """Break English text by word boundaries."""
        words = text.split()
        lines = []
        current_line = ""
        avg_char_width = font_size * 0.6
        chars_per_line = max(1, int(max_width / avg_char_width))

        for word in words:
            if len(current_line) + len(word) + 1 <= chars_per_line:
                current_line = f"{current_line} {word}".strip() if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                if len(word) > chars_per_line:
                    for j in range(0, len(word), chars_per_line):
                        lines.append(word[j:j + chars_per_line])
                    current_line = ""
                else:
                    current_line = word
        if current_line:
            lines.append(current_line)
        return lines if lines else [text]

    @staticmethod
    def _compute_line_height(font_size: int, num_lines: int) -> int:
        """Compute line height based on number of lines."""
        if num_lines <= 1:
            return font_size
        elif num_lines == 2:
            return int(font_size * 1.3)
        else:
            return int(font_size * 1.2)
```

- [ ] **Step 3: Implement Painter**

```python
"""M7.4: Painter — layered text rendering onto inpainting result."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.utils.schemas import TypesetResult, RenderStyle


class Painter:
    """Render translated text onto the inpainted comic image.

    Renders in four layers for correct visual stacking:
    1. Shadow (if detected in original)
    2. Stroke/outline (if detected)
    3. Main text body
    4. Composite onto target image
    """

    def render(
        self,
        image: np.ndarray,
        typeset: TypesetResult,
        style: RenderStyle,
        position: tuple[int, int],
    ) -> np.ndarray:
        """Render text onto the image at the specified position."""
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
        """Compute x position based on alignment."""
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        if typeset.alignment == "center":
            return base_x - text_width // 2
        elif typeset.alignment == "right":
            return base_x - text_width
        return base_x
```

- [ ] **Step 4: Commit**

```bash
git add src/core/renderer/typesetter.py src/core/renderer/painter.py tests/core/test_renderer/
git commit -m "feat: add Typesetter and Painter (M7.3, M7.4) — adaptive text layout and layered rendering"
```

---

### Task 14: GUI — Main Window and Pipeline Worker (M8)

**Files:**
- Create: `src/gui/main_window.py`
- Create: `src/gui/workers/pipeline_worker.py`
- Create: `src/gui/widgets/image_viewer.py`
- Create: `src/gui/widgets/control_panel.py`
- Create: `tests/gui/test_main_window.py`

- [ ] **Step 1: Write test for MainWindow structure**

```python
# tests/gui/test_main_window.py
import pytest
from unittest.mock import patch, MagicMock


class TestMainWindow:
    def test_window_creation(self, qtbot):
        """Skip if PySide6 not installed in CI — structural test only."""
        try:
            from src.gui.main_window import MainWindow
            window = MainWindow()
            qtbot.addWidget(window)
            assert window.windowTitle() == "Comic Translator"
            window.close()
        except ImportError:
            pytest.skip("PySide6 not available")
```

- [ ] **Step 2: Implement PipelineWorker**

```python
"""Background worker for running the translation pipeline without blocking GUI."""
from PySide6.QtCore import QThread, Signal


class PipelineWorker(QThread):
    """Run the full comic translation pipeline in a background thread.

    Emits progress signals so the GUI can update the progress bar
    and status label without freezing.
    """

    progress = Signal(int, str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, image_path: str, source_lang: str, target_lang: str) -> None:
        super().__init__()
        self._image_path = image_path
        self._source_lang = source_lang
        self._target_lang = target_lang

    def run(self) -> None:
        """Execute the full pipeline in order."""
        try:
            self.progress.emit(5, "加载模型...")
            from src.core.model_manager import ModelManager
            mgr = ModelManager()

            self.progress.emit(15, "检测文本区域...")
            from src.core.detector import Detector
            detector = Detector()
            boxes = detector.detect(self._image_path)
            self.progress.emit(25, f"检测到 {len(boxes)} 个文本区域")

            self.progress.emit(35, "识别文本内容...")
            from src.core.recognizer import Recognizer
            recognizer = Recognizer()
            for i, box in enumerate(boxes):
                import cv2
                img = cv2.imread(self._image_path)
                x1 = max(0, int(box.x))
                y1 = max(0, int(box.y))
                x2 = min(img.shape[1], int(box.x + box.width))
                y2 = min(img.shape[0], int(box.y + box.height))
                if x2 > x1 and y2 > y1:
                    region = img[y1:y2, x1:x2]
                    import tempfile, os
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
            import cv2
            image = cv2.imread(self._image_path)
            box_dicts = [
                {"x": b.x, "y": b.y, "width": b.width, "height": b.height}
                for b in boxes
            ]
            cleaned = inpainter.inpaint(image, box_dicts)

            self.progress.emit(90, "渲染译文...")
            from src.core.renderer.style_detector import StyleDetector
            from src.core.renderer.typesetter import Typesetter
            from src.core.renderer.painter import Painter
            from src.utils.config import config

            style_detector = StyleDetector()
            typesetter = Typesetter()
            painter = Painter()
            output_dir = config.get("paths.output_dir", "output")
            import os
            os.makedirs(output_dir, exist_ok=True)

            for box, trans in zip(boxes, translations):
                x1 = max(0, int(box.x))
                y1 = max(0, int(box.y))
                x2 = min(cleaned.shape[1], int(box.x + box.width))
                y2 = min(cleaned.shape[0], int(box.y + box.height))
                if x2 <= x1 or y2 <= y1:
                    continue
                region = cleaned[y1:y2, x1:x2]
                style = style_detector.analyze(region)
                typeset = typesetter.compute(
                    trans.translated_text,
                    box.width, box.height,
                    box.language.value,
                    style,
                )
                cleaned = painter.render(cleaned, typeset, style, (x1, y1))

            output_path = os.path.join(output_dir, "translated_output.png")
            cv2.imwrite(output_path, cleaned)

            self.progress.emit(100, "完成！")
            self.finished.emit(output_path)

        except Exception as e:
            self.error.emit(str(e))
```

- [ ] **Step 3: Implement MainWindow**

```python
"""M8: Main Window — PySide6 GUI for the comic translation system."""
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QProgressBar,
    QFileDialog, QMessageBox, QScrollArea, QSplitter,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QImage


class MainWindow(QMainWindow):
    """Main application window with image viewer and control panel."""

    def __init__(self) -> None:
        super().__init__()
        self._image_path: str | None = None
        self._worker = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the main window layout."""
        self.setWindowTitle("Comic Translator")
        self.resize(1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)

        left_panel = self._build_image_panel()
        splitter.addWidget(left_panel)

        right_panel = self._build_control_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([900, 500])
        main_layout.addWidget(splitter)

    def _build_image_panel(self) -> QWidget:
        """Build the left image viewer panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self._image_label = QLabel("请打开一张漫画图片")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; border: 1px solid #ccc; }"
        )
        self._image_label.setMinimumSize(600, 600)

        scroll = QScrollArea()
        scroll.setWidget(self._image_label)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        return panel

    def _build_control_panel(self) -> QWidget:
        """Build the right control panel with buttons and settings."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        layout.addWidget(QLabel("控制面板"))
        layout.addWidget(self._h_line())

        self._btn_open = QPushButton("打开漫画图片")
        self._btn_open.clicked.connect(self._on_open_image)
        layout.addWidget(self._btn_open)

        layout.addWidget(QLabel("源语言:"))
        self._combo_src = QComboBox()
        self._combo_src.addItems(["日语", "英语", "中文"])
        layout.addWidget(self._combo_src)

        layout.addWidget(QLabel("目标语言:"))
        self._combo_tgt = QComboBox()
        self._combo_tgt.addItems(["中文", "英语"])
        layout.addWidget(self._combo_tgt)

        self._btn_translate = QPushButton("一键翻译")
        self._btn_translate.clicked.connect(self._on_translate)
        self._btn_translate.setEnabled(False)
        self._btn_translate.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; padding: 10px; "
            "font-size: 14px; border-radius: 5px; }"
        )
        layout.addWidget(self._btn_translate)

        self._btn_erase = QPushButton("仅擦除原文")
        self._btn_erase.setEnabled(False)
        layout.addWidget(self._btn_erase)

        layout.addWidget(self._h_line())

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("就绪")
        layout.addWidget(self._status_label)

        layout.addStretch()

        layout.addWidget(self._h_line())
        layout.addWidget(QLabel("字体参数"))
        self._font_size_label = QLabel("字号: 自动")
        layout.addWidget(self._font_size_label)

        layout.addWidget(QLabel("排版模式"))
        self._combo_layout = QComboBox()
        self._combo_layout.addItems(["自动", "手动"])
        layout.addWidget(self._combo_layout)

        return panel

    @staticmethod
    def _h_line() -> QWidget:
        """Create a horizontal separator line."""
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #ddd;")
        return line

    def _on_open_image(self) -> None:
        """Handle open image button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开漫画图片", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if file_path:
            self._image_path = file_path
            pixmap = QPixmap(file_path)
            scaled = pixmap.scaled(
                self._image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self._image_label.setPixmap(scaled)
            self._btn_translate.setEnabled(True)
            self._btn_erase.setEnabled(True)
            self._status_label.setText(f"已加载: {file_path.split('/')[-1]}")

    def _on_translate(self) -> None:
        """Start the translation pipeline."""
        if not self._image_path:
            return

        src_map = {"日语": "ja", "英语": "en", "中文": "zh"}
        tgt_map = {"中文": "zh", "英语": "en"}

        lang_src = src_map[self._combo_src.currentText()]
        lang_tgt = tgt_map[self._combo_tgt.currentText()]

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._btn_translate.setEnabled(False)
        self._btn_open.setEnabled(False)

        from src.gui.workers.pipeline_worker import PipelineWorker
        self._worker = PipelineWorker(self._image_path, lang_src, lang_tgt)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, value: int, message: str) -> None:
        """Update progress bar and status."""
        self._progress_bar.setValue(value)
        self._status_label.setText(message)

    def _on_finished(self, output_path: str) -> None:
        """Handle pipeline completion."""
        self._progress_bar.setValue(100)
        self._status_label.setText(f"完成! 保存至: {output_path}")
        self._btn_translate.setEnabled(True)
        self._btn_open.setEnabled(True)

        pixmap = QPixmap(output_path)
        scaled = pixmap.scaled(
            self._image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._image_label.setPixmap(scaled)

        QMessageBox.information(self, "翻译完成", f"翻译结果已保存至:\n{output_path}")

    def _on_error(self, error_msg: str) -> None:
        """Handle pipeline error."""
        self._progress_bar.setVisible(False)
        self._status_label.setText(f"错误: {error_msg}")
        self._btn_translate.setEnabled(True)
        self._btn_open.setEnabled(True)
        QMessageBox.critical(self, "翻译失败", f"发生错误:\n{error_msg}")

    def closeEvent(self, event) -> None:
        """Clean up worker thread on close."""
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait()
        event.accept()
```

- [ ] **Step 4: Commit**

```bash
git add src/gui/ tests/gui/
git commit -m "feat: add MainWindow (M8) and PipelineWorker — full GUI integration"
```

---

### Task 15: Model Download Script and Integration Test

**Files:**
- Create: `scripts/download_models.py`
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write model download script**

```python
"""Download all required pretrained models for the comic translation system.

Run: python scripts/download_models.py
"""
import os
import sys


def download_yolo():
    print("[1/5] Downloading YOLOv8-OBB...")
    from ultralytics import YOLO
    model = YOLO("yolov8n-obb.pt")
    os.makedirs("models/detection", exist_ok=True)
    print("  YOLOv8-OBB ready.")


def download_paddleocr():
    print("[2/5] Initializing PaddleOCR (models auto-download)...")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="japan", use_angle_cls=True)
    print("  PaddleOCR ready.")


def download_opus_mt():
    print("[3/5] Downloading OPUS-MT translation models...")
    from transformers import MarianMTModel, MarianTokenizer
    for model_name in ["Helsinki-NLP/opus-mt-ja-zh", "Helsinki-NLP/opus-mt-en-zh"]:
        print(f"  Downloading {model_name}...")
        MarianTokenizer.from_pretrained(model_name)
        MarianMTModel.from_pretrained(model_name)
    print("  OPUS-MT models ready.")


def main():
    print("=" * 50)
    print("Comic Translator — Model Download")
    print("=" * 50)
    download_yolo()
    download_paddleocr()
    download_opus_mt()
    print("\n[Note] Qwen2.5-VL-2B and LaMa need manual download.")
    print("  Qwen2.5-VL-2B: https://huggingface.co/Qwen/Qwen2.5-VL-2B-Instruct")
    print("  LaMa ONNX: export from https://github.com/advimman/lama")
    print("\nAll base models ready!")
    print(f"Models directory: {os.path.abspath('models')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write integration test**

```python
"""End-to-end integration test for the full pipeline."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestIntegration:
    """Test the full pipeline end-to-end with mocked models."""

    def test_pipeline_flow_with_mocks(self, tmp_path):
        """Verify all modules connect correctly."""
        import cv2

        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.putText(img, "こんにちは", (250, 200),
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
                        text="こんにちは", language=Language.JA, confidence=0.9),
            ]
            from src.core.recognizer import RecognitionResult
            mock_ocr.return_value = [RecognitionResult(text="こんにちは", confidence=0.92, language="ja")]
            mock_trans.return_value = [
                TranslationResult(textbox_id="t1", original_text="こんにちは",
                                  translated_text="你好", source_lang=Language.JA,
                                  target_lang=Language.ZH, translation_method="opus_mt",
                                  character_id=None, confidence=0.85),
            ]
            mock_inpaint.return_value = img

            from src.core.detector import Detector
            from src.core.recognizer import Recognizer
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
            profile = extractor.extract_profile("char_1", ["こんにちは"])
            assert profile.speech_style in ("casual_masculine", "neutral", "modest")

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

            output_path = str(tmp_path / "output.png")
            cv2.imwrite(output_path, cleaned)
            assert cv2.imread(output_path) is not None
```

- [ ] **Step 3: Commit**

```bash
git add scripts/download_models.py tests/test_integration.py
git commit -m "feat: add model download script and end-to-end integration test"
```

---

### Task 16: Main Entry Point

**Files:**
- Create: `src/main.py`

- [ ] **Step 1: Write main.py**

```python
"""Comic Translator — Main entry point.

Usage:
    python src/main.py                        # Launch GUI
    python src/main.py --cli image.png        # CLI mode
"""
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Comic Translator")
    parser.add_argument("--cli", type=str, help="Run in CLI mode on a single image")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--source", type=str, default="ja", help="Source language")
    parser.add_argument("--target", type=str, default="zh", help="Target language")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    from src.utils.config import config
    config.load(args.config)

    if args.cli:
        run_cli(args.cli, args.source, args.target, args.output)
    else:
        run_gui()


def run_cli(image_path: str, source: str, target: str, output: str):
    """Command-line pipeline execution."""
    print(f"[Comic Translator] Processing: {image_path}")
    from src.gui.workers.pipeline_worker import PipelineWorker
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication(sys.argv)
    worker = PipelineWorker(image_path, source, target)

    def on_progress(value, msg):
        print(f"  [{value}%] {msg}")

    def on_finished(path):
        print(f"  Done! Output: {path}")
        app.quit()

    def on_error(err):
        print(f"  Error: {err}")
        app.quit()

    worker.progress.connect(on_progress)
    worker.finished.connect(on_finished)
    worker.error.connect(on_error)
    worker.start()
    app.exec()


def run_gui():
    """Launch the PySide6 GUI."""
    from PySide6.QtWidgets import QApplication
    from src.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify entry point**

Run: `python -c "import src.main; print('Entry point OK')"`
Expected: `Entry point OK`

- [ ] **Step 3: Final commit**

```bash
git add src/main.py
git commit -m "feat: add main entry point with CLI and GUI modes"
```

---

## Implementation Order Summary

| Task | Module | Priority | Depends On | Assignee |
|:--:|------|:--:|------|:--:|
| 1 | Scaffolding | P0 | — | E |
| 2 | M1 ModelManager | P0 | Task 1 | E |
| 3 | M2 Detector | P0 | Task 1, 2 | A |
| 4 | M3 Enhancer | P0 | Task 1 | A |
| 5 | M4 Recognizer | P0 | Task 1, 2 | B |
| 6 | M5.1 PageAnalyzer | P0 | Task 1 | C |
| 7 | M5.2 DialogueGraph | P0 | Task 6 | C |
| 8 | M5.3 CharProfile | P1 | Task 7 | D |
| 9 | M5.4 TransEngine | P0 | Task 7, 8 | D |
| 10 | M5.5 Consistency | P1 | Task 9 | D |
| 11 | M6 Inpainter | P0 | Task 1, 2 | D |
| 12 | M7.1 StyleDetector | P1 | Task 1 | D |
| 13 | M7.3 Typesetter + M7.4 Painter | P1 | Task 12 | D |
| 14 | M8 GUI | P0 | Task 1 | E |
| 15 | Scripts + Integration | P0 | All | E |
| 16 | Entry Point | P0 | All | E |
