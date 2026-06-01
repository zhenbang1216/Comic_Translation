"""M1：模型管理器 — 所有模型的懒加载、缓存和生命周期管理。"""
from typing import Any


class ModelManager:
    """单例管理器，用于懒加载模型实例。

    模型按 "类别:名称" 作为键存储，首次访问时加载，
    模块间共享以避免重复占用内存。
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
        """获取模型实例，如未加载则先加载。"""
        key = f"{category}:{name}"
        if key not in self._models:
            self._models[key] = self._load_model(category, name, mock_path)
        return self._models[key]

    def _load_model(
        self, category: str, name: str, mock_path: str | None = None
    ) -> Any:
        """根据类别和名称加载对应的模型。"""
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
            raise ValueError(f"未知的模型类别/名称: {category}/{name}")

    def release_model(self, category: str, name: str) -> None:
        """释放指定模型的内存。"""
        key = f"{category}:{name}"
        self._models.pop(key, None)

    def release_all(self) -> None:
        """释放所有已加载模型。"""
        self._models.clear()
