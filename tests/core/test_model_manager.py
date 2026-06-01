"""ModelManager单元测试。"""
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
        with patch.object(mgr, "_load_model", return_value=mock_model):
            mgr.get_model("detection", "yolo_obb")
        assert "detection:yolo_obb" in mgr._models

    def test_get_model_returns_cached_on_second_call(self):
        mgr = ModelManager()
        mgr._models.clear()
        mock_model = MagicMock()
        mgr._models["detection:yolo_obb"] = mock_model
        with patch.object(mgr, "_load_model") as mock_load:
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

    def test_unknown_category_raises_error(self):
        mgr = ModelManager()
        mgr._models.clear()
        with pytest.raises(ValueError, match="未知"):
            mgr.get_model("unknown_cat", "unknown_name")
