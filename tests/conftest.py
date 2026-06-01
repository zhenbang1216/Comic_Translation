"""pytest共享夹具。"""
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
    cv2.putText(img, "Hello text", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    img_path = str(tmp_path / "sample.png")
    cv2.imwrite(img_path, img)
    return img_path
