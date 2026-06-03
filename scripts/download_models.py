"""下载漫画翻译系统所需的全部预训练模型。

运行: python scripts/download_models.py
"""
import os
import sys
from pathlib import Path


def download_yolo():
    print("[1/5] 下载 YOLOv8-OBB...")
    from ultralytics import YOLO
    YOLO("yolov8n-obb.pt")
    os.makedirs("models/detection", exist_ok=True)
    print("  YOLOv8-OBB 就绪。")


def download_paddleocr():
    print("[2/5] 初始化 PaddleOCR（模型自动下载）...")
    from paddleocr import PaddleOCR
    PaddleOCR(lang="japan", use_angle_cls=True)
    print("  PaddleOCR 就绪。")


def download_opus_mt():
    print("[3/5] 下载 OPUS-MT 翻译模型...")
    from transformers import MarianMTModel, MarianTokenizer
    for model_name in ["Helsinki-NLP/opus-mt-ja-zh", "Helsinki-NLP/opus-mt-en-zh"]:
        print(f"  下载 {model_name}...")
        MarianTokenizer.from_pretrained(model_name)
        MarianMTModel.from_pretrained(model_name)
    print("  OPUS-MT 模型就绪。")


def download_lama():
    print("[4/5] 下载 LaMa ONNX 修复模型...")
    import urllib.request

    lama_dir = Path("models/inpainting")
    lama_dir.mkdir(parents=True, exist_ok=True)
    output_path = lama_dir / "big-lama.onnx"

    if output_path.exists():
        print(f"  LaMa 模型已存在: {output_path}")
        return

    urls = [
        "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.onnx",
    ]

    for url in urls:
        try:
            print(f"  下载 {url} ...")
            urllib.request.urlretrieve(url, str(output_path))
            print(f"  LaMa 模型下载完成: {output_path}")
            return
        except Exception as e:
            print(f"  下载失败: {e}")

    print("  [警告] LaMa 自动下载失败，请手动下载到 models/inpainting/big-lama.onnx")
    print("  下载地址: https://github.com/advimman/lama")


def main():
    print("=" * 50)
    print("Comic Translator — 模型下载")
    print("=" * 50)
    download_yolo()
    download_paddleocr()
    download_opus_mt()
    download_lama()
    print("\n[注意] Qwen2.5-VL-2B 需要手动下载。")
    print("  Qwen2.5-VL-2B: https://huggingface.co/Qwen/Qwen2.5-VL-2B-Instruct")
    print("\n全部基础模型就绪！")
    print(f"模型目录: {os.path.abspath('models')}")


if __name__ == "__main__":
    main()
