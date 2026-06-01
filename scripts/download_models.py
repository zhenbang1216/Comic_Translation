"""下载漫画翻译系统所需的全部预训练模型。

运行: python scripts/download_models.py
"""
import os
import sys


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


def main():
    print("=" * 50)
    print("Comic Translator — 模型下载")
    print("=" * 50)
    download_yolo()
    download_paddleocr()
    download_opus_mt()
    print("\n[注意] Qwen2.5-VL-2B 和 LaMa 需要手动下载。")
    print("  Qwen2.5-VL-2B: https://huggingface.co/Qwen/Qwen2.5-VL-2B-Instruct")
    print("  LaMa ONNX: 从 https://github.com/advimman/lama 导出")
    print("\n全部基础模型就绪！")
    print(f"模型目录: {os.path.abspath('models')}")


if __name__ == "__main__":
    main()
