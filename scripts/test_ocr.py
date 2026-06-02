"""OCR测试脚本 — 支持单张图片或整个文件夹批量识别。

用法:
    conda activate comic
    python scripts/test_ocr.py input/                              # 识别input文件夹所有图片
    python scripts/test_ocr.py input/ --lang en                    # 指定英语
    python scripts/test_ocr.py input/你的图片.png                  # 单张图片
    python scripts/test_ocr.py input/你的图片.png --lang ch         # 单张+指定语言
"""
import sys
import argparse
from pathlib import Path

# 支持的图片格式
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="漫画OCR文字识别测试")
    parser.add_argument("path", type=str, help="图片路径 或 文件夹路径")
    parser.add_argument("--lang", type=str, default="japan",
                        choices=["japan", "en", "ch"],
                        help="识别语言 (默认: japan)")
    args = parser.parse_args()

    target = Path(args.path)

    # 收集要处理的图片列表
    if target.is_dir():
        images = sorted([
            f for f in target.iterdir()
            if f.suffix.lower() in IMAGE_EXTS
        ])
        if not images:
            print(f"文件夹 '{target}' 下没有支持的图片文件")
            print(f"支持格式: {', '.join(IMAGE_EXTS)}")
            sys.exit(1)
        print(f"找到 {len(images)} 张图片，批量处理...\n")
    elif target.is_file():
        images = [target]
    else:
        print(f"错误: '{args.path}' 不是有效的文件或文件夹")
        sys.exit(1)

    # 加载模型（只加载一次）
    from paddleocr import PaddleOCR

    print("加载PaddleOCR模型...")
    ocr = PaddleOCR(
        lang=args.lang,
        use_doc_orientation_classify=False,
        use_textline_orientation=False,
    )
    print("开始识别...\n")

    # 逐个处理
    for idx, img_path in enumerate(images, 1):
        print(f"{'='*50}")
        print(f"[{idx}/{len(images)}] {img_path.name}")
        print(f"{'='*50}")

        results = ocr.predict(str(img_path))
        item = results[0]

        texts = item["rec_texts"]
        scores = item["rec_scores"]
        boxes = item["rec_polys"]

        if len(texts) == 0:
            print("  未检测到文字\n")
        else:
            print(f"  检测到 {len(texts)} 个文本区域:\n")
            for i, (text, score, box) in enumerate(zip(texts, scores, boxes)):
                x, y = int(box[0][0]), int(box[0][1])
                print(f"    [{i+1}] \"{text}\"  置信度: {score:.2f}  位置: ({x}, {y})")
            print()

    print(f"全部完成！共处理 {len(images)} 张图片。")


if __name__ == "__main__":
    main()
