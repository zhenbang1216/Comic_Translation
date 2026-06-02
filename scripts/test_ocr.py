"""OCR测试脚本 — 快速测试图片中的文字识别。

用法:
    conda activate comic
    python scripts/test_ocr.py input/你的漫画图片.png          # 日语
    python scripts/test_ocr.py input/你的漫画图片.png --lang en  # 英语
    python scripts/test_ocr.py input/你的漫画图片.png --lang ch  # 中文
"""
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="漫画OCR识别测试")
    parser.add_argument("image", type=str, help="图片路径")
    parser.add_argument("--lang", type=str, default="japan",
                        choices=["japan", "en", "ch"],
                        help="识别语言 (默认: japan)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"错误: 找不到图片 '{args.image}'")
        print(f"提示: 把漫画图片放到 input/ 文件夹下")
        sys.exit(1)

    print(f"识别图片: {args.image}")
    print(f"语言: {args.lang}")
    print("-" * 40)

    from paddleocr import PaddleOCR

    print("加载PaddleOCR模型...")
    ocr = PaddleOCR(
        lang=args.lang,
        use_doc_orientation_classify=False,
        use_textline_orientation=False,
    )

    print("识别中...\n")
    results = ocr.predict(str(image_path))
    item = results[0]

    texts = item["rec_texts"]
    scores = item["rec_scores"]
    boxes = item["rec_polys"]

    print(f"检测到 {len(texts)} 个文本区域:\n")
    for i, (text, score, box) in enumerate(zip(texts, scores, boxes)):
        x, y = int(box[0][0]), int(box[0][1])
        print(f"  [{i+1}] \"{text}\"  置信度: {score:.2f}  位置: ({x}, {y})")

    print(f"\n完成！")


if __name__ == "__main__":
    main()
