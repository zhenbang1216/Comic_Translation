"""全流程测试脚本 — 检测+识别+可视化。支持调参。

把漫画图片丢到 input/，一键运行检测和OCR，输出带标注框的可视化图片。

用法:
    conda activate comic
    python scripts/test_full.py input/                          # 默认参数（推荐先用这个）
    python scripts/test_full.py input/ --thresh 0.1             # 更多检测框
    python scripts/test_full.py input/ --thresh 0.5             # 更严格（更少框）
    python scripts/test_full.py input/ --rec-score 0.7          # 只显示置信度≥0.7的结果
    python scripts/test_full.py input/ --lang en                # 英语漫画

调参速查:
    --thresh     检测敏感度（0.1=多框, 0.5=少框, 默认0.3）
    --box-thresh  框的置信度阈值（默认0.6）
    --rec-score   只显示高于此分数的结果（默认0.0=全显示）
    --lang        语言: japan / en / ch
"""
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def draw_boxes(image, results, min_score, output_path):
    """在图片上绘制检测框和识别文字。"""
    img = image.copy()
    texts = results.get("rec_texts", [])
    scores = results.get("rec_scores", [])
    boxes = results.get("rec_polys", [])

    drawn = 0
    for i, (text, score, box) in enumerate(zip(texts, scores, boxes)):
        if score < min_score:
            continue
        pts = box.astype(np.int32)
        # 高置信=绿色，低置信=黄色
        color = (0, 255, 0) if score >= 0.7 else (0, 255, 255)
        cv2.polylines(img, [pts], True, color, 2)

        x, y = pts[0][0], pts[0][1] - 8
        label = f"{text} ({score:.2f})"
        cv2.putText(img, label, (x, max(y, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        drawn += 1

    cv2.imwrite(output_path, img)
    print(f"  可视化保存至: {output_path} (绘制 {drawn}/{len(texts)} 个框)")


def main():
    parser = argparse.ArgumentParser(description="漫画文字检测+OCR可视化")
    parser.add_argument("path", type=str, help="图片路径 或 文件夹路径")
    parser.add_argument("--lang", type=str, default="japan",
                        choices=["japan", "en", "ch"], help="识别语言 (默认japan)")
    parser.add_argument("--thresh", type=float, default=0.3,
                        help="检测阈值 (0.1~0.5, 越低框越多, 默认0.3)")
    parser.add_argument("--box-thresh", type=float, default=0.6,
                        help="框置信度阈值 (默认0.6)")
    parser.add_argument("--rec-score", type=float, default=0.0,
                        help="只显示≥此分数的识别结果 (默认0.0=全显示)")
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_dir():
        images = sorted([f for f in target.iterdir() if f.suffix.lower() in IMAGE_EXTS])
    elif target.is_file():
        images = [target]
    else:
        print(f"错误: '{args.path}' 不存在")
        sys.exit(1)

    if not images:
        print(f"没有找到支持的图片 ({', '.join(IMAGE_EXTS)})")
        sys.exit(1)

    out_dir = Path("output/detection")
    out_dir.mkdir(parents=True, exist_ok=True)

    from paddleocr import PaddleOCR
    print(f"加载 PaddleOCR (thresh={args.thresh}, box_thresh={args.box_thresh}, lang={args.lang})...")
    ocr = PaddleOCR(
        lang=args.lang,
        use_doc_orientation_classify=False,
        use_textline_orientation=False,
        text_det_thresh=args.thresh,
        text_det_box_thresh=args.box_thresh,
    )

    print(f"处理 {len(images)} 张图片...\n")

    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] {img_path.name}")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  无法读取图片，跳过\n")
            continue

        results = ocr.predict(str(img_path))
        item = results[0]
        texts = item["rec_texts"]
        scores = item["rec_scores"]

        # 按分数过滤
        filtered = [(t, s) for t, s in zip(texts, scores) if s >= args.rec_score]
        print(f"  检测到 {len(texts)} 个文本区域，过滤后 {len(filtered)} 个 (≥{args.rec_score})")

        if filtered:
            print(f"  识别结果:")
            for t, s in filtered[:15]:
                print(f"    \"{t}\" ({s:.2f})")
            if len(filtered) > 15:
                print(f"    ... 共 {len(filtered)} 条")

            out_name = img_path.stem + "_detected.png"
            draw_boxes(img, item, args.rec_score, str(out_dir / out_name))

        print()


if __name__ == "__main__":
    main()
