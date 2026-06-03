"""全流程端到端测试 — 检测+识别+翻译+修复+渲染。

用法:
    conda activate comic
    python scripts/test_full_pipeline.py input/                        # 日→英
    python scripts/test_full_pipeline.py input/ --target zh            # 日→中
"""
import sys, os
import argparse
from pathlib import Path
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from _utils import load_translator, translate, inpaint_text, render_text

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="漫画翻译全流程 — 检测+OCR+翻译+修复+渲染")
    parser.add_argument("path", type=str, help="图片路径 或 文件夹路径")
    parser.add_argument("--lang", type=str, default="japan",
                        choices=["japan", "en", "ch"], help="源语言")
    parser.add_argument("--target", type=str, default="en",
                        choices=["en", "zh"], help="目标语言")
    parser.add_argument("--rec-score", type=float, default=0.5,
                        help="只处理识别分数≥此值的文字")
    parser.add_argument("--font", type=str, default=None,
                        help="字体文件路径 (如 C:/Windows/Fonts/arial.ttf)")
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_dir():
        images = sorted([f for f in target.iterdir() if f.suffix.lower() in IMAGE_EXTS])
    elif target.is_file():
        images = [target]
    else:
        print(f"错误: '{args.path}' 不存在")
        sys.exit(1)

    # ===== 加载模型 =====
    print("=== 加载模型 ===\n")
    from paddleocr import PaddleOCR
    print("1/3 PaddleOCR...")
    ocr = PaddleOCR(lang=args.lang, use_doc_orientation_classify=False,
                    use_textline_orientation=False)

    lang_map = {"japan": "ja", "en": "en", "ch": "zh"}
    src_iso = lang_map[args.lang]

    print("2/3 翻译模型...")
    if args.target == "zh":
        tok_ja_en, mod_ja_en = load_translator("ja-en")
        tok_en_zh, mod_en_zh = load_translator("en-zh")
        print("  链式: JA → EN → ZH")
    else:
        tok, mod = load_translator(f"{src_iso}-{args.target}")
        print(f"  {src_iso.upper()} → {args.target.upper()}")

    print("3/3 修复+渲染: OpenCV + PIL (内置)\n")

    # ===== 输出目录 =====
    out_dir = Path("output/pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== 处理 =====
    print(f"=== 处理 {len(images)} 张图片 ===\n")
    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] {img_path.name}")

        img = cv2.imread(str(img_path))
        if img is None:
            print("  读取失败\n")
            continue

        # Step 1: OCR
        results = ocr.predict(str(img_path))
        item = results[0]
        texts = item["rec_texts"]
        scores = item["rec_scores"]
        boxes = item["rec_polys"]

        filtered = [(t, s, b) for t, s, b in zip(texts, scores, boxes) if s >= args.rec_score]
        print(f"  OCR: {len(texts)} 区 → 过滤 {len(filtered)} 个")

        if not filtered:
            print("  跳过\n")
            continue

        # Step 2: 翻译
        translations = []
        for t, s, b in filtered:
            if args.target == "zh":
                en = translate(t, tok_ja_en, mod_ja_en)
                zh = translate(en, tok_en_zh, mod_en_zh)
                translations.append((zh, b))
            else:
                tr = translate(t, tok, mod)
                translations.append((tr, b))

        # Step 3: 修复
        print("  修复原文字...")
        cleaned = inpaint_text(img, [b for _, _, b in filtered])

        # Step 4: 渲染
        print(f"  渲染 {len(translations)} 条译文...")
        rendered = cleaned.copy()
        for trans_text, box in translations:
            if trans_text.strip():
                rendered = render_text(rendered, trans_text, box, args.font)

        out_name = img_path.stem + "_translated.png"
        out_path = str(out_dir / out_name)
        cv2.imwrite(out_path, rendered)
        print(f"  输出: {out_path}\n")

    print(f"全部完成！结果在 {out_dir}/")


if __name__ == "__main__":
    main()
