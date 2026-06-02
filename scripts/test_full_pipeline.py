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
import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# ---------- 翻译 ----------
def load_translator(lang_pair):
    from transformers import MarianMTModel, MarianTokenizer
    model_name = f"Helsinki-NLP/opus-mt-{lang_pair}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate(text, tokenizer, model):
    if not text or not text.strip():
        return ""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------- 修复 ----------
def inpaint_text(img, boxes, padding=4):
    """用OpenCV修复去除文字区域。"""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for box in boxes:
        pts = box.astype(np.int32)
        x1 = max(0, pts[:, 0].min() - padding)
        y1 = max(0, pts[:, 1].min() - padding)
        x2 = min(img.shape[1], pts[:, 0].max() + padding)
        y2 = min(img.shape[0], pts[:, 1].max() + padding)
        mask[y1:y2, x1:x2] = 255
    return cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# ---------- 渲染 ----------
def render_text(img, text, box, font_path=None):
    """在修复后的区域渲染译文。"""
    pts = box.astype(np.int32)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    w, h = pts[:, 0].max() - x1, pts[:, 1].max() - y1

    # 尝试加载字体
    font_size = min(max(10, int(h * 0.6)), 36)
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 计算位置
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    # 如果文字太宽，缩小字号或换行
    if tw > w * 1.3:
        lines = []
        words = text.split()
        line = ""
        for word in words:
            test = f"{line} {word}".strip()
            tb = draw.textbbox((0, 0), test, font=font)
            if tb[2] - tb[0] <= w * 1.3:
                line = test
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
        if not lines:
            lines = [text]
    else:
        lines = [text]

    line_h = font_size + 4
    start_y = y1 + max(0, (h - len(lines) * line_h) // 2)

    for i, line in enumerate(lines):
        tb = draw.textbbox((0, 0), line, font=font)
        lw = tb[2] - tb[0]
        lx = x1 + max(0, (w - lw) // 2)
        ly = start_y + i * line_h

        # 黑色文字 + 白色描边
        draw.text((lx, ly), line, fill=(0, 0, 0), font=font,
                  stroke_width=1, stroke_fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


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
