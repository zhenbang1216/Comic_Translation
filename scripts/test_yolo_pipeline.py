"""YOLO检测 + PaddleOCR识别 + 翻译 集成管线。

用法:
    conda activate comic
    python scripts/test_yolo_pipeline.py input/                        # 日→英
    python scripts/test_yolo_pipeline.py input/ --target zh            # 日→中
"""
import sys, os, argparse
from pathlib import Path
import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="YOLO检测+OCR+翻译管线")
    parser.add_argument("path", help="图片或文件夹路径")
    parser.add_argument("--model", default="runs/detect/comic_2-2/weights/best.pt",
                        help="YOLO模型路径")
    parser.add_argument("--lang", default="japan", choices=["japan","en","ch"])
    parser.add_argument("--target", default="en", choices=["en","zh"])
    parser.add_argument("--conf", type=float, default=0.3,
                        help="YOLO置信度阈值(默认0.3)")
    parser.add_argument("--rec-score", type=float, default=0.5,
                        help="OCR置信度阈值")
    parser.add_argument("--render", action="store_true",
                        help="输出修复+渲染后的翻译图")
    args = parser.parse_args()

    target = Path(args.path)
    images = (sorted([f for f in target.iterdir() if f.suffix.lower() in IMAGE_EXTS])
              if target.is_dir() else [target])
    if not images:
        print("无图片"); sys.exit(1)

    # ===== 加载模型 =====
    print("=== 加载模型 ===\n")
    from ultralytics import YOLO
    print(f"1/3 YOLO: {args.model}")
    yolo = YOLO(args.model)

    from paddleocr import PaddleOCR
    print("2/3 PaddleOCR (仅识别)...")
    ocr = PaddleOCR(lang=args.lang, use_doc_orientation_classify=False,
                    use_textline_orientation=False)

    from transformers import MarianMTModel, MarianTokenizer
    print("3/3 翻译模型...")
    lang_map = {"japan": "ja", "en": "en", "ch": "zh"}
    src = lang_map[args.lang]

    def load_mt(pair):
        m = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{pair}")
        t = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{pair}")
        return t, m

    if args.target == "zh":
        t1, m1 = load_mt("ja-en"); t2, m2 = load_mt("en-zh")
    else:
        t1, m1 = load_mt(f"{src}-{args.target}"); t2 = m2 = None

    # ===== 处理 =====
    out_dir = Path("output/yolo_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)
    import tempfile

    for idx, img_path in enumerate(images, 1):
        print(f"\n[{idx}/{len(images)}] {img_path.name}")
        img = cv2.imread(str(img_path))

        # Step 1: YOLO检测
        results = yolo.predict(str(img_path), conf=args.conf, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            print("  YOLO: 未检测到文字区域")
            continue

        print(f"  YOLO: {len(boxes)} 个文字区域")

        # Step 2: 逐个区域OCR识别
        texts = []
        for i, box in enumerate(boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            conf = float(boxes.conf[i]) if boxes.conf is not None else 0
            region = img[max(0,y1):min(img.shape[0],y2), max(0,x1):min(img.shape[1],x2)]
            if region.size == 0:
                continue

            # 保存临时文件做OCR
            from uuid import uuid4
            tmp_path = str(out_dir / f"_tmp_{uuid4().hex[:8]}.png")
            cv2.imwrite(tmp_path, region)
            rec = ocr.predict(tmp_path)
            cv2.waitKey(1)
            os.unlink(tmp_path)

            item = rec[0]
            if item["rec_texts"] and item["rec_scores"][0] >= args.rec_score:
                t = item["rec_texts"][0]
                s_ocr = item["rec_scores"][0]
                texts.append((t, s_ocr, (x1, y1, x2-x1, y2-y1), conf, box))

        print(f"  OCR: {len(texts)} 个有效文本")

        # Step 3: 翻译
        translations = []
        for t, s_ocr, bbox, conf_yolo, box_raw in texts:
            if args.target == "zh":
                en = _translate(t, t1, m1)
                zh = _translate(en, t2, m2)
                result = zh
            else:
                result = _translate(t, t1, m1)
            translations.append((t, result, bbox, conf_yolo, s_ocr))
            print(f"    {t} → {result}")

        # Step 4: 渲染(可选)
        if args.render and translations:
            cleaned = _inpaint(img, [(x,y,w,h) for _,_,(x,y,w,h),_,_ in translations])
            rendered = cleaned.copy()
            for orig, trans, (x,y,w,h), _, _ in translations:
                rendered = _render(rendered, trans, x, y, w, h)

            out_path = str(out_dir / (img_path.stem + "_translated.png"))
            cv2.imwrite(out_path, rendered)
            print(f"  渲染: {out_path}")

    print(f"\n完成!")


def _translate(text, tokenizer, model):
    if not text or not text.strip():
        return ""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _inpaint(img, boxes, padding=4):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for x, y, w, h in boxes:
        x1, y1 = max(0, x-padding), max(0, y-padding)
        x2, y2 = min(img.shape[1], x+w+padding), min(img.shape[0], y+h+padding)
        mask[y1:y2, x1:x2] = 255
    return cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def _render(img, text, x, y, w, h):
    from PIL import Image, ImageDraw, ImageFont
    font_size = max(10, min(int(h*0.6), 28))
    try:
        font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    tb = draw.textbbox((0,0), text, font=font)
    tw = tb[2]-tb[0]
    lx = x + max(0, (w-tw)//2)
    ly = y + max(0, (h-font_size)//2)
    draw.text((lx, ly), text, fill=(0,0,0), font=font, stroke_width=1, stroke_fill=(255,255,255))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    main()
