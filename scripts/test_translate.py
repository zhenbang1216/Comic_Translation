"""端到端翻译测试 — OCR检测+识别+翻译。

把漫画图片丢到 input/，一键完成从图片到翻译的全流程。

用法:
    conda activate comic
    python scripts/test_translate.py input/                           # 日→英
    python scripts/test_translate.py input/ --target zh               # 日→中（链式翻译）
"""
import sys
import argparse
from pathlib import Path
import cv2

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_translator(lang_pair):
    """加载OPUS-MT翻译模型。"""
    from transformers import MarianMTModel, MarianTokenizer
    model_name = f"Helsinki-NLP/opus-mt-{lang_pair}"
    print(f"  加载 {model_name}...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


def translate(text, tokenizer, model):
    """翻译单条文本。"""
    if not text or not text.strip():
        return ""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="漫画图片端到端翻译")
    parser.add_argument("path", type=str, help="图片路径 或 文件夹路径")
    parser.add_argument("--lang", type=str, default="japan",
                        choices=["japan", "en", "ch"], help="源语言")
    parser.add_argument("--target", type=str, default="en",
                        choices=["en", "zh"], help="目标语言 (默认en)")
    parser.add_argument("--rec-score", type=float, default=0.5,
                        help="只翻译识别分数≥此值的文字 (默认0.5)")
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
        print(f"没有找到图片，支持格式: {', '.join(IMAGE_EXTS)}")
        sys.exit(1)

    # ----- 加载 OCR -----
    from paddleocr import PaddleOCR
    print("=== 加载模型 ===\n")
    print("1/2 加载 PaddleOCR...")
    ocr = PaddleOCR(
        lang=args.lang,
        use_doc_orientation_classify=False,
        use_textline_orientation=False,
    )

    # ----- 加载翻译 -----
    print("2/2 加载翻译模型...")
    # PaddleOCR lang → ISO 代码映射
    lang_map = {"japan": "ja", "en": "en", "ch": "zh"}
    src_iso = lang_map[args.lang]

    if args.target == "zh":
        # 链式: 日→英→中
        tok_ja_en, mod_ja_en = load_translator("ja-en")
        tok_en_zh, mod_en_zh = load_translator("en-zh")
        print("  链式翻译: JA → EN → ZH\n")
    else:
        tok_main, mod_main = load_translator(f"{src_iso}-{args.target}")
        print(f"  直接翻译: {src_iso.upper()} → {args.target.upper()}\n")

    # ----- 处理图片 -----
    print(f"=== 处理 {len(images)} 张图片 ===\n")

    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] {img_path.name}")

        # OCR
        results = ocr.predict(str(img_path))
        item = results[0]
        texts = item["rec_texts"]
        scores = item["rec_scores"]

        # 过滤低分数
        filtered = [(t, s) for t, s in zip(texts, scores) if s >= args.rec_score]
        print(f"  OCR: {len(texts)} 区域 → 过滤后 {len(filtered)} 个 (≥{args.rec_score})")

        if not filtered:
            print("  无有效文本\n")
            continue

        # 翻译
        print(f"  翻译结果:")
        for t, s in filtered:
            if args.target == "zh":
                en = translate(t, tok_ja_en, mod_ja_en)
                zh = translate(en, tok_en_zh, mod_en_zh)
                print(f"    {t}  →  {en}  →  {zh}")
            else:
                translated = translate(t, tok_main, mod_main)
                print(f"    {t}  →  {translated}")

        print()


if __name__ == "__main__":
    main()
