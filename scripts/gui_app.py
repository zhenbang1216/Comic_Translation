"""漫画翻译 GUI — PySide6 图形界面。

用法:
    conda activate comic
    python scripts/gui_app.py
"""
import sys, os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QProgressBar, QFileDialog,
    QMessageBox, QScrollArea, QSplitter, QCheckBox,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage


# ==================== 管线工作线程 ====================
class PipelineThread(QThread):
    progress = Signal(int, str)
    finished = Signal(str, str)
    error = Signal(str)

    def __init__(self, img_path, src_lang, tgt_lang, rec_score):
        super().__init__()
        self.img_path = img_path
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.rec_score = rec_score

    def run(self):
        try:
            self.progress.emit(10, "加载OCR模型...")
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(
                lang=self.src_lang,
                use_doc_orientation_classify=False,
                use_textline_orientation=False,
            )

            self.progress.emit(30, "识别文字...")
            img = cv2.imread(self.img_path)
            results = ocr.predict(self.img_path)
            item = results[0]
            texts = item["rec_texts"]
            scores = item["rec_scores"]
            boxes = item["rec_polys"]

            filtered = [(t, s, b) for t, s, b in zip(texts, scores, boxes) if s >= self.rec_score]

            if not filtered:
                self.error.emit("未检测到有效文字")
                return

            self.progress.emit(50, "加载翻译模型(约30秒)...")
            from transformers import MarianMTModel, MarianTokenizer
            lang_map = {"japan": "ja", "en": "en", "ch": "zh"}
            src_iso = lang_map[self.src_lang]

            tok_main, mod_main = None, None
            tok_en_zh, mod_en_zh = None, None

            if self.tgt_lang == "zh":
                self.progress.emit(52, "加载 ja→en 模型...")
                tok_ja_en, mod_ja_en = self._load_mt("ja-en")
                self.progress.emit(55, "加载 en→zh 模型...")
                tok_en_zh, mod_en_zh = self._load_mt("en-zh")
            else:
                self.progress.emit(52, f"加载 {src_iso}→{self.tgt_lang} 模型...")
                tok_main, mod_main = self._load_mt(f"{src_iso}-{self.tgt_lang}")

            self.progress.emit(60, "翻译中...")
            translations = []
            for i, (t, s, b) in enumerate(filtered):
                if self.tgt_lang == "zh":
                    en = self._translate(t, tok_ja_en, mod_ja_en)
                    zh = self._translate(en, tok_en_zh, mod_en_zh)
                    translations.append((t, zh, b))
                else:
                    tr = self._translate(t, tok_main, mod_main)
                    translations.append((t, tr, b))

                if i % 5 == 0:
                    self.progress.emit(60 + int(20 * i / len(filtered)), f"翻译中 {i+1}/{len(filtered)}...")

            self.progress.emit(82, "修复原文字...")
            cleaned = self._inpaint(img, [b for _, _, b in filtered])

            self.progress.emit(90, "渲染译文...")
            rendered = cleaned.copy()
            for original, translated, box in translations:
                if translated.strip():
                    rendered = self._render(rendered, translated, box)

            out_dir = Path("output/gui")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(out_dir / (Path(self.img_path).stem + "_translated.png"))
            cv2.imwrite(out_path, rendered)

            # 生成对照文本
            log_path = str(out_dir / (Path(self.img_path).stem + "_log.txt"))
            with open(log_path, "w", encoding="utf-8") as f:
                for orig, trans, _ in translations:
                    f.write(f"{orig} → {trans}\n")

            self.progress.emit(100, "完成!")
            self.finished.emit(out_path, log_path)

        except Exception as e:
            self.error.emit(str(e))

    @staticmethod
    def _load_mt(lang_pair):
        from transformers import MarianMTModel, MarianTokenizer
        model_name = f"Helsinki-NLP/opus-mt-{lang_pair}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return tokenizer, model

    @staticmethod
    def _translate(text, tokenizer, model):
        if not text or not text.strip():
            return ""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs, max_length=128)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    @staticmethod
    def _inpaint(img, boxes, padding=4):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for box in boxes:
            pts = box.astype(np.int32)
            x1 = max(0, pts[:, 0].min() - padding)
            y1 = max(0, pts[:, 1].min() - padding)
            x2 = min(img.shape[1], pts[:, 0].max() + padding)
            y2 = min(img.shape[0], pts[:, 1].max() + padding)
            mask[y1:y2, x1:x2] = 255
        return cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    @staticmethod
    def _render(img, text, box, font_size=14):
        pts = box.astype(np.int32)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        w = max(pts[:, 0].max() - x1, 1)
        h = max(pts[:, 1].max() - y1, 1)
        font_size = max(10, min(int(h * 0.6), 32))

        try:
            font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 简单断行
        lines = [text]
        tb = draw.textbbox((0, 0), text, font=font)
        if tb[2] - tb[0] > w:
            mid = len(text) // 2
            lines = [text[:mid], text[mid:]]

        line_h = font_size + 4
        start_y = y1 + max(0, (h - len(lines) * line_h) // 2)

        for i, line in enumerate(lines):
            tb = draw.textbbox((0, 0), line, font=font)
            lx = x1 + max(0, (w - (tb[2] - tb[0])) // 2)
            ly = start_y + i * line_h
            draw.text((lx, ly), line, fill=(0, 0, 0), font=font,
                      stroke_width=1, stroke_fill=(255, 255, 255))

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._img_path = None
        self._thread = None
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Comic Translator - 漫画翻译")
        self.resize(1400, 850)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)

        # === 左侧：图片预览 ===
        left = QWidget()
        ll = QVBoxLayout(left)
        self._img_label = QLabel("请打开一张漫画图片\n\n支持 jpg/png/bmp/webp")
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setStyleSheet("QLabel { background: #f5f5f5; border: 1px dashed #ccc; }")
        self._img_label.setMinimumSize(700, 600)
        scroll = QScrollArea()
        scroll.setWidget(self._img_label)
        scroll.setWidgetResizable(True)
        ll.addWidget(scroll)
        splitter.addWidget(left)

        # === 右侧：控制面板 ===
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setSpacing(8)

        rl.addWidget(QLabel("<b>漫画翻译</b>"))

        self._btn_open = QPushButton("📂 打开图片")
        self._btn_open.clicked.connect(self._on_open)
        rl.addWidget(self._btn_open)

        rl.addWidget(QLabel("源语言:"))
        self._combo_src = QComboBox()
        self._combo_src.addItems(["日语", "英语", "中文"])
        rl.addWidget(self._combo_src)

        rl.addWidget(QLabel("目标语言:"))
        self._combo_tgt = QComboBox()
        self._combo_tgt.addItems(["英语", "中文"])
        rl.addWidget(self._combo_tgt)

        self._check_save = QCheckBox("保留原图对照(PNG+文本)")
        self._check_save.setChecked(True)
        rl.addWidget(self._check_save)

        self._btn_run = QPushButton("🚀 开始翻译")
        self._btn_run.clicked.connect(self._on_run)
        self._btn_run.setEnabled(False)
        self._btn_run.setStyleSheet(
            "QPushButton { background: #4CAF50; color: white; padding: 12px; "
            "font-size: 15px; border-radius: 6px; }"
            "QPushButton:hover { background: #45a049; }"
            "QPushButton:disabled { background: #ccc; }"
        )
        rl.addWidget(self._btn_run)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        rl.addWidget(self._progress)

        self._status = QLabel("就绪")
        rl.addWidget(self._status)

        rl.addStretch()
        splitter.addWidget(right)
        splitter.setSizes([950, 450])
        main_layout.addWidget(splitter)

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开漫画图片", "",
            "图片 (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if path:
            self._img_path = path
            pixmap = QPixmap(path)
            scaled = pixmap.scaled(self._img_label.size(),
                                   Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._img_label.setPixmap(scaled)
            self._btn_run.setEnabled(True)
            self._status.setText(f"已加载: {Path(path).name}")

    def _on_run(self):
        if not self._img_path:
            return

        src_map = {"日语": "japan", "英语": "en", "中文": "ch"}
        tgt_map = {"英语": "en", "中文": "zh"}

        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._btn_run.setEnabled(False)
        self._btn_open.setEnabled(False)

        self._thread = PipelineThread(
            self._img_path,
            src_map[self._combo_src.currentText()],
            tgt_map[self._combo_tgt.currentText()],
            rec_score=0.5,
        )
        self._thread.progress.connect(self._on_progress)
        self._thread.finished.connect(self._on_finished)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _on_progress(self, value, msg):
        self._progress.setValue(value)
        self._status.setText(msg)

    def _on_finished(self, img_path, log_path):
        self._progress.setValue(100)
        self._status.setText(f"完成! {img_path}")

        pixmap = QPixmap(img_path)
        scaled = pixmap.scaled(self._img_label.size(),
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._img_label.setPixmap(scaled)

        self._btn_run.setEnabled(True)
        self._btn_open.setEnabled(True)

        msg = f"翻译完成!\n\n图片: {img_path}"
        if self._check_save.isChecked():
            msg += f"\n对照文本: {log_path}"

        QMessageBox.information(self, "完成", msg)

    def _on_error(self, msg):
        self._progress.setVisible(False)
        self._status.setText(f"错误: {msg}")
        self._btn_run.setEnabled(True)
        self._btn_open.setEnabled(True)
        QMessageBox.critical(self, "错误", msg)

    def closeEvent(self, event):
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
