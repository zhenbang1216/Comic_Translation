"""M8：主窗口 — 漫画翻译系统的PySide6图形界面。"""
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QProgressBar, QCheckBox,
    QFileDialog, QMessageBox, QScrollArea, QSplitter,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap


class MainWindow(QMainWindow):
    """主应用窗口，包含图像查看器和控制面板。"""

    def __init__(self) -> None:
        super().__init__()
        self._image_path: str | None = None
        self._worker = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        """构建主窗口布局。"""
        self.setWindowTitle("Comic Translator - 漫画翻译")
        self.resize(1400, 850)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)

        # === 左侧：图片预览 ===
        left = QWidget()
        ll = QVBoxLayout(left)
        self._image_label = QLabel("请打开一张漫画图片\n\n支持 jpg/png/bmp/webp")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet(
            "QLabel { background: #f5f5f5; border: 1px dashed #ccc; }"
        )
        self._image_label.setMinimumSize(700, 600)
        scroll = QScrollArea()
        scroll.setWidget(self._image_label)
        scroll.setWidgetResizable(True)
        ll.addWidget(scroll)
        splitter.addWidget(left)

        # === 右侧：控制面板 ===
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setSpacing(8)

        rl.addWidget(QLabel("<b>漫画翻译</b>"))

        self._btn_open = QPushButton("打开漫画图片")
        self._btn_open.clicked.connect(self._on_open_image)
        rl.addWidget(self._btn_open)

        rl.addWidget(QLabel("源语言:"))
        self._combo_src = QComboBox()
        self._combo_src.addItems(["日语", "英语", "中文"])
        rl.addWidget(self._combo_src)

        rl.addWidget(QLabel("目标语言:"))
        self._combo_tgt = QComboBox()
        self._combo_tgt.addItems(["中文", "英语"])
        rl.addWidget(self._combo_tgt)

        self._check_save = QCheckBox("保留原图对照(PNG+文本)")
        self._check_save.setChecked(True)
        rl.addWidget(self._check_save)

        self._btn_translate = QPushButton("一键翻译")
        self._btn_translate.clicked.connect(self._on_translate)
        self._btn_translate.setEnabled(False)
        self._btn_translate.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; padding: 10px; "
            "font-size: 14px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #ccc; }"
        )
        rl.addWidget(self._btn_translate)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        rl.addWidget(self._progress_bar)

        self._status_label = QLabel("就绪")
        rl.addWidget(self._status_label)

        rl.addStretch()
        splitter.addWidget(right)
        splitter.setSizes([950, 450])
        main_layout.addWidget(splitter)

    def _on_open_image(self) -> None:
        """处理打开图片按钮点击。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开漫画图片", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if file_path:
            self._image_path = file_path
            pixmap = QPixmap(file_path)
            scaled = pixmap.scaled(
                self._image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self._image_label.setPixmap(scaled)
            self._btn_translate.setEnabled(True)
            self._status_label.setText(f"已加载: {Path(file_path).name}")

    def _on_translate(self) -> None:
        """启动翻译管线。"""
        if not self._image_path:
            return
        src_map = {"日语": "ja", "英语": "en", "中文": "zh"}
        tgt_map = {"中文": "zh", "英语": "en"}
        lang_src = src_map[self._combo_src.currentText()]
        lang_tgt = tgt_map[self._combo_tgt.currentText()]

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._btn_translate.setEnabled(False)
        self._btn_open.setEnabled(False)

        from src.gui.workers.pipeline_worker import PipelineWorker
        self._worker = PipelineWorker(self._image_path, lang_src, lang_tgt)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, value: int, message: str) -> None:
        self._progress_bar.setValue(value)
        self._status_label.setText(message)

    def _on_finished(self, translated_path: str, yolo_det_path: str) -> None:
        self._progress_bar.setValue(100)
        self._status_label.setText(f"完成! 翻译图: {translated_path}")

        pixmap = QPixmap(translated_path)
        scaled = pixmap.scaled(
            self._image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._image_label.setPixmap(scaled)

        self._btn_translate.setEnabled(True)
        self._btn_open.setEnabled(True)

        QMessageBox.information(self, "完成",
            f"YOLO检测图: {yolo_det_path}\n翻译结果: {translated_path}")

    def _on_error(self, error_msg: str) -> None:
        self._progress_bar.setVisible(False)
        self._status_label.setText(f"错误: {error_msg}")
        self._btn_translate.setEnabled(True)
        self._btn_open.setEnabled(True)
        QMessageBox.critical(self, "翻译失败", f"发生错误:\n{error_msg}")

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait()
        event.accept()
