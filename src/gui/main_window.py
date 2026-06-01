"""M8：主窗口 — 漫画翻译系统的PySide6图形界面。"""
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QProgressBar,
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
        self.resize(1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        left_panel = self._build_image_panel()
        splitter.addWidget(left_panel)
        right_panel = self._build_control_panel()
        splitter.addWidget(right_panel)
        splitter.setSizes([900, 500])
        main_layout.addWidget(splitter)

    def _build_image_panel(self) -> QWidget:
        """构建左侧图像预览面板。"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self._image_label = QLabel("请打开一张漫画图片")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; border: 1px solid #ccc; }"
        )
        self._image_label.setMinimumSize(600, 600)

        scroll = QScrollArea()
        scroll.setWidget(self._image_label)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        return panel

    def _build_control_panel(self) -> QWidget:
        """构建右侧控制面板。"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        layout.addWidget(QLabel("控制面板"))

        self._btn_open = QPushButton("打开漫画图片")
        self._btn_open.clicked.connect(self._on_open_image)
        layout.addWidget(self._btn_open)

        layout.addWidget(QLabel("源语言:"))
        self._combo_src = QComboBox()
        self._combo_src.addItems(["日语", "英语", "中文"])
        layout.addWidget(self._combo_src)

        layout.addWidget(QLabel("目标语言:"))
        self._combo_tgt = QComboBox()
        self._combo_tgt.addItems(["中文", "英语"])
        layout.addWidget(self._combo_tgt)

        self._btn_translate = QPushButton("一键翻译")
        self._btn_translate.clicked.connect(self._on_translate)
        self._btn_translate.setEnabled(False)
        self._btn_translate.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; padding: 10px; "
            "font-size: 14px; border-radius: 5px; }"
        )
        layout.addWidget(self._btn_translate)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)
        self._status_label = QLabel("就绪")
        layout.addWidget(self._status_label)
        layout.addStretch()

        layout.addWidget(QLabel("排版模式"))
        self._combo_layout = QComboBox()
        self._combo_layout.addItems(["自动", "手动"])
        layout.addWidget(self._combo_layout)
        return panel

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
            self._status_label.setText(f"已加载: {file_path.split('/')[-1]}")

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

    def _on_finished(self, output_path: str) -> None:
        self._progress_bar.setValue(100)
        self._status_label.setText(f"完成！保存至: {output_path}")
        self._btn_translate.setEnabled(True)
        self._btn_open.setEnabled(True)
        pixmap = QPixmap(output_path)
        scaled = pixmap.scaled(
            self._image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._image_label.setPixmap(scaled)
        QMessageBox.information(self, "翻译完成", f"翻译结果已保存至:\n{output_path}")

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
