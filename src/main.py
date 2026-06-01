"""Comic Translator — 主入口点。

用法:
    python src/main.py                        # 启动GUI
    python src/main.py --cli image.png        # CLI模式
"""
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Comic Translator - 漫画翻译系统")
    parser.add_argument("--cli", type=str, help="CLI模式：翻译单张图片")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--source", type=str, default="ja", help="源语言")
    parser.add_argument("--target", type=str, default="zh", help="目标语言")
    parser.add_argument("--output", type=str, default="output", help="输出目录")
    args = parser.parse_args()

    from src.utils.config import config
    config.load(args.config)

    if args.cli:
        run_cli(args.cli, args.source, args.target, args.output)
    else:
        run_gui()


def run_cli(image_path: str, source: str, target: str, output: str):
    """命令行管线执行。"""
    print(f"[Comic Translator] 处理: {image_path}")
    from src.gui.workers.pipeline_worker import PipelineWorker
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication(sys.argv)
    worker = PipelineWorker(image_path, source, target)

    def on_progress(value, msg):
        print(f"  [{value}%] {msg}")

    def on_finished(path):
        print(f"  完成！输出: {path}")
        app.quit()

    def on_error(err):
        print(f"  错误: {err}")
        app.quit()

    worker.progress.connect(on_progress)
    worker.finished.connect(on_finished)
    worker.error.connect(on_error)
    worker.start()
    app.exec()


def run_gui():
    """启动PySide6图形界面。"""
    from PySide6.QtWidgets import QApplication
    from src.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
