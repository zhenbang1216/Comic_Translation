# Comic Translator — 漫画翻译系统

基于深度学习的端到端自动化漫画文本翻译系统，融合 YOLO 目标检测、PaddleOCR 文字识别、OPUS-MT 神经机器翻译及图像修复渲染技术，实现漫画图像"一键翻译"。

## 环境配置

```bash
# 创建环境（需要 Python 3.10）
conda create -n comic python=3.10 -y
conda activate comic

# 安装依赖
pip install torch ultralytics paddlepaddle==3.0.0 paddleocr transformers PySide6 opencv-python Pillow pyyaml pytest scikit-learn sentencepiece
```

## 快速开始

```bash
conda activate comic

# 1. OCR文字检测+识别（带可视化框）
python scripts/test_full.py input/

# 2. OCR+翻译
python scripts/test_translate.py input/

# 3. 全流程：OCR→翻译→修复→渲染
python scripts/test_full_pipeline.py input/

# 4. GUI图形界面
python scripts/gui_app.py
```

## 项目结构

```
├── src/              # 核心模块（9个模块，4层架构）
│   ├── core/         #   算法管线层（检测/增强/OCR/翻译/修复/渲染）
│   ├── gui/          #   界面层（PySide6）
│   └── utils/        #   工具层（配置/数据模式）
├── scripts/          # 可运行脚本
│   ├── test_full.py          # OCR+可视化
│   ├── test_translate.py     # OCR+翻译
│   ├── test_full_pipeline.py # 全流程
│   └── gui_app.py           # GUI界面
├── tests/            # 测试
├── configs/          # 配置文件
├── docs/             # 设计文档
└── plan.md           # 技术方案
```

## 文档

| 文档 | 说明 |
|------|------|
| `plan.md` | 完整技术方案（选型+模块+可行性+时间规划） |
| `实现计划.md` | 中文实现计划（16个任务，含完整代码） |
| `docs/superpowers/plans/` | 英文实现计划 |

## 模型下载

首次运行会自动下载以下模型到本地缓存：

| 模型 | 用途 | 大小 |
|------|------|:--:|
| PP-OCRv5_server_det | 文字检测 | ~5MB |
| PP-OCRv5_server_rec | 日文识别 | ~15MB |
| OPUS-MT ja→en | 日→英翻译 | ~300MB |
| OPUS-MT en→zh | 英→中翻译 | ~300MB |
| YOLOv8n | 气泡检测 | ~6MB |

## 训练自定义模型

```bash
# 准备Roboflow格式数据集，放入 Training_dateset/
yolo detect train data="Training_dateset/你的数据集/data.yaml" model=yolov8n.pt epochs=50 name=my_model
```

模型保存在 `runs/detect/my_model/weights/best.pt`。

## 技术栈

- **检测**: YOLOv8 / PaddleOCR
- **OCR**: PP-OCRv5 + SVTR
- **翻译**: OPUS-MT (ja↔en, en↔zh)
- **修复**: OpenCV Inpainting
- **渲染**: PIL
- **GUI**: PySide6

## 团队

大学生项目，5人组。详见 `plan.md` 分工与时间规划。
