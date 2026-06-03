# Comic Translator — 端到端漫画翻译系统

基于深度学习的自动化漫画文本翻译系统。融合目标检测、OCR识别、神经机器翻译及图像修复渲染技术，实现漫画图像"一键翻译"。

## 六大创新

| 创新 | 命名 | 核心技术 |
|:--:|------|------|
| 一 | **双管线检测** (OBB+DBNet) | YOLOv8-OBB 旋转框 + DBNet 分割，兼顾竖排文字与弧形文字 |
| 二 | **TACE 三级级联增强** | CLAHE → SVTR抗模糊 → Real-ESRGAN | 按需增强，零冗余开销 |
| 三 | **SVTR 多语种OCR** | PP-OCRv5 + SVTR 全Transformer架构，中日英漫���文本识别 |
| 四 | **CADGT 角色感知对话图翻译** | 五阶段翻译管线：页面理解→对话图→角色画像→上下文翻译→一致性校验 |
| 五 | **LaMa 快速修复** | FFT全局感受野，网点纹理/速度线/渐变背景无痕修复 |
| 六 | **SAAT 风格感知自适应排版渲染** | 风格检测→空间计算→二分搜索排版→分层渲染→双输出 |

## 技术选型

| 管线阶段 | 选定方案 |
|:--:|------|
| 文本检测 | YOLOv8-OBB（主力）+ DBNet（辅助） |
| 图像增强 | TACE 三级级联：CLAHE → SVTR直识 → Real-ESRGAN |
| OCR识别 | PP-OCRv5 + SVTR |
| 机器翻译 | **CADGT** 五阶段管线（Qwen2.5-VL-2B）+ 两阶段混合调度 |
| 图像修复 | LaMa（Large Mask Inpainting） |
| 译文渲染 | **SAAT** 风格感知自适应排版渲染 |
| GUI交互 | QT6 PySide6 |

## 快速开始

### 环境配置

```bash
# Python 3.10
conda create -n comic python=3.10 -y
conda activate comic

pip install torch ultralytics paddlepaddle==3.0.0 paddleocr transformers PySide6 opencv-python Pillow pyyaml pytest scikit-learn sentencepiece
```

### 运行

```bash
conda activate comic

# GUI 图形界面（YOLO检测 → OCR → CADGT翻译 → LaMa修复 → SAAT渲染）
python scripts/gui_app.py

# CLI 命令行
python src/main.py --cli input/your_image.jpg --source ja --target zh

# 下载模型
python scripts/download_models.py
```

### 训练

```bash
# 准备Roboflow格式数据集
yolo detect train data="数据集/data.yaml" model=yolov8n.pt epochs=50 name=my_model
```

## 项目结构

```
├── src/core/          # 算法管线（9个模块，4层架构）
│   ├── detector.py               # M2 文本检测
│   ├── enhancer.py               # M3 TACE三级增强
│   ├── recognizer.py             # M4 OCR识别
│   ├── translator/               # M5 CADGT翻译（5个子模块）
│   ├── inpainter.py              # M6 图像修复
│   └── renderer/                 # M7 SAAT渲染（5个子模块）
├── src/gui/           # PySide6界面（M8）
│   ├── main_window.py
│   └── workers/
├── src/utils/         # 工具（配置/数据模式）
├── scripts/           # 可运行脚本
├── tests/             # 测试
├── configs/           # 配置文件
├── docs/              # 设计文档
├── plan.md            # 完整技术方案
└── 实现计划.md         # 中文实现计划
```

## 文档

| 文档 | 说明 |
|------|------|
| `plan.md` | 完整技术方案（选型论证+模块划分+可行性+时间规划） |
| `plan_mind_map.md` | 思维导图版（可导入幕布） |
| `实现计划.md` | 16 个可执行任务，含完整代码 |
| `docs/superpowers/plans/` | 英文实现计划 |

## 团队

5人组。分工与时间规划详见 `plan.md`。

## License

MIT
