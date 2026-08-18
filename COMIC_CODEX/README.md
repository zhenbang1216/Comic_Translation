# COMIC_CODEX

COMIC_CODEX 是面向 Windows 笔记本的漫画智能翻译与自动嵌字桌面软件。

## 项目边界

- Git 仓库：`E:\Create\project\COMIC\COMIC_CODEX`
- 新工程根目录：`E:\Create\project\COMIC\COMIC_CODEX\COMIC_CODEX`
- 数据资产：`E:\Create\Data\COMIC\COMIC_CODEX`
- 目标设备：AMD Ryzen 9、RTX 4070 Laptop（按 8GB 显存设计）、16GB 内存
- 默认模式：完全离线、零付费；在线 API 仅为可选增强

旧工程 `Comic_Translation0` 和 `Comic_cccdx0` 只作为迁移来源，不在新主线实施中修改。

项目总体方案位于：
`E:\Create\Data\COMIC\COMIC_CODEX\COMIC_CODEX_项目总体方案.md`。

## 开发环境

```powershell
Set-Location 'E:\Create\project\COMIC\COMIC_CODEX\COMIC_CODEX'
& 'E:\huangjingwei\SOFT\VisionTrainPlus_Envs\Miniconda3\envs\morning_agent\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
$env:COMIC_CODEX_DATA_ROOT = 'E:\Create\Data\COMIC\COMIC_CODEX'
```

## 基础阶段验证

```powershell
.\.venv\Scripts\python.exe scripts\verify_environment.py
.\.venv\Scripts\python.exe -m pytest -q --cov=src/comic_codex --cov-report=term-missing
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\mypy.exe src
```

详细实施计划位于：
`E:\Create\Data\COMIC\COMIC_CODEX\2026-08-16-01-foundation-implementation-plan.md`。

