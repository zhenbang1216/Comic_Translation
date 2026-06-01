"""模块间通信的数据模式定义。"""
from dataclasses import dataclass, field
from enum import Enum


class Language(str, Enum):
    JA = "ja"
    ZH = "zh"
    EN = "en"


class SceneType(str, Enum):
    BATTLE = "battle"
    DAILY = "daily"
    ROMANCE = "romance"
    COMEDY = "comedy"
    SERIOUS = "serious"
    UNKNOWN = "unknown"


@dataclass
class TextBox:
    """M2检测器输出 — 检测到的文本区域。"""
    id: str
    x: float
    y: float
    width: float
    height: float
    angle: float = 0.0
    text: str = ""
    language: Language = Language.JA
    confidence: float = 0.0
    character_id: str | None = None
    style_params: dict | None = None
    bubble_bbox: tuple[float, float, float, float] | None = None


@dataclass
class CharacterProfile:
    """M5阶段3输出 — 角色的说话风格档案。"""
    id: str
    pronouns: list[str] = field(default_factory=list)
    sentence_endings: list[str] = field(default_factory=list)
    speech_style: str = "neutral"
    estimated_role: str = "unknown"


@dataclass
class DialogueEdge:
    """对话图中的边 — 对话流向。"""
    from_bubble_id: str
    to_bubble_id: str
    relation: str = "response"


@dataclass
class TranslationResult:
    """M5输出 — 翻译结果。"""
    textbox_id: str
    original_text: str
    translated_text: str
    source_lang: Language
    target_lang: Language
    translation_method: str
    character_id: str | None = None
    confidence: float = 0.0


@dataclass
class RenderStyle:
    """M7.1输出 — 检测到的原文字视觉风格。"""
    text_color: tuple[int, int, int] = (0, 0, 0)
    stroke_color: tuple[int, int, int] | None = None
    stroke_width: int = 0
    shadow_offset: tuple[int, int] | None = None
    shadow_color: tuple[int, int, int] | None = None
    estimated_font_size: int = 14
    font_weight: str = "regular"
    alignment: str = "center"


@dataclass
class TypesetResult:
    """M7.3输出 — 计算得到的排版参数。"""
    font_size: int
    lines: list[str]
    line_height: int
    start_x: int
    start_y: int
    alignment: str
