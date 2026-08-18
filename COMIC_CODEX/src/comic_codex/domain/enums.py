from enum import StrEnum


class ReadingDirection(StrEnum):
    LTR = "ltr"
    RTL = "rtl"
    VERTICAL = "vertical"


class ProcessingStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    NEEDS_REVIEW = "needs_review"
    COMPLETE = "complete"
    FAILED = "failed"


class QualityFlag(StrEnum):
    LOW_DETECTION_CONFIDENCE = "low_detection_confidence"
    LOW_OCR_CONFIDENCE = "low_ocr_confidence"
    BLURRED_SOURCE = "blurred_source"
    TRANSLATION_RISK = "translation_risk"
    RENDER_OVERFLOW = "render_overflow"
    PROCESSING_FAILED = "processing_failed"


class TextKind(StrEnum):
    DIALOGUE = "dialogue"
    NARRATION = "narration"
    SOUND_EFFECT = "sound_effect"
    UNKNOWN = "unknown"

