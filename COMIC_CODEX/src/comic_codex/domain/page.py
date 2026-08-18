from pathlib import Path
from typing import Self
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from comic_codex.domain.enums import ProcessingStatus, QualityFlag, TextKind
from comic_codex.domain.geometry import Polygon


class OCRCandidate(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    provider: str
    source_variant: str


class TranslationEntry(BaseModel):
    source_text: str
    target_text: str | None = None
    provider: str | None = None
    user_locked: bool = False


class TextRegion(BaseModel):
    id: UUID
    polygon: Polygon
    detection_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    ocr_candidates: list[OCRCandidate] = Field(default_factory=list)
    selected_ocr_index: int | None = None
    translation: TranslationEntry | None = None
    quality_flags: set[QualityFlag] = Field(default_factory=set)
    text_kind: TextKind = TextKind.UNKNOWN

    @model_validator(mode="after")
    def selected_candidate_must_exist(self) -> Self:
        if self.selected_ocr_index is None:
            return self
        if not 0 <= self.selected_ocr_index < len(self.ocr_candidates):
            raise ValueError("selected OCR index does not reference a candidate")
        return self

    def selected_text(self) -> str | None:
        if self.selected_ocr_index is None:
            return None
        return self.ocr_candidates[self.selected_ocr_index].text


class ComicPage(BaseModel):
    id: UUID
    source_image: Path
    page_index: int = Field(ge=0)
    status: ProcessingStatus = ProcessingStatus.PENDING
    text_regions: list[TextRegion] = Field(default_factory=list)
