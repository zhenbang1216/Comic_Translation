from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AfterValidator, BaseModel, Field, model_validator

from comic_codex.domain.page import ComicPage


def _non_blank(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("title must not be blank")
    return normalized


Title = Annotated[str, AfterValidator(_non_blank)]
LanguageCode = Annotated[str, Field(pattern=r"^[a-z]{2,3}(-[A-Z]{2})?$")]


class Chapter(BaseModel):
    id: UUID
    title: Title
    pages: list[ComicPage] = Field(default_factory=list)

    @model_validator(mode="after")
    def page_ids_and_indexes_must_be_unique(self) -> Self:
        page_ids = [page.id for page in self.pages]
        page_indexes = [page.page_index for page in self.pages]
        if len(page_ids) != len(set(page_ids)):
            raise ValueError("chapter contains duplicate page IDs")
        if len(page_indexes) != len(set(page_indexes)):
            raise ValueError("chapter contains duplicate page indexes")
        return self


class ComicProject(BaseModel):
    schema_version: Literal[1] = 1
    id: UUID
    title: Title
    source_language: LanguageCode
    target_language: LanguageCode
    chapters: list[Chapter] = Field(default_factory=list)

    @model_validator(mode="after")
    def chapter_and_page_ids_must_be_unique(self) -> Self:
        chapter_ids = [chapter.id for chapter in self.chapters]
        if len(chapter_ids) != len(set(chapter_ids)):
            raise ValueError("project contains duplicate chapter IDs")

        page_ids = [page.id for chapter in self.chapters for page in chapter.pages]
        if len(page_ids) != len(set(page_ids)):
            raise ValueError("project contains duplicate page IDs")
        return self

    def find_page(self, page_id: UUID) -> ComicPage:
        for chapter in self.chapters:
            for page in chapter.pages:
                if page.id == page_id:
                    return page
        raise KeyError(str(page_id))

