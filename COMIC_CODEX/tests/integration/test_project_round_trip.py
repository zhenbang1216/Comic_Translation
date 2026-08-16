from pathlib import Path

from comic_codex.domain.geometry import Point, Polygon
from comic_codex.domain.page import OCRCandidate, TextRegion, TranslationEntry
from comic_codex.domain.project import ComicProject
from comic_codex.project_store.store import JsonProjectStore


def test_project_round_trip_preserves_processing_state(
    tmp_path: Path, comic_project: ComicProject
) -> None:
    page = comic_project.chapters[0].pages[0]
    page.text_regions.append(
        TextRegion(
            id=page.id,
            polygon=Polygon(
                points=(Point(x=0, y=0), Point(x=4, y=0), Point(x=4, y=8))
            ),
            detection_confidence=0.88,
            ocr_candidates=[
                OCRCandidate(
                    text="こんにちは",
                    confidence=0.93,
                    provider="paddleocr",
                    source_variant="original",
                )
            ],
            selected_ocr_index=0,
            translation=TranslationEntry(
                source_text="こんにちは", target_text="你好", provider="offline"
            ),
        )
    )
    path = tmp_path / "chapter.comicproj"
    store = JsonProjectStore()

    store.save(comic_project, path)
    restored = store.load(path)

    assert restored == comic_project
    assert restored.chapters[0].pages[0].text_regions[0].selected_text() == "こんにちは"
