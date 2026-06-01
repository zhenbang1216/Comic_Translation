"""M5阶段3：角色画像提取器 — 从对话文本构建人格卡片。"""
import json
from pathlib import Path
from src.utils.schemas import CharacterProfile


class CharacterProfileExtractor:
    """从对话文本中提取角色说话风格档案。

    分析自称词使用、句尾习惯和说话模式，
    构建可持久化的角色画像以驱动翻译风格。
    档案保存到磁盘，同一部漫画的后续页面可复用。
    """

    PRONOUN_MAP = {
        "オレ": "casual_masculine",
        "俺": "casual_masculine",
        "僕": "modest",
        "わたし": "neutral_feminine",
        "私": "formal",
        "あたし": "casual_feminine",
        "わし": "elderly",
        "俺様": "arrogant",
    }

    ENDING_STYLE_MAP = {
        "ぜ": "casual_masculine",
        "ぞ": "casual_masculine",
        "だ": "casual",
        "だぜ": "rough_casual",
        "です": "polite",
        "ます": "polite",
        "わ": "feminine",
        "ね": "friendly",
        "かしら": "feminine_wondering",
    }

    def extract_profile(
        self, character_id: str, lines: list[str]
    ) -> CharacterProfile:
        """从角色的对话行中提取角色画像。"""
        pronouns = self._extract_pronouns(lines)
        endings = self._extract_sentence_endings(lines)
        speech_style = self._infer_speech_style(pronouns, endings)
        role = self._infer_role(pronouns, speech_style)

        return CharacterProfile(
            id=character_id,
            pronouns=pronouns,
            sentence_endings=endings,
            speech_style=speech_style,
            estimated_role=role,
        )

    def _extract_pronouns(self, lines: list[str]) -> list[str]:
        found = []
        for line in lines:
            for pronoun in self.PRONOUN_MAP:
                if pronoun in line and pronoun not in found:
                    found.append(pronoun)
        return found

    def _extract_sentence_endings(self, lines: list[str]) -> list[str]:
        found = []
        for line in lines:
            for ending in sorted(self.ENDING_STYLE_MAP, key=len, reverse=True):
                if line.rstrip().endswith(ending) and ending not in found:
                    found.append(ending)
                    break
        return found

    def _infer_speech_style(
        self, pronouns: list[str], endings: list[str]
    ) -> str:
        style_scores: dict[str, int] = {}
        for pronoun in pronouns:
            style = self.PRONOUN_MAP.get(pronoun, "")
            if style:
                style_scores[style] = style_scores.get(style, 0) + 1
        for ending in endings:
            style = self.ENDING_STYLE_MAP.get(ending, "")
            if style:
                style_scores[style] = style_scores.get(style, 0) + 1
        if not style_scores:
            return "neutral"
        return max(style_scores, key=lambda k: style_scores[k])

    def _infer_role(self, pronouns: list[str], speech_style: str) -> str:
        if any(p in pronouns for p in ["俺様"]):
            return "antagonist"
        if any(p in pronouns for p in ["オレ", "俺"]):
            return "protagonist"
        if speech_style in ("polite", "formal"):
            return "supporting"
        return "unknown"

    def save_profile(self, profile: CharacterProfile, directory: str) -> None:
        """将角色画像持久化到磁盘。"""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        filepath = dir_path / f"{profile.id}.json"
        data = {
            "id": profile.id,
            "pronouns": profile.pronouns,
            "sentence_endings": profile.sentence_endings,
            "speech_style": profile.speech_style,
            "estimated_role": profile.estimated_role,
        }
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_profile(self, character_id: str, directory: str) -> CharacterProfile | None:
        """加载之前保存的角色画像。"""
        filepath = Path(directory) / f"{character_id}.json"
        if not filepath.exists():
            return None
        data = json.loads(filepath.read_text(encoding="utf-8"))
        return CharacterProfile(
            id=data["id"],
            pronouns=data.get("pronouns", []),
            sentence_endings=data.get("sentence_endings", []),
            speech_style=data.get("speech_style", "neutral"),
            estimated_role=data.get("estimated_role", "unknown"),
        )
