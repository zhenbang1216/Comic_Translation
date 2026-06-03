"""M5阶段1：页面分析器 — 场景分类、阅读顺序、气泡特征提取、角色分配。"""
from src.utils.schemas import TextBox, SceneType


SCENE_KEYWORDS: dict[SceneType, list[str]] = {
    SceneType.BATTLE: [
        "戦う", "倒す", "攻撃", "必殺", "敵", "戦闘", "勝負", "力",
        "やられる", "やっつける", "ぶっ飛ばす", "死", "殺", "斬",
        "fight", "attack", "defeat", "battle", "kill", "destroy", "power", "enemy",
    ],
    SceneType.ROMANCE: [
        "好き", "愛", "恋", "キス", "デート", "告白", "抱きしめ",
        "love", "kiss", "date", "heart", "dear", "sweet", "hug",
    ],
    SceneType.COMEDY: [
        "笑", "バカ", "アホ", "ギャグ", "面白", "あはは", "うける", "笑える",
        "lol", "haha", "funny", "joke",
    ],
    SceneType.SERIOUS: [
        "責任", "約束", "運命", "真実", "誓", "覚悟", "許",
        "serious", "promise", "truth", "fate", "destiny", "resolve",
    ],
    SceneType.DAILY: [
        "朝", "ご飯", "学校", "仕事", "買い物", "宿題", "おはよう",
        "おやすみ", "いただきます", "天気", "晩",
        "school", "homework", "morning", "weather", "dinner",
    ],
}


class PageAnalyzer:
    """分析漫画页面以理解布局和上下文。

    确定阅读顺序（日漫：右→左），分类场景类型，
    提取每个气泡的视觉特征供下游对话图构建使用。
    """

    def determine_reading_order(self, boxes: list[TextBox]) -> list[TextBox]:
        """按日漫阅读顺序排序文本框（从右上方开始，从右向左，从上向下）。"""
        if not boxes:
            return []
        row_threshold = 60
        sorted_by_y = sorted(boxes, key=lambda b: b.y)
        rows = []
        current_row = [sorted_by_y[0]]
        for box in sorted_by_y[1:]:
            if abs(box.y - current_row[0].y) < row_threshold:
                current_row.append(box)
            else:
                rows.append(sorted(current_row, key=lambda b: b.x, reverse=True))
                current_row = [box]
        rows.append(sorted(current_row, key=lambda b: b.x, reverse=True))

        ordered = []
        for row in rows:
            ordered.extend(row)
        return ordered

    def classify_scene(self, image=None, texts: list[str] | None = None) -> SceneType:
        """使用关键词启发式分类页面场景类型。

        优先使用文本关键词匹配，如无文本则返回UNKNOWN。
        完整的CLIP集成在P2阶段添加。
        """
        if not texts:
            return SceneType.UNKNOWN

        combined = " ".join(texts)
        scores: dict[SceneType, int] = {}
        for scene_type, keywords in SCENE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in combined)
            if score > 0:
                scores[scene_type] = score

        if not scores:
            return SceneType.UNKNOWN

        return max(scores, key=scores.get)

    def extract_bubble_features(self, box: TextBox) -> dict:
        """提取文本气泡的视觉特征用于角色归属。"""
        return {
            "width": box.width,
            "height": box.height,
            "aspect_ratio": box.width / max(box.height, 1),
            "angle": box.angle,
            "position_x": box.x,
            "position_y": box.y,
        }

    def assign_characters_by_tail(
        self, boxes: list[TextBox], image_path: str = ""
    ) -> list[TextBox]:
        """通过文本模式分析将文本框分配给角色。

        分析每个气泡文本中的自称词和句尾习惯，
        相同模式的文本分配给同一角色ID。
        完整尾巴检测使用CV轮廓分析留待后续。
        """
        from src.core.translator.character_profile import CharacterProfileExtractor

        pronoun_map = CharacterProfileExtractor.PRONOUN_MAP
        ending_map = CharacterProfileExtractor.ENDING_STYLE_MAP

        signature_to_char: dict[tuple, str] = {}
        char_counter = 0

        for box in boxes:
            if not box.text or not box.text.strip():
                box.character_id = None
                continue

            text = box.text
            found_pronouns = tuple(
                p for p in sorted(pronoun_map) if p in text
            )
            found_endings = tuple(
                e for e in sorted(ending_map, key=len, reverse=True)
                if text.rstrip().endswith(e)
            )

            signature = (found_pronouns, found_endings)

            if not found_pronouns and not found_endings:
                box.character_id = None
            elif signature in signature_to_char:
                box.character_id = signature_to_char[signature]
            else:
                char_id = f"char_{chr(ord('A') + char_counter)}"
                signature_to_char[signature] = char_id
                box.character_id = char_id
                char_counter += 1

        return boxes
