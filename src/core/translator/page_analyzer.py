"""M5阶段1：页面分析器 — 场景分类、阅读顺序、气泡特征提取。"""
from src.utils.schemas import TextBox, SceneType


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

    def classify_scene(self, image) -> SceneType:
        """使用轻量启发式或CLIP分类页面场景类型。

        当前返回UNKNOWN。完整的CLIP集成在P2阶段添加。
        """
        return SceneType.UNKNOWN

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
        self, boxes: list[TextBox], image_path: str
    ) -> list[TextBox]:
        """通过检测气泡尾巴将文本框分配给角色。

        漫画中气泡尾巴指向说话的角色。
        当前为启发式占位，完整尾巴检测使用CV轮廓分析。
        """
        for box in boxes:
            box.character_id = None
        return boxes
