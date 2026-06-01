"""M5阶段2：对话图构建器 — 构建对话结构。"""
from src.utils.schemas import TextBox, DialogueEdge


class DialogueGraph:
    """从有序文本框构建对话图。

    按角色分组气泡，建立对话边（谁回应谁），
    导出为JSON结构供翻译引擎使用。
    """

    def build(
        self, boxes: list[TextBox]
    ) -> tuple[list[TextBox], list[DialogueEdge]]:
        """从有序文本框构建对话图。

        返回 (节点, 边)，节点是富化的文本框，
        边表示角色间的对话流向。
        """
        if not boxes:
            return [], []

        edges = self._build_edges(boxes)
        return boxes, edges

    def _group_by_character(self, boxes: list[TextBox]) -> dict[str, list[TextBox]]:
        """按角色ID分组文本框。"""
        groups: dict[str, list[TextBox]] = {}
        for box in boxes:
            cid = box.character_id or "unknown"
            if cid not in groups:
                groups[cid] = []
            groups[cid].append(box)
        return groups

    def _build_edges(self, boxes: list[TextBox]) -> list[DialogueEdge]:
        """在相邻的不同角色气泡间创建对话边。

        角色A说话后角色B说话 = 回应关系。
        同一角色连续气泡 = 延续关系。
        """
        edges = []
        for i in range(len(boxes) - 1):
            current = boxes[i]
            next_box = boxes[i + 1]
            if current.character_id and next_box.character_id:
                if current.character_id != next_box.character_id:
                    edges.append(DialogueEdge(
                        from_bubble_id=current.id,
                        to_bubble_id=next_box.id,
                        relation="response",
                    ))
                else:
                    edges.append(DialogueEdge(
                        from_bubble_id=current.id,
                        to_bubble_id=next_box.id,
                        relation="continuation",
                    ))
        return edges

    def export_json(
        self, nodes: list[TextBox], edges: list[DialogueEdge]
    ) -> dict:
        """导出对话图为JSON可序列化字典。"""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "text": n.text,
                    "language": n.language.value,
                    "character_id": n.character_id,
                    "position": {"x": n.x, "y": n.y, "w": n.width, "h": n.height},
                }
                for n in nodes
            ],
            "edges": [
                {
                    "from": e.from_bubble_id,
                    "to": e.to_bubble_id,
                    "relation": e.relation,
                }
                for e in edges
            ],
        }
