"""M5阶段5：一致性校验器 — 翻译后验证与重译触发。"""
from collections import defaultdict
from src.utils.schemas import TranslationResult


class ConsistencyChecker:
    """翻译后一致性验证。

    检查项：
    - 角色语气一致性（同一角色，相同语气）
    - 术语统一性（同一术语，相同译法）

    未通过的气泡触发重新翻译。
    """

    def check(
        self, results: list[TranslationResult]
    ) -> tuple[list[TranslationResult], list[str]]:
        """运行全部一致性检查。

        返回 (通过的结果列表, 失败的气泡ID列表)。
        失败的ID应使用调整后的上下文重新翻译。
        """
        if not results:
            return [], []

        failed_ids: set[str] = set()

        voice_issues = self._check_character_consistency(results)
        failed_ids.update(voice_issues)

        term_issues = self._check_terminology_consistency(results)
        failed_ids.update(term_issues)

        passed = [r for r in results if r.textbox_id not in failed_ids]
        return passed, list(failed_ids)

    def _check_character_consistency(
        self, results: list[TranslationResult]
    ) -> list[str]:
        """检查同一角色的翻译语气是否一致。"""
        issues = []
        char_groups: dict[str, list[TranslationResult]] = defaultdict(list)
        for r in results:
            if r.character_id:
                char_groups[r.character_id].append(r)

        for char_id, char_results in char_groups.items():
            if len(char_results) < 2:
                continue
            lengths = [len(r.translated_text) for r in char_results]
            avg_len = sum(lengths) / len(lengths)
            for r, length in zip(char_results, lengths):
                if avg_len > 0 and length > avg_len * 2.5:
                    issues.append(r.textbox_id)
        return issues

    def _check_terminology_consistency(
        self, results: list[TranslationResult]
    ) -> list[str]:
        """检查相同术语是否翻译一致。"""
        issues = []
        term_map: dict[str, str] = {}
        for r in results:
            key = r.original_text.strip()
            if key in term_map:
                if r.translated_text.strip() != term_map[key]:
                    issues.append(r.textbox_id)
            else:
                term_map[key] = r.translated_text.strip()
        return issues
