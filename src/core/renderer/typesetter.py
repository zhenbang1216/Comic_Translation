"""M7.3：排版器 — 带二分搜索字号的自适应排版引擎。"""
from PIL import ImageFont
from src.utils.schemas import TypesetResult, RenderStyle


class Typesetter:
    """为翻译后的文字计算最优排版参数。

    使用二分搜索找到气泡空间内容纳的最大字号，
    然后应用语言感知的换行策略。
    """

    def compute(
        self,
        text: str,
        available_width: int,
        available_height: int,
        language: str,
        style: RenderStyle,
    ) -> TypesetResult:
        """为给定文字和空间计算最优排版方案。"""
        if not text:
            return TypesetResult(
                font_size=12, lines=[], line_height=0,
                start_x=0, start_y=0, alignment="center"
            )

        font_size = self._binary_search_font_size(
            text, available_width, available_height, language
        )

        lines = self._break_lines(text, available_width, language, font_size)

        line_height = self._compute_line_height(font_size, len(lines))
        total_height = len(lines) * font_size + (len(lines) - 1) * (line_height - font_size)
        start_y = max(0, int((available_height - total_height) // 2))
        start_x = 0

        return TypesetResult(
            font_size=font_size,
            lines=lines,
            line_height=line_height,
            start_x=start_x,
            start_y=start_y,
            alignment=style.alignment,
        )

    def _binary_search_font_size(
        self, text: str, max_width: int, max_height: int, language: str
    ) -> int:
        """二分搜索最大可容纳字号。"""
        low, high = 8, 48
        best = 10

        while low <= high:
            mid = (low + high) // 2
            if self._fits(text, max_width, max_height, mid, language):
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return max(8, min(48, best))

    def _fits(
        self, text: str, max_width: int, max_height: int, font_size: int, language: str
    ) -> bool:
        """检查给定字号下文字是否能放入指定空间。"""
        lines = self._break_lines(text, max_width, language, font_size)
        total_height = len(lines) * font_size + (len(lines) - 1) * max(1, font_size // 4)
        return total_height <= max_height

    def _break_lines(
        self, text: str, max_width: int, language: str, font_size: int
    ) -> list[str]:
        """将文字断行为适合max_width的行。"""
        if language in ("zh", "ja"):
            return self._break_cjk(text, max_width, font_size)
        else:
            return self._break_word(text, max_width, font_size)

    def _break_cjk(
        self, text: str, max_width: int, font_size: int
    ) -> list[str]:
        """CJK文字按字符断行，附带中文标点规则。"""
        char_width = font_size
        chars_per_line = max(1, max_width // char_width)
        lines = []
        i = 0
        while i < len(text):
            end = min(i + chars_per_line, len(text))
            if end < len(text) and text[end] in "。、，！？；：」』）":
                end -= 1
            if end == i:
                end = i + 1
            lines.append(text[i:end])
            i = end
        return lines if lines else [text]

    def _break_word(
        self, text: str, max_width: int, font_size: int
    ) -> list[str]:
        """英文文字按单词边界断行。"""
        words = text.split()
        lines = []
        current_line = ""
        avg_char_width = font_size * 0.6
        chars_per_line = max(1, int(max_width / avg_char_width))

        for word in words:
            if len(current_line) + len(word) + 1 <= chars_per_line:
                current_line = f"{current_line} {word}".strip() if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                if len(word) > chars_per_line:
                    for j in range(0, len(word), chars_per_line):
                        lines.append(word[j:j + chars_per_line])
                    current_line = ""
                else:
                    current_line = word
        if current_line:
            lines.append(current_line)
        return lines if lines else [text]

    @staticmethod
    def _compute_line_height(font_size: int, num_lines: int) -> int:
        """根据行数计算行高。"""
        if num_lines <= 1:
            return font_size
        elif num_lines == 2:
            return int(font_size * 1.3)
        else:
            return int(font_size * 1.2)
