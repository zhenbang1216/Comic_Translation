"""M5阶段4：翻译引擎 — 两阶段混合翻译 + 上下文感知。"""
from src.utils.schemas import TextBox, TranslationResult, CharacterProfile, Language
from src.utils.config import config


class TranslationEngine:
    """两阶段混合翻译引擎。

    简单气泡（短、无依赖）：OPUS-MT 快速直译（~50ms）
    复杂气泡（长、有上下文）：Qwen2.5-VL-2B 多模态翻译（~3s）
    """

    def __init__(self) -> None:
        from src.core.model_manager import ModelManager
        self._mgr = ModelManager()
        self._simple_threshold = config.get("pipeline.translation.simple_bubble_threshold", 0.6)
        self._max_context_chars = config.get("pipeline.translation.max_vlm_context_chars", 512)

    def translate_all(
        self,
        boxes: list[TextBox],
        profiles: dict[str, CharacterProfile],
        scene: str,
        target_lang: Language,
    ) -> list[TranslationResult]:
        """翻译页面上的所有文本框，带上下文感知。"""
        results = []
        previous_translations: list[TranslationResult] = []

        for box in boxes:
            if self._is_simple_bubble(box.text):
                result = self._translate_simple(box, target_lang)
            else:
                result = self._translate_complex(
                    box, boxes, profiles, scene, target_lang, previous_translations
                )
            results.append(result)
            previous_translations.append(result)

        return results

    def _is_simple_bubble(self, text: str) -> bool:
        """判断气泡是否足够简单以使用OPUS-MT直译。"""
        if len(text) <= 15 and "\n" not in text:
            return True
        question_words = ["何", "なぜ", "どうして", "か？", "ですか"]
        if any(q in text for q in question_words):
            return True
        return False

    def _translate_simple(
        self, box: TextBox, target_lang: Language
    ) -> TranslationResult:
        """使用OPUS-MT快速翻译。"""
        source_lang = box.language
        translated = self._translate_opus(box.text, source_lang.value, target_lang.value)
        return TranslationResult(
            textbox_id=box.id,
            original_text=box.text,
            translated_text=translated,
            source_lang=source_lang,
            target_lang=target_lang,
            translation_method="opus_mt",
            character_id=box.character_id,
            confidence=0.85,
        )

    def _translate_complex(
        self,
        box: TextBox,
        all_boxes: list[TextBox],
        profiles: dict[str, CharacterProfile],
        scene: str,
        target_lang: Language,
        previous: list[TranslationResult],
    ) -> TranslationResult:
        """使用Qwen2.5-VL-2B进行五维上下文感知翻译。"""
        context = self._build_context(box, all_boxes, profiles, scene)
        translated = self._translate_vlm(box.text, context, box.language.value, target_lang.value)
        return TranslationResult(
            textbox_id=box.id,
            original_text=box.text,
            translated_text=translated,
            source_lang=box.language,
            target_lang=target_lang,
            translation_method="vlm",
            character_id=box.character_id,
            confidence=0.9,
        )

    def _build_context(
        self,
        box: TextBox,
        all_boxes: list[TextBox],
        profiles: dict[str, CharacterProfile],
        scene: str,
    ) -> dict:
        """构建五维翻译上下文：场景、角色、历史、对话、视觉。"""
        profile = profiles.get(box.character_id or "", None)
        character_info = ""
        if profile:
            character_info = (
                f"角色性格: {profile.speech_style}, "
                f"自称: {', '.join(profile.pronouns) if profile.pronouns else '未知'}"
            )

        neighboring_texts = []
        for other in all_boxes:
            if other.id != box.id and other.character_id == box.character_id:
                neighboring_texts.append(other.text)

        return {
            "scene": scene,
            "character_id": box.character_id,
            "character_info": character_info,
            "neighboring_dialogue": neighboring_texts[-3:],
            "target_language": "zh",
            "instruction": "将以下日文漫画台词翻译成自然的中文口语，保持角色语气一致。",
        }

    def _translate_opus(self, text: str, source_lang: str, target_lang: str) -> str:
        """使用OPUS-MT MarianMT模型翻译。"""
        model_key = f"opus_mt_{source_lang}_{target_lang}"
        try:
            model_data = self._mgr.get_model("translation", model_key)
            tokenizer = model_data["tokenizer"]
            model = model_data["model"]
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_length=128)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result
        except Exception:
            return text

    def _translate_vlm(self, text: str, context: dict, source_lang: str, target_lang: str) -> str:
        """使用Qwen2.5-VL-2B进行带完整上下文的翻译。

        VLM不可用时自动降级为OPUS-MT，OPUS也失败则返回原文。
        """
        prompt = f"""{context.get('instruction', '')}

场景: {context.get('scene', '未知')}
{context.get('character_info', '')}

原文: {text}

翻译成{target_lang}:"""

        try:
            model_data = self._mgr.get_model("translation", "vlm")
            model = model_data["model"]
            processor = model_data["processor"]

            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text_prompt = processor.apply_chat_template(messages, tokenize=False)
            inputs = processor(text=[text_prompt], return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=256)
            result = processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )[0]
            return result.strip()
        except Exception:
            try:
                return self._translate_opus(text, source_lang, target_lang)
            except Exception:
                return text
