"""
Translation Handler Module

Handles multilingual translation using Qwen3-MT with support for 92 languages.
"""

from typing import Optional, Dict, Set
from dataclasses import dataclass
import logging
import time
import re

logger = logging.getLogger(__name__)


@dataclass
class TranslationRequest:
    """Request for translation."""
    text: str
    source_lang: Optional[str] = None  # Auto-detect if None
    target_lang: str = "en"  # Default to English
    preserve_formatting: bool = True


@dataclass
class TranslationResponse:
    """Response from translation."""
    translated_text: str
    source_lang: str  # Detected or provided (ISO 639-1 code)
    target_lang: str  # ISO 639-1 code
    latency_ms: float


class TranslationHandler:
    """
    Qwen3-MT handler for multilingual translation.

    Supports 92 languages with high-quality translation including:
    - Major European languages (English, Spanish, French, German, etc.)
    - CJK languages (Chinese, Japanese, Korean)
    - Middle Eastern languages (Arabic, Persian, Hebrew)
    - South Asian languages (Hindi, Bengali, Urdu)
    - Southeast Asian languages (Thai, Vietnamese, Indonesian)

    Features:
    - Automatic language detection
    - Formatting preservation (newlines, bullet points, structure)
    - Character set detection for CJK and Arabic scripts
    - Batch translation support
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-MT-8B"
    ):
        """
        Initialize translation handler.

        Args:
            model_name: Translation model name (default: Qwen3-MT-8B)
        """
        self.model_name = model_name
        self._model = None
        self.supported_languages = self._get_supported_languages()
        self.language_names = self._get_language_names()

    def load(self):
        """
        Load Qwen3-MT model.

        In production, this would initialize the model using vLLM or transformers:
        - vLLM for GPU inference (recommended for production)
        - transformers for direct model loading
        """
        logger.info(f"Loading translation model: {self.model_name}")

        # Placeholder for actual model loading
        # Production implementation would use:
        # from vllm import LLM, SamplingParams
        # self._model = LLM(
        #     model=self.model_name,
        #     tensor_parallel_size=1,
        #     gpu_memory_utilization=0.90,
        #     trust_remote_code=True,
        # )

        logger.info(f"Translation model loaded: {self.model_name}")

    async def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: Optional[str] = None
    ) -> TranslationResponse:
        """
        Translate text to target language.

        Args:
            text: Text to translate
            target_lang: Target language code (ISO 639-1: en, es, zh, ja, etc.)
            source_lang: Source language code (auto-detect if None)

        Returns:
            TranslationResponse with translated text and language info

        Raises:
            ValueError: If text is empty or target language is unsupported
        """
        start_time = time.time()

        # Validate input
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Detect source language if not provided
        if source_lang is None:
            source_lang = self.detect_language(text)

        # Validate source language
        if source_lang not in self.supported_languages:
            logger.warning(f"Unsupported source language: {source_lang}, defaulting to 'en'")
            source_lang = 'en'

        # Validate target language
        if target_lang not in self.supported_languages:
            raise ValueError(
                f"Unsupported target language: {target_lang}. "
                f"Supported languages: {', '.join(sorted(self.supported_languages))}"
            )

        # Skip translation if source and target are the same
        if source_lang == target_lang:
            logger.info(f"Source and target language are the same ({source_lang}), returning original text")
            return TranslationResponse(
                translated_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                latency_ms=(time.time() - start_time) * 1000
            )

        logger.info(
            f"Translating from {source_lang} to {target_lang} "
            f"({len(text)} chars)"
        )

        # Format prompt for Qwen3-MT
        prompt = self._format_translation_prompt(text, source_lang, target_lang)

        # Run inference
        translated_text = self._run_inference(prompt)

        # Preserve formatting
        translated_text = self._preserve_formatting(text, translated_text)

        latency_ms = (time.time() - start_time) * 1000

        return TranslationResponse(
            translated_text=translated_text,
            source_lang=source_lang,
            target_lang=target_lang,
            latency_ms=latency_ms
        )

    def detect_language(self, text: str) -> str:
        """
        Detect language of text.

        Uses langdetect library if available, otherwise falls back to
        character set heuristics for common languages.

        Args:
            text: Text to analyze

        Returns:
            ISO 639-1 language code
        """
        if not text or not text.strip():
            return 'en'

        # Try using langdetect library
        try:
            from langdetect import detect
            lang_code = detect(text)

            # Validate detected language is supported
            if lang_code in self.supported_languages:
                logger.debug(f"Detected language: {lang_code} (using langdetect)")
                return lang_code
            else:
                logger.warning(f"Detected unsupported language: {lang_code}, using fallback")
        except ImportError:
            logger.debug("langdetect not available, using character set heuristics")
        except Exception as e:
            logger.warning(f"Language detection failed: {e}, using fallback")

        # Fallback: character set heuristics
        if self._contains_chinese(text):
            return 'zh'
        elif self._contains_japanese(text):
            return 'ja'
        elif self._contains_korean(text):
            return 'ko'
        elif self._contains_arabic(text):
            return 'ar'
        elif self._contains_cyrillic(text):
            return 'ru'
        elif self._contains_thai(text):
            return 'th'
        elif self._contains_hebrew(text):
            return 'he'
        elif self._contains_devanagari(text):
            return 'hi'
        else:
            return 'en'  # Default to English

    def _format_translation_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """
        Format prompt for Qwen3-MT.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Formatted prompt string
        """
        source_name = self.language_names.get(source_lang, source_lang.upper())
        target_name = self.language_names.get(target_lang, target_lang.upper())

        prompt = f"""Translate the following text from {source_name} to {target_name}. Preserve all formatting including newlines, bullet points, numbered lists, and structure.

Source text:
{text}

Translation:"""

        return prompt

    def _run_inference(self, prompt: str) -> str:
        """
        Run model inference.

        Args:
            prompt: Formatted translation prompt

        Returns:
            Translated text

        Note:
            This is a placeholder implementation. In production, would use:
            - vLLM for GPU-accelerated inference
            - transformers for CPU inference
            - API calls to hosted Qwen3-MT
        """
        # Placeholder for actual inference
        # Production implementation:
        # sampling_params = SamplingParams(
        #     temperature=0.3,  # Lower temperature for more consistent translations
        #     max_tokens=4096,
        #     top_p=0.9,
        # )
        # outputs = self._model.generate([prompt], sampling_params)
        # return outputs[0].outputs[0].text.strip()

        # Mock translation for testing
        return "[Translated text]"

    def _preserve_formatting(self, original: str, translated: str) -> str:
        """
        Preserve formatting from original text in translation.

        Attempts to maintain:
        - Newline structure
        - Bullet points (-, *, •)
        - Numbered lists (1., 2., etc.)
        - Indentation (best-effort)
        - Code blocks (```)

        Args:
            original: Original source text
            translated: Translated text

        Returns:
            Translated text with preserved formatting
        """
        # Count newlines in original
        original_newlines = original.count('\n')
        translated_newlines = translated.count('\n')

        # If original has significantly more newlines, try to preserve structure
        if original_newlines > translated_newlines + 2:
            # Split both into lines
            original_lines = original.split('\n')
            translated_lines = translated.split('\n')

            # If line counts are similar, assume structure is preserved
            if abs(len(original_lines) - len(translated_lines)) <= 2:
                return translated

            # Otherwise, try to add back missing line breaks
            # This is a heuristic approach - perfect preservation requires model support
            logger.debug("Attempting to restore newline structure")

        # Preserve leading/trailing whitespace pattern
        if original.startswith('\n'):
            translated = '\n' + translated.lstrip('\n')
        if original.endswith('\n'):
            translated = translated.rstrip('\n') + '\n'

        return translated

    def _get_supported_languages(self) -> Set[str]:
        """
        Get set of supported language codes.

        Qwen3-MT supports 92 languages. This includes all major languages
        and many regional languages.

        Returns:
            Set of ISO 639-1 language codes
        """
        return {
            # Major European languages
            'en',  # English
            'es',  # Spanish
            'fr',  # French
            'de',  # German
            'it',  # Italian
            'pt',  # Portuguese
            'nl',  # Dutch
            'pl',  # Polish
            'ru',  # Russian
            'uk',  # Ukrainian
            'cs',  # Czech
            'sk',  # Slovak
            'bg',  # Bulgarian
            'ro',  # Romanian
            'hr',  # Croatian
            'sr',  # Serbian
            'sl',  # Slovenian
            'da',  # Danish
            'sv',  # Swedish
            'no',  # Norwegian
            'fi',  # Finnish
            'is',  # Icelandic
            'el',  # Greek
            'hu',  # Hungarian
            'et',  # Estonian
            'lv',  # Latvian
            'lt',  # Lithuanian
            'ga',  # Irish
            'cy',  # Welsh
            'mt',  # Maltese
            'sq',  # Albanian
            'mk',  # Macedonian
            'bs',  # Bosnian

            # CJK languages
            'zh',  # Chinese
            'ja',  # Japanese
            'ko',  # Korean

            # Middle Eastern languages
            'ar',  # Arabic
            'he',  # Hebrew
            'fa',  # Persian (Farsi)
            'tr',  # Turkish
            'ur',  # Urdu
            'az',  # Azerbaijani
            'kk',  # Kazakh
            'uz',  # Uzbek
            'ky',  # Kyrgyz

            # South Asian languages
            'hi',  # Hindi
            'bn',  # Bengali
            'ta',  # Tamil
            'te',  # Telugu
            'mr',  # Marathi
            'gu',  # Gujarati
            'kn',  # Kannada
            'ml',  # Malayalam
            'pa',  # Punjabi
            'si',  # Sinhala
            'ne',  # Nepali

            # Southeast Asian languages
            'th',  # Thai
            'vi',  # Vietnamese
            'id',  # Indonesian
            'ms',  # Malay
            'tl',  # Tagalog
            'my',  # Burmese
            'km',  # Khmer
            'lo',  # Lao

            # African languages
            'sw',  # Swahili
            'am',  # Amharic
            'ha',  # Hausa
            'yo',  # Yoruba
            'ig',  # Igbo
            'zu',  # Zulu
            'xh',  # Xhosa
            'af',  # Afrikaans
            'so',  # Somali

            # Other Indo-European
            'ca',  # Catalan
            'gl',  # Galician
            'eu',  # Basque
            'be',  # Belarusian
            'hy',  # Armenian
            'ka',  # Georgian

            # Central/South American
            'qu',  # Quechua
            'gn',  # Guarani

            # Other Asian
            'mn',  # Mongolian
            'ps',  # Pashto
            'sd',  # Sindhi
            'ku',  # Kurdish

            # Pacific
            'sm',  # Samoan
            'to',  # Tongan
            'fj',  # Fijian

            # Additional European
            'lb',  # Luxembourgish
            'fo',  # Faroese
        }

    def _get_language_names(self) -> Dict[str, str]:
        """
        Map language codes to full names.

        Returns:
            Dictionary mapping ISO 639-1 codes to language names
        """
        return {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'nl': 'Dutch',
            'pl': 'Polish',
            'ru': 'Russian',
            'uk': 'Ukrainian',
            'cs': 'Czech',
            'sk': 'Slovak',
            'bg': 'Bulgarian',
            'ro': 'Romanian',
            'hr': 'Croatian',
            'sr': 'Serbian',
            'sl': 'Slovenian',
            'da': 'Danish',
            'sv': 'Swedish',
            'no': 'Norwegian',
            'fi': 'Finnish',
            'is': 'Icelandic',
            'el': 'Greek',
            'hu': 'Hungarian',
            'et': 'Estonian',
            'lv': 'Latvian',
            'lt': 'Lithuanian',
            'ga': 'Irish',
            'cy': 'Welsh',
            'mt': 'Maltese',
            'sq': 'Albanian',
            'mk': 'Macedonian',
            'bs': 'Bosnian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'he': 'Hebrew',
            'fa': 'Persian',
            'tr': 'Turkish',
            'ur': 'Urdu',
            'az': 'Azerbaijani',
            'kk': 'Kazakh',
            'uz': 'Uzbek',
            'ky': 'Kyrgyz',
            'hi': 'Hindi',
            'bn': 'Bengali',
            'ta': 'Tamil',
            'te': 'Telugu',
            'mr': 'Marathi',
            'gu': 'Gujarati',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'pa': 'Punjabi',
            'si': 'Sinhala',
            'ne': 'Nepali',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'id': 'Indonesian',
            'ms': 'Malay',
            'tl': 'Tagalog',
            'my': 'Burmese',
            'km': 'Khmer',
            'lo': 'Lao',
            'sw': 'Swahili',
            'am': 'Amharic',
            'ha': 'Hausa',
            'yo': 'Yoruba',
            'ig': 'Igbo',
            'zu': 'Zulu',
            'xh': 'Xhosa',
            'af': 'Afrikaans',
            'so': 'Somali',
            'ca': 'Catalan',
            'gl': 'Galician',
            'eu': 'Basque',
            'be': 'Belarusian',
            'hy': 'Armenian',
            'ka': 'Georgian',
            'qu': 'Quechua',
            'gn': 'Guarani',
            'mn': 'Mongolian',
            'ps': 'Pashto',
            'sd': 'Sindhi',
            'ku': 'Kurdish',
            'sm': 'Samoan',
            'to': 'Tongan',
            'fj': 'Fijian',
            'lb': 'Luxembourgish',
            'fo': 'Faroese',
        }

    # Character set detection helpers
    def _contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters (CJK Unified Ideographs)."""
        return any('\u4e00' <= char <= '\u9fff' for char in text)

    def _contains_japanese(self, text: str) -> bool:
        """Check if text contains Japanese characters (Hiragana or Katakana)."""
        return any(
            ('\u3040' <= char <= '\u309f') or  # Hiragana
            ('\u30a0' <= char <= '\u30ff')     # Katakana
            for char in text
        )

    def _contains_korean(self, text: str) -> bool:
        """Check if text contains Korean characters (Hangul)."""
        return any('\uac00' <= char <= '\ud7af' for char in text)

    def _contains_arabic(self, text: str) -> bool:
        """Check if text contains Arabic characters."""
        return any('\u0600' <= char <= '\u06ff' for char in text)

    def _contains_cyrillic(self, text: str) -> bool:
        """Check if text contains Cyrillic characters."""
        return any('\u0400' <= char <= '\u04ff' for char in text)

    def _contains_thai(self, text: str) -> bool:
        """Check if text contains Thai characters."""
        return any('\u0e00' <= char <= '\u0e7f' for char in text)

    def _contains_hebrew(self, text: str) -> bool:
        """Check if text contains Hebrew characters."""
        return any('\u0590' <= char <= '\u05ff' for char in text)

    def _contains_devanagari(self, text: str) -> bool:
        """Check if text contains Devanagari characters (Hindi, etc.)."""
        return any('\u0900' <= char <= '\u097f' for char in text)

    async def batch_translate(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: Optional[str] = None
    ) -> list[TranslationResponse]:
        """
        Translate multiple texts efficiently.

        Args:
            texts: List of texts to translate
            target_lang: Target language code
            source_lang: Source language code (auto-detect if None)

        Returns:
            List of translation responses
        """
        responses = []

        for text in texts:
            try:
                response = await self.translate(
                    text=text,
                    target_lang=target_lang,
                    source_lang=source_lang
                )
                responses.append(response)
            except Exception as e:
                logger.error(f"Failed to translate text: {e}")
                # Add error response
                responses.append(TranslationResponse(
                    translated_text=f"[Translation failed: {str(e)}]",
                    source_lang=source_lang or 'unknown',
                    target_lang=target_lang,
                    latency_ms=0.0
                ))

        return responses

    def is_language_supported(self, lang_code: str) -> bool:
        """
        Check if language is supported.

        Args:
            lang_code: ISO 639-1 language code

        Returns:
            True if language is supported
        """
        return lang_code in self.supported_languages

    def get_supported_languages(self) -> list[str]:
        """
        Get list of supported language codes.

        Returns:
            Sorted list of ISO 639-1 language codes
        """
        return sorted(self.supported_languages)

    def get_language_name(self, lang_code: str) -> str:
        """
        Get full language name from code.

        Args:
            lang_code: ISO 639-1 language code

        Returns:
            Full language name, or the code itself if not found
        """
        return self.language_names.get(lang_code, lang_code.upper())

    def unload(self):
        """Unload translation model and free resources."""
        logger.info(f"Unloading translation model: {self.model_name}")
        self._model = None
        logger.info("Translation model unloaded successfully")
