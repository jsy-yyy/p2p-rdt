import logging
from pathlib import Path

from elefant.text_tokenizer.text_tokenizer import (
    GemmaTextTokenizer,
    DummyTextTokenizer,
    LingbotTextTokenizer,
)
from elefant.text_tokenizer.config import (
    TextTokenizerConfig,
    DummyTextTokenizerConfig,
    GemmaTextTokenizerConfig,
    LingbotTextTokenizerConfig,
)
from elefant.text_tokenizer.base_text_tokenizer import TextBaseTokenizer


LOGGER = logging.getLogger(__name__)


def _get_lingbot_text_tokenizer() -> LingbotTextTokenizer:
    config = LingbotTextTokenizerConfig()
    model_root = Path(config.model_root)
    if not model_root.is_dir():
        raise FileNotFoundError(
            f"LingBot text tokenizer root not found: {model_root}"
        )
    return LingbotTextTokenizer(config)


def get_text_tokenizer(config: TextTokenizerConfig | None) -> TextBaseTokenizer | None:
    if not config:
        return None
    elif config.text_tokenizer_name == "gemma":
        try:
            return GemmaTextTokenizer(GemmaTextTokenizerConfig())
        except Exception as exc:
            LOGGER.warning(
                "Failed to initialize Gemma text tokenizer; falling back to LingBot UMT5 text tokenizer. Error: %s",
                exc,
            )
            return _get_lingbot_text_tokenizer()
    elif config.text_tokenizer_name in {"lingbot", "umt5"}:
        return _get_lingbot_text_tokenizer()
    elif config.text_tokenizer_name == "dummy":
        return DummyTextTokenizer(DummyTextTokenizerConfig())
    else:
        raise ValueError(f"Unknown text tokenizer: {config.text_tokenizer_name}")
