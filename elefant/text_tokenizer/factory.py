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
_LOCAL_GEMMA_MODEL_ROOT = Path("checkpoints/embeddinggemma-300M")
_GEMMA_TOKENIZER_NAMES = {
    "gemma",
    "ggemma",
    "gemma-300m",
    "ggemma-300m",
    "embeddinggemma",
    "embeddinggemma-300m",
}
_LINGBOT_TOKENIZER_NAMES = {"lingbot", "umt5"}


def _resolve_local_or_remote_model_name_or_path(
    configured_model_name_or_path: str | None,
    default_model_name_or_path: str,
) -> str:
    if configured_model_name_or_path:
        return configured_model_name_or_path
    if _LOCAL_GEMMA_MODEL_ROOT.is_dir():
        return str(_LOCAL_GEMMA_MODEL_ROOT)
    return default_model_name_or_path



def _build_gemma_config(config: TextTokenizerConfig) -> GemmaTextTokenizerConfig:
    default_cfg = GemmaTextTokenizerConfig()
    return GemmaTextTokenizerConfig(
        model_id=_resolve_local_or_remote_model_name_or_path(
            config.model_name_or_path,
            default_cfg.model_id,
        ),
        max_position_embeddings=(
            config.max_position_embeddings
            if config.max_position_embeddings is not None
            else default_cfg.max_position_embeddings
        ),
    )



def _build_lingbot_config(config: TextTokenizerConfig) -> LingbotTextTokenizerConfig:
    default_cfg = LingbotTextTokenizerConfig()
    return LingbotTextTokenizerConfig(
        model_root=config.model_name_or_path or default_cfg.model_root,
        max_position_embeddings=(
            config.max_position_embeddings
            if config.max_position_embeddings is not None
            else default_cfg.max_position_embeddings
        ),
    )



def _get_lingbot_text_tokenizer(config: TextTokenizerConfig) -> LingbotTextTokenizer:
    lingbot_config = _build_lingbot_config(config)
    model_root = Path(lingbot_config.model_root)
    if not model_root.is_dir():
        raise FileNotFoundError(
            f"LingBot text tokenizer root not found: {model_root}"
        )
    return LingbotTextTokenizer(lingbot_config)



def get_text_tokenizer(config: TextTokenizerConfig | None) -> TextBaseTokenizer | None:
    if not config:
        return None

    tokenizer_name = (config.text_tokenizer_name or "").strip().lower()
    if not tokenizer_name:
        return None

    if tokenizer_name in _GEMMA_TOKENIZER_NAMES:
        gemma_config = _build_gemma_config(config)
        LOGGER.info(
            "Initializing Gemma text tokenizer from %s (max_position_embeddings=%s)",
            gemma_config.model_id,
            gemma_config.max_position_embeddings,
        )
        return GemmaTextTokenizer(gemma_config)

    if tokenizer_name in _LINGBOT_TOKENIZER_NAMES:
        lingbot_config = _build_lingbot_config(config)
        LOGGER.info(
            "Initializing LingBot/UMT5 text tokenizer from %s (max_position_embeddings=%s)",
            lingbot_config.model_root,
            lingbot_config.max_position_embeddings,
        )
        return _get_lingbot_text_tokenizer(config)

    if tokenizer_name == "dummy":
        return DummyTextTokenizer(DummyTextTokenizerConfig())

    raise ValueError(f"Unknown text tokenizer: {config.text_tokenizer_name}")
