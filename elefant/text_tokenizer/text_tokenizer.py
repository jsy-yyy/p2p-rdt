from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from transformers import T5TokenizerFast, UMT5EncoderModel

from elefant.text_tokenizer.base_text_tokenizer import TextBaseTokenizer
from elefant.text_tokenizer.config import (
    GemmaTextTokenizerConfig,
    DummyTextTokenizerConfig,
    LingbotTextTokenizerConfig,
)

try:
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
except Exception:
    prompt_clean = None



def _mean_pool_last_hidden_state(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    summed = torch.sum(hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class DummyTextTokenizer(TextBaseTokenizer):
    # this is a dummy text tokenizer used to pass text
    def __init__(self, config: DummyTextTokenizerConfig):
        super().__init__(config)
        self.embed_dim = 2048
        self.n_text_tokens = 1

    def tokenize(self, text: str) -> dict[str, torch.Tensor]:
        input_ids = torch.ones(1, self.embed_dim)
        attention_mask = torch.ones(1, self.embed_dim)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if input_ids.ndim == 3:
            batch_size, n_steps, _ = input_ids.shape
        elif input_ids.ndim == 2:
            batch_size, _ = input_ids.shape
            n_steps = 1
        else:
            raise ValueError(f"Invalid input_ids shape: {input_ids.shape}")
        return torch.ones(
            batch_size,
            n_steps,
            self.n_text_tokens,
            self.embed_dim,
            device=input_ids.device,
        )

    def get_n_text_tokens(self) -> int:
        return self.n_text_tokens

    def get_text_embed_dim(self) -> int:
        return self.embed_dim


class GemmaTextTokenizer(TextBaseTokenizer):
    def __init__(self, config: GemmaTextTokenizerConfig):
        super().__init__(config)
        self.gemma_model = SentenceTransformer(config.model_id, device="cpu").eval()
        self.tokenizer = self.gemma_model.tokenizer
        self.gemma_embedding = self.gemma_model[0].auto_model.eval()
        self.embed_dim = self.gemma_model.get_sentence_embedding_dimension()
        del self.gemma_model
        self.n_text_tokens = 1
        self.max_position_embeddings = config.max_position_embeddings

    def tokenize(self, text: str) -> dict[str, torch.Tensor]:
        texts = [text] if isinstance(text, str) else list(text)
        return self.tokenizer(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_position_embeddings,
        )

    @torch.compiler.disable
    @torch.inference_mode()
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if input_ids.ndim == 3:
            batch_size, n_steps, text_dim = input_ids.shape
        elif input_ids.ndim == 2:
            batch_size, text_dim = input_ids.shape
            n_steps = 1
        else:
            raise ValueError(f"Invalid input_ids shape: {input_ids.shape}")

        encoder_device = next(self.gemma_embedding.parameters()).device
        input_ids = input_ids.reshape(-1, text_dim).to(encoder_device)
        attention_mask = attention_mask.reshape(-1, text_dim).to(encoder_device)
        text_features = self.gemma_embedding(
            input_ids, attention_mask
        ).last_hidden_state
        sentence_embedding = _mean_pool_last_hidden_state(text_features, attention_mask)
        sentence_embedding = sentence_embedding.to(dtype=torch.float32)

        return sentence_embedding.reshape(
            batch_size,
            n_steps,
            self.n_text_tokens,
            self.embed_dim,
        )

    def get_n_text_tokens(self) -> int:
        return self.n_text_tokens

    def get_text_embed_dim(self) -> int:
        return self.embed_dim


class LingbotTextTokenizer(TextBaseTokenizer):
    """Use the same local UMT5 text encoder assets as lingbot-va."""

    def __init__(self, config: LingbotTextTokenizerConfig):
        super().__init__(config)
        model_root = Path(config.model_root)
        tokenizer_path = model_root / "tokenizer"
        text_encoder_path = model_root / "text_encoder"
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"LingBot tokenizer path not found: {tokenizer_path}")
        if not text_encoder_path.is_dir():
            raise FileNotFoundError(
                f"LingBot text encoder path not found: {text_encoder_path}"
            )

        self.tokenizer = T5TokenizerFast.from_pretrained(str(tokenizer_path))
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            str(text_encoder_path),
            torch_dtype=torch.bfloat16,
        ).eval()
        self.embed_dim = int(self.text_encoder.config.d_model)
        self.n_text_tokens = 1
        self.max_position_embeddings = config.max_position_embeddings

    def _clean_prompts(self, texts: list[str]) -> list[str]:
        if prompt_clean is None:
            return texts
        return [prompt_clean(text) for text in texts]

    def tokenize(self, text: str) -> dict[str, torch.Tensor]:
        texts = [text] if isinstance(text, str) else list(text)
        texts = self._clean_prompts(texts)
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_position_embeddings,
        )

    @torch.compiler.disable
    @torch.inference_mode()
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if input_ids.ndim == 3:
            batch_size, n_steps, text_dim = input_ids.shape
        elif input_ids.ndim == 2:
            batch_size, text_dim = input_ids.shape
            n_steps = 1
        else:
            raise ValueError(f"Invalid input_ids shape: {input_ids.shape}")

        encoder_device = next(self.text_encoder.parameters()).device
        input_ids = input_ids.reshape(-1, text_dim).to(encoder_device)
        attention_mask = attention_mask.reshape(-1, text_dim).to(encoder_device)
        hidden_state = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        sentence_embedding = _mean_pool_last_hidden_state(hidden_state, attention_mask)
        sentence_embedding = sentence_embedding.to(dtype=torch.float32)
        return sentence_embedding.reshape(
            batch_size,
            n_steps,
            self.n_text_tokens,
            self.embed_dim,
        )

    def get_n_text_tokens(self) -> int:
        return self.n_text_tokens

    def get_text_embed_dim(self) -> int:
        return self.embed_dim
