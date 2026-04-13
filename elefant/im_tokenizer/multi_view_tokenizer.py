import torch

from elefant.im_tokenizer.base_tokenizer import ImageBaseTokenizer
from elefant.torch import eager_assert


class MultiViewTokenConcatTokenizer(ImageBaseTokenizer):
    def __init__(self, base_tokenizer: ImageBaseTokenizer, num_views: int):
        super().__init__(base_tokenizer.config)
        if num_views <= 0:
            raise ValueError(f"num_views must be positive, got {num_views}.")
        self.base_tokenizer = base_tokenizer
        self.num_views = int(num_views)

    def get_n_img_tokens(self) -> int:
        return self.base_tokenizer.get_n_img_tokens() * self.num_views

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.ndim == 5:
            if self.num_views != 1:
                raise ValueError(
                    f"Expected 6D multi-view input with {self.num_views} views, got shape {tuple(input_tensor.shape)}."
                )
            return self.base_tokenizer(input_tensor)

        if input_tensor.ndim != 6:
            raise ValueError(
                f"MultiViewTokenConcatTokenizer expects 6D input, got shape {tuple(input_tensor.shape)}."
            )

        B, T, V, C, H, W = input_tensor.shape
        if V != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} camera views, got {V} for shape {tuple(input_tensor.shape)}."
            )

        tokenizer_in = input_tensor.reshape(B, T * V, C, H, W)
        tokens = self.base_tokenizer(tokenizer_in)
        per_view_tokens = self.base_tokenizer.get_n_img_tokens()
        eager_assert(
            tokens.shape,
            (B, T * V, per_view_tokens, tokens.shape[-1]),
        )
        return tokens.reshape(B, T, V * per_view_tokens, tokens.shape[-1])
