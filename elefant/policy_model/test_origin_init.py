import pytest
import torch

from elefant.policy_model.stage3_finetune import _adapt_origin_tensor_to_target


def test_rejects_generic_shape_adaptation_for_non_whitelisted_weights():
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    target = torch.zeros((3, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="Cannot adapt origin checkpoint tensor"):
        _adapt_origin_tensor_to_target(
            source_key="robot_action_in_proj.weight",
            source_tensor=source,
            target_key="robot_action_in_proj.weight",
            target_tensor=target,
        )


def test_allows_whitelisted_img_pos_token_repeat():
    source = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    target = torch.zeros((1, 4, 4), dtype=torch.float32)

    adapted, mode = _adapt_origin_tensor_to_target(
        source_key="bc_transformer.image_tokenizer.img_pos_tokens",
        source_tensor=source,
        target_key="bc_transformer.image_tokenizer.base_tokenizer.img_pos_tokens",
        target_tensor=target,
    )

    assert mode == "repeat_img_tokens_x2"
    assert adapted.shape == target.shape
    assert torch.equal(adapted[:, :2], source)
    assert torch.equal(adapted[:, 2:], source)
