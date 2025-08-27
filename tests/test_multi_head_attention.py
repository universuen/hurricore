import torch
from hurricore.models import MultiHeadAttention


def test_multi_head_attention_without_rope():
    model = MultiHeadAttention(128, 8, use_rope=False)
    model.eval()

    # Original inputs
    q = torch.randn(8, 16, 128)
    k = torch.randn(8, 32, 128)
    v = torch.randn(8, 32, 128)
    mask = torch.ones(8, 16, 32, dtype=torch.int)
    mask[:, :, -1] = 0  # Every q cannot see the final kv in every batch

    # Test 1: Shape of the result
    result = model(q, k, v, mask)
    assert result.shape == (8, 16, 128), f"Unexpected result shape: {result.shape}"

    # Test 2: Insensitivity to key-value changes
    k[:, -1] = torch.ones(128)
    v[:, -1] = torch.ones(128)
    new_result = model(q, k, v, mask)
    assert torch.allclose(result, new_result), "Results differ after key-value change!"

    # Test 3: Order insensitivity
    permutation = torch.randperm(k.shape[1])  # Random permutation of sequence length
    k_shuffled = k[:, permutation]
    v_shuffled = v[:, permutation]
    mask_shuffled = mask.clone()
    mask_shuffled = mask_shuffled[:, :, permutation]  # Permute mask accordingly

    shuffled_result = model(q, k_shuffled, v_shuffled, mask_shuffled)

    assert torch.allclose(result.mean(), shuffled_result.mean()), "Results differ after sequence reordering!"


if __name__ == "__main__":
    test_multi_head_attention_without_rope()
