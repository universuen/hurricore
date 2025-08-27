import torch
from hurricore.models import TransformerDecoder


def test_transformer_encoder():
    model = TransformerDecoder(
        num_layers=3,
        embedding_dim=512,
    ).eval()
    memory = torch.rand(2, 16, 512)
    query = torch.rand(2, 4, 512)
    cross_attn_mask = torch.ones(2, 4, 16, dtype=torch.int)
    cross_attn_mask[0, :, 8:] = 0
    output_a = model(query, memory, cross_attn_mask=cross_attn_mask)
    memory[0, 8:] = 0
    output_b = model(query, memory, cross_attn_mask=cross_attn_mask)
    assert torch.allclose(output_a, output_b)


if __name__ == "__main__":
    test_transformer_encoder()
