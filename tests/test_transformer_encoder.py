import torch
from hurricore.models import TransformerEncoder


def test_transformer_encoder():
    model = TransformerEncoder(
        num_layers=3,
        embedding_dim=512,
    ).eval()
    x = torch.rand(2, 16, 512)
    mask = torch.ones(2, 16)
    mask[0, -3:] = 0
    mask = mask.unsqueeze(1).expand(-1, 16, -1)
    output_a = model(x, mask)
    x[0, -3:] = 0
    output_b = model(x, mask)
    assert torch.allclose(output_a[0, :-3], output_b[0, :-3])
    assert not torch.allclose(output_a[0, -4:], output_b[0, -4:])

    # Test RoPE
    x = torch.rand(2, 16, 512)
    mask = torch.ones(2, 16)
    output_a = model(x, mask.unsqueeze(1).expand(-1, 16, -1))
    x = torch.cat(
        [
            torch.zeros(2, 1, 512),
            x
        ],
        dim=1,
    )
    mask = torch.cat(
        [
            torch.zeros(2, 1),
            mask
        ],
        dim=1,
    )
    output_b = model(x, mask.unsqueeze(1).expand(-1, 17, -1))
    assert torch.allclose(output_a, output_b[:, 1:, :], atol=1e-6)


if __name__ == "__main__":
    test_transformer_encoder()