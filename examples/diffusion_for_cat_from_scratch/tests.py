import torch

from unet import(
    patchify,
    unpatchify,
    SinusoidalPositionalEmbedding,
    ViT,
    DownSampling,
    UpSampling,
    UNet,
)


@torch.no_grad()
def test_patchify_and_unpatchify():
    x = torch.rand(32, 3, 256, 256)
    x_patches = patchify(x, 64)
    assert x_patches.shape == (32, 16, 3, 64, 64)
    x_reconstructed = unpatchify(x_patches, 256)
    assert torch.allclose(x, x_reconstructed), "Reconstructed images are not equal to the original images"


@torch.no_grad()
def test_sinusoidal_positional_embedding():
    x = torch.rand(32, 16, 256)
    pos_emb = SinusoidalPositionalEmbedding()
    x = pos_emb(x)
    assert x.shape == (32, 16, 256), "SinusoidalPositionalEmbedding is not added correctly"


@torch.no_grad()
def test_vit():
    x = torch.rand(32, 3, 256, 256)
    vit = ViT()
    x = vit(x)
    assert x.shape == (32, 3, 256, 256), "ViT is not working correctly"


@torch.no_grad()
def test_down_sampling():
    x = torch.rand(32, 64, 128, 128)
    down_sampling = DownSampling(64, 128)
    x = down_sampling(x)
    assert x.shape == (32, 128, 64, 64), "DownSampling layer is not working correctly"


@torch.no_grad()
def test_up_sampling():
    x = torch.rand(32, 64, 128, 128)
    up_sampling = UpSampling(64, 128)
    x = up_sampling(x)
    assert x.shape == (32, 128, 256, 256), "UpSampling layer is not working correctly"


@torch.no_grad()
def test_u_net():
    x = torch.rand(1, 3, 256, 256).cuda()
    u_net = UNet(16, 64, 16).cuda()
    x = u_net(x)
    assert x.shape == (1, 3, 256, 256), "UNet is not working correctly"
