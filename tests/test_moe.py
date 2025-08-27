import shutil
from pathlib import Path
from dataclasses import dataclass

import torch
import onnxruntime

from hurricore.models import MoE


@dataclass
class Config:
    batch_size: int = 16
    num_tokens: int = 100
    emb_dim: int = 128
    expert_dim: int = 64


def test_moe_output_shape():
    config = Config()
    model = MoE(input_dim=config.emb_dim, expert_dim=config.expert_dim)
    input_data = torch.randn(config.batch_size, config.num_tokens, config.emb_dim)
    output = model(input_data)
    assert output.shape == (config.batch_size, config.num_tokens, config.emb_dim)


def test_moe_auto_load_balancing():
    config = Config()
    model = MoE(input_dim=config.emb_dim, expert_dim=config.expert_dim).eval()
    assert torch.allclose(model.gate.routing_bias, torch.zeros_like(model.gate.routing_bias))
    input_data = torch.randn(config.batch_size, config.num_tokens, config.emb_dim)
    _ = model(input_data)
    assert torch.allclose(model.gate.routing_bias, torch.zeros_like(model.gate.routing_bias))
    model.train()
    input_data = torch.randn(config.batch_size, config.num_tokens, config.emb_dim)
    _ = model(input_data)
    assert not torch.allclose(model.gate.routing_bias, torch.zeros_like(model.gate.routing_bias))


def test_moe_exporting():
    config = Config()
    model = MoE(input_dim=config.emb_dim, expert_dim=config.expert_dim).eval()
    model = torch.jit.script(model)
    dummy_input = torch.randn(config.batch_size, config.num_tokens, config.emb_dim)

    # Export the model to ONNX
    onnx_path = Path(__file__).parent / "moe.onnx"
    if onnx_path.exists():
        onnx_path.unlink()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'num_tokens'},
            'output': {0: 'batch_size', 1: 'num_tokens'}
        },
    )
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {'input': dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_output = torch.from_numpy(ort_outputs[0])

    # Get the PyTorch model output
    with torch.no_grad():
        pytorch_output = model(dummy_input)

    # Compare the outputs
    tolerance = 1e-6
    assert torch.allclose(pytorch_output, onnx_output, atol=tolerance, rtol=tolerance)
    another_input = torch.randn(config.batch_size + 1, config.num_tokens - 1, config.emb_dim)
    ort_inputs_another = {'input': another_input.numpy()}
    ort_outputs_another = ort_session.run(None, ort_inputs_another)
    onnx_output_another = torch.from_numpy(ort_outputs_another[0])

    with torch.no_grad():
        pytorch_output_another = model(another_input)
    assert torch.allclose(pytorch_output_another, onnx_output_another, atol=tolerance, rtol=tolerance)
    
    onnx_path.unlink()
    

if __name__ == "__main__":
    test_moe_output_shape()
    test_moe_auto_load_balancing()
    test_moe_exporting()
