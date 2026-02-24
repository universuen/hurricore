<div align="center">

# 🌀 Hurricore

**A Universal Deep Learning Training Framework for Any Model, Any Task**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples)

</div>

---

## 📖 Overview

**Hurricore** is a universal deep learning training framework built on top of PyTorch and Hugging Face Accelerate. At its core, the abstract `HurricoreTrainer` class provides a highly flexible foundation that can handle **almost any type of model** in **any training paradigm** - from computer vision to NLP, from supervised learning to reinforcement learning, from classification to generation tasks.

While the framework is general-purpose and model-agnostic, we provide comprehensive implementations and examples focused on **Large Language Models (LLMs)** for both **Supervised Fine-Tuning (SFT)** and **Reinforcement Learning (RL)**, as these represent some of the most demanding and popular use cases in modern deep learning.

### Why Hurricore?

- 🌐 **Truly Universal**: The abstract `HurricoreTrainer` core can handle any model architecture and training paradigm
- 🚀 **Easy to Use**: Simple, intuitive API that gets you started in minutes
- 🎯 **Production Ready**: Battle-tested components for real-world applications
- 🔧 **Highly Flexible**: Modular design allows easy customization and extension
- ⚡ **Performance Optimized**: Built-in support for DeepSpeed, mixed precision, and distributed training
- 🧠 **RL Support**: Advanced reinforcement learning algorithms (GRPO, DAPO, GSPO) for LLM alignment
- 🎨 **Multi-Modal**: Native support for both text-only and vision-language models
- 📦 **Rich Examples**: Comprehensive LLM SFT/RL examples to get you started quickly

---

## ✨ Features

### Core Architecture

- **🏗️ Universal Training Framework**
  - Abstract `HurricoreTrainer` base class for maximum flexibility
  - Supports any PyTorch model architecture
  - Works with any training paradigm (supervised, unsupervised, RL, etc.)
  - Extensible hook system for custom training logic
  - Model-agnostic design - bring your own model!

### Featured Implementations (LLM Focus)

- **🤖 LLM Training**
  - Supervised Fine-Tuning (SFT)
  - Reinforcement Learning (GRPO, DAPO, GSPO)
  - Custom reward functions
  - vLLM integration for fast inference

- **👁️ Vision-Language Models**
  - Multi-modal training support
  - Image-text alignment
  - VLM-specific collators and processors

- **📊 Advanced Training Features**
  - DeepSpeed integration (ZeRO stages)
  - Mixed precision training (fp16/bf16)
  - Gradient accumulation
  - Distributed training with Accelerate
  - Custom learning rate schedulers

- **🎯 Classification Tasks**
  - Multi-class classification
  - Multi-label classification
  - Focal loss and logits adjustment
  - Balanced and imbalanced dataset handling

- **🔌 Extensible Components**
  - Custom datasets (LMDB, balanced, filtered, packed)
  - Flexible collators for different tasks
  - Hook system for training customization
  - Built-in evaluation metrics

- **📈 Monitoring & Logging**
  - TensorBoard integration
  - Comprehensive logging system
  - Checkpoint management
  - Training history tracking

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- PyTorch 2.0+

### Quick Install

```bash
# Clone the repository
git clone https://github.com/universuen/hurricore.git
cd hurricore

# Install the package
bash install.sh

# Or install with project dependencies
bash install.sh your_project_name
```

### Manual Installation

```bash
# Install core dependencies
pip install -e .

# Install with optional project dependencies
pip install -e .[projects]
```

### Dependencies

Core dependencies include:
- `torch` - PyTorch deep learning framework
- `accelerate` - Distributed training made easy
- `transformers` - Hugging Face transformers
- `deepspeed` - Training optimization
- `vllm` - Fast LLM inference
- `tensorboard` - Training visualization
- And more (see `pyproject.toml`)

---

## 🎯 Quick Start

> **Note**: While Hurricore can handle any model type, the following examples focus on LLM training (SFT and RL) as they demonstrate the framework's capabilities in demanding scenarios. You can easily adapt these patterns to your own models and tasks by extending the `HurricoreTrainer` base class.

### Example 1: Supervised Fine-Tuning (SFT) for LLM

```python
from pathlib import Path
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset

from hurricore.trainers import LLMTrainer
from hurricore.collators import LLMSFTCollator
from hurricore.utils import split_dataset


class ConversationDataset(Dataset):
    def __getitem__(self, idx: int) -> list[dict]:
        return [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    
    def __len__(self):
        return 1000


# Setup accelerator with DeepSpeed
accelerator = Accelerator(
    gradient_accumulation_steps=4,
    mixed_precision="bf16",
    deepspeed_plugin=DeepSpeedPlugin(zero_stage=1)
)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

# Prepare datasets
dataset = ConversationDataset()
train_ds, val_ds = split_dataset(dataset, training_ratio=0.9)

# Create trainer
trainer = LLMTrainer(
    experiment_name="my_sft_experiment",
    data_folder_path=Path("./data"),
    num_epochs=3,
    per_device_batch_size=2,
    learning_rate=1e-5,
    model=model,
    training_dataset=train_ds,
    validation_dataset=val_ds,
    accelerator=accelerator,
    collate_fn=LLMSFTCollator(tokenizer=tokenizer),
)

# Start training
trainer.run()
```

### Example 2: Reinforcement Learning with GRPO

```python
import re
from hurricore.trainers import LLMGRPOTrainer
from hurricore.collators import LLMRLCollator


class MathRLTrainer(LLMGRPOTrainer):
    def compute_reward(self, completion: str, target: str) -> float:
        """Custom reward function for math problems"""
        try:
            answer = re.findall(r"<answer>(.*?)</answer>", completion)[0]
            penalty = abs(int(answer) - int(target))
            return 1000 - penalty
        except:
            return -100


# Create RL trainer
trainer = MathRLTrainer(
    experiment_name="math_rl",
    data_folder_path=Path("./data"),
    num_epochs=3,
    per_device_batch_size=1,
    learning_rate=1e-6,
    model=model,
    tokenizer=tokenizer,
    training_dataset=train_ds,
    validation_dataset=val_ds,
    accelerator=accelerator,
    collate_fn=LLMRLCollator(tokenizer=tokenizer, enable_thinking=True),
    use_vllm=True,  # Use vLLM for fast inference
)

trainer.run()
```
---

## 📚 Documentation

### Key Components

#### Trainers

**Base Trainer:**
- **`HurricoreTrainer`**: Abstract base class - extend this to create trainers for any model and task

**LLM-Focused Implementations:**
- **`LLMTrainer`**: Standard supervised fine-tuning for LLMs
- **`LLMGRPOTrainer`**: Group Relative Policy Optimization for RL
- **`LLMDAPOTrainer`**: Direct Alignment from Preferences Optimization
- **`LLMGSPOTrainer`**: Guided Self-Play Optimization

**Other Task Trainers:**
- **`MultiClassClassificationTrainer`**: Multi-class classification tasks
- **`MultiLabelClassificationTrainer`**: Multi-label classification tasks

#### Collators

- **`LLMSFTCollator`**: Collator for LLM supervised fine-tuning
- **`LLMRLCollator`**: Collator for LLM reinforcement learning
- **`VLMSFTCollator`**: Collator for vision-language model SFT
- **`VLMRLCollator`**: Collator for vision-language model RL

#### Datasets

- **`LMDBDataset`**: High-performance LMDB-backed dataset
- **`BalancedDataset`**: Automatically balance class distributions
- **`PackedDataset`**: Pack multiple samples for efficiency
- **`FilteredDataset`**: Apply custom filtering logic
- **`CycleDataset`**: Infinite cycling through data

#### Hooks

Hooks allow you to customize training behavior at different stages:

- **`CheckpointHook`**: Save model checkpoints
- **`LoggingHook`**: Log training metrics
- **`TensorBoardHook`**: TensorBoard integration
- **`LRSchedulerHook`**: Learning rate scheduling
- **`ValidationLossHook`**: Compute validation loss
---

## 💡 Examples

Check out the `examples/` directory for complete working examples focused on LLM training:

- **`sft_llm.py`**: Supervised fine-tuning for language models
- **`rl_llm.py`**: Reinforcement learning with GRPO
- **`sft_vlm.py`**: Vision-language model fine-tuning
- **`rl_vlm.py`**: Vision-language model RL training

> **Extending to Other Models**: These examples demonstrate the framework's usage with LLMs, but the same patterns apply to any model type. Simply extend `HurricoreTrainer` and implement your custom training logic.

### Running Examples

```bash
# Run SFT example
python examples/sft_llm.py

# Run RL example
python examples/rl_llm.py
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built on top of [PyTorch](https://pytorch.org/)
- Powered by [Hugging Face Accelerate](https://github.com/huggingface/accelerate)
- Integrated with [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- Fast inference with [vLLM](https://github.com/vllm-project/vllm)

---

## 📧 Contact

- **Author**: universuen
- **Email**: universuen@gmail.com
- **GitHub**: [https://github.com/universuen/hurricore](https://github.com/universuen/hurricore)

---

## ⚠️ Status

**Note**: This repository is currently under active development. APIs may change between versions. Please check the changelog for updates.

---

