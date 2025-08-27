from __future__ import annotations

import os
import itertools
from collections import defaultdict
import random
import sys
import inspect
import argparse
from types import ModuleType
from importlib import util, import_module
from pathlib import Path
from datetime import datetime
from typing import Iterable, Any, Callable
from io import BytesIO
import pickle
import hashlib

import torch
from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np
from tensorboard import program
from tqdm import tqdm
from PIL import Image
import requests

from hurricore.utils.environment import env


def split_list_into_chunks(original_list, chunk_size):
    chunks = []
    for i in range(0, len(original_list), chunk_size):
        chunks.append(original_list[i:i + chunk_size])
    return chunks



def decouple_vlm_inputs_to_vllm_format(
        text_inputs: list[str],
        image_inputs: list[Image.Image],
        image_token: str = "<|image_pad|>",
        *args, **kwargs,
    ) -> list[dict]:
    model_inputs = []
    current_image_idx = 0

    for i, prompt_text in enumerate(text_inputs):
        num_images = prompt_text.count(image_token)
        multi_modal_data = dict()
        
        if num_images > 0:
            prompt_images = image_inputs[current_image_idx : current_image_idx + num_images]
            multi_modal_data["image"] = prompt_images
            current_image_idx += num_images

        model_inputs.append(
            {
                "prompt": prompt_text,
                "multi_modal_data": multi_modal_data,
            }
        )

    return model_inputs

def make_table(
    headers: list[str],
    rows: list[list],
    float_format: str = ".4f",
) -> str:
    # Helper to format a cell
    def format_cell(cell):
        if isinstance(cell, float) and float_format is not None:
            return format(cell, float_format)
        return str(cell)

    # Calculate the maximum width of each column
    cols = [headers] + rows
    col_widths = [
        max(len(format_cell(cell)) for cell in col)
        for col in zip(*cols)
    ]

    # Helper to format a row
    def format_row(row):
        return "| " + " | ".join(format_cell(cell).ljust(width) for cell, width in zip(row, col_widths)) + " |"

    # Build the table as a list of strings
    lines = []
    lines.append(format_row(headers))
    lines.append("|" + "|".join("-" * (width + 2) for width in col_widths) + "|")
    for row in rows:
        lines.append(format_row(row))
    return "\n".join(lines)


def assign_args(self: Any, locals: dict) -> None:
    for name, value in locals.items():
        if name != 'self':  # Skip self
            setattr(self, name, value)


def reverse_dict(d: dict) -> dict:
    return {v: k for k, v in d.items()}


def hash_anything(item: Any) -> str:
    return hashlib.md5(pickle.dumps(item)).hexdigest()


def get_random_user_agent():
    browsers = [
        "Mozilla/5.0",
        "Mozilla/4.0",
        "Mozilla/3.0",
        "Opera/9.80",
        "Mozilla/5.0 (KHTML, like Gecko)",
        "Mozilla/5.0 (AppleWebKit)"
    ]
    
    platforms = [
        "Windows NT 10.0; Win64; x64",
        "Windows NT 6.1; Win64; x64",
        "Macintosh; Intel Mac OS X 10_15_7",
        "Macintosh; Intel Mac OS X 12_3",
        "Linux; Android 11",
        "Linux; Android 9",
        "iPhone; CPU iPhone OS 15_2 like Mac OS X",
        "iPad; CPU OS 14_7 like Mac OS X",
        "Windows NT 6.3; Trident/7.0; rv:11.0",
        "Windows NT 5.1; SV1"
    ]
    
    engines = [
        "Gecko/20100101 Firefox/90.0",
        "Chrome/91.0.4472.124 Safari/537.36",
        "Safari/605.1.15",
        "Edge/91.0.864.54",
        "Opera/74.0.3911.203",
        ".NET CLR 1.1.4322",
        ".NET CLR 2.0.50727",
        "KHTML, like Gecko",
        "AppleWebKit/537.36",
        "Trident/7.0"
    ]

    browser = random.choice(browsers)
    platform = random.choice(platforms)
    engine = random.choice(engines)
    user_agent = f"{browser} ({platform}) {engine}"
    return user_agent



def find_first_sublist_index(full_list, sub_list) -> int:
    if not sub_list or len(sub_list) > len(full_list):
        return -1
    sub_len = len(sub_list)
    for i in range(len(full_list) - sub_len + 1):
        if full_list[i:i + sub_len] == sub_list:
            return i
    return -1


def find_last_sublist_index(full_list, sub_list) -> int:
    if not sub_list or len(sub_list) > len(full_list):
        return -1  
    sub_len = len(sub_list)
    for i in range(len(full_list) - sub_len, -1, -1):
        if full_list[i:i + sub_len] == sub_list:
            return i
    return -1


def find_all_sublist_indices(full_list, sub_list) -> list[int]:
    if not sub_list or len(sub_list) > len(full_list):
        return []
    sub_len = len(sub_list)
    indices = []
    for i in range(len(full_list) - sub_len + 1):
        if full_list[i:i + sub_len] == sub_list:
            indices.append(i)
    return indices


def deduplicate_with_order(seq: Iterable) -> list:
    return list(dict.fromkeys(seq))


def split_list(main_list: list, sub_list: list) -> list:
    """
    Splits a list by a sublist, similar to string.split() method.
    """
    if not sub_list:
        return [main_list]
    result = []
    current = []
    i = 0
    while i < len(main_list):
        if main_list[i:i + len(sub_list)] == sub_list:
            result.append(current)
            current = []
            i += len(sub_list)
        else:
            current.append(main_list[i])
            i += 1
    result.append(current)
    return result



def get_casaulty_matrix(batch_size, seq_len: int) -> torch.Tensor:
    """
    Returns a lower triangular matrix of shape (batch_size, seq_len, seq_len) with ones on the diagonal.
    """
    return torch.tril(torch.ones(batch_size, seq_len, seq_len))


def get_ith_item_from_iterator(iterator: Iterable, i: int) -> Any:
    """
    Returns the i-th item from an iterator.
    """
    item = next(itertools.islice(iterator, i, i + 1))
    return item


def find_latest_checkpoint_folder(checkpoints_folder_path: Path | str) -> Path:
    checkpoints_folder_path = Path(checkpoints_folder_path)
    checkpoints = checkpoints_folder_path.glob("ckpt_step_*")
    steps = [int(checkpoint.name.split("_")[-1]) for checkpoint in checkpoints]
    if len(steps) == 0:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_folder_path}")
    latest_step = max(steps)
    return checkpoints_folder_path / f"ckpt_step_{latest_step}"


def is_deepspeed_zero3(accelerator) -> bool:
    if accelerator.state.deepspeed_plugin is not None \
    and accelerator.state.deepspeed_plugin.zero_stage == 3:
        return True
    return False


def get_time_stamp() -> str:
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")


def auto_name(iterable: Iterable) -> list[str]:
    names = []
    names_cnt = {}
    for i in iterable:
        name = i.__class__.__name__
        if name in names_cnt:
            names_cnt[name] += 1
            name = f'{name}_{names_cnt[name]}'
        else:
            names_cnt[name] = 0
        names.append(name)
    return names


def format_large_number(num_params: int) -> str:
    if num_params >= 1e9: 
        return f'{num_params / 1e9:.2f} B'
    elif num_params >= 1e6: 
        return f'{num_params / 1e6:.2f} M'
    elif num_params >= 1e3: 
        return f'{num_params / 1e3:.2f} K'
    else: 
        return str(num_params)


def count_model_parameters(model: torch.nn.Module, trainable_params_only: bool = False) -> int:
    """
    Counts the total or trainable parameters of a PyTorch model.
    """
    total_params = 0
    for param in model.parameters():
        if not trainable_params_only or param.requires_grad:
            total_params += param.numel()
    return total_params


def get_params_details_table(*models: tuple[torch.nn.Module], format_numbers: bool = True) -> str:
    if len(models) == 0:
        return "\n\tNo models provided.\n"

    model_names = auto_name(models)
    param_data = []
    max_total_str_len = 0
    max_trainable_str_len = 0

    for name, model in zip(model_names, models):
        total_params = count_model_parameters(model)
        trainable_params = count_model_parameters(model, trainable_params_only=True)

        if format_numbers:
            total_params_str = format_large_number(total_params)
            trainable_params_str = format_large_number(trainable_params)
        else:
            total_params_str = str(total_params)
            trainable_params_str = str(trainable_params)

        param_data.append({
            'name': name,
            'total_params': total_params, # Store original for potential later use
            'trainable_params': trainable_params, # Store original
            'total_params_str': total_params_str,
            'trainable_params_str': trainable_params_str
        })
        max_total_str_len = max(max_total_str_len, len(total_params_str))
        max_trainable_str_len = max(max_trainable_str_len, len(trainable_params_str))

    # Determine column widths
    header_model = "Model"
    header_total = "Total Params"
    header_trainable = "Trainable Params"

    # Add padding (e.g., 2 spaces)
    name_col_width = max(len(header_model), max(len(d['name']) for d in param_data)) + 2
    # Ensure param columns are at least as wide as their headers
    total_col_width = max(len(header_total), max_total_str_len) + 2
    trainable_col_width = max(len(header_trainable), max_trainable_str_len) + 2

    # Construct table parts
    table_header = f"\t{header_model:<{name_col_width}} | {header_total:>{total_col_width}} | {header_trainable:>{trainable_col_width}}"
    # Adjust divider length based on the final header length (minus the initial tab)
    table_divider = '\t' + '-' * (len(table_header) - 1) # -1 for the initial \t

    table_rows = [table_header, table_divider]
    for data in param_data:
        row = f"\t{data['name']:<{name_col_width}} | {data['total_params_str']:>{total_col_width}} | {data['trainable_params_str']:>{trainable_col_width}}"
        table_rows.append(row)

    full_table = '\n'.join(table_rows)
    return f'\n{full_table}\n'


def generate_sinusoid_positional_embeddings(length: int, embedding_dim: int) -> torch.Tensor:
    pos = torch.arange(length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embedding_dim))
    pos_emb = torch.zeros(length, embedding_dim)
    pos_emb[:, 0::2] = torch.sin(pos * div_term)
    pos_emb[:, 1::2] = torch.cos(pos * div_term)
    return pos_emb.unsqueeze(0)  # [1, length, embedding_dim]


def chunkify(seq: list, chunk_size: int) -> list[list]:
    num_chunks = int(np.ceil(len(seq) / chunk_size))
    chunks = [seq[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    return chunks


def launch_tensor_board(folder_path, port: int = 6006) -> str:
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', str(folder_path), '--port', str(port)])
    url = tb.launch()
    return url


def summarize_tensor(
        tensor: torch.Tensor,
        print_summary: bool = True,
    ) -> dict[str, any]:
    summary = {
        'shape': tensor.shape,
        'min': tensor.min().item(),
        'max': tensor.max().item(),
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
        'dtype': tensor.dtype,
        'device': tensor.device,
        'requires_grad': tensor.requires_grad,
    }
    if print_summary:
        print(f"{'Property':<15} {'Value'}")
        print("-" * 30)
        for key, value in summary.items():
            print(f"{key:<15} {value: .2f}" if isinstance(value, float) else f"{key:<15} {value}")
    return summary


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.device("npu")
    return torch.device("cpu")


def get_device_utilization() -> float:
    try:
        if torch.cuda.is_available():
            return torch.cuda.utilization()
        if hasattr(torch, 'npu') and torch.npu.is_available():
            return torch.npu.utilization()
    except Exception as e: 
        print(f"Error getting device utilization: {e}")
    return float("nan")


def get_device_memory_info() -> tuple[int, int]:
    try:
        if torch.cuda.is_available():
            return torch.cuda.mem_get_info()
        if hasattr(torch, 'npu') and torch.npu.is_available():
            return torch.npu.mem_get_info()
    except Exception as e:
        print(f"Error getting device memory info: {e}")
    return 0, 0


def build_label_indices_map(
        dataset: Dataset, 
        get_label: callable = lambda x: x[1],
        batch_size: int = 32,
    ) -> dict[str, int]:
    data_loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        num_workers=os.cpu_count(),
        collate_fn=lambda x: x,
    )
    label_index = defaultdict(list)
    for i, batch in enumerate(tqdm(data_loader, desc="Building label indices map")):
        for j, item in enumerate(batch):
            label = get_label(item)
            label_index[label].append(i * batch_size + j)
    # Sort the indices
    for label in label_index:
        label_index[label].sort()
    return label_index


def split_dataset(
        dataset: Dataset, 
        training_ratio: float = 0.8,
        get_label: Callable = None,
        batch_size: int = 32,
        random_seed: int = 42,
    ) -> tuple[Dataset, Dataset]:
    random.seed(random_seed)
    if get_label is None:
        print("get_label function not provided. Using random split.")
        all_indices = list(range(len(dataset)))
        random.shuffle(all_indices)
        num_training_indices = int(training_ratio * len(all_indices))
        training_indices = all_indices[:num_training_indices]
        validation_indices = all_indices[num_training_indices:]
    else:
        print("Using label-based split.")
        label_index = build_label_indices_map(
            dataset=dataset, 
            get_label=get_label, 
            batch_size=batch_size
        )
        training_indices = []
        validation_indices = []
        for indices in label_index.values():
            num_training_indices = int(training_ratio * len(indices))
            if num_training_indices == len(indices):
                num_training_indices -= 1
            if num_training_indices == 0:
                num_training_indices += 1
            random.shuffle(indices)
            training_indices.extend(indices[:num_training_indices])
            validation_indices.extend(indices[num_training_indices:])
        
    training_dataset = Subset(dataset, training_indices)
    validation_dataset = Subset(dataset, validation_indices)
    return training_dataset, validation_dataset


def get_list_mean(list_: list[int | float]) -> float:
    return sum(list_) / (len(list_) + 1e-6)


def find_start_and_end_index(long_t: torch.Tensor, short_t: torch.Tensor) -> int:
        for i in range(len(long_t) - len(short_t) + 1):
            if torch.all(long_t[i:i + len(short_t)] == short_t):
                return i, i + len(short_t)
        return -1, -1


def freeze_model(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def download_image(url: str) -> Image.Image:
    response = requests.get(
        url=url,
        headers={
            "User-Agent": get_random_user_agent(),
        },
        timeout=10,
    )
    with BytesIO(response.content) as img_data:
        image = Image.open(img_data).convert('RGB')
    return image
 


def get_trainable_layer_names(model: torch.nn.Module) -> list[str]:
    results = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only print layers, not containers
            is_trainable = any(p.requires_grad for p in module.parameters())
            if is_trainable:
                results.append(f"{model.__class__.__name__}.{name}")
    return results


def get_file_name() -> str:
    caller_frame_record = inspect.stack()[1]
    module_path = caller_frame_record.filename
    return Path(module_path).stem


def set_cuda_visible_devices(*device_indices: tuple[int]) -> None:
    assert all(isinstance(i, int) for i in device_indices), 'device_indices must be integers'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_indices))


def load_config(path: str = None, accept_cmd_args: bool = True) -> ModuleType:
    if accept_cmd_args:
        help_msg = (
            "CONFIG can be any of the following formats:\n"
            "- python_module.config\n"
            "- local/file/path/to/config.py\n"
            "- https://url/to/config.py"
        )
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('-c', '--config', type=str, default=path, help=help_msg)
        args = parser.parse_args()
        path = args.config
    
    if path is None:
        raise ImportError("No config path provided")
    
    env.config_path = path
    
    assert isinstance(path, str), "path must be a string"
    
    temp_file_path = None
    if path.startswith("http"):
        # handle URL
        response = requests.get(path)
        if response.status_code != 200:
            raise ImportError(f"Cannot download the module from {path}")
        with open(Path(__file__).parents[2] / '_temp_config_from_url.py', 'w') as tmp_file:
            temp_file_path = tmp_file.name
            tmp_file.write(response.text)
            module_path = tmp_file.name
            module_name = Path(tmp_file.name).stem
        spec = util.spec_from_file_location(module_name, module_path)
    elif  "/" in path or "\\" in path:
        # handle filesystem path
        path = Path(path)
        module_path = str(path)
        module_name = Path(path).stem
        spec = util.spec_from_file_location(module_name, module_path)
    else:
        # handle module import string
        try:
            return import_module(path)
        except ImportError as e:
            raise ImportError(f"Cannot import module using import string {path}") from e

    if spec is None:
        raise ImportError(f"Cannot import module from {path}")
    
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    if temp_file_path is not None:
        os.remove(temp_file_path)
    
    return module
