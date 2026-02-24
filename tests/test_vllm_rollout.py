import os

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from accelerate import Accelerator

from hurricore.utils import VLLMRollout
from hurricore.collators import VLMRLCollator


def test_vllm_rollout():
    accelerator = Accelerator()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    print(model.config.name_or_path)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    collator = VLMRLCollator(
        processor=processor,
        generation_mode=True,
    )

    samples = [
        [
            {
                "role": "user",
                "content": [Image.new("RGB", (256, 256), "red"), "What is the color of this image?"]
            },
        ],
        [
            {
                "role": "user",
                "content": f"My index is {accelerator.process_index}, what is my index?"
            }
        ]
    ]
    vllm_conversations = collator(samples)["vllm_conversations"]
    from pprint import pprint
    pprint(vllm_conversations)

    os.environ["LOCAL_RANK"] = str(accelerator.process_index)
    os.environ["WORLD_SIZE"] = str(accelerator.num_processes)

    vllm_rollout = VLLMRollout(
        model=model,
        vllm_kwargs={
            "max_model_len": 16 * 1024,
            "gpu_memory_utilization": 0.3,
            "max_num_seqs": 8,
            "tensor_parallel_size": 1,
        },
    )
    completions_ids = vllm_rollout(
        vllm_conversations=vllm_conversations,
    )
    for completion_ids in completions_ids:
        print(processor.decode(completion_ids, skip_special_tokens=True))

    
if __name__ == "__main__":
    test_vllm_rollout()
