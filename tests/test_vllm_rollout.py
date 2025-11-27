import os

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from accelerate import Accelerator

from hurricore.utils import QwenVLSFTCollator, convert_vlm_inputs_to_vllm_prompts, VLLMRollout


def test_vllm_rollout():
    accelerator = Accelerator()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    print(model.config.name_or_path)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    collator = QwenVLSFTCollator(
        processor=processor,
        generation_mode=True,
        tokenize=False,
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
    data = collator(samples)
    vllm_model_inputs = convert_vlm_inputs_to_vllm_prompts(
        text_inputs=data["text_inputs"],
        image_inputs=data["image_inputs"],
    )
    from pprint import pprint
    pprint(vllm_model_inputs)

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
    completions = vllm_rollout.generate(
        vllm_prompts=vllm_model_inputs,
    )
    for completion in completions:
        print(completion)

    
if __name__ == "__main__":
    test_vllm_rollout()
