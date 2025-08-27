import os
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

from vllm import LLM, SamplingParams
from transformers import PreTrainedModel


class VLLMRollout:
    def __init__(
        self,
        model: PreTrainedModel,
        vllm_kwargs: dict = None,
    ) -> None:
        self.model = model
        self.vllm_kwargs = vllm_kwargs or dict()
        os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")

        self.vllm = LLM(
            model=model.config.name_or_path, 
            distributed_executor_backend="external_launcher",
            enable_sleep_mode=True,
            dtype=model.dtype,
            **self.vllm_kwargs,
        )
        self.update_weights()


    def update_weights(self, model: PreTrainedModel = None) -> None:
        model = model or self.model
        for param_name, param in model.named_parameters():
            inner_model = self.vllm.llm_engine.model_executor.driver_worker.model_runner.model
            inner_model.load_weights([(param_name, param.data)])
        self.vllm.reset_prefix_cache()
    

    def generate(
            self, 
            vllm_prompts: list[str] | list[dict], 
            sampling_params: SamplingParams = None,
        ) -> list[str]:
        outputs = self.vllm.generate(
            prompts=vllm_prompts, 
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        completions = [output.outputs[0].text for output in outputs]
        return completions
    
    
    def sleep(self) -> None:
        self.vllm.sleep(level=1)
    

    def wake_up(self) -> None:
        self.vllm.wake_up()



if __name__ == "__main__":
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from PIL import Image
    from accelerate import Accelerator

    from hurricore.utils import QwenVLSFTCollator, decouple_vlm_inputs_to_vllm_format


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
    vllm_model_inputs = decouple_vlm_inputs_to_vllm_format(
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

    
    

