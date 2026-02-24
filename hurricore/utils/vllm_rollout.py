import os
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

import deepspeed
from vllm import LLM, SamplingParams
from transformers import PreTrainedModel

from hurricore.utils import deepspeed_zero3_enabled


class VLLMRollout:
    def __init__(
        self,
        model: PreTrainedModel,
        vllm_kwargs: dict = None,
    ) -> None:
        self.model = model
        self.vllm_kwargs = vllm_kwargs or dict()
        os.environ["RANK"] = os.environ.get("RANK", "0")
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
        inner_model = self.vllm.llm_engine.model_executor.driver_worker.worker.model_runner.model

        if deepspeed_zero3_enabled():
            import deepspeed
            def weight_generator():
                for name, param in model.named_parameters():
                    if hasattr(param, 'ds_active_sub_modules'):
                        param.ds_active_sub_modules.clear()
                    with deepspeed.zero.GatheredParameters([param], modifier_rank=None):
                        yield name, param       
            inner_model.load_weights(weight_generator())
        else:
            inner_model.load_weights(model.named_parameters())
            
        self.vllm.reset_prefix_cache()
    

    def __call__(
            self, 
            vllm_conversations: list[list[dict]], 
            sampling_params: SamplingParams = None,
            return_text: bool = True,
        ) -> list[str] | list[list[int]]:
        outputs = self.vllm.chat(
            vllm_conversations, 
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        if return_text:
            return [output.outputs[0].text for output in outputs]
        return [output.outputs[0].token_ids for output in outputs]
    
    
    def sleep(self) -> None:
        self.vllm.sleep(level=1)
    

    def wake_up(self) -> None:
        self.vllm.wake_up()
