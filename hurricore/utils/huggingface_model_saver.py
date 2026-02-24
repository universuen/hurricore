from pathlib import Path

from transformers import AutoProcessor
from peft import PeftModel


class HuggingfaceModelSaver:
    def __init__(self, trainer, folder_path: Path):
        self.trainer = trainer
        self.folder_path = folder_path

    def save(self, name: str) -> Path:
        """
        Saves the full model (base + merged adapters) and processor.
        Handles distributed state gathering and LoRA merging automatically.
        """
        save_path = self.folder_path / name
        self.trainer.accelerator.wait_for_everyone()
        
        model = self.trainer.model
        unwrapped = self.trainer.accelerator.unwrap_model(model)
        is_peft = isinstance(unwrapped, PeftModel)

        if is_peft:
            unwrapped.merge_adapter()

        state_dict = self.trainer.accelerator.get_state_dict(model)

        if is_peft:
            unwrapped.unmerge_adapter()

        if self.trainer.accelerator.is_main_process:
            self._save_disk_operations(unwrapped, state_dict, save_path, is_peft)

        self.trainer.accelerator.wait_for_everyone()
        return save_path


    def _save_disk_operations(self, model, state_dict, save_path: Path, is_peft: bool):
        save_path.mkdir(parents=True, exist_ok=True)
        
        if is_peft:
            model_to_save = model.base_model.model
            final_state_dict = {
                k.replace("base_model.model.", ""): v 
                for k, v in state_dict.items()
            }
            active_adapter = model.active_adapter
            if isinstance(active_adapter, list): active_adapter = active_adapter[0]
            base_model_path = model.peft_config[active_adapter].base_model_name_or_path
            
        else:
            model_to_save = model
            final_state_dict = state_dict
            base_model_path = getattr(model, "name_or_path", None)
        model_to_save.save_pretrained(
            save_path, 
            state_dict=final_state_dict
        )
        if base_model_path:
            try:
                processor = AutoProcessor.from_pretrained(base_model_path)
                processor.save_pretrained(save_path)
            except Exception:
                pass
