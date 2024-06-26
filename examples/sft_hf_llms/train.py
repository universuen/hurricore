import os

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

from hurricore.trainers import HFLLMTrainer
from hurricore.utils import Logger, launch, import_config, HFLLMITCollator

from zhihu_qa_dataset import ZhihuQADataset

# import config from module path
config = import_config('configs.opt_350m')

""" Optional:
import config from file path
`config = import_config('examples/sft_hf_llms/configs/opt_350m.py')`
import config from url
`config = import_config('https://raw.githubusercontent.com/universuen/hurricore/main/examples/sft_hf_llms/configs/opt_350m.py')`
"""


def main():
    # setup logger and accelerator
    accelerator = Accelerator(**config.AcceleratorConfig())
    logger = Logger(**config.LoggerConfig())
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if accelerator.is_main_process:
        logger.info('Set TOKENIZERS_PARALLELISM=false to prevent dead lock.')
    # setup tokenizer, model, dataset and dataloader
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForCausalLM.from_pretrained(config.model_name)
        dataset = ZhihuQADataset()
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model.resize_token_embeddings(len(tokenizer))
    data_loader = DataLoader(
        dataset=dataset,
        collate_fn=HFLLMITCollator(
            tokenizer=tokenizer, 
            **config.CollatorConfig(),
        ),
        **config.DataLoaderConfig(),
    )
    # setup optimizer and lr scheduler
    optimizer = AdamW(
        params=model.parameters(),
        **config.OptimizerConfig(),
    )
    num_steps_per_epoch = len(data_loader)
    num_epochs = config.TrainerConfig().num_epochs
    gradient_accumulation_steps = accelerator.gradient_accumulation_steps
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=num_steps_per_epoch * num_epochs // gradient_accumulation_steps,
    )
    # setup trainer and run
    trainer = HFLLMTrainer(
        model=model, 
        data_loader=data_loader, 
        optimizer=optimizer, 
        logger=logger, 
        accelerator=accelerator,
        tokenizer=tokenizer,
        lr_scheduler=scheduler,
        lr_scheduler_mode='per_step',
        **config.TrainerConfig(),
    )
    trainer.run()

launch(main, **config.LaunchConfig())
