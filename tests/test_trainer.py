import context

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.trainers.sft_trainer import SFTTrainer
from src.datasets.zhihu_qa_dataset import ZhihuQADataset
from src.logger import Logger


trainer = SFTTrainer(
    model=None,
    tokenizer=None,
    dataset=None,
    optimizer=None,
    scheduler=None,
    logger=None,
)

trainer._print('Hello from print')
trainer.logger = Logger('test')
trainer._print('Hello from logger.info', 'info')

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
trainer.tokenizer = tokenizer
# batch = trainer.collate_fn(
#             [
#                 ('Hello', "Thank you"),
#                 ('Hello Hello', "Thank you Thank you"),
#                 ('Wow', 'Fantastic'),
#             ]
#         )
# print(batch)

trainer.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map='auto')
trainer.model.resize_token_embeddings(len(tokenizer))
trainer.dataset = ZhihuQADataset()
trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), 1e-3)
trainer.scheduler = torch.optim.lr_scheduler.LinearLR(trainer.optimizer)

trainer.run(batch_size=1, max_len=100)
