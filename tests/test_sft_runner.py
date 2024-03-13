import context

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.runners.sft_runner import SFTRunner
from src.datasets.zhihu_qa_dataset import ZhihuQADataset
from src.logger import Logger


runner = SFTRunner(
    model=None,
    tokenizer=None,
    dataset=None,
    optimizer=None,
    scheduler=None,
    logger=None,
)

runner._print('Hello from print')
runner.logger = Logger('test')
runner._print('Hello from logger.info', 'info')

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
runner.tokenizer = tokenizer
# batch = runner.collate_fn(
#             [
#                 ('Hello', "Thank you"),
#                 ('Hello Hello', "Thank you Thank you"),
#                 ('Wow', 'Fantastic'),
#             ]
#         )
# print(batch)

runner.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map='auto')
runner.model.resize_token_embeddings(len(tokenizer))
runner.dataset = ZhihuQADataset()
runner.optimizer = torch.optim.AdamW(runner.model.parameters(), 1e-3)
runner.scheduler = torch.optim.lr_scheduler.LinearLR(runner.optimizer)

runner.train(batch_size=1, max_len=512)
