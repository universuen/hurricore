from __future__ import annotations

from typing import Literal, Any

import torch

from hurricore.trainers import StandardTrainer
from hurricore.hooks import ClassificationEvaluationHook
from hurricore.utils import ClassificationEvaluator


class MultiLabelClassificationTrainer(StandardTrainer):
    def __init__(
        self,
        # New in this class
        num_labels: int,
        evaluation_average_method: Literal['micro', 'macro', 'weighted', 'none'] = 'none',
        threshold: float = 0.5,
        # Task specific hyper parameters
        num_epochs: int = 3,
        per_device_batch_size: int = 32,
        learning_rate: float = 5e-5,
        **kwargs,
    ) -> None:
        super().__init__(
            num_labels=num_labels,
            evaluation_average_method=evaluation_average_method,
            threshold=threshold,
            num_epochs=num_epochs,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            **kwargs,
        )
        self.criterion = torch.nn.CrossEntropyLoss()
    

    def setup_hooks(self) -> None:
        super().setup_hooks()
        self.init_context.hooks.append(
            ClassificationEvaluationHook(
                trainer=self,
                data_loader=self.init_context.validation_data_loader,
                interval=self.init_context.evaluation_interval,  
                evaluator=ClassificationEvaluator(
                    num_labels=self.init_context.num_labels,
                    average=self.init_context.evaluation_average_method,
                    threshold=self.init_context.threshold,
                ),
            )
        )
    
    
    def compute_loss(self, batch: dict) -> torch.Tensor:
        assert type(batch) is dict, f"Invalid batch type: {type(batch)}"
        assert "model_inputs" in batch.keys(), f"`model_inputs` not in batch keys: {batch.keys()}"
        assert type(batch['model_inputs']) is dict, f"Invalid model_inputs type: {type(batch['model_inputs'])}. Expected dict"
        assert "labels" in batch.keys(), f"`labels` not in batch keys: {batch.keys()}"
        assert type(batch['labels']) is torch.Tensor, f"Invalid labels type: {type(batch['labels'])}. Expected torch.Tensor"
        model_inputs, labels = batch['model_inputs'], batch['labels']
        logits = self.model(**model_inputs) if type(model_inputs) is dict else self.model(model_inputs)
        assert logits.shape[-1] == self.init_context.num_labels, f"Logits shape: {logits.shape}, num_labels: {self.init_context.num_labels}"
        loss = self.criterion(logits, labels)
        return loss
