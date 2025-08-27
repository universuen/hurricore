from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Precision, Recall, F1Score, AUROC, AveragePrecision, Accuracy

from hurricore.utils import make_table


class ClassificationEvaluator:
    def __init__(
            self,
            task: Literal["binary", "multiclass", "multilabel"] = "binary",
            num_classes: int = None,
            num_labels: int = None,
            average: Literal["micro", "macro", "weighted", "none"] = "none",
            threshold: float = 0.5,
        ):
        assert task in ["binary", "multiclass", "multilabel"], 'task should be one of "binary", "multiclass", "multilabel".'
        self.metrics = MetricCollection(
            {
                'AUPRC': AveragePrecision(
                    task = task,
                    num_classes = num_classes,
                    num_labels = num_labels,
                    average = average if average != "micro" else None,
                ),
                'AUROC': AUROC(
                    task = task,
                    num_classes = num_classes,
                    num_labels = num_labels,
                    average = average if average != "micro" else None,
                ),
                'F1': F1Score(
                    task = task,
                    num_classes = num_classes,
                    num_labels = num_labels,
                    average = average,
                    threshold = threshold,
                ),
                'Precision': Precision(
                    task = task,
                    num_classes = num_classes,
                    num_labels = num_labels,
                    average = average,
                    threshold = threshold,
                ),
                'Recall': Recall(
                    task = task,
                    num_classes = num_classes,
                    num_labels = num_labels,
                    average = average,
                    threshold = threshold,
                ),
                'Accuracy': Accuracy(
                    task = task,
                    num_classes = num_classes,
                    num_labels = num_labels,
                    average = average,
                    threshold = threshold,
                )
            }
        )
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.task = task
        self.average = average
        self.threshold = threshold


    def get_probs_and_labels(self, model: nn.Module, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


    def __call__(
            self, 
            model: nn.Module,
            data_loader: DataLoader, 
            print_results: bool = False,
        ) -> dict:
        device = next(model.parameters()).device
        self.metrics = self.metrics.to(device)
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                probs, labels = self.get_probs_and_labels(model, batch)
                probs = probs.to(device)
                labels = labels.to(device)
                assert type(probs) is torch.Tensor, 'probs should be a Tensor.'
                assert type(labels) is torch.Tensor, 'labels should be a Tensor.'
                self.metrics.update(probs, labels)
            results: dict[str, torch.Tensor] = self.metrics.compute()
            self.metrics.reset()
        results = {
            k: v.tolist()
            for k, v in results.items()
        }
        if print_results:
            first_value = next(iter(results.values()))
            if isinstance(first_value, list):
                formatted_results = [[metric, *values] for metric, values in results.items()]
                headers = ["Metric"] + [f"Class {i}" for i in range(len(results[next(iter(results))]))]
            else:
                formatted_results = [[metric, values] for metric, values in results.items()]
                headers = ["Metric", "Value"]
            headers = ["Metric"] + [f"Class {i}" for i in range(len(results[next(iter(results))]))]
            print(make_table(headers=headers, rows=formatted_results))
        return results
