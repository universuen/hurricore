from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hurricore.hooks import HurricoreHook
from hurricore.trainers import HurricoreTrainer
from hurricore.utils import env, ClassificationEvaluator, make_table


class ClassificationEvaluationHook(HurricoreHook):
    def __init__(
            self, 
            trainer: HurricoreTrainer,
            data_loader: DataLoader,
            evaluator: ClassificationEvaluator,
            interval: int = 10,
        ) -> None:
        super().__init__(
            trainer=trainer,
            data_loader=data_loader,
            evaluator=evaluator,
            interval=interval,
        )
        self.evaluator = evaluator
        self.interval = interval
    

    def execute_on_trainer_init_end(self) -> None:
        self.data_loader = self.trainer.accelerator.prepare(self.init_context.data_loader)
        if self.trainer.accelerator.is_main_process:
            self.data_loader = tqdm(self.data_loader)

    
    def evaluate(self) -> dict:
        return self.evaluator(self.trainer.original_model, self.data_loader)


    def execute_on_step_end(self) -> None:
        global_step = self.trainer.training_context.global_step + 1
        if global_step % self.interval != 0:
            return
        env.logger.info(f'Evaluating model at step {global_step}...')
        results = self.evaluate()
        self._log_results(results)
    
    
    def execute_on_training_end(self) -> None:
        global_step = self.trainer.training_context.global_step + 1
        env.logger.info(f'Evaluating model at step {global_step}...')
        results=self.evaluate()
        self._log_results(results)
    

    def _log_results(self, results: dict[str, torch.Tensor]) -> None:
        first_value = next(iter(results.values()))
        if isinstance(first_value, list):
            formatted_results = [[metric, *values] for metric, values in results.items()]
            headers = ["Metric"] + [f"Class {i}" for i in range(len(results[next(iter(results))]))]
        else:
            formatted_results = [[metric, values] for metric, values in results.items()]
            headers = ["Metric", "Value"]
        table_string = make_table(headers=headers, rows=formatted_results)
        env.logger.info(f"Evaluation results:\n{table_string}")
        for metric_name, metric_value in results.items():
            if isinstance(metric_value, float):
                if metric_value == float('nan'):
                    metric_value = 0.0
                env.tensor_board_writer.add_scalar(f'Metric/{metric_name}_{self.evaluator.average}_avg', metric_value, self.trainer.training_context.global_step)
                env.history_recorder.record(f'metric_{metric_name}_{self.evaluator.average}_avg', metric_value, self.trainer.training_context.global_step)

            elif isinstance(metric_value, list):
                for i, value in enumerate(metric_value):
                    if value == float('nan'):
                        value = 0.0
                    env.tensor_board_writer.add_scalar(f'Metric/{metric_name}_class_{i}', value, self.trainer.training_context.global_step)
                    env.history_recorder.record(f'metric_{metric_name}_class_{i}', value, self.trainer.training_context.global_step)
            else:
                raise ValueError(f"Unsupported metric value type: {type(metric_value)}")
