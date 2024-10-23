import collections
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import json
import warnings
from logging import StreamHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from tqdm.auto import tqdm

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from transformers import Trainer
from transformers.trainer_utils import speed_metrics
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow


class CustomTrainer(Trainer):
    def __init__(
        self,
        model,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        path_to_best_model="models/model",
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
        )
        self.path_to_best_model = path_to_best_model

    def evaluate(
        self,
        train_dataset=None,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        self._memory_tracker.start()

        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = (
            self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        )

        train_output = eval_loop(
            train_dataloader,
            description="Evaluation on train",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix="train",
        )

        eval_output = eval_loop(
            eval_dataloader,
            description="Evaluation on dev",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix="eval",
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size

        train_output.metrics.update(
            speed_metrics(metric_key_prefix, start_time, train_output.num_samples)
        )

        eval_output.metrics.update(
            speed_metrics(metric_key_prefix, start_time, eval_output.num_samples)
        )

        self.log(train_output.metrics)
        self.log(eval_output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, train_output.metrics
        )
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, eval_output.metrics
        )

        self._memory_tracker.stop_and_update_metrics(train_output.metrics)
        self._memory_tracker.stop_and_update_metrics(eval_output.metrics)

        dic = {
            "Training metrics": train_output.metrics,
            "Validation metrics": eval_output.metrics,
        }
        print(eval_output.metrics.keys())
        eval_accuracy = eval_output.metrics["eval_accuracy"]

        # store model if performance improved
        if self.state.best_model_checkpoint is None or eval_accuracy > self.state.best_metric:
            self.state.best_model_checkpoint = self.state.global_step
            self.state.best_metric = eval_accuracy
            self._save_model(self.path_to_best_model)
            self._store_metadata_best_model()
        return dic

    def _store_metadata_best_model(self):
        """
        Store metadata of best model to .txt file.
        """
        # change extension of path
        path, ext = os.path.splitext(self.path_to_best_model)
        path_to_metadata = f"{path}.txt"

        # create metadata string
        metadata = f"Best metric: {self.state.best_metric}, global_step: {self.state.best_model_checkpoint}"

        # store metadata
        with open(path_to_metadata, "w") as fp:
            fp.write(metadata)

    def _save_model(self, output_dir: Optional[str] = None):
        """
        Stores the best model found so far.
        """
        print("Storing best model")
        super().save_model(output_dir)
