# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch

from dataclasses import dataclass, field
from pathlib import Path

from litgpt.data import SFTDataset, DPODataset
from litgpt.data.alpaca import Alpaca

from torch.utils.data import DataLoader
from typing import Union

from litgpt.prompts import PromptStyle
from .preference_datasets import get_collate_fn, get_hh


@dataclass
class HH(Alpaca):
    """Alpaca2k data module for supervised finetuning."""

    val_split_fraction: float = 0.05  # to get exactly 100 validation samples,
    """The fraction of the dataset to use for the validation dataset. The rest is used for training."""
    download_dir: Path = Path("./data/hh-rlhf")
    """The directory in which the downloaded datasetgets saved."""
    repo_id: str = field(repr=False, default="Anthropic/hh-rlhf")
    """The name of the dataset file to download."""
    prompt_style: Union[str, PromptStyle] = "default"

    max_seq_length: int = 512
    max_prompt_length: int = 256

    def prepare_data(self) -> None:
        from datasets import load_dataset

        load_dataset(self.repo_id, cache_dir=self.download_dir) #, split='train[:2%]')

    def setup(self, stage: str = "") -> None:

        train_data = get_hh(split='train', cache_dir=self.download_dir)
        test_data = get_hh(split='test', cache_dir=self.download_dir)

        self.train_dataset = DPODataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            max_prompt_length=self.max_prompt_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
            truncation_mode='keep_end',
            chosen_str="chosen",
            rejected_str="rejected",
        )
        self.test_dataset = DPODataset(
            data=test_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            max_prompt_length=self.max_prompt_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
            truncation_mode='keep_end',
            chosen_str="chosen",
            rejected_str="rejected",
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_collate_fn(self.tokenizer),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_collate_fn(self.tokenizer),
        )