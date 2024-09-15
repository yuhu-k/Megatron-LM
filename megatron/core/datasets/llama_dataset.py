import logging
import os
import time
from typing import Dict, Optional, Tuple

import numpy
import torch

from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.utils import Split
from megatron.core.utils import log_single_rank
from .gpt_dataset import (
    GPTDataset, 
    GPTDatasetConfig, 
    _PAD_TOKEN_ID, 
    _get_ltor_masks_and_position_ids, 
    logger, 
    _build_document_index, 
    _build_shuffle_index,
)
import random
from torchtune.datasets import alpaca_dataset, alpaca_cleaned_dataset
from torchtune.models.llama2._tokenizer import Llama2Tokenizer


class LLaMADataset(GPTDataset):
    """The base GPT dataset

    Args:
        indexed_dataset (IndexedDataset): The IndexedDataset around which to build the GPTDataset

        dataset_path (Optional[str]): The real path on disk to the dataset, for bookkeeping

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (Optional[int]): The number of samples to draw from the indexed dataset. When None, build as many samples as correspond to one epoch.

        index_split (Split): The indexed_indices Split

        config (GPTDatasetConfig): The config
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: Optional[str],
        indexed_indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        # super().__init__(
        #     indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        # )
        self.config = config
        # self.length = len(self.dataset)
        # self.__build_shuffle_indices()
        tokenizer = Llama2Tokenizer("/tmp2/Megatron-LM/tokenizer.model")
        self.dataset = alpaca_cleaned_dataset(
            tokenizer=tokenizer,
            max_seq_len=config.sequence_length
        )
        self.iter = iter(self.dataset)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Abstract method implementation

        Args:
            idx (Optioal[int]): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        if idx is None:
            # Batch padding sequence so the index does not matter
            #text, _ = self._query_document_sample_shuffle_indices(0)
            batch = next(self.iter)
            tokens = batch["tokens"]
            labels = batch["labels"]
        else:
            #text, _ = self._query_document_sample_shuffle_indices(idx)
            batch = self.dataset[idx]
            tokens = batch["tokens"]
            labels = batch["labels"]

        # text = torch.from_numpy(text).long()
        # if self.config.add_extra_token_to_sequence:
        #     tokens = text[:-1].contiguous()
        #     labels = text[1:].contiguous()
        # else:
        #     tokens = text
        #     labels = torch.roll(text, shifts=-1, dims=0)
        #     labels[-1] = self._pad_token_id
        tokens = torch.as_tensor(tokens, dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
        #         tokens,
        #         self.config.tokenizer.eod,
        #         self.config.reset_position_ids,
        #         self.config.reset_attention_mask,
        #         self.config.eod_mask_loss,
        #         self.config.create_attention_mask,
        #     )

        # # For padded sequences, mask the loss
        # loss_mask[labels == self._pad_token_id] = 0.0

        # # For padded sequences, ensure the embedding layer can map the token ID
        # tokens[tokens == self._pad_token_id] = 0
        # labels[labels == self._pad_token_id] = 0

        # Batch padding sequence so we mask the loss
        # if idx is None:
        #     loss_mask = torch.zeros_like(loss_mask)

        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": [],
                "loss_mask": [],
                "position_ids": [],
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": [],
                "position_ids": [],
            }

    def _query_document_sample_shuffle_indices(
        self, idx: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get the text (token ids) and document ids for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The text ids and document ids
        """
        
        samples, ids = self.get_sample_and_positionid(idx)

        return samples, ids
    
    def __build_shuffle_indices(self):
        self.indices = list(range(self.length))

        random.shuffle(self.indices)
        
    def get_sample_and_positionid(self, idx:int):
        document_ids = []
        shuffle_id = self.indices[idx % self.length]
        document_ids.append(shuffle_id)
        tokens_seq = self.dataset.get(idx=shuffle_id)
        if len(tokens_seq) > self.config.sequence_length:
            tokens_seq = tokens_seq[:self.config.sequence_length+1]

        return numpy.array(tokens_seq, dtype=numpy.int64), numpy.array(document_ids, dtype=numpy.int64)
    
    
