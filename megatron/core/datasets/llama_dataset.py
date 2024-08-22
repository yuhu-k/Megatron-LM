import logging
import os
import time
from typing import Dict, Optional, Tuple

import numpy
import torch

from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.utils import Split
from megatron.core.utils import log_single_rank
from megatron.training import get_args
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
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )

        args = get_args()
        self.leng = []
        self.batch_num = args.micro_batch_size
        self.max_seq_len = args.seq_length
        self.length = len(self.dataset)
        self.__build_shuffle_indices()

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Abstract method implementation

        Args:
            idx (Optioal[int]): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        if idx is None:
            # Batch padding sequence so the index does not matter
            text, _ = self._query_document_sample_shuffle_indices(0)
        else:
            text, _ = self._query_document_sample_shuffle_indices(idx)

        text = torch.from_numpy(text).long()
        if self.config.add_extra_token_to_sequence:
            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()
        else:
            tokens = text
            labels = torch.roll(text, shifts=-1, dims=0)
            labels[-1] = self._pad_token_id
        attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                tokens,
                self.config.tokenizer.eod,
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
                self.config.create_attention_mask,
            )
        # if (
        #     not self.masks_and_position_ids_are_cacheable
        #     or not self.masks_and_position_ids_are_cached
        # ):
        #     attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
        #         tokens,
        #         self.config.tokenizer.eod,
        #         self.config.reset_position_ids,
        #         self.config.reset_attention_mask,
        #         self.config.eod_mask_loss,
        #         self.config.create_attention_mask,
        #     )
        #     if self.masks_and_position_ids_are_cacheable:
        #         self.cached_attention_mask = attention_mask
        #         self.cached_loss_mask = loss_mask
        #         self.cached_position_ids = position_ids
        #         self.masks_and_position_ids_are_cached = True
        # else:
        #     attention_mask = self.cached_attention_mask
        #     loss_mask = self.cached_loss_mask
        #     position_ids = self.cached_position_ids
        
        # For padded sequences, mask the loss
        loss_mask[labels == self._pad_token_id] = 0.0

        # For padded sequences, ensure the embedding layer can map the token ID
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        # Batch padding sequence so we mask the loss
        if idx is None:
            loss_mask = torch.zeros_like(loss_mask)

        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }
    def __query_document_sample_shuffle_indices_no_leng(
        self, idx:int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get the text (token ids) and document ids for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The text ids and document ids
        """
        # Do the shuffle mapping
        idx = self.shuffle_index[idx]

        # Get the beginning and end documents and offsets
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        document_ids = []
        sample_parts = []

        # Sample spans a single document
        if doc_index_beg == doc_index_end:
            # Add the document id
            document_ids.append(self.document_index[doc_index_beg])

            # Add the entire sample
            sample_parts.append(
                self.dataset.get(
                    self.document_index[doc_index_beg],
                    offset=doc_index_beg_offset,
                    length=doc_index_end_offset
                    - doc_index_beg_offset
                    + self.config.add_extra_token_to_sequence,
                )
            )

        # Sample spans multiple documents
        else:
            for i in range(doc_index_beg, doc_index_end + 1):
                # Add the document id
                document_ids.append(self.document_index[i])

                # Add the sample part
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = (
                    None
                    if i < doc_index_end
                    else doc_index_end_offset + self.config.add_extra_token_to_sequence
                )
                sample_parts.append(
                    self.dataset.get(self.document_index[i], offset=offset, length=length)
                )
        assert len(document_ids) == len(
            sample_parts
        ), f"len(document_ids) ({len(document_ids)}) != len(sample_parts) ({len(sample_parts)})"

        # length = sum(map(len, sample_parts))

        # # Pad the sample if necessary
        # if length < (self.config.sequence_length + self.config.add_extra_token_to_sequence):
        #     sample_parts.append(
        #         [self._pad_token_id]
        #         * (self.config.sequence_length + self.config.add_extra_token_to_sequence - length)
        #     )
    
        # l = len(max(sample_parts, key=len))
        # for i in range(len(sample_parts)):
        #     l2 = len(sample_parts[i])
        #     if l2 < l:
        #         padding = numpy.zeros((l - l2,) + sample_parts[i].shape[1:], dtype=sample_parts[i].dtype)
        #         #sample_parts[i].extend([self._pad_token_id]*(l-len(sample_parts[i])))
        #         sample_parts[i] = numpy.concatenate([sample_parts[i], padding], axis=0)
        import random

        # 從範圍 [1, 10] 中隨機選取一個整數
        #random_integer = random.randint(0, len(sample_parts)-1)
        random_integer = len(sample_parts)-1
        return (
            numpy.concatenate(sample_parts[random_integer:], dtype=numpy.int64),
            numpy.array(document_ids[random_integer:], dtype=numpy.int64),
        )
        # return (
        #     numpy.concatenate(sample_parts, dtype=numpy.int64),
        #     numpy.array(document_ids, dtype=numpy.int64),
        # )
    def _query_document_sample_shuffle_indices(
        self, idx: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get the text (token ids) and document ids for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The text ids and document ids
        """
        
        if not(len(self.leng) > int(idx/self.batch_num) and self.leng[int(idx/self.batch_num)] != None):
            while len(self.leng) <= int(idx/self.batch_num):
                self.leng.append(None)
                
            base = int(idx/self.batch_num)*self.batch_num
            for i in range(base, base+self.batch_num):
                #sample, _ = self.__query_document_sample_shuffle_indices_no_leng(i)
                sample, _ = self.get_sample_and_positionid(i)
                if self.leng[int(idx/self.batch_num)] == None or len(sample) > self.leng[int(idx/self.batch_num)]:
                    self.leng[int(idx/self.batch_num)] = len(sample)
                    
        l = self.leng[int(idx/self.batch_num)]
        samples, ids = self.get_sample_and_positionid(idx)
        if len(samples) < l:
            tmp = [self._pad_token_id] * (l - len(samples))
            samples = numpy.concatenate([samples, tmp], dtype=numpy.int64)
            ids = numpy.concatenate([ids, tmp], dtype=numpy.int64)
        return samples, ids
    
    def __build_shuffle_indices(self):
        self.indices = list(range(self.length))
        #random.shuffle(self.indices)
        
    def get_sample_and_positionid(self, idx:int):
        document_ids = []
        shuffle_id = self.indices[idx]
        document_ids.append(shuffle_id)
        tokens_seq = self.dataset.get(idx=shuffle_id)
        if len(tokens_seq) > self.max_seq_len:
            tokens_seq = tokens_seq[:self.max_seq_len]
        return numpy.array(tokens_seq, dtype=numpy.int64), numpy.array(document_ids, dtype=numpy.int64)