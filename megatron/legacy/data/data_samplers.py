# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Dataloaders."""

from typing import List, Tuple, Union
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from megatron.training import get_args
from megatron.core import mpu
from functools import partial
from megatron.core.datasets.llama_dataset import LLaMADataset
import torch.nn.functional as F
CROSS_ENTROPY_IGNORE_IDX = -100

# TokenPair is a pair (tuple) of two lists: tokenized text inputs and labels.


def build_pretraining_data_loader(dataset, consumed_samples):
    """Build dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()

    # Megatron sampler
    if args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding)
    elif args.dataloader_type == "external":
        # External dataloaders are passed through. User is expected to provide a
        # torch-compatible dataloader and define samplers, if needed.
        return dataset
    else:
        raise Exception('{} dataloader type is not supported.'.format(
                args.dataloader_type))

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       persistent_workers=True if args.num_workers > 0 else False,
                                       collate_fn=partial(
                                            padded_collate,
                                            padding_idx=0,
                                            config=dataset.config
                                        ) if type(dataset) == LLaMADataset else None ,
                                       )

class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class RandomSeedDataset(Dataset):

    def __init__(self, dataset):
        args = get_args()
        self.base_seed = args.seed
        self.curr_seed = args.seed
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.curr_seed = self.base_seed + epoch

    def __getitem__(self, idx):
        seed = idx + self.curr_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return self.dataset[idx]


class MegatronPretrainingRandomSampler:

    def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, data_sharding):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                           * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) \
                                * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = \
                torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.data_parallel_rank::self.data_parallel_size]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []

def padded_collate(
    batch: List[dict],
    padding_idx: int = 0,
    config = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors.

    Args:
        batch {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }: A list of tuples containing input, label pairs.
        padding_idx (int): Padding index for input ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.

    Returns:
        Collated input and label tensors.

    Example:
        >>> token_pairs = [
        >>>    ([1, 2, 3], [4, 5, 6]),
        >>>    ([7,], [10,],),
        >>> ]
        >>> inputs, labels = padded_collate(
        >>>    batch=token_pairs,
        >>>    padding_idx=padding_idx,
        >>>    ignore_idx=ignore_idx,
        >>> )
        >>> inputs
        >>> tensor([[1, 2, 3], [7, 0, 0]])
        >>> labels
        >>> tensor([[4,5,6], [10,-100,-100]])
    """
    import numpy
    def padding(inputs: list[torch.Tensor], padding_value: int):

        max_len = 0
        for input in inputs:
            max_len = len(input) if len(input) > max_len else max_len
        samples = []
        for input in inputs:
            if len(input) < max_len:
                tmp = torch.full((max_len - len(input),), padding_value)
                input = torch.concatenate([input, tmp],dim=0)
            samples.append(input)
        #result = torch.stack(samples)
        return samples

    def padding_atten(inputs: list[torch.Tensor], padding_value: int):

        max_len = 0
        for input in inputs:
            max_len = input.size()[1] if input.size()[1] > max_len else max_len
        samples = []
        for input in inputs:
            if input.size()[1] < max_len:
                # (left, right, top, bottom) 的 padding 設定
                padding = (0, max_len - input.size()[1], 0, max_len - input.size()[1])
                input = F.pad(input, padding, value = padding_value)
            samples.append(input)
        #result = torch.stack(samples)
        return samples
            
    input_ids = padding(
        [x["tokens"] for x in batch],
        padding_value=padding_idx,
    )
    labels = padding(
        [x["labels"] for x in batch],
        padding_value=padding_idx,
    )
    
    from torch.utils.data import _utils
    from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids
    tmp_list = []
    for i in range(len(batch)):
        attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
            input_ids[i],
            config.tokenizer.eod,
            config.reset_position_ids,
            config.reset_attention_mask,
            config.eod_mask_loss,
            config.create_attention_mask,
        )
        # For padded sequences, mask the loss
        loss_mask[labels == -1] = 0.0
        
        # For padded sequences, ensure the embedding layer can map the token ID
        input_ids[i][input_ids[i] == -1] = 0
        labels[i][labels == -1] = 0
        
        if batch[0].get("attention_mask") != None:
            tmp_list.append(
                {"tokens":input_ids[i], 
                "labels": labels[i],
                "attention_mask" : attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
                }
            )
        else:
            tmp_list.append(
                {"tokens":input_ids[i], 
                "labels": labels[i],
                "loss_mask": loss_mask,
                "position_ids": position_ids,
                }
            )
    return _utils.collate.default_collate(tmp_list)