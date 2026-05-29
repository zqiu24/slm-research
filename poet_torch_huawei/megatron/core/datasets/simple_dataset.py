# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import hashlib
import json
import logging
import os
from collections import OrderedDict
from typing import Any, Callable, Iterable, List, Optional, Type, Union, Tuple, Dict

import numpy
import torch

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, _PAD_TOKEN_ID, _get_ltor_masks_and_position_ids
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.object_storage_utils import ObjectStorageConfig, is_object_storage_path
from megatron.core.datasets.utils import Split, normalize
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


class SimpleBlendedDataset(torch.utils.data.Dataset):
    """Simple concatenating class for a set of MegatronDataset instances

    This class does not shuffle the order of data files. Instead, it reads each file
    in the order passed in. For __getitem__, the dataset_id and dataset_sample_id
    can be obtained via simple integer division and modulo operations.

    Args:
        datasets (List[MegatronDataset]): The MegatronDataset instances to blend

        config (BlendedMegatronDatasetConfig): The config
    """

    def __init__(
        self,
        datasets: List[MegatronDataset],
        weights: List[Union[int, float]],
        size: Optional[int],
        config: BlendedMegatronDatasetConfig,
    ) -> None:
        self.datasets = datasets
        self.split = self.datasets[0].index_split
        self.config = config

        # Build cumulative size array for fast lookup
        self.dataset_sizes = [len(dataset) for dataset in self.datasets]
        self.dataset_cumsum = numpy.cumsum([0] + self.dataset_sizes)
        self.total_size = self.dataset_cumsum[-1]

        unique_identifiers = OrderedDict()
        unique_identifiers["class"] = type(self).__name__
        unique_identifiers["datasets"] = [dataset.unique_identifiers for dataset in self.datasets]
        unique_identifiers["split"] = self.split.name

        self.unique_description = json.dumps(
            unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers
        )
        self.unique_description_hash = hashlib.md5(
            self.unique_description.encode("utf-8"), usedforsecurity=False
        ).hexdigest()

        log_single_rank(
            logger, logging.INFO, f"> SimpleBlendedDataset total samples: {self.total_size}"
        )
        for i, size in enumerate(self.dataset_sizes):
            log_single_rank(
                logger, logging.INFO, f"> dataset {i} size: {size}"
            )

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> Dict[str, Union[int, numpy.ndarray]]:
        """Get item using simple integer division and modulo

        Args:
            idx (int): The global index into the blended dataset

        Returns:
            Dict[str, Union[int, numpy.ndarray]]: The sample data
        """
        # Find which dataset this index belongs to
        dataset_id = self._get_dataset_id(idx)
        sample_id_in_dataset = idx - self.dataset_cumsum[dataset_id]
        return {"dataset_id": dataset_id, **self.datasets[dataset_id][sample_id_in_dataset]}

    def _get_dataset_id(self, idx: int) -> int:
        """Get the dataset id for a global index

        Args:
            idx (int): The global index

        Returns:
            int: The dataset id
        """
        # Use numpy searchsorted for O(log n) lookup
        return numpy.searchsorted(self.dataset_cumsum, idx, side='right') - 1


class SimpleGPTDataset(MegatronDataset):
    """Simple GPT dataset that reads all data and checks sequence length

    This class reads all sequences and verifies that each sequence has length
    equal to config.sequence_length. It returns data directly by index without
    complex shuffling or blending.

    Args:
        indexed_dataset (IndexedDataset): The IndexedDataset around which to build the SimpleGPTDataset

        dataset_path (Optional[str]): The real path on disk to the dataset, for bookkeeping

        indices (numpy.ndarray): The set of the sequence indices to expose (optional)

        config (GPTDatasetConfig): The config
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: Optional[str],
        indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(
            indexed_dataset, dataset_path, indices, num_samples, index_split, config
        )

        try:
            self._pad_token_id = self.config.tokenizer.pad
        except Exception:
            self._pad_token_id = _PAD_TOKEN_ID

        # Verify sequence lengths
        self._verify_sequence_lengths()

        # Pre-compute which indices are cacheable
        self.masks_and_position_ids_are_cacheable = not any(
            [
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
            ]
        )
        self.masks_and_position_ids_are_cached = False
        self.cached_attention_mask = None
        self.cached_loss_mask = None
        self.cached_position_ids = None

    def _verify_sequence_lengths(self):
        """Verify that all sequences have the expected length"""
        expected_sequence_length = self.config.sequence_length+1
        sequence_lengths = self.dataset.sequence_lengths
        if not numpy.all(sequence_lengths == expected_sequence_length):
            # Find sequences with non-matching lengths
            mask = sequence_lengths != expected_sequence_length
            mismatched_indices = numpy.where(mask)[0]
            mismatched_lengths = sequence_lengths[mask]

            # Log details
            log_single_rank(
                logger,
                logging.WARNING,
                f"Some sequences in SimpleGPTDataset do not match sequence_length {expected_sequence_length}"
            )
            log_single_rank(
                logger,
                logging.WARNING,
                f"Found {mismatched_indices.shape[0]} mismatched sequences"
            )

            if mismatched_indices.shape[0] <= 10:
                for idx, length in zip(mismatched_indices, mismatched_lengths):
                    log_single_rank(
                        logger,
                        logging.WARNING,
                        f"  Sequence index {idx}: length {length} (expected {expected_sequence_length})"
                    )
            else:
                for idx, length in zip(mismatched_indices[:5], mismatched_lengths[:5]):
                    log_single_rank(
                        logger,
                        logging.WARNING,
                        f"  Sequence index {idx}: length {length} (expected {expected_sequence_length})"
                    )
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"  ... and {mismatched_indices.shape[0] - 5} more"
                )

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: IndexedDataset) -> int:
        """Return the number of sequences in the underlying IndexedDataset

        Args:
            low_level_dataset (IndexedDataset): The underlying IndexedDataset

        Returns:
            int: The number of unique sequences
        """
        return low_level_dataset.sequence_lengths.shape[0]

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> IndexedDataset:
        """Build the low level dataset

        Args:
            dataset_path (str): The real path prefix to the IndexedDataset .bin and .idx files

            config (GPTDatasetConfig): The config

        Returns:
            IndexedDataset: The underlying IndexedDataset
        """
        if is_object_storage_path(dataset_path):
            assert config.object_storage_cache_path is not None
            return IndexedDataset(
                dataset_path,
                multimodal=False,
                mmap=config.mmap_bin_files,
                object_storage_config=ObjectStorageConfig(
                    path_to_idx_cache=config.object_storage_cache_path
                ),
            )
        return IndexedDataset(dataset_path, multimodal=False, mmap=config.mmap_bin_files)

    def __len__(self) -> int:
        """Return the number of samples in the dataset

        Returns:
            int: The length of the dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample directly by index

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        # The index equals the actual index in the IndexedDataset
        actual_idx = idx

        # Get the sequence from the indexed dataset
        text = self.dataset[actual_idx]
        text = torch.from_numpy(text.copy()).long()

        tokens = text[:-1].contiguous()
        labels = text[1:].contiguous()

        if (
            not self.masks_and_position_ids_are_cacheable
            or not self.masks_and_position_ids_are_cached
        ):
            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                tokens,
                self.config.tokenizer.eod,
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
                self.config.create_attention_mask,
            )
            if self.masks_and_position_ids_are_cacheable:
                self.cached_attention_mask = attention_mask
                self.cached_loss_mask = loss_mask
                self.cached_position_ids = position_ids
                self.masks_and_position_ids_are_cached = True
        else:
            attention_mask = self.cached_attention_mask
            loss_mask = self.cached_loss_mask
            position_ids = self.cached_position_ids

        # For padded sequences, mask the loss
        loss_mask[labels == self._pad_token_id] = 0.0

        # For padded sequences, ensure the embedding layer can map the token ID
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

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


MidLevelDataset = MegatronDataset

TopLevelDataset = Union[SimpleBlendedDataset, MidLevelDataset]

DistributedDataset = Union[
    TopLevelDataset, MidLevelDataset, LowLevelDataset, torch.utils.data.Dataset
]


class SimpleBlendedMegatronDatasetBuilder(BlendedMegatronDatasetBuilder):
    def __init__(
        self,
        cls: Type[MidLevelDataset],
        sizes: List[int],
        is_built_on_rank: Callable,
        config: BlendedMegatronDatasetConfig,
    ):
        super().__init__(
            cls, sizes, is_built_on_rank, config
        )

    def _build_blended_dataset_splits(self) -> List[Optional[TopLevelDataset]]:
        """Build all dataset splits according to the provided blend(s)

        See the BlendedMegatronDatasetBuilder.build alias for more information.

        Returns:
            List[Optional[TopLevelDataset]]: A list containing a dataset instance (or None) per
                split
        """
        ##
        # Return fake "mock" datasets
        ##
        if self.config.mock:
            split = self.config.split_matrix
            try:
                return self._build_megatron_dataset_splits(None, split, self.sizes)
            except Exception as error:
                raise Exception(
                    f"{self.cls.__name__} failed to build as a mock data generator"
                ) from error

        ##
        # All splits come from the same distribution
        ##
        elif self.config.blend:
            prefixes, weights = self.config.blend

            # Use all datasets for training
            split = [(0, 1.0), None, None]

            # Blend consists of a single prefix
            if len(prefixes) == 1 and weights is None:
                return self._build_megatron_dataset_splits(prefixes[0], split, self.sizes)

            sizes_per_dataset_buffer = [[None for split in Split] for prefix in prefixes]

            # Build each dataset in parallel
            megatron_datasets = self._build_megatron_datasets_parallel(
                prefixes, split, sizes_per_dataset_buffer
            )

            # Build the top-level datasets
            blended_datasets = [None] * len(Split)
            for i in range(len(Split)):
                if split[i] is not None:
                    blended_datasets[i] = self.build_generic_dataset(
                        SimpleBlendedDataset,
                        self.is_built_on_rank,
                        True,  # synchronize_ranks, default behavior to build on rank-0 first
                        megatron_datasets[i],
                        None,
                        None,
                        self.config,
                    )

            return blended_datasets

        ##
        # Each split comes from a separate distribution
        ##
        else:
            blended_datasets = [None] * len(Split)
            for i in range(len(Split)):
                split_spoof = [None] * len(Split)
                split_spoof[i] = (0.0, 1.0)
                sizes_spoof = [0] * len(Split)
                sizes_spoof[i] = self.sizes[i]

                # Blend is provided for the split
                blend = self.config.blend_per_split[i]
                if blend is not None:
                    prefixes, weights = blend
                    if weights is not None:
                        weights = normalize(weights)

                    # Blend consists of a sigle prefix
                    if len(prefixes) == 1:
                        blended_datasets[i] = self._build_megatron_dataset_splits(
                            prefixes[0], split_spoof, sizes_spoof
                        )[i]
                        continue
                    elif self.config.multiple_validation_sets and i == Split.valid.value:
                        # handle multiple validation sets
                        validation_datasets = []
                        if self.config.full_validation:
                            # verify that size is None, which causes a single epoch dataset
                            # to be built
                            assert sizes_spoof[i] is None
                        for prefix in prefixes:
                            ds = self._build_megatron_dataset_splits(
                                prefix, split_spoof, sizes_spoof
                            )[i]
                            validation_datasets.append(ds)
                        blended_datasets[i] = validation_datasets
                        continue

                    sizes_per_dataset_buffer = [
                        [None for split in Split] for prefix in prefixes
                    ]

                    # Build each dataset in parallel
                    megatron_datasets = self._build_megatron_datasets_parallel(
                        prefixes, split_spoof, sizes_per_dataset_buffer
                    )[i]

                    blended_datasets[i] = self.build_generic_dataset(
                        SimpleBlendedDataset,
                        self.is_built_on_rank,
                        True,  # synchronize_ranks, default behavior to build on rank-0 first
                        megatron_datasets,
                        None,
                        None,
                        self.config,
                    )

            return blended_datasets

    def _build_megatron_dataset_splits(
        self,
        dataset_path: Optional[str],
        split: List[float],
        sizes: List[int],
        synchronize_ranks: bool = True,
    ) -> List[Optional[MidLevelDataset]]:
        """Build each MidLevelDataset split from a single LowLevelDataset

        Args:
            dataset_path (Optional[str]): The path on disk which defines the underlying
                LowLevelDataset, or None for mock dataset classes

            split (List[Tuple[float, float]]): The dataset split matrix

            sizes (List[int]): The number of total samples to draw from each split

            synchronize_ranks (bool): Whether to call barrier for rank-0 / barrier / other-ranks
                behavior. Set to False when we enforce this behavior at higher level.

        Returns:
            List[Optional[MidLevelDataset]]: The MidLevelDataset (or None) per split
        """
        # short-cut if we are not building on this rank
        if torch.distributed.is_initialized() and not self.is_built_on_rank():
            for i in range(len(Split)):
                if split[i] is not None and synchronize_ranks:
                    torch.distributed.barrier()
            return [None] * len(Split)

        # Build the low level dataset
        low_level_dataset = self.cls.build_low_level_dataset(dataset_path, self.config)

        # Build the split indices for the low level dataset
        num_elements = self.cls.numel_low_level_dataset(low_level_dataset)

        # Build the mid level dataset
        mid_level_datasets = []
        for i, _split in enumerate(Split):
            if split[i] is None:
                mid_level_datasets.append(None)
            else:
                mid_level_datasets.append(
                    self.build_generic_dataset(
                        self.cls,
                        self.is_built_on_rank,
                        synchronize_ranks,
                        low_level_dataset,
                        dataset_path,
                        None,
                        None,
                        _split,
                        self.config,
                    )
                )

        return mid_level_datasets
