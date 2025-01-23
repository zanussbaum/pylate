from torch.utils.data import BatchSampler, DataLoader, Dataset, ConcatDataset
import torch
from typing import Iterator, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformerTrainer


class SingleDatasetBatchSampler(BatchSampler):
    """
    A batch sampler that samples from a single dataset per batch and handles distribution across GPUs.

    Args:
        datasets (List[Dataset]): List of datasets to sample from
        batch_size (int): Global batch size (will be divided across GPUs)
        drop_last (bool): Whether to drop the last incomplete batch
        generator (Optional[torch.Generator]): Random number generator
    """

    def __init__(
        self,
        datasets: List[Dataset],
        global_batch_size: int,
        drop_last: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        self.datasets = datasets
        self.global_batch_size = global_batch_size
        self.drop_last = drop_last
        self.generator = generator or torch.Generator()

        # Calculate dataset sizes and create index mappings
        self.dataset_sizes = [len(dataset) for dataset in datasets]
        self.cumsum_sizes = np.cumsum([0] + self.dataset_sizes).tolist()
        self.total_size = sum(self.dataset_sizes)

        # Create shuffled indices for each dataset
        self.indices_per_dataset = [
            torch.randperm(size, generator=self.generator).tolist()
            for size in self.dataset_sizes
        ]
        self.current_positions = [0] * len(datasets)

    def __iter__(self) -> Iterator[List[int]]:
        while True:
            # Randomly select a dataset
            dataset_idx = torch.randint(
                len(self.datasets), size=(1,), generator=self.generator
            ).item()

            # Get indices for the current dataset
            dataset_indices = self.indices_per_dataset[dataset_idx]
            current_pos = self.current_positions[dataset_idx]

            # Check if we need to reshuffle
            if current_pos + self.global_batch_size > len(dataset_indices):
                if not self.drop_last and current_pos < len(dataset_indices):
                    # Yield remaining indices if we don't drop last
                    yield [
                        idx + self.cumsum_sizes[dataset_idx]
                        for idx in dataset_indices[current_pos:]
                    ]

                # Reshuffle indices
                self.indices_per_dataset[dataset_idx] = torch.randperm(
                    self.dataset_sizes[dataset_idx], generator=self.generator
                ).tolist()
                dataset_indices = self.indices_per_dataset[dataset_idx]
                current_pos = 0

            # Get batch indices
            batch_indices = [
                idx + self.cumsum_sizes[dataset_idx]
                for idx in dataset_indices[
                    current_pos : current_pos + self.global_batch_size
                ]
            ]

            # Update position
            self.current_positions[dataset_idx] = current_pos + self.global_batch_size

            yield batch_indices

    @property
    def batch_size(self) -> int:
        return self.global_batch_size

    def __len__(self) -> int:
        if self.drop_last:
            return sum(size // self.global_batch_size for size in self.dataset_sizes)
        else:
            return sum(
                (size + self.global_batch_size - 1) // self.global_batch_size
                for size in self.dataset_sizes
            )


class ContrastiveDistributedTrainer(SentenceTransformerTrainer):
    """
    A trainer that samples from a single dataset per batch and distributes data across GPUs.
    """

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training DataLoader with SingleDatasetBatchSampler.
        """
        if self.train_dataset is None:
            raise ValueError("Training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        generator = torch.Generator()
        if self.args.seed:
            generator.manual_seed(self.args.seed)

        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }

        if isinstance(train_dataset, dict):
            # Convert dictionary of datasets to list
            datasets = list(train_dataset.values())
            train_dataset = ConcatDataset(datasets)

            batch_sampler = SingleDatasetBatchSampler(
                datasets=datasets,
                global_batch_size=self.args.train_batch_size,
                drop_last=self.args.dataloader_drop_last,
                generator=generator,
            )
            dataloader_params["batch_sampler"] = batch_sampler

        else:
            # Single dataset case
            batch_sampler = SingleDatasetBatchSampler(
                datasets=[train_dataset],
                batch_size=self.args.train_batch_size,
                drop_last=self.args.dataloader_drop_last,
                generator=generator,
            )
            dataloader_params["batch_sampler"] = batch_sampler

        self.accelerator.even_batches = False
        self._train_dataloader = self.accelerator.prepare(
            DataLoader(train_dataset, **dataloader_params)
        )
        return self._train_dataloader
