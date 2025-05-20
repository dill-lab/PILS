import multiprocessing
import os
import shutil
from typing import Callable

import datasets
import torch

datasets.disable_caching()


def get_world_size() -> int:
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1


def get_num_proc() -> int:
    world_size: int = get_world_size()
    try:
        # os.sched_getaffinity respects schedulers, unlike cpu_count(), but it's only available
        # on some Unix platforms, so we support both!
        return len(os.sched_getaffinity(0)) // world_size  # type: ignore[attr-defined]
    except AttributeError:
        return multiprocessing.cpu_count() // world_size


def dataset_map_multi_worker(
    dataset: datasets.Dataset, map_fn: Callable, *args, **kwargs
) -> datasets.Dataset:

    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        kwargs["num_proc"] = kwargs.get("num_proc", get_num_proc())
    except (RuntimeError, ValueError):
        # In non-distributed mode, just run regular map()
        kwargs["num_proc"] = kwargs.get("num_proc", get_num_proc())
        return dataset.map(map_fn, *args, **kwargs)
    datasets.disable_caching()

    cache_path = os.environ.get(
        "PILS_CACHE", os.path.expanduser("~/.cache/inversion")
    )
    ds_shard_filepaths = [
        os.path.join(cache_path, f"{dataset._fingerprint}_subshard_{w}.cache")
        for w in range(0, world_size)
    ]
    print(f"\tworker {rank} saving sub-shard to {ds_shard_filepaths[rank]}")
    ds_shard = dataset.shard(
        num_shards=world_size,
        index=rank,
        contiguous=True,
    )
    ds_shard = ds_shard.map(map_fn, *args, **kwargs)
    ds_shard.save_to_disk(ds_shard_filepaths[rank])
    print("rank", rank, "saving:", ds_shard_filepaths[rank])
    torch.distributed.barrier()
    full_dataset = datasets.concatenate_datasets(
        [datasets.load_from_disk(p) for p in ds_shard_filepaths]
    )
    torch.distributed.barrier()
    print("rank", rank, "deleting:", ds_shard_filepaths[rank])
    shutil.rmtree(ds_shard_filepaths[rank])
    return full_dataset


class MockEmbedder:
    embedder_dim: int

    def __init__(self, embedder_dim: int):
        self.embedder_dim = embedder_dim

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], self.embedder_dim),
            dtype=torch.float32,
            device=input_ids.device,
        )

    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], self.embedder_dim),
            dtype=torch.float32,
            device=input_ids.device,
        )
