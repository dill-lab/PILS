import logging
import os
import random
from typing import Dict, List

import datasets
import torch

from pils.run_args import DataArguments
from pils.utils import dataset_map_multi_worker, get_num_proc

STATIC_USER_PROMPT = "You are?"

def retain_dataset_columns(
    d: datasets.Dataset, allowed_columns: List[str]
) -> datasets.Dataset:
    column_names_to_remove = [c for c in d.features if c not in allowed_columns]
    return d.remove_columns(column_names_to_remove)


def create_omi_ex(ex: Dict[str, str]) -> Dict[str, str]:
    ex["text"] = ex["user"]
    return ex


def create_ompi_ex(ex: Dict[str, str]) -> Dict[str, str]:
    ex["user"] = ex["user"].strip()
    ex["system"] = ex["system"].strip()
    ex["text"] = ex["system"] + "\n\n" + ex["user"]
    ex["prefix"] = ex["system"] + "\n\n"
    ex["suffix"] = ex["user"]
    return ex


def get_world_size() -> int:
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1

def load_one_million_instructions() -> datasets.Dataset:
    # has only "train" split, and "system" (system prompt)
    # and "user" (user input) columns
    dataset_dict = datasets.load_dataset("wentingzhao/one-million-instructions")
    dataset_dict = dataset_map_multi_worker(dataset_dict, create_ompi_ex)

    return dataset_dict["train"]


def load_awesomegpt_prompts() -> datasets.DatasetDict:
    custom_data_root = os.environ.get("CUSTOM_DATASET_ROOT")
    assert custom_data_root, "`awesomegpt_prompts` could not be loaded. Please set the environ variable `CUSTOM_DATASET_ROOT`"
    #train_ds = datasets.load_from_disk(os.path.join(custom_data_root, "train", "awesomegpt_prompts"))
    test_ds = datasets.load_from_disk(os.path.join(custom_data_root, "test", "awesomegpt_prompts"))
    train_ds = test_ds.select(range(50))
    test_ds = test_ds.select(range(50, len(test_ds)))
    dataset_dict = datasets.DatasetDict()
    dataset_dict["train"] = train_ds
    dataset_dict["validation"] = test_ds

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    def decode(sample):
        #sample["user"] = "Give me 16 short sentences that best describe yourself. Start with \"1:\""
        sample["user"] = STATIC_USER_PROMPT
        sample["system"] = tokenizer.decode([i for i in sample["system_prompt"] if i!= -100], skip_special_tokens=True)

        sample["text"] = sample["system"] + "\n\n" + sample["user"]
        sample["prefix"] = sample["system"] + "\n\n"
        sample["suffix"] = sample["user"]
        return sample

    return dataset_dict.map(decode)


def load_real_gpts() -> datasets.DatasetDict:
    custom_data_root = os.environ.get("CUSTOM_DATASET_ROOT")
    assert custom_data_root, "`awesomegpt_prompts` could not be loaded. Please set the environ variable `CUSTOM_DATASET_ROOT`"
    #train_ds = datasets.load_from_disk(os.path.join(custom_data_root, "train", "awesomegpt_prompts"))
    test_ds = datasets.load_from_disk(os.path.join(custom_data_root, "test", "real_gpts_arrow"))
    train_ds = test_ds.select(range(50))
    test_ds = test_ds.select(range(50, len(test_ds)))
    dataset_dict = datasets.DatasetDict()
    dataset_dict["train"] = train_ds
    dataset_dict["validation"] = test_ds

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    def decode(sample):
        #sample["user"] = "Give me 16 short sentences that best describe yourself. Start with \"1:\""
        sample["user"] = STATIC_USER_PROMPT
        sample["system"] = tokenizer.decode([i for i in sample["system_prompt"] if i!= -100], skip_special_tokens=True)

        sample["text"] = sample["system"] + "\n\n" + sample["user"]
        sample["prefix"] = sample["system"] + "\n\n"
        sample["suffix"] = sample["user"]
        return sample

    return dataset_dict.map(decode)


def load_anthropic_toxic_prompts() -> datasets.Dataset:
    d = datasets.load_dataset("wentingzhao/anthropic-hh-first-prompt")["train"]
    d = d.rename_column("user", "text")
    return d


def dataset_from_args(data_args: DataArguments) -> datasets.DatasetDict:
    """Loads a dataset from data_args create in `run_args`."""
    if data_args.dataset_name == "one_million_instructions":
        raw_datasets = load_one_million_instructions()
        raw_datasets = raw_datasets.train_test_split(test_size=0.01)
        raw_datasets["validation"] = raw_datasets["test"]
    elif data_args.dataset_name == "awesomegpt_prompts":
        raw_datasets = load_awesomegpt_prompts()
    elif data_args.dataset_name == "real_gpts":
        raw_datasets = load_real_gpts()
    else:
        raise ValueError(f"unsupported dataset {data_args.dataset_name}")
    return raw_datasets


def load_python_code_instructions_18k_alpaca() -> datasets.Dataset:
    d = datasets.load_dataset("iamtarun/python_code_instructions_18k_alpaca")["train"]
    d = d.rename_column("instruction", "text")
    return d

def load_standard_val_datasets() -> datasets.DatasetDict:
    """Loads a pre-defined set of standard val datasets."""
    d = {
        "anthropic_toxic_prompts": load_anthropic_toxic_prompts(),
        "python_code_alpaca": load_python_code_instructions_18k_alpaca(),
        "awesomegpt_prompts": load_awesomegpt_prompts()["validation"],
        "real_gpts": load_real_gpts()["validation"],
    }
    d = {k: retain_dataset_columns(v, ["text", "prefix", "suffix"]) for k, v in d.items()}

    return datasets.DatasetDict(d)
