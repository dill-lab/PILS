from typing import Any, Dict

import torch
import torch.nn as nn
import transformers

EMBEDDER_MODEL_NAMES = [
    "llama2-random_k-alr",
    "llama2-random_k-clr",
    "llama2_chat-random_k-alr",
    "llama3_chat-random_k-alr",
]


FREEZE_STRATEGIES = ["decoder", "encoder_and_decoder", "encoder", "none"]
EMBEDDING_TRANSFORM_STRATEGIES = ["repeat"]


def get_device():
    """
    Function that checks
    for GPU availability and returns
    the appropriate device.
    :return: torch.device
    """
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    device = torch.device(dev)
    return device


device = get_device()


def disable_dropout(model: nn.Module):
    dropout_modules = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for m in dropout_modules:
        m.p = 0.0
    print(
        f"Disabled {len(dropout_modules)} "
        f"dropout modules from model type {type(model)}"
    )


def freeze_params(model: nn.Module):
    total_num_params = 0
    for name, params in model.named_parameters():
        params.requires_grad = False
        total_num_params += params.numel()
    # print(f"Froze {total_num_params} params from model type {type(model)}")


def mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


def max_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.max(dim=1).values
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


def stack_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.reshape((B, S * D))  # stack along seq length
    assert pooled_outputs.shape == (B, S * D)
    return pooled_outputs


def load_embedder_and_tokenizer(
    name: str,
    torch_dtype: str,
    use_hidden_states: bool = False,
    **kwargs,
):
    # TODO make abstract/argparse for it etc.
    # name = "gpt2" #### <--- TEMP. For debugging. Delete!
    if use_hidden_states:
        if name == "llama2-random_k-alr":
            from pils.embedders.embeddings import Llama2RandomKALREmbedder

            model = Llama2RandomKALREmbedder(
                max_length=kwargs["max_length"],
                max_new_tokens=kwargs["max_new_tokens"],
                extra_tokens=kwargs["extra_tokens"],
                torch_dtype=torch_dtype,
            )
            tokenizer = model.tokenizer
        elif name == "llama2_chat-random_k-alr":
            from pils.embedders.embeddings import Llama2ChatRandomKALREmbedder

            model = Llama2ChatRandomKALREmbedder(
                max_length=kwargs["max_length"],
                max_new_tokens=kwargs["max_new_tokens"],
                extra_tokens=kwargs["extra_tokens"],
                torch_dtype=torch_dtype,
            )
            tokenizer = model.tokenizer
        elif name == "llama3_chat-random_k-alr":
            from pils.embedders.embeddings import Llama3ChatRandomKALREmbedder

            model = Llama3ChatRandomKALREmbedder(
                max_length=kwargs["max_length"],
                max_new_tokens=kwargs["max_new_tokens"],
                extra_tokens=kwargs["extra_tokens"],
                torch_dtype=torch_dtype,
            )
            tokenizer = model.tokenizer

        else:
            raise Exception(f"hidden states are not supported for {name}")
        return model, tokenizer
    # model = torch.compile(model)
    return model, tokenizer


def load_encoder_decoder(
    model_name: str, lora: bool = False
) -> transformers.AutoModelForSeq2SeqLM:
    model_kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
    }
    if lora:
        model_kwargs.update(
            {
                "load_in_8bit": True,
                "device_map": "auto",
            }
        )
    return transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_name, **model_kwargs
    )


def load_tokenizer(name: str, max_length: int) -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        name,
        padding="max_length",
        truncation="max_length",
        max_length=max_length,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Disable super annoying warning:
    # https://github.com/huggingface/transformers/issues/22638
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    return tokenizer
