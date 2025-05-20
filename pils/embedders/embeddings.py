from abc import ABC
from types import SimpleNamespace
from typing import List, Literal
import warnings

import torch
from torch import nn
import transformers
from transformers import AutoTokenizer


class Embedder(nn.Module):
    def __init__(self, max_length: int, max_new_tokens: int):
        super(Embedder, self).__init__()

        self.model, self.tokenizer = self.load_model_and_tokenizer()
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

    def train(self, mode):
        warnings.warn(
            "Tried to set a mode. This model is permanently set in eval mode")
        return super().train(mode=False)

    def __call__(self, *args, **kwargs):
        return self.get_hidden_states(*args, **kwargs)


class TransformedHiddenStateEmbedder(Embedder, ABC):

    def extract_hidden_state_from_logprobs(self, logprobs):
        raise NotImplementedError

    def get_hidden_states(self, embedder_input_ids, embedder_attention_mask):
        logprobs, chosen_tokens = self.get_logprobs(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
        )
        return {"embeddings": self.extract_hidden_state_from_logprobs(logprobs),
                "chosen_tokens": chosen_tokens}

    def get_logprobs(self, embedder_input_ids, embedder_attention_mask):
        device = next(self.model.parameters()).device
        embedder_input_ids = embedder_input_ids.to(device)
        embedder_attention_mask = embedder_attention_mask.to(device)
        output = self.model.generate(
            input_ids=embedder_input_ids,
            attention_mask=embedder_attention_mask,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=1,
            top_p=None,
            pad_token_id=self.tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True,
        )

        ##!!  this part is usually in lms and not in embedder.
        logits = torch.cat([i.unsqueeze(1) for i in output.scores], dim=1)
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        chosen_tokens = output.sequences[:, -
                                         self.max_new_tokens:].unsqueeze(-1)
        return logprobs, chosen_tokens


class RandomTransformALREmbedder(TransformedHiddenStateEmbedder, ABC):

    def extract_hidden_state_from_logprobs(self, logprobs):
        alr = logprobs[:, :, 1:] - logprobs[:, :, 0:1]  # B, T, V
        if not hasattr(self, "transform"):
            g = torch.Generator()
            g.manual_seed(666)
            self.transform = torch.randn(
                self.model.config.vocab_size - 1,
                self.config.hidden_size,
                generator=g,
            ).to(logprobs.device)
        hidden_states = alr @ self.transform  # B, T, D

        return hidden_states


class Llama2KTokensEmbedder(TransformedHiddenStateEmbedder, ABC):
    def __init__(
        self,
        max_length: int,
        max_new_tokens: int,
        extra_tokens: int,
        torch_dtype: Literal["float32", "float16", "bfloat16"],
    ):
        self.torch_dtype = torch_dtype
        super(Llama2KTokensEmbedder, self).__init__(
            max_length=max_length, max_new_tokens=max_new_tokens
        )
        assert extra_tokens >= 0
        self.extra_tokens = extra_tokens

        self.config = SimpleNamespace(
            hidden_size=self.model.config.hidden_size + extra_tokens
        )

    def load_model_and_tokenizer(self):

        if self.torch_dtype == "float32":
            self.torch_dtype = torch.float32
        elif self.torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16

        # bnb_config = transformers.BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=self.torch_dtype,
            # quantization_config=bnb_config,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return model, tokenizer


class Llama2RandomKALREmbedder(Llama2KTokensEmbedder):

    def extract_hidden_state_from_logprobs(self, logprobs):

        if not hasattr(self, "chosen_tokens"):
            g = torch.Generator()
            g.manual_seed(666)
            self.chosen_tokens = torch.randperm(
                self.model.config.vocab_size,
                generator=g,
            )[
                : self.config.hidden_size + 1
            ]  # alr will remove one

        logprobs = logprobs[:, :, self.chosen_tokens]
        alr = logprobs[:, :, 1:] - logprobs[:, :, 0:1]  # B, T, V
        return alr


class Llama2ChatRandomKALREmbedder(Llama2KTokensEmbedder):

    def load_model_and_tokenizer(self):

        if self.torch_dtype == "float32":
            self.torch_dtype = torch.float32
        elif self.torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16

        # bnb_config = transformers.BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            torch_dtype=self.torch_dtype,
            # quantization_config=bnb_config,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return model, tokenizer

    def extract_hidden_state_from_logprobs(self, logprobs):

        if not hasattr(self, "chosen_tokens"):
            g = torch.Generator()
            g.manual_seed(666)
            self.chosen_tokens = torch.randperm(
                self.model.config.vocab_size,
                generator=g,
            )[
                : self.config.hidden_size + 1
            ]  # alr will remove one

        logprobs = logprobs[:, :, self.chosen_tokens]
        alr = logprobs[:, :, 1:] - logprobs[:, :, 0:1]  # B, T, V
        return alr


class Llama3ChatRandomKALREmbedder(Llama2KTokensEmbedder):

    def load_model_and_tokenizer(self):

        if self.torch_dtype == "float32":
            self.torch_dtype = torch.float32
        elif self.torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16

        # bnb_config = transformers.BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=self.torch_dtype,
            # quantization_config=bnb_config,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return model, tokenizer

    def extract_hidden_state_from_logprobs(self, logprobs):

        if not hasattr(self, "chosen_tokens"):
            g = torch.Generator()
            g.manual_seed(666)
            self.chosen_tokens = torch.randperm(
                self.model.config.vocab_size,
                generator=g,
            )[
                : self.config.hidden_size + 1
            ]  # alr will remove one

        logprobs = logprobs[:, :, self.chosen_tokens]
        alr = logprobs[:, :, 1:] - logprobs[:, :, 0:1]  # B, T, V
        return alr

