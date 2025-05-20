import copy
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers
from sentence_transformers import SentenceTransformer

from pils.models.config import InversionConfig
from pils.models.model_utils import (
    FREEZE_STRATEGIES,
    disable_dropout,
    freeze_params,
    load_embedder_and_tokenizer,
    load_encoder_decoder,
    load_tokenizer,
    mean_pool,
)
logger = logging.getLogger(__name__)


class InversionModel(transformers.PreTrainedModel):
    config_class = InversionConfig
    embedder: nn.Module
    embedder_tokenizer: transformers.PreTrainedTokenizer  # embedder's tokenizer
    encoder_decoder: transformers.AutoModelForSeq2SeqLM
    encoder_decoder_lora: bool  # Whether to use LoRA for the encoder-decoder model
    tokenizer: transformers.PreTrainedTokenizer  # encoder_decoder's tokenizer
    embedding_transform: nn.Module  # Module that transformers embedder output into encoder-decoder input
    bottleneck_dim: int  # Bottleneck dimension for embedding_transform
    embedder_dim: int  # Hidden dimension of embedding model
    use_frozen_embeddings_as_input: bool  # Whether to train/evaluate on frozen embeddings

    def __init__(self, config: InversionConfig):
        super().__init__(config=config)

        use_frozen_embeddings_as_input = config.use_frozen_embeddings_as_input

        encoder_decoder = load_encoder_decoder(
            model_name=config.model_name_or_path,
            lora=config.use_lora,
        )

        embedder, embedder_tokenizer = self.load_embedder_and_tokenizer(config)

        tokenizer = load_tokenizer(
            config.model_name_or_path,
            max_length=config.max_seq_length,
        )

        self.encoder_decoder = encoder_decoder  # .to_bettertransformer()
        ######################################################

        self.embedder_is_decoder = False

        if isinstance(embedder, SentenceTransformer):
            self.embedder_dim = embedder.get_sentence_embedding_dimension()
            bottleneck_dim = self.embedder_dim
        else:
            self.embedder_dim = embedder.config.hidden_size
            bottleneck_dim = self.embedder_dim
        self.use_frozen_embeddings_as_input = use_frozen_embeddings_as_input
        self.bottleneck_dim = bottleneck_dim
        ######################################################
        self.tokenizer = tokenizer
        self.embedder = embedder
        for param in self.embedder.parameters():
            param.requires_grad = False

        self.embedder.eval()

        self.embedder_tokenizer = embedder_tokenizer
        # self.freeze(freeze_strategy=config.freeze_strategy)

    @property
    def embedder_device(self) -> torch.device:
        return next(self.embedder.parameters()).device

