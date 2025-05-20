import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING

from vec2text.models import (
    EMBEDDER_MODEL_NAMES,
    EMBEDDING_TRANSFORM_STRATEGIES,
    FREEZE_STRATEGIES,
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

DATASET_NAMES = [
    "one_million_instructions",
    "awesomegpt_prompts",
    "real_gpts"
]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        ###
        # huggingface.co/facebook/dpr-ctx_encoder-single-nq-base
        ###
        default="t5-base",
        metadata={
            "help": (
                "The model checkpoint for weights initialization .Don't set if you want to train a model from scratch."
            )
        },
    )
    embedder_model_name: str = field(
        ###
        # huggingface.co/facebook/dpr-ctx_encoder-single-nq-base
        ###
        default="gtr_base",
        metadata={
            "help": "Model to get embeddings from (locally)",
            "choices": EMBEDDER_MODEL_NAMES,
        },
    )
    embedder_torch_dtype: str = field(
        default="float32",
        metadata={
            "help": "torch dtype of embedder",
            "choices": ["float32", "float16", "bfloat16"],
        },
    )
    encoder_dropout_disabled: bool = field(
        default=False, metadata={"help": "Disable dropout on T5 encoder"}
    )
    decoder_dropout_disabled: bool = field(
        default=False, metadata={"help": "Disable dropout on T5 decoder"}
    )

    max_seq_length: int = field(
        default=128, metadata={"help": "Maximum sequence length for tokenizer"}
    )
    max_new_tokens: int = field(
        default=42,
        metadata={"help": "Maximum new tokens to generate for hidden states"},
    )
    extra_tokens: int = field(
        default=-1, metadata={"help": "Extra tokens to sample in `random_k` embedders"}
    )
    hidden_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of tokens to use for logprobs. Defaults to embedder's hidden size"
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    use_frozen_embeddings_as_input: bool = field(
        default=False,
        metadata={
            "help": "Whether to pass a 'frozen_embedding' column and train on that instead of generating embeddings on-the-fly"
        },
    )
    pretrained_path: Optional[str] = field(
            default=None,
            metadata={
                "help": "checkpoint path to load model weights"
                })

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="msmarco",
        metadata={
            "choices": DATASET_NAMES,
            "help": "The name of the dataset to use (via the datasets library).",
        },
    )
    max_eval_samples: int = field(
        default=1000,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    use_less_data: int = field(
        default=-1,
        metadata={
            "help": {"Use a small amount of the training/eval data (for testing)"}
        },
    )

    def __post_init__(self):
        if self.dataset_name is None:
            raise ValueError("Need a dataset name.")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # https://github.com/huggingface/transformers/blob/e82c1cb78e178519060b9391214727be75a218ca/src/transformers/training_args.py#L121
    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Output directory for training saves. If not set, will output to saves/<random hash>."
        },
    )
    steps_per_epoch: int = field(
        default=500_000,
        metadata={"required": False, "help": "Size of pseudo-training set."},
    )
    num_train_epochs: float = field(
        default=30.0,
        metadata={"required": False, "help": "Number of epochs for training"},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW on the backbone model."},
    )
    use_wandb: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to log to Weights & Biases."}
    )
    report_to: str = "wandb"
    per_device_train_batch_size: int = field(
        default=128, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": ("Whether to use bf16 (mixed) precision instead of 32-bit.")},
    )
    # torch_compile: bool = True # for torch 2

    ##################### Experimental Settings ####################
    experiment: str = field(
        default="inversion",
        metadata={
            "required": False,
            "help": "Which experiment to run (defines model, loss func, dataset...) ",
            "choices": [
                "inversion",  # our model: projects and feeds to encoder-decoder
                "inversion_from_hidden_states",
            ],
        },
    )
    exp_name: str = field(
        default="",
        metadata={
            "required": False,
            "help": "Name to identify this specific run of an experiment",
        },
    )
    exp_group_name: str = field(
        default="",
        metadata={
            "required": False,
            "help": "Name to identify this sweep / series of experiments",
        },
    )

    # Need to *not* remove unused columns so we keep query_attention_mask, etc.
    # which huggingface doesn't think we need.
    remove_unused_columns: bool = False

    # Do evaluation and logging on certain num steps.
    evaluation_strategy: str = "steps"
    logging_strategy: str = "steps"
    save_strategy: str = "steps"

    save_total_limit: int = 2  # Maximum number of checkpoints to save.

    warmup_steps: int = field(
        default=4000, metadata={"help": "Number of steps of warmup"}
    )
    logging_steps: int = field(
        default=400, metadata={"help": "Number of steps between logging metrics"}
    )
    save_steps: int = field(
        default=4000,
        metadata={"help": "Number of steps per save"},
    )
    eval_steps: int = field(
        default=40000,
        metadata={
            "help": "Number of steps between eval (will be scaled as if batch size is 32)"
        },
    )
    mock_embedder: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, will delete the embedder and replace all embedder logits with"
                " zeros once training starts. You probably don't want to do this. But "
                " if you precomputed all the embeddings for train and val, this will"
                " work fine, except the embedding-based metrics (just cosine similarity"
                " I think) will be broken."
            )
        },
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DistributedDataParallel`."
            )
        },
    )

    include_inputs_for_metrics: bool = True

    def __setattr__(self, name, value):
        super(transformers.TrainingArguments, self).__setattr__(name, value)

    def __post_init__(self):
        super().__post_init__()
        self._frozen = True
        self.report_to = (
            ["wandb"] if (self.use_wandb and (self.local_rank <= 0)) else []
        )
        self.dataloader_pin_memory = True
        num_workers = torch.cuda.device_count()
        os.environ["RAYON_RS_NUM_CPUS"] = str(
            num_workers
        )  # Sets threads for hf tokenizers
        self.dataloader_num_workers = num_workers
        print(f"Set num workers to {num_workers}")

        self.dataloader_drop_last = False

        # Scale logging steps proportional to batch size.
        self.warmup_steps = round(self.warmup_steps * (32 / self.train_batch_size))
        self.logging_steps = round(self.logging_steps * (32 / self.train_batch_size))
        self.eval_steps = round(self.eval_steps * (32 / self.train_batch_size))
        self.save_steps = round(self.save_steps * (32 / self.train_batch_size))

        # defaults from SentenceTransformers
        # lr 2e-5
        self.adam_epsilon = 1e-6

        self.group_by_length = True
        self.length_column_name = "length"

        self.load_best_model_at_end = True
        self.greater_is_better = False

        self.do_eval = False
        # self.ddp_backend = "gloo"
