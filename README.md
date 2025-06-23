# Better Language Model Inversion by Compactly Representing Next-Token Distributions

Code for our [paper](https://arxiv.org/abs/2506.17090)

## Requirements
We recommend using `python=3.11.11`

```bash
wget "https://zenodo.org/records/12759549/files/prompt2output_datasets.zip?download=1" -O prompt2output_datasets.zip
unzip prompt2output_datasets.zip
pip install .
```

## Training
```bash
export CUSTOM_DATASET_ROOT=$(pwd)/datasets/
python pils/run.py --per_device_train_batch_size 250\
           --per_device_eval_batch_size 250\
           --max_seq_length 64\
           --num_train_epochs 100\
           --max_eval_samples 1000\
           --eval_steps 250\
           --warmup_steps 250\
           --learning_rate 0.0002\
           --dataset_name one_million_instructions\
           --model_name_or_path t5-base\
           --use_wandb=1\
           --experiment inversion_from_hidden_states\
           --bf16=1\
           --embedder_torch_dtype bfloat16\
           --lr_scheduler_type constant_with_warmup\
           --use_frozen_embeddings_as_input 1 --mock_embedder 1\
           --embedder_model_name llama2_chat-random_k-alr\
           --max_new_tokens 32\
           --output_dir /path/to/save/dir/\
           --exp_group_name llama2-chat\
           --extra_tokens 100
```

Possible values for `embedder_model_name`:

- `llama2-random_k-alr`
- `llama2_chat-random_k-alr`
- `llama3_chat-random_k-alr`
 


## Finetuning
```bash
export CUSTOM_DATASET_ROOT=$(pwd)/datasets/
python pils/run.py --per_device_train_batch_size 50\
           --per_device_eval_batch_size 50\
           --max_seq_length 64\
           --num_train_epochs 100\
           --max_eval_samples 1000\
           --logging_steps 4\
           --eval_steps 10\
           --warmup_steps 1\
           --learning_rate 0.0001\
           --dataset_name awesomegpt_prompts\
           --model_name_or_path t5-base\
           --use_wandb=1\
           --experiment inversion_from_hidden_states\
           --bf16=1\
           --embedder_torch_dtype bfloat16\
           --lr_scheduler_type constant_with_warmup\
           --use_frozen_embeddings_as_input 1 --mock_embedder 1\
           --embedder_model_name llama2_chat-random_k-alr\
           --max_new_tokens 32\
           --output_dir /path/to/save/dir/\
           --exp_group_name llama2-chat\
           --extra_tokens 100\
           --pretrained_path /path/to/model
```

# Cite our work

```
@misc{nazir2025betterlanguagemodelinversion,
      title={Better Language Model Inversion by Compactly Representing Next-Token Distributions}, 
      author={Murtaza Nazir and Matthew Finlayson and John X. Morris and Xiang Ren and Swabha Swayamdipta},
      year={2025},
      eprint={2506.17090},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.17090}, 
}
```
