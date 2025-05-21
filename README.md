# pils

To train:
```bash
python pils/run.py --per_device_train_batch_size 250\
           --per_device_eval_batch_size 250\
           --max_seq_length 64\
           --num_train_epochs 100\
           --max_eval_samples 1000\
           --eval_steps 250\
           --warmup_steps 250\
           --learning_rate 0.0002\
           --dataset_name one_million_instructions\
           --logging_steps 32\
           --model_name_or_path t5-base\
           --use_wandb=1\
           --experiment inversion_from_hidden_states\
           --bf16=1\
           --embedder_torch_dtype bfloat16\
           --lr_scheduler_type constant_with_warmup\
           --use_frozen_embeddings_as_input 1 --mock_embedder 1\
           --embedder_model_name llama2_chat-random_k-alr\
           --max_new_tokens 16\
           --output_dir /path/to/save/dir/\
           --exp_group_name llama2-chat\
           --extra_tokens 100
```
