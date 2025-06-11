output_model_name="UniHGKR-7B-pretrained"
llm_pretrain_data="../../UniHGKR_training_data/UniHGKR_7b_pretrain_data.jsonl"

nohup torchrun --nproc_per_node 8 \
run.py \
--output_dir ./${output_model_name} \
--model_name_or_path BAAI/LLARA-pretrain \
--train_data ${llm_pretrain_data} \
--learning_rate 1e-5 \
--num_train_epochs 1 \
--per_device_train_batch_size 384 \
--gradient_accumulation_steps 1 \
--dataloader_drop_last True \
--cutoff_len 128 \
--logging_steps 1 \
--save_steps 500000 \
--save_total_limit 1 \
--gradient_checkpointing \
--ddp_find_unused_parameters False \
--use_flash_attn False \
--deepspeed ../stage1.json \
--warmup_ratio 0.1 \
--remove_stop_words True \
--use_lora False \
--bf16 \
--cache_dir ./LMs \
--token ... > logs/${output_model_name}.log 2>&1 &