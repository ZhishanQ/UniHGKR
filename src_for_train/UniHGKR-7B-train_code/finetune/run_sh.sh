base_model="../pretrain/UniHGKR-7B-pretrained"
output_model_name="UniHGKR-7B"
dataset_file="../../UniHGKR_training_data/UniHGKR_7b_finetune_data.jsonl"

nohup torchrun --nproc_per_node 8 \
run.py \
--output_dir ./${output_model_name} \
--model_name_or_path ${base_model} \
--train_data ${dataset_file} \
--learning_rate 2e-4 \
--num_train_epochs 1 \
--per_device_train_batch_size 64 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.01 \
--query_max_len 64 \
--passage_max_len 160 \
--train_group_size 8 \
--logging_steps 10 \
--save_steps 50000 \
--save_total_limit 1 \
--ddp_find_unused_parameters False \
--negatives_cross_device \
--gradient_checkpointing \
--deepspeed ../stage1.json \
--warmup_ratio 0.1 \
--fp16 \
--cache_dir ./LMs \
--token ...  > logs/${output_model_name}.log 2>&1 &