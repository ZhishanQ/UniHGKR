models=("UniHGKR-7B")
test_set=("test")
sources=("all" "text" "kb" "info" "table")
# if you just want to test "retrieve from all sources", you can set sources=("all")

for model in "${models[@]}"; do
    for cur_source in "${sources[@]}"; do
        nohup python -u ST_code_util_mgpu_batch_ins.py --gpus 1 \
        --model_name ./${model} \
        --convmix_test_set "test" \
        --save_embedding 0 \
        --debug_flag 0 \
        --use_domain_tips 1 \
        --use_diverse_ins 1 \
        --batch_size 16 \
        --use_single_source ${cur_source} > logs/${model}_full_ins_test_${cur_source}.log 2>&1 &
        wait
    done
done