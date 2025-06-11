#!/bin/bash

MODEL_NAME="UniHGKR-7B" # path to UniHGKR-7B
CONVMIX_TEST_SET="test"
BATCH_SIZE=32

for i in {0..7}
do
  nohup python -u llara_run_compmix_fp16_part.py --gpus $i \
  --model_name $MODEL_NAME \
  --convmix_test_set $CONVMIX_TEST_SET \
  --batch_size $BATCH_SIZE \
  --split_index $i > logs/${MODEL_NAME}_test_fp16_split_$i.log 2>&1 &
done
