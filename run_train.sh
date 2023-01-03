pool_type="cls"
gpu='0,1,2,3,4,5,6,7'
bert_model="bert-base-multilingual-cased"
output_dir="output_model/"$bert_model"_"$pool_type""

mkdir -p $output_dir

num_gpu=8
batch_size=8

train_data="data"

CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.launch --master_port 2025 --nproc_per_node $num_gpu run_training.py \
  --output_dir $output_dir \
  --model_name_or_path $bert_model \
  --do_train \
  --save_steps 1000 \
  --model_type bert \
  --per_device_train_batch_size $batch_size \
  --gradient_accumulation_steps 1 \
  --warmup_ratio 0.1 \
  --learning_rate 2e-5 \
  --num_train_epochs 40 \
  --dataloader_drop_last \
  --overwrite_output_dir \
  --dataloader_num_workers 10 \
  --max_seq_length 256 \
  --train_dir $train_data \
  --weight_decay 0.01 \
  --pool_type $pool_type --fp16 # fp16是指模型参数用float point16来存储
