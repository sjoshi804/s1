export CUDA_VISIBLE_DEVICES=0,1,2,3

uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-7B-Instruct"
lr=1e-5
min_lr=0
epochs=5
micro_batch_size=1
push_to_hub=false
gradient_accumulation_steps=1
max_steps=-1
gpu_count=4

deepspeed --num_gpus ${gpu_count} \
train/ds_sft.py \
--deepspeed train/ds_config.json \
--per_device_train_batch_size=${micro_batch_size} \
--per_device_eval_batch_size=${micro_batch_size} \
--gradient_accumulation_steps=${gradient_accumulation_steps} \
--num_train_epochs=${epochs} \
--max_steps=${max_steps} \
--train_file_path="simplescaling/s1K-1.1_tokenized" \
--model_name=${base_model} \
--warmup_ratio=0.05 \
--bf16=True \
--eval_strategy="steps" \
--eval_steps=50 \
--logging_steps=1 \
--save_steps=100 \
--lr_scheduler_type="cosine" \
--learning_rate=${lr} \
--weight_decay=1e-4 \
--adam_beta1=0.9 \
--adam_beta2=0.95 \
--output_dir="ckpts/s1-1_7b_${uid}" \
--hub_model_id="simplescaling/s1-1_7b-${uid}" \
--push_to_hub=${push_to_hub} \
--save_only_model=True