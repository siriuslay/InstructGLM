#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=2,3
name=cora-7b

output=HDDDATA/wyd/$name

PYTHONPATH=$PYTHONPATH:./llama_cora_src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port 12321 \
    llama_cora_src/pretrain.py \
        --distributed --multiGPU \
        --seed 42 \
	--gradient_accumulation_steps 8 \
        --train Cora \
        --valid Cora \
        --batch_size 8 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --num_workers 8 \
        --clip_grad_norm 1.0 \
        --losses 'link,classification' \
        --backbone '/HDDDATA/wyd/7B' \
        --data_path '/HDDDATA/wyd' \
        --output $output ${@:2} \
        --epoch 2 \
	--weight_decay 0 \
        --max_text_length 512 \
        --gen_max_length 64 \
	--lr 0.00008
