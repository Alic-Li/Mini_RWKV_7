#!/bin/bash
################################################################################
#
# This will generate the initial model, and save it to the output folder
#
################################################################################
#
# Please firstly create data folder & Download minipile (1498226207 tokens, around 3GB)
# mkdir -p data
# wget --continue -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
# wget --continue -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin
#
################################################################################
#
MODEL_TYPE="x070" # x060 => rwkv-6.0
#
N_LAYER="8"
N_EMBD="512"
#
CTX_LEN="512" # !!! change magic_prime if you change ctx_len !!!
PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE # set output folder
#
################################################################################
#
# magic_prime = the largest 3n+2 prime smaller than datalen/ctxlen-1 (= 1498226207/512-1 = 2926222.06 in this case) = 2926181 in this case
# use https://www.dcode.fr/prime-numbers-search
#
MY_EXIT_TOKENS="441795307"
MAGIC_PRIME="862811"
DATA_FILE="data/pretrain_hq"
DATA_TYPE="binidx"
#
python train.py \
 --wandb "" \
 --accelerator cpu \
 --adam_eps 1e-8 \
 --beta1 0.9 \
 --beta2 0.99 \
 --ctx_len $CTX_LEN \
 --data_file $DATA_FILE \
 --data_type $DATA_TYPE \
 --devices 1 \
 --epoch_begin 0 \
 --epoch_count 1 \
 --epoch_save 1 \
 --grad_cp 1 \
 --head_size 64 \
 --lr_final 1e-5 \
 --lr_init 1e-5 \
 --magic_prime $MAGIC_PRIME \
 --micro_bsz 1 \
 --my_exit_tokens $MY_EXIT_TOKENS \
 --my_testing $MODEL_TYPE \
 --n_embd $N_EMBD \
 --n_layer $N_LAYER \
 --num_nodes 1 \
 --precision bf16 \
 --proj_dir $PROJ_DIR \
 --strategy deepspeed_stage_2 \
 --train_stage 1 \
 --vocab_size 6400 \
 --warmup_steps 10 \
 --weight_decay 0 \
