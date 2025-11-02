## configs of different datasets
#cfg=$1
#
## model settings
#imsize=256
#num_workers=4
#batch_size_per_gpu=32
#stamp=normal
#train=True
#
## resume training
#resume_epoch=1
#resume_model_path=./saved_models/bird/base_z_dim100_bird_256_2022_06_04_23_20_33/
#
## DDP settings
#multi_gpus=True
#nodes=1
#master_port=11111
#
## You can set CUDA_VISIBLE_DEVICES=0,1,2... to accelerate the training process if you have multiple GPUs
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$nodes --master_port $master_port src/train.py \
#                    --stamp $stamp \
#                    --cfg $cfg \
#                    --batch_size $batch_size_per_gpu \
#                    --num_workers $num_workers \
#                    --imsize $imsize \
#                    --resume_epoch $resume_epoch \
#                    --resume_model_path $resume_model_path \
#                    --train $train \
#                    --multi_gpus $multi_gpus \

cfg=$1
batch_size=16 #64--64G

state_epoch=1
pretrained_model_path='./saved_models/coco/model0_gpu8MP_False_coco_256_2024_06_24_10_14_54'
log_dir='new'

multi_gpus=True
mixed_precision=False

nodes=8

num_workers=8
master_port=11266
stamp=gpu${nodes}MP_${mixed_precision}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=$nodes src/train.py \
                    --stamp $stamp \
                    --cfg $cfg \
                    --mixed_precision $mixed_precision \
                    --log_dir $log_dir \
                    --batch_size $batch_size \
                    --state_epoch $state_epoch \
                    --num_workers $num_workers \
                    --multi_gpus $multi_gpus \
                    --pretrained_model_path $pretrained_model_path \


#cfg=$1
#batch_size=16 #64--64G
#
#state_epoch=1
#pretrained_model_path='../saved_models/data/model_save_file'
#log_dir='new'
#
#multi_gpus=False
#mixed_precision=False
#
#nodes=1
#
#num_workers=8
#master_port=11266
#stamp=gpu${nodes}MP_${mixed_precision}
#
#CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=$nodes src/train.py \
#                    --stamp $stamp \
#                    --cfg $cfg \
#                    --mixed_precision $mixed_precision \
#                    --log_dir $log_dir \
#                    --batch_size $batch_size \
#                    --state_epoch $state_epoch \
#                    --num_workers $num_workers \
#                    --multi_gpus $multi_gpus \
#                    --pretrained_model_path $pretrained_model_path \


