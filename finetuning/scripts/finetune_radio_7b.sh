#!/bin/bash
#SBATCH -A a-a02
#SBATCH --job-name=radvlm-full  # Job name
#SBATCH --nodes=32    # Number of nodes
#SBATCH --ntasks-per-node=1                  # Number of tasks per node (1 process per node)
#SBATCH --gpus-per-task=4                  # Number of GPUs per ta
#SBATCH --time=12:00:00                      # Time limit
#SBATCH --output=job_outputs/%x.txt    # Standard output and error log
#SBATCH --mem=460000
#SBATCH --partition=normal  

# Initialization
set -x
cat $0
export MASTER_PORT=29500
#export MASTER_ADDR=$(hostname)
export MASTER_ADDR=$(scontrol show hostname | head -n 1)
export HF_HOME=$SCRATCH/huggingface_home

PROMPT_VERSION="qwen_1_5"

RUN_NAME=${SLURM_JOB_NAME}
echo "RUN_NAME: ${RUN_NAME}"
CKPT_PATH="lmms-lab/llava-onevision-qwen2-7b-si" # this could also be the previous stage checkpoint

NUM_EPOCHS=1
LR=1e-5
SAVE_STEPS=200

WORKDIR="$SCRATCH/code/RadVLM"

# Run main script
srun -ul  --environment=llava_env_clariden bash -c "
  cd $WORKDIR  # Change cwd and run the main training script.
  export PYTHONPATH=$WORKDIR/finetuning
  export WANDB_API_KEY=81291d9e2d99efeb2a4e3f4d507abe879e646a22
  TORCHRUN_ARGS=\"
   --node-rank=\${SLURM_PROCID} \
   --master-addr=\${MASTER_ADDR} \
   --master-port=\${MASTER_PORT} \
   --nnodes=\${SLURM_NNODES} \
   --nproc-per-node=4 \
  \"

 ACCELERATE_CPU_AFFINITY=1  torchrun \${TORCHRUN_ARGS} finetuning/llava/train/train_mem.py \
    --deepspeed finetuning/scripts/zero3.json \
    --model_name_or_path $CKPT_PATH \
    --version ${PROMPT_VERSION} \
    --data_path radvlm/data/llava_datasets/all_train.json \
    --image_folder . \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
   --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints \"(1x1),...,(6x6)\" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir="$SCRATCH/checkpoints/${RUN_NAME}" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy \"no\" \
    --save_strategy \"steps\" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 1 \
    --learning_rate $LR \
    --warmup_ratio 0.03 \
    --lr_scheduler_type \"cosine\" \
    --weight_decay 0. \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend \"inductor\" \
    --dataloader_drop_last True \
    --frames_upbound 32
"


