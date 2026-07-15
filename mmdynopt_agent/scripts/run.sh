#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

export BASE_MODEL='/path/to/your/base_model'
export PROJECT_NAME='MMDynOpt'
export EXPERIMENT_NAME='MMDynOpt'

export RAY_PORT=6662
export RAY_DASHBOARD_PORT=6676
export RAY_TEMP_DIR=/tmp/ray_mmdynopt

export TRAIN_DATA_PATH=datasets/train.parquet
export VAL_DATA_PATH=datasets/val.parquet

export NO_PROXY=""
export no_proxy=$NO_PROXY

export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_ALGO=Ring
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export RAY_grpc_keepalive_time_ms=300
export RAY_grpc_keepalive_timeout_ms=300

export WANDB_MODE=offline
export WANDB_DIR=./wandb_logs

export RAY_raylet_start_wait_time_s=300
export RAY_gcs_server_request_timeout_seconds=60

SPILL_DIR=${RAY_TEMP_DIR}/spill

mkdir -p ${SPILL_DIR}

OLD_PID=$(lsof -ti :${RAY_PORT} 2>/dev/null || true)

if [ -n "$OLD_PID" ]; then
    echo "Port ${RAY_PORT} in use (PID: ${OLD_PID}), cleaning up..."
    kill -9 $OLD_PID || true
    sleep 2
fi

rm -rf ${RAY_TEMP_DIR}
mkdir -p ${RAY_TEMP_DIR}

set -xe

unset RAY_ADDRESS

NODE_IP=$(hostname -i | awk '{print $1}')

ray start --head \
    --node-ip-address=${NODE_IP} \
    --num-cpus=8 \
    --num-gpus=4 \
    --port=${RAY_PORT} \
    --include-dashboard=True \
    --dashboard-port=${RAY_DASHBOARD_PORT} \
    --dashboard-host=0.0.0.0 \
    --temp-dir=${RAY_TEMP_DIR} \
    --object-spilling-directory=${SPILL_DIR} \
    --object-store-memory=8000000000

export RAY_ADDRESS="${NODE_IP}:${RAY_PORT}"

sleep 10

ray status --address="${NODE_IP}:${RAY_PORT}" || { echo "Ray start failed, exiting"; exit 1; }

python3 -m mmdynopt_agent.trainer.multimodal.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA_PATH \
    data.val_files=$VAL_DATA_PATH \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.image_key=images \
    data.user_prompt_round_1=mmdynopt_agent/prompts/initial_prompt.pkl \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_sigmoid_decay_warmup=True \
    actor_rollout_ref.actor.optim.lr_sigmoid_decay_ratio=0.95 \
    actor_rollout_ref.actor.optim.lr_sigmoid_decay_warmup_steps=45 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_multi_turn_response_mask=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_multiturn_mmdynopt \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_gen_round=5 \
    actor_rollout_ref.rollout.response_length_total=8192 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.search.topk=5 \
    actor_rollout_ref.rollout.search.parallel_tool_call=True \
    actor_rollout_ref.rollout.search.parallel_tool_call_threads=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    'trainer.logger=[wandb,console]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    +trainer.max_actor_ckpt_to_keep=3 \
    +trainer.max_critic_ckpt_to_keep=3 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    +trainer.search_penalty=0.1 \
    +trainer.format_penalty=0.1 \
    +trainer.reward_mode="EM" \
    +trainer.val_before_train=True \
    +algorithm.filter_groups.enable=False
