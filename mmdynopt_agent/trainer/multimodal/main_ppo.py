import hydra
import ray

from mmdynopt_agent.trainer.multimodal.ray_trainer import RayPPOTrainer


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config, compute_score=None):
    if not ray.is_initialized():
        ray.init(
                address="auto",
                ignore_reinit_error=True,
                runtime_env={
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "true",
                        "NCCL_DEBUG": "WARN"
                    }
                }
        )

    ray.get(main_task.remote(config, compute_score))


@ray.remote(num_cpus=1)
def main_task(config, compute_score=None):
    from pprint import pprint
    from omegaconf import OmegaConf
    from verl.utils.fs import copy_to_local

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    from verl.utils import hf_processor, hf_tokenizer

    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)

    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import CriticWorker
        from mmdynopt_agent.workers.multimodal.fsdp_workers import ActorRolloutRefWorker

        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from mmdynopt_agent.trainer.multimodal.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic:       ray.remote(CriticWorker),
        Role.RefPolicy:    ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic:       global_pool_id,
        Role.RefPolicy:    global_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_manager_name = config.reward_model.get("reward_manager", "naive")

    if reward_manager_name == 'naive':
        from mmdynopt_agent.workers.multimodal.reward import MMDynOptRewardManager
        reward_manager_cls = MMDynOptRewardManager

    elif reward_manager_name == 'prime':
        from verl.workers.reward_manager import PrimeRewardManager
        reward_manager_cls = PrimeRewardManager

    else:
        raise NotImplementedError

    reward_fn = reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=0,
        compute_score=compute_score
    )

    val_reward_fn = reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=1,
        compute_score=compute_score
    )

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping
    )

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )

    trainer.init_workers()

    if config.actor_rollout_ref.rollout.name == "vllm_multiturn_mmdynopt":
        assert config.actor_rollout_ref.actor.use_multi_turn_response_mask == True, \
            "Set `actor_rollout_ref.actor.use_multi_turn_response_mask` to refine " \
            "`response_mask` correctly in update_policy()"
    else:
        assert config.actor_rollout_ref.actor.use_multi_turn_response_mask == False, \
            "Set `actor_rollout_ref.actor.use_multi_turn_response_mask` to False " \
            "for non-multi-turn scenario"

    trainer.fit()


if __name__ == '__main__':
    main()
