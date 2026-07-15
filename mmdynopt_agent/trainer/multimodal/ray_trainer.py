import os
import json
import re
from mmdynopt_agent.utils.reward_score_mm.mmdynopt_reward import extract_solution
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pprint import pprint
from typing import Type

import numpy as np
import pandas as pd
import tqdm
from omegaconf import OmegaConf, open_dict
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import compute_timing_metrics, reduce_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    Role,
    _timer,
    apply_kl_penalty,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)

from mmdynopt_agent.monkey_patch.monkey_patch import create_colocated_worker_cls_patch
from mmdynopt_agent.trainer.multimodal.core_algos import compute_grpo_outcome_advantage
from mmdynopt_agent.utils.dataset.mm_rl_dataset import RLHFDataset, collate_fn

WorkerType = Type[Worker]
import torch


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True, tokenizer=None):
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    total_count = sequence_score.size(0)
    search_cnt_text_total, search_cnt_image_total = 0, 0
    search_cnt_text, search_cnt_image, search_cnt_mix = 0, 0, 0
    search_fail_text, search_fail_image = 0, 0
    responses_after_first_user_prompt = batch.batch['responses']
    assert total_count == responses_after_first_user_prompt.shape[0], "B*R != Total Num of Rollout Responses"
    
    for idx, response in enumerate(responses_after_first_user_prompt):
        _resp_length = response.size(0)
        if 'multi_turn_response_mask' in batch.batch:
            _resp_mask = batch.batch['multi_turn_response_mask'][idx][-_resp_length:]
            response, response_non_assistant = response[_resp_mask == 1], response[_resp_mask < 0.1]
        response_non_assistant = tokenizer.decode(response_non_assistant)
        if '[Text Search Results] There is an error' in response_non_assistant:
            search_fail_text += 1
        if '[Image Search Results] There is an error' in response_non_assistant:
            search_fail_image += 1
        if "[Text Search Results]" in response_non_assistant:
            search_cnt_text_total += 1
        if "[Image Search Results]" in response_non_assistant:
            search_cnt_image_total += 1
        if "[Text Search Results]" in response_non_assistant and "[Image Search Results]" not in response_non_assistant:
            search_cnt_text += 1
        if "[Image Search Results]" in response_non_assistant and "[Text Search Results]" not in response_non_assistant:
            search_cnt_image += 1
        if "[Image Search Results]" in response_non_assistant and "[Text Search Results]" in response_non_assistant:
            search_cnt_mix += 1

    search_ratio_text = search_cnt_text / total_count
    search_ratio_image = search_cnt_image / total_count
    search_ratio_mix = search_cnt_mix / total_count
    fail_ratio_text = search_fail_text / (search_cnt_text_total + 1e-5)
    fail_ratio_image = search_fail_image / (search_cnt_image_total + 1e-5)
    fp = 0.1
    if 'extra_info' in batch.non_tensor_batch and 'format_penalty' in batch.non_tensor_batch['extra_info'][0]:
        fp = batch.non_tensor_batch['extra_info'][0]['format_penalty']
    correct_threshold = fp + 1e-4
    count_correct = torch.sum(sequence_score > correct_threshold).item()
    answer_acc = count_correct / total_count

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    if 'multi_turn_response_mask' in batch.batch:
        response_length = batch.batch['multi_turn_response_mask'].sum(dim=1).float()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    f1_scores = []
    llm_call_counts = []
    if 'extra_info' in batch.non_tensor_batch:
        for info in batch.non_tensor_batch['extra_info']:
            if isinstance(info, dict):
                if 'f1' in info:
                    f1_scores.append(info['f1'])
                if 'n_llm_calls' in info:
                    llm_call_counts.append(info['n_llm_calls'])
    
    f1_metrics = {}
    if f1_scores:
        f1_tensor = torch.tensor(f1_scores, dtype=torch.float32)
        f1_metrics = {
            'critic/mmdynopt/f1/mean': torch.mean(f1_tensor).detach().item(),
            'critic/mmdynopt/f1/max': torch.max(f1_tensor).detach().item(),
            'critic/mmdynopt/f1/min': torch.min(f1_tensor).detach().item(),
        }
    
    llm_call_metrics = {}
    if llm_call_counts:
        llm_call_tensor = torch.tensor(llm_call_counts, dtype=torch.float32)
        llm_call_metrics = {
            'critic/mmdynopt/llm_calls/mean': torch.mean(llm_call_tensor).detach().item(),
            'critic/mmdynopt/llm_calls/max': torch.max(llm_call_tensor).detach().item(),
            'critic/mmdynopt/llm_calls/min': torch.min(llm_call_tensor).detach().item(),
        }

    metrics = {
        'critic/mmdynopt/correct_threshold': correct_threshold,
        'critic/mmdynopt/search_fail_ratio_text': fail_ratio_text,
        'critic/mmdynopt/search_fail_ratio_image': fail_ratio_image,
        'critic/mmdynopt/search_ratio_text': search_ratio_text,
        'critic/mmdynopt/search_ratio_image': search_ratio_image,
        'critic/mmdynopt/search_ratio_mix': search_ratio_mix,
        'critic/mmdynopt/answer_acc': answer_acc,
        'critic/score/mean': torch.mean(sequence_score).detach().item(),
        'critic/score/max': torch.max(sequence_score).detach().item(),
        'critic/score/min': torch.min(sequence_score).detach().item(),
        'critic/rewards/mean': torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max': torch.max(sequence_reward).detach().item(),
        'critic/rewards/min': torch.min(sequence_reward).detach().item(),
        'critic/advantages/mean': torch.mean(valid_adv).detach().item(),
        'critic/advantages/max': torch.max(valid_adv).detach().item(),
        'critic/advantages/min': torch.min(valid_adv).detach().item(),
        'critic/returns/mean': torch.mean(valid_returns).detach().item(),
        'critic/returns/max': torch.max(valid_returns).detach().item(),
        'critic/returns/min': torch.min(valid_returns).detach().item(),
        **(
            {
                'critic/values/mean': torch.mean(valid_values).detach().item(),
                'critic/values/max': torch.max(valid_values).detach().item(),
                'critic/values/min': torch.min(valid_values).detach().item(),
                'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        'response_length/mean': torch.mean(response_length).detach().item(),
        'response_length/max': torch.max(response_length).detach().item(),
        'response_length/min': torch.min(response_length).detach().item(),
        'response_length/clip_ratio': torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        'prompt_length/mean': torch.mean(prompt_length).detach().item(),
        'prompt_length/max': torch.max(prompt_length).detach().item(),
        'prompt_length/min': torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio': torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        **f1_metrics,
        **llm_call_metrics,
    }
    return metrics


def compute_advantage(
    data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, grpo_normalize=True
):
    if adv_estimator == AdvantageEstimator.GRPO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index,
            grpo_normalize=grpo_normalize,
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@dataclass
class ResourcePoolManager:
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]


class RayPPOTrainer:

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
    ):

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert (
                    config.algorithm.kl_ctrl.horizon > 0
                ), f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(
                    init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                    target_kl=config.algorithm.kl_ctrl.target_kl,
                    horizon=config.algorithm.kl_ctrl.horizon,
                )
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.0)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert (
            real_train_batch_size % n_gpus == 0
        ), f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(
                    f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                    f"'{name}.micro_batch_size_per_gpu'."
                )

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.micro_batch_size' AND "
                    f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                    f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated)."
                )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
            )

            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if (
                config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1
                or config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1
            ):
                assert (
                    config.actor_rollout_ref.model.use_remove_padding
                ), "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert (
                    config.critic.model.use_remove_padding
                ), "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        self.train_dataset = RLHFDataset(
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get('image_key', 'images'),
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation='error',
            user_prompt_round_1=self.config.data.get('user_prompt_round_1', None),
        )
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=self.config.data.num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = RLHFDataset(
            parquet_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get('image_key', 'images'),
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation='error',
            user_prompt_round_1=self.config.data.get('user_prompt_round_1', None),
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert (
            len(self.val_dataloader) == 1
        ), "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations_to_wandb(self, inputs, outputs, scores, reward_models=[], image_urls=[]):
        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. '
            )
            return

        import numpy as np
        import wandb

        also_log_ground_truth_and_image_urls = len(reward_models) > 0 and len(image_urls) > 0

        if also_log_ground_truth_and_image_urls:
            samples = list(zip(inputs, outputs, scores, reward_models, image_urls))
        else:
            samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])

        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[:generations_to_log]

        if also_log_ground_truth_and_image_urls:
            columns = ["step", "input_text", "output_text", "score", "reward_model", "image_url"]
        else:
            columns = ["step", "input_text", "output_text", "score"]

        if not hasattr(self, 'validation_table'):
            self.validation_table = pd.DataFrame(columns=columns)

        row_data = []
        for sample in samples:
            if also_log_ground_truth_and_image_urls:
                reward_model = sample[3]
                if 'candidate_answers' not in reward_model:
                    reward_model['candidate_answers'] = '[]'
                row = {
                    "step": self.global_steps,
                    "input_text": sample[0],
                    "output_text": sample[1],
                    "score": sample[2],
                    "reward_model": reward_model,
                    "image_url": sample[4],
                }
            else:
                row = {"step": self.global_steps, "input_text": sample[0], "output_text": sample[1], "score": sample[2]}
            row_data.append(row)

        new_df = pd.DataFrame(row_data)
        self.validation_table = pd.concat([self.validation_table, new_df], ignore_index=True)
        
        if self.config.trainer.get('val_only', False) and self.config.trainer.val_only_save_dir is not None:
            os.makedirs(self.config.trainer.val_only_save_dir, exist_ok=True)
            save_path = os.path.join(
                self.config.trainer.val_only_save_dir, f"val_result_{len(self.validation_table)}.json"
            )
            self.validation_table.to_json(save_path, orient="records", indent=2)
            print(f'validation generation saved to local: {save_path}')

        wandb.log({"val/generations": wandb.Table(dataframe=self.validation_table)}, step=self.global_steps)
        print('validation generation saved to wandb table')

    def _validate(self):

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Validate Starts ..."
        )

        reward_tensor_lst = []
        data_source_lst = []

        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_reward_models = []
        sample_image_url = []

        all_results = []
        _is_val_only = self.config.trainer.get('val_only', False)

        val_f1_lst = []
        val_n_interactions_lst = []
        val_llm_input_tokens_lst = []
        val_llm_output_tokens_lst = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            if 'reward_model' in test_batch.non_tensor_batch:
                sample_reward_models.extend(list(test_batch.non_tensor_batch['reward_model']))
            if 'image_urls' in test_batch.non_tensor_batch:
                sample_image_url.extend(list(test_batch.non_tensor_batch['image_urls']))

            if 'multi_modal_data' in test_batch.non_tensor_batch.keys():
                if 'image_urls' in test_batch.non_tensor_batch:
                    test_gen_batch = test_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'image_urls'],
                    )
                else:
                    test_gen_batch = test_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data'],
                    )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            if 'extra_info' not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch['extra_info'] = [{} for _ in range(len(test_batch.batch))]
            for item_id in range(len(test_batch.batch)):
                if 'search_penalty' in self.config.trainer:
                    test_batch.non_tensor_batch['extra_info'][item_id][
                        'search_penalty'
                    ] = self.config.trainer.search_penalty
                if 'format_penalty' in self.config.trainer:
                    test_batch.non_tensor_batch['extra_info'][item_id][
                        'format_penalty'
                    ] = self.config.trainer.format_penalty
                if 'reward_mode' in self.config.trainer:
                    assert self.config.trainer.reward_mode in [
                        'EM',
                        'SubEM',
                    ], f'reward mode {self.config.trainer.reward_mode} not recognized, please use EM or SubEM'
                    test_batch.non_tensor_batch['extra_info'][item_id]['reward_mode'] = self.config.trainer.reward_mode
                if 'use_search_count_penalty' in self.config.trainer:
                    use_search_count_penalty = self.config.trainer.use_search_count_penalty
                    test_batch.non_tensor_batch['extra_info'][item_id][
                        'use_search_count_penalty'
                    ] = use_search_count_penalty
            test_batch.non_tensor_batch['extra_info'] = np.array(test_batch.non_tensor_batch['extra_info'])
            reward_tensor = self.val_reward_fn(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

            for _ei in test_batch.non_tensor_batch.get('extra_info', []):
                _ei = _ei if isinstance(_ei, dict) else {}
                val_f1_lst.append(float(_ei.get('f1', 0.0)))
                val_n_interactions_lst.append(float(_ei.get('n_llm_calls', 0)))
                val_llm_input_tokens_lst.append(float(_ei.get('llm_prompt_len', 0)))
                val_llm_output_tokens_lst.append(float(_ei.get('llm_response_len', 0)))

            if _is_val_only:
                batch_size_cur = reward_tensor.shape[0]
                for _i in range(batch_size_cur):
                    _data_source = test_batch.non_tensor_batch.get(
                        'data_source', ['unknown'] * batch_size_cur
                    )[_i]
                    _reward_model = test_batch.non_tensor_batch.get(
                        'reward_model', [{}] * batch_size_cur
                    )[_i]
                    _ground_truth = (
                        _reward_model.get('ground_truth', '')
                        if isinstance(_reward_model, dict)
                        else ''
                    )
                    _image_urls = test_batch.non_tensor_batch.get(
                        'image_urls', [None] * batch_size_cur
                    )[_i]
                    _score = scores[_i]
                    _extra_info = test_batch.non_tensor_batch.get(
                        'extra_info', [{}] * batch_size_cur
                    )[_i]
                    _f1 = (
                        _extra_info.get('f1', None)
                        if isinstance(_extra_info, dict)
                        else None
                    )
                    _global_idx = len(all_results)
                    _response_str = sample_outputs[_global_idx]
                    _input_str = sample_inputs[_global_idx]
                    _image_urls_serializable = (
                        _image_urls.tolist()
                        if isinstance(_image_urls, np.ndarray)
                        else _image_urls
                    )
                    _predicted_answer = extract_solution(_response_str)
                    _n_interactions = int(_extra_info.get('n_llm_calls', 0)) if isinstance(_extra_info, dict) else 0
                    _ei = _extra_info if isinstance(_extra_info, dict) else {}
                    _llm_input_tokens = int(_ei.get('llm_input_tokens', 0))
                    _llm_output_tokens = int(_ei.get('llm_output_tokens', 0))
                    _agent_input_tokens = int(_ei.get('agent_input_tokens', 0))
                    _agent_output_tokens_sum = int(_ei.get('agent_output_tokens_sum', 0))
                    _llm_per_round_input = list(_ei.get('llm_per_round_input', []))
                    _llm_per_round_output = list(_ei.get('llm_per_round_output', []))
                    _agent_per_round_input = list(_ei.get('agent_per_round_input', []))
                    _agent_per_round_output = list(_ei.get('agent_per_round_output', []))
                    _n_agent_rounds = int(_ei.get('n_agent_rounds', 0))
                    _prompt_len = test_batch.batch['prompts'][_i].shape[-1]
                    _agent_output_tokens = int(test_batch.batch['attention_mask'][_i][_prompt_len:].sum().item())
                    all_results.append({
                        'data_id': _global_idx,
                        'question': _input_str,
                        'predicted_answer': _predicted_answer,
                        'ground_truth': _ground_truth,
                        'score': _score,
                        'f1': _f1,
                        'data_source': _data_source,
                        'image_urls': _image_urls_serializable,
                        'response': _response_str,
                        'n_interactions': _n_interactions,
                        'llm_input_tokens': _llm_input_tokens,
                        'llm_output_tokens': _llm_output_tokens,
                        'agent_output_tokens': _agent_output_tokens,
                        'agent_input_tokens': _agent_input_tokens,
                        'agent_output_tokens_sum': _agent_output_tokens_sum,
                        'llm_per_round_input': _llm_per_round_input,
                        'llm_per_round_output': _llm_per_round_output,
                        'agent_per_round_input': _agent_per_round_input,
                        'agent_per_round_output': _agent_per_round_output,
                        'n_agent_rounds': _n_agent_rounds,
                    })

        self._maybe_log_val_generations_to_wandb(
            inputs=sample_inputs,
            outputs=sample_outputs,
            scores=sample_scores,
            reward_models=sample_reward_models,
            image_urls=sample_image_url,
        )

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)

        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/{data_source}/reward'] = np.mean(rewards)
            fp = 0.1
            if 'format_penalty' in self.config.trainer:
                fp = self.config.trainer.format_penalty
            correct_threshold = fp + 1e-4
            correct_cnt = sum(1 for x in rewards if x > correct_threshold)
            metric_dict[f'val/{data_source}/score'] = correct_cnt / len(rewards)
            _ds_indices = [j for j, ds in enumerate(data_sources) if ds == data_source]
            metric_dict[f'val/{data_source}/f1'] = np.mean([val_f1_lst[j] for j in _ds_indices]) if _ds_indices else 0.0
            metric_dict[f'val/{data_source}/n_interactions'] = np.mean([val_n_interactions_lst[j] for j in _ds_indices]) if _ds_indices else 0.0
            metric_dict[f'val/{data_source}/llm_input_tokens'] = np.mean([val_llm_input_tokens_lst[j] for j in _ds_indices]) if _ds_indices else 0.0
            metric_dict[f'val/{data_source}/llm_output_tokens'] = np.mean([val_llm_output_tokens_lst[j] for j in _ds_indices]) if _ds_indices else 0.0
            search_cnt_text_total, search_cnt_image_total = 0, 0
            search_cnt_text, search_cnt_image, search_cnt_mix = 0, 0, 0
            search_fail_text, search_fail_image = 0, 0
            responses_after_first_user_prompt = test_batch.batch['responses']
            for idx, response in enumerate(responses_after_first_user_prompt):
                _resp_length = response.size(0)
                if 'multi_turn_response_mask' in test_batch.batch:
                    _resp_mask = test_batch.batch['multi_turn_response_mask'][idx][-_resp_length:]
                    response, response_non_assistant = response[_resp_mask == 1], response[_resp_mask < 0.1]
                response_non_assistant = self.tokenizer.decode(response_non_assistant)
                if (
                    "[Text Search Results]" in response_non_assistant
                    and "[Image Search Results]" not in response_non_assistant
                ):
                    search_cnt_text += 1
                if (
                    "[Image Search Results]" in response_non_assistant
                    and "[Text Search Results]" not in response_non_assistant
                ):
                    search_cnt_image += 1
                if (
                    "[Image Search Results]" in response_non_assistant
                    and "[Text Search Results]" in response_non_assistant
                ):
                    search_cnt_mix += 1
                if "[Text Search Results]" in response_non_assistant:
                    search_cnt_text_total += 1
                if "[Image Search Results]" in response_non_assistant:
                    search_cnt_image_total += 1
                if '[Text Search Results] There is an error' in response_non_assistant:
                    search_fail_text += 1
                if '[Image Search Results] There is an error' in response_non_assistant:
                    search_fail_image += 1
            search_ratio_text = search_cnt_text / len(rewards)
            search_ratio_image = search_cnt_image / len(rewards)
            search_ratio_mix = search_cnt_mix / len(rewards)
            fail_ratio_text = search_fail_text / (search_cnt_text_total + 1e-5)
            fail_ratio_image = search_fail_image / (search_cnt_image_total + 1e-5)
            metric_dict[f'val/{data_source}/correct_threshold'] = correct_threshold
            metric_dict[f'val/{data_source}/search_ratio_text'] = search_ratio_text
            metric_dict[f'val/{data_source}/search_ratio_image'] = search_ratio_image
            metric_dict[f'val/{data_source}/search_ratio_mix'] = search_ratio_mix
            metric_dict[f'val/{data_source}/search_fail_ratio_text'] = fail_ratio_text
            metric_dict[f'val/{data_source}/search_fail_ratio_image'] = fail_ratio_image
            metric_dict[f'val/{data_source}/rewards_len'] = len(rewards)
            metric_dict[f'val/{data_source}/responses_after_first_user_prompt_len'] = len(
                responses_after_first_user_prompt
            )

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Validate Ends ...")

        if _is_val_only and self.config.trainer.get('val_only_save_dir', None):
            _save_dir = self.config.trainer.val_only_save_dir
            os.makedirs(_save_dir, exist_ok=True)

            def _numpy_default(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

            _res_path = os.path.join(_save_dir, 'res.json')
            with open(_res_path, 'w', encoding='utf-8') as _f:
                json.dump(all_results, _f, ensure_ascii=False, indent=2, default=_numpy_default)
            print(f'[val_only] res.json saved to {_res_path} ({len(all_results)} samples)')

            _log_path = os.path.join(_save_dir, 'log.txt')
            with open(_log_path, 'w', encoding='utf-8') as _f:
                _f.write(
                    f"Val file: {self.config.data.val_files}\n"
                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Total samples: {len(all_results)}\n"
                    f"{'=' * 80}\n\n"
                )
                for _r in all_results:
                    _f.write(f"\n{'#' * 80}\n")
                    _f.write(f"Data ID: {_r['data_id']}\n")
                    _f.write(f"Question: {_r['question']}\n")
                    _f.write(f"Ground Truth: {_r['ground_truth']}\n")
                    _f.write(f"{'#' * 80}\n\n")

                    _resp = _r['response'] or ''
                    _turns = re.split(r'(?:^|(?<=\n))assistant\n', _resp)
                    _turn_num = 0
                    for _t_idx, _turn_text in enumerate(_turns):
                        _turn_text = _turn_text.strip()
                        if not _turn_text:
                            continue
                        _turn_text = re.sub(r'^assistant\n', '', _turn_text).strip()
                        if not _turn_text:
                            continue
                        _turn_num += 1
                        _f.write(f"{'=' * 60}\n")
                        _f.write(f"Turn {_turn_num}\n\n")

                        _display = _turn_text
                        _display = _display.replace('<think>', '\n<Think>\n')
                        _display = _display.replace('</think>', '\n')
                        _display = _display.replace('<prompt>', '\n<Prompt>\n')
                        _display = _display.replace('</prompt>', '\n')
                        _display = _display.replace('<response>', '\n<Response>\n')
                        _display = _display.replace('</response>', '\n')
                        _display = _display.replace('<answer>', '\n<Answer>\n')
                        _display = _display.replace('</answer>', '\n')
                        _f.write(_display.strip() + '\n\n')

                    _f.write(f"{'=' * 80}\n")
                    _f.write(f"Predicted Answer: {_r['predicted_answer']}\n")
                    _f.write(f"Score: score={_r['score']}  f1={_r['f1']}\n")
                    _f.write(f"{'=' * 80}\n")

                _f.write(
                    f"\n{'=' * 80}\n"
                    f"Validation complete  Total: {len(all_results)}\n"
                    f"Results: {_res_path}\n"
                )
            print(f'[val_only] log.txt saved to {_log_path}')

        return metric_dict

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role='actor_rollout',
            )
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role='ref'
            )
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls_patch(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f'global_step_{self.global_steps}'
        )
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        )
        remove_previous_ckpt_in_save = self.config.trainer.get('remove_previous_ckpt_in_save', False)
        if remove_previous_ckpt_in_save:
            print(
                'Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead'
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get('max_actor_ckpt_to_keep', None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get('max_critic_ckpt_to_keep', None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, 'latest_checkpointed_iteration.txt'
        )
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        self.global_steps += 1
        
        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0

        progress_bar = tqdm.tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                if 'multi_modal_data' in new_batch.non_tensor_batch.keys():
                    if (
                        'image_urls' in new_batch.non_tensor_batch
                    ):
                        gen_batch = new_batch.pop(
                            batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                            non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'image_urls'],
                        )
                    else:
                        gen_batch = new_batch.pop(
                            batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                            non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data'],
                        )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )
                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Rollout Starts ..."
                    )
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        del gen_batch
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Rollout Ends ..."
                    )

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch['uid'] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with _timer('reward', timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)
                        if 'extra_info' not in new_batch.non_tensor_batch:
                            new_batch.non_tensor_batch['extra_info'] = [{} for _ in range(len(new_batch.batch))]
                        for item_id in range(len(new_batch.batch)):
                            if 'search_penalty' in self.config.trainer:
                                step_search_penalty = self.config.trainer.search_penalty
                                if 'search_penalty_warmup_steps' in self.config.trainer:
                                    step_search_penalty = -self.config.trainer.search_penalty
                                    step_search_penalty += min(
                                        (self.global_steps - 1) / self.config.trainer.search_penalty_warmup_steps, 1
                                    ) * (2 * self.config.trainer.search_penalty)
                                new_batch.non_tensor_batch['extra_info'][item_id][
                                    'search_penalty'
                                ] = step_search_penalty
                            if 'format_penalty' in self.config.trainer:
                                new_batch.non_tensor_batch['extra_info'][item_id][
                                    'format_penalty'
                                ] = self.config.trainer.format_penalty
                            if 'reward_mode' in self.config.trainer:
                                assert self.config.trainer.reward_mode in [
                                    'EM',
                                    'SubEM',
                                ], f'reward mode {self.config.trainer.reward_mode} not recognized, please use EM or SubEM'
                                new_batch.non_tensor_batch['extra_info'][item_id][
                                    'reward_mode'
                                ] = self.config.trainer.reward_mode
                            if 'use_search_count_penalty' in self.config.trainer:
                                use_search_count_penalty = self.config.trainer.use_search_count_penalty
                                new_batch.non_tensor_batch['extra_info'][item_id][
                                    'use_search_count_penalty'
                                ] = use_search_count_penalty
                        new_batch.non_tensor_batch['extra_info'] = np.array(new_batch.non_tensor_batch['extra_info'])

                        print(
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Reward_fn Starts ..."
                        )
                        reward_tensor = self.reward_fn(new_batch)
                        print(
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Reward_fn Ends ..."
                        )
                        new_batch.batch['token_level_scores'] = reward_tensor

                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch['token_level_rewards'] = new_batch.batch['token_level_scores']

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:
                        metric_name = self.config.algorithm.filter_groups.metric
                        if (
                            metric_name == "seq_final_reward"
                        ):
                            new_batch.batch["seq_final_reward"] = (
                                new_batch.batch['token_level_scores'].sum(dim=-1).numpy()
                            )
                        elif (
                            metric_name == "seq_reward"
                        ):
                            new_batch.batch["seq_reward"] = new_batch.batch['token_level_scores'].sum(dim=-1).numpy()

                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch['uid'], new_batch.batch[metric_name]
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch['uid']):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        if batch is None:
                            batch = new_batch
                        else:
                            batch = DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f'{num_gen_batches=}. Keep generating...')
                                continue
                            else:
                                raise ValueError(
                                    f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
                                )
                        else:
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Compute log_prob Starts ..."
                    )
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Compute log_prob Ends ..."
                    )

                    if self.use_reference_policy:
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        if 'grpo_denormalize' in self.config.trainer and self.config.trainer.grpo_denormalize:
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                grpo_normalize=False,
                            )
                        else:
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                grpo_normalize=True,
                            )
                    
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Update Actor Starts ..."
                    )
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Update Actor Ends ..."
                    )

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, tokenizer=self.tokenizer))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                timing_raw = defaultdict(float)
                metrics["train/num_gen_batches"] = num_gen_batches
                if 'search_penalty' in self.config.trainer:
                    metrics["train/search_penalty"] = step_search_penalty

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    return
                
                progress_bar.update(1)
                self.global_steps += 1

                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
