from collections import defaultdict

import numpy as np
import torch
import verl.utils.torch_functional as verl_F


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange, traj_level_loss=False):
    if not traj_level_loss:
        cliprange_high, cliprange_low = cliprange
        negative_approx_kl = log_prob - old_log_prob
        ratio = torch.exp(negative_approx_kl)
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange_low, 1.0 + cliprange_high)

        pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
        pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    else:
        cliprange_high, cliprange_low = cliprange
        negative_approx_kl = log_prob - old_log_prob
        ratio = torch.exp(negative_approx_kl)
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)
        negative_approx_kl_clipped = torch.clamp(
            negative_approx_kl, np.log(1 - cliprange_low), np.log(1 + cliprange_high)
        )
        ratio_adj = torch.exp(torch.sum(negative_approx_kl_clipped * eos_mask, dim=1, keepdim=True))
        ratio_adj = torch.clamp(ratio_adj, 1.0 - cliprange_low, 1.0 + cliprange_high)
        nonzero_mask = advantages != 0
        first_nonzero_indices = nonzero_mask.float().argmax(dim=1, keepdim=True)
        advantages_scalars = advantages.gather(1, first_nonzero_indices)
        pg_losses = -advantages_scalars * ratio_adj
        pg_losses2 = -advantages_scalars * torch.clamp(ratio_adj, 1.0 - cliprange_low, 1.0 + cliprange_high)
        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).float())

    return pg_loss, pg_clipfrac, ppo_kl


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    grpo_normalize=True,
):
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        if grpo_normalize:
            for i in range(bsz):
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            for i in range(bsz):
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores
