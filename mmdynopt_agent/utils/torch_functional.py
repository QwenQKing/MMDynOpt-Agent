from typing import Dict, List, Optional, Union

import torch
import torch.distributed
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from verl.utils.torch_functional import logprobs_from_logits_v2

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


def logprobs_from_logits(logits, labels):
    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logits = logits.reshape(-1, last_dim)
        labels = labels.reshape(-1)
        output = logprobs_from_logits_flash_attn(logits, labels)
        output = output.view(*batch_dim)
    else:
        output = logprobs_from_logits_v2(logits, labels)
    return output


def logprobs_from_logits_flash_attn(logits, labels):
    output = cross_entropy_loss(logits, labels)
    assert isinstance(
        output, tuple
    ), "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    return -output[0]


def logprobs_from_logits_naive(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logpy = gather_from_labels(logp, labels)
    return logpy


def logprobs_from_logits_v2(logits: torch.FloatTensor, labels):
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.stack([torch.logsumexp(l, dim=-1) for l in logits])
        logprobs_labels = logits_labels - logsumexp_values
    else:
        logprobs_labels = []
        for row_logits, row_labels in zip(logits, labels):
            row_logprobs = F.log_softmax(row_logits, dim=-1)
            row_logprobs_labels = row_logprobs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            logprobs_labels.append(row_logprobs_labels)
        logprobs_labels = torch.stack(logprobs_labels)
    return logprobs_labels


import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_sigmoid_decay_schedule(
    optimizer: Optimizer,
    lr_low: float,
    num_warmup_steps: int,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        k = (num_warmup_steps * 0.1) ** -1
        lr_high = 1
        lr = (lr_high - lr_low) / (1 + math.exp(k * (current_step - num_warmup_steps))) + lr_low
        return lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_final_eos_mask(response_id: torch.Tensor, eos_token: Union[int, List[int]] = 2, dtype=torch.int64):
    if isinstance(eos_token, int):
        eos_token = [eos_token]

    eos_mask = torch.zeros_like(response_id, dtype=torch.bool)
    for token in eos_token:
        eos_mask |= response_id.eq(token)

    eos_mask = eos_mask.long()

    reversed_mask = torch.flip(eos_mask, dims=[1])
    reversed_cumsum = torch.cumsum(reversed_mask, dim=1)

    new_mask = reversed_cumsum.gt(0)
    new_mask = torch.flip(new_mask, dims=[1])

    return new_mask.to(dtype)


def get_eos_mask(response_id: torch.Tensor, eos_token: Union[int, List[int]] = 2, dtype=torch.int64):
    if isinstance(eos_token, int):
        eos_token = [eos_token]

    eos_mask = torch.zeros_like(response_id, dtype=torch.bool)
    for token in eos_token:
        eos_mask |= response_id.eq(token)

    eos_mask = eos_mask.long()
    eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
    eos_mask = torch.logical_not(eos_mask).to(dtype)
    return eos_mask
