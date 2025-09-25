import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as dist

from torch import nn


def discretized_mix_logistic_loss(
    pred: torch.Tensor,
    data: torch.Tensor,
    num_classes: int = 256,
    log_scale_min: float = -6.0,
):
    """Discretized mix of logistic distributions loss.
    Note that it is assumed that input is scaled to [-1, 1]
    Args:
        pred: Tensor [batch_size, channels, width, height], predicted output.
        data: Tensor [batch_size, 1, width, height], Target.
    Returns:
        Tensor loss
    """
    # [Batch_size, width, height, channel]
    pred = pred.permute(0, 2, 1).to(torch.float32)
    data = data.permute(0, 2, 1).to(torch.float32)
    # Number of logistic distributions
    nr_mix = pred.shape[-1] // 3
    # unpack paramteres: distribution probability, mean, log scale
    logit_probs = pred[..., :nr_mix]
    mean = pred[..., nr_mix : 2 * nr_mix]
    log_scales = torch.clamp(pred[..., 2 * nr_mix : 3 * nr_mix], min=log_scale_min, max=3.0)

    # Repeat data with nr_mix channels
    data = data * torch.ones((1, 1, 1, nr_mix)).type_as(data)

    centered_data = data - mean
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_data + 1.0 / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_data - 1.0 / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)

    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_data
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(
        torch.clamp(cdf_delta, min=1e-8)
    ) + (1.0 - inner_inner_cond) * (
        log_pdf_mid - torch.log(torch.tensor((num_classes - 1) / 2))
    )
    inner_cond = (data > 0.999).float()
    inner_out = (
        inner_cond * log_one_minus_cdf_min + (1.0 - inner_cond) * inner_inner_out
    )
    cond = (data < -0.999).float()
    log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out
    log_probs = log_probs + F.log_softmax(logit_probs, dim=-1)

    losses = -torch.mean(torch.logsumexp(log_probs, dim=-1), dim=1)
    return losses.mean().to(torch.float16)


class Scaler(nn.Module):
    def __init__(self, init_min: float = math.inf, init_max: float = -math.inf):
        super().__init__()
        self.register_buffer("min", torch.tensor(init_min).float())
        self.register_buffer("max", torch.tensor(init_max).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        min_val = self.min.to(x.device)
        max_val = self.max.to(x.device)
        return (x - min_val) / max((max_val - min_val), 1e-4) * 2 - 1

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        min_val = self.min.to(x.device)
        max_val = self.max.to(x.device)
        return (x + 1) / 2 * (max_val - min_val) + min_val


def adaptive_update_hook(module: Scaler, input):
    x = input[0]
    if module.training:
        module.min.fill_(torch.min(module.min, x.min()))
        module.max.fill_(torch.max(module.max, x.max()))


def get_scaler(adaptive: bool = True, **kwargs) -> Scaler:
    scaler = Scaler(**kwargs)
    if adaptive:
        scaler.register_forward_pre_hook(adaptive_update_hook)
    return scaler

def get_individual_scaler(x):
    x_min = torch.amin(x, dim=(-1, -2))
    x_max = torch.amax(x, dim=(-1, -2))
    diff = x_max - x_min
    return (x - x_min) / diff * 2 - 1


def logistic_dist(a, b):
    base_distribution = dist.Uniform(0, 1)
    transforms = [dist.SigmoidTransform().inv, dist.AffineTransform(loc=a, scale=b)]
    return dist.TransformedDistribution(base_distribution, transforms)

def mix_logistic_loss(x, y, min_scale=-7.):
    x = x.permute(0, 2, 1)
    y = y.permute(0, 2, 1)
    nr_mix = x.shape[-1] // 3
    y = y * torch.ones((1, 1, nr_mix)).type_as(y)
    logit_probs = x[..., :nr_mix]
    mean = x[..., nr_mix : 2 * nr_mix]
    log_scales = torch.clamp(x[..., 2 * nr_mix : 3 * nr_mix].to(torch.float32), min_scale)
    z = (y - mean).to(torch.float32) / torch.exp(log_scales)
    mix_probs = F.log_softmax(logit_probs, dim=-1).to(torch.float32)
    log_probs = -log_scales - 2. * F.softplus(-z)
    loss = -torch.logsumexp(log_probs + mix_probs, dim=-1).mean(dim=1)
    return loss.mean().to(torch.float16)


def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=pos.dtype, device=pos.device)
    omega = omega / embed_dim * 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.zeros(len(pos), len(omega) * 2, device=out.device, dtype=out.dtype)  # (M, D)
    emb[:, 0::2] = emb_sin
    emb[:, 1::2] = emb_cos
    return emb


def get_restore_indices(mask):
    """
    mask: [B, T], 0 is keep, 1 is remove
    return:
        ids_restore: [B, L], the indices to restore original order
    """
    B, T = mask.shape
    all_indices = torch.arange(T, device=mask.device).expand(B, T)
    unmasked = all_indices[~mask].view(B, -1)
    masked = all_indices[mask].view(B, -1)
    perm = torch.cat([unmasked, masked], dim=1)
    ids_restore = torch.empty_like(perm)
    batch_ids = torch.arange(B).unsqueeze(1).expand_as(perm)
    ids_restore[batch_ids, perm] = all_indices
    return ids_restore