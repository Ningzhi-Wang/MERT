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
    log_scale_min: float = -7.0,
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
    pred = pred.permute(0, 2, 1)
    data = data.permute(0, 2, 1)
    # Number of logistic distributions
    nr_mix = pred.shape[-1] // 3
    # unpack paramteres: distribution probability, mean, log scale
    logit_probs = pred[..., :nr_mix]
    mean = pred[..., nr_mix : 2 * nr_mix]
    log_scales = torch.clamp(pred[..., 2 * nr_mix : 3 * nr_mix], min=log_scale_min)

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

    inner_inner_cond = (cdf_delta > 1e-4).float()
    inner_inner_out = inner_inner_cond * torch.log(
        torch.clamp(cdf_delta, min=1e-4)
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
    return losses.mean()


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


def logistic_dist(a, b):
    base_distribution = dist.Uniform(0, 1)
    transforms = [dist.SigmoidTransform().inv, dist.AffineTransform(loc=a, scale=b)]
    return dist.TransformedDistribution(base_distribution, transforms)

def mix_logistic_loss(x, y):
    x = x.permute(0, 2, 1)
    y = y.permute(0, 2, 1).squeeze(-1)
    nr_mix = x.shape[-1] // 3
    logit_probs = x[..., :nr_mix]
    mean = x[..., nr_mix : 2 * nr_mix]
    scales = F.softplus(x[..., 2 * nr_mix :])
    log_probs = F.log_softmax(logit_probs, dim=-1)
    mix = dist.Categorical(logits=log_probs)
    comp = logistic_dist(mean, scales)
    mixture_distribution = dist.MixtureSameFamily(mix, comp)
    loss = -mixture_distribution.log_prob(y)
    return loss.mean()

