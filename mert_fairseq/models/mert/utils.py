import torch
import numpy as np
import torch.nn.functional as F


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
    pred = pred.permute(0, 2, 3, 1)
    data = data.permute(0, 2, 3, 1)
    # Number of logistic distributions
    nr_mix = pred.shape[-1] // 3
    # unpack paramteres: distribution probability, mean, log scale
    logit_probs = pred[:, :, :, :nr_mix]
    mean = pred[:, :, :, nr_mix : 2 * nr_mix]
    log_scales = torch.clamp(pred[:, :, :, 2 * nr_mix : 3 * nr_mix], min=log_scale_min)

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
        torch.clamp(cdf_delta, min=1e-12)
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

    losses = -torch.sum(torch.logsumexp(log_probs, dim=-1), [1, 2]) / pred.shape[1:3].numel()
    return losses.mean()
