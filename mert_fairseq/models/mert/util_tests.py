import torch
import pytest

from .utils import get_1d_sincos_pos_embed, mix_logistic_loss, get_restore_indices

################################################
# Unit tests for Positoional Embedding
################################################

def test_shape():
    out = get_1d_sincos_pos_embed(8, torch.tensor([0, 1, 2, 3]))
    assert out.shape == (4, 8)

def test_position_zero():
    out = get_1d_sincos_pos_embed(6, torch.tensor([0]))
    expected = torch.tensor([[0., 1., 0., 1., 0., 1.]])
    assert torch.allclose(out, expected, atol=1e-6), f"Expected {expected}, but got {out}"

def test_sin_cos_identity():
    out = get_1d_sincos_pos_embed(6, torch.tensor([5]))
    sin_cos_pairs = out.view(-1, 2)
    values = (sin_cos_pairs**2).sum(dim=1)
    assert torch.allclose(values, torch.ones(3), atol=1e-6)

def test_different_positions():
    out = get_1d_sincos_pos_embed(8, torch.tensor([1, 2]))
    assert not torch.allclose(out[0], out[1])

def test_large_position():
    out = get_1d_sincos_pos_embed(16, torch.tensor([10**6]))
    assert torch.isfinite(out).all()


################################################
# Unit tests for Mixture of Logistic Loss
################################################
def test_output_is_scalar():
    N, D, n_mix = 2, 4, 3
    C = 3 * n_mix
    pred = torch.randn(N, C, D)
    target = torch.randn(N, 1, D)
    loss = mix_logistic_loss(pred, target, min_scale=-3)
    assert loss.ndim == 0, "Loss should return a scalar (average over batch)"

def test_minimum_scale_enforced():
    N, D, n_mix = 1, 2, 2
    C = 3 * n_mix
    pred = torch.zeros(N, C, D)  # raw scales = 0
    target = torch.zeros(N, 1, D)
    loss = mix_logistic_loss(pred, target, min_scale=-5)
    assert torch.isfinite(loss), "Loss should not be NaN or Inf"

def test_same_target_same_loss():
    """Batch should average to the same value if all samples are identical."""
    N, D, n_mix = 2, 3, 1
    C = 3 * n_mix
    pred = torch.tensor([[[0.0, 0.0, 0.0],
                          [1.0, 1.0, 1.0],
                          [0.0, 0.0, 0.0]]], dtype=torch.float32)  # (1, 3, 3)
    pred = pred.repeat(N, 1, 1)  # (N, C, D)
    target_1 = torch.ones(N, 1, D)
    target_2 = torch.ones(N, 1, D) 
    loss_1 = mix_logistic_loss(pred, target_1, min_scale=-2)
    loss_2 = mix_logistic_loss(pred, target_2, min_scale=-2)
    assert torch.isclose(loss_1, loss_2, atol=1e-4)

def test_one_mixture_component_equivalence():
    """With n_mix=1, should reduce to standard logistic NLL."""
    N, D, n_mix = 1, 2, 1
    C = 3 * n_mix
    logits = torch.zeros(N, 1, D)    # mixture log prob = 0
    means  = torch.zeros(N, 1, D)    # mean = 0
    scales = torch.zeros(N, 1, D)    # log(scale)=0
    pred   = torch.cat([logits, means, scales], dim=1)
    target = torch.zeros(N, 1, D)
    loss = mix_logistic_loss(pred, target, min_scale=-2)
    # logistic pdf at 0: log(1/4) = -log(4) ≈ -1.386
    expected = torch.log(torch.tensor(4.0, dtype=torch.float16))
    assert torch.allclose(loss, expected, atol=1e-4), f"Expected {expected}, but got {loss}"

def test_gradient_backprop():
    """Loss should support gradient backpropagation."""
    N, D, n_mix = 2, 3, 2
    C = 3 * n_mix
    pred = torch.randn(N, C, D, requires_grad=True)
    target = torch.randn(N, 1, D)
    loss = mix_logistic_loss(pred, target, min_scale=-3)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


################################################
# Unit tests for Restore Indices
################################################
@pytest.mark.parametrize("mask", [
    # Single batch, some masked
    torch.tensor([[0, 1, 0, 1]], dtype=torch.bool),

    # Single batch, all unmasked
    torch.tensor([[0, 0, 0, 0]], dtype=torch.bool),

    # Single batch, all masked
    torch.tensor([[1, 1, 1, 1]], dtype=torch.bool),

    # Multi-batch with different patterns
    torch.tensor([
        [0, 1, 0, 1],
        [1, 0, 0, 1],
    ], dtype=torch.bool),

    # Larger T
    torch.tensor([
        [0, 1, 1, 0, 1],
        [1, 1, 0, 0, 1],
    ], dtype=torch.bool),
])
def test_restore_indices(mask):
    B, T = mask.shape
    device = mask.device

    ids_restore = get_restore_indices(mask)

    # (1) shape check
    assert ids_restore.shape == (B, T)

    # (2) Permute [unmasked | masked]
    all_idx = torch.arange(T, device=device).expand(B, T)
    unmasked = all_idx[~mask].view(B, -1)
    masked = all_idx[mask].view(B, -1)
    perm = torch.cat([unmasked, masked], dim=1)

    # (3) Apply restore to perm → should recover original indices
    restored = torch.gather(perm, dim=1, index=ids_restore)

    for b in range(B):
        assert torch.equal(restored[b], all_idx[b]), f"Batch {restored[b], ids_restore[b]} failed"