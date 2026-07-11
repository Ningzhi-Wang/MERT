# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""MERT-specific Transformer encoder variants and attention layers."""

import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.distributed.fully_sharded_data_parallel import FullyShardedDataParallel
from fairseq.models.wav2vec.utils import pad_to_multiple
from fairseq.models.wav2vec.wav2vec2 import (
    ConformerWav2Vec2EncoderLayer,
    TransformerEncoder,
    TransformerSentenceEncoderLayer,
    TransformerSentenceEncoderWithAdapterLayer,
)
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.utils import index_put

if TYPE_CHECKING:
    from .mert_model import MERTConfig


logger = logging.getLogger(__name__)


def _normalize_position_ids(
    position_ids: Optional[torch.Tensor],
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Return per-example token positions, preserving positions after masking."""
    if position_ids is None:
        return torch.arange(sequence_length, device=device).unsqueeze(0).expand(
            batch_size, -1
        )
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    if position_ids.size(0) == 1 and batch_size > 1:
        position_ids = position_ids.expand(batch_size, -1)
    if position_ids.shape != (batch_size, sequence_length):
        raise ValueError(
            "position_ids must have shape "
            f"({batch_size}, {sequence_length}), got {tuple(position_ids.shape)}"
        )
    return position_ids.to(device=device, dtype=torch.long)


def _apply_rope(
    x: torch.Tensor, position_ids: torch.Tensor, base: float
) -> torch.Tensor:
    """Apply rotary position embeddings to ``[batch, heads, time, head_dim]``."""
    head_dim = x.size(-1)
    if head_dim % 2:
        raise ValueError("RoPE requires an even attention head dimension")

    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32)
            / head_dim
        )
    )
    angles = position_ids.to(dtype=torch.float32).unsqueeze(-1) * inv_freq
    cos = angles.cos().to(dtype=x.dtype).unsqueeze(1)
    sin = angles.sin().to(dtype=x.dtype).unsqueeze(1)

    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1
    )
    return rotated.flatten(-2)


class TransformerEncoder_extend(TransformerEncoder):
    def __init__(
        self,
        args: "MERTConfig",
        skip_pos_conv: bool = False,
        causal: bool = False,
        self_attn_type: str = "standard",
        mwmha_window_sizes: Optional[List[int]] = None,
    ):
        if causal and args.layer_type != "transformer":
            raise NotImplementedError(
                "Causal attention is only implemented for transformer layers in TransformerEncoder_extend."
            )
        self.causal = causal
        self.self_attn_type = self_attn_type
        self.mwmha_window_sizes = mwmha_window_sizes
        self.use_rope = getattr(args, "pos_type", "conv") == "rope"
        self.rope_base = getattr(args, "rope_base", 10000.0)
        super().__init__(args, skip_pos_conv=skip_pos_conv or self.use_rope)

        if args.deepnorm:
            # if is_encoder_decoder:
            #     init_scale = (
            #         math.pow(
            #             math.pow(args.encoder_layers, 4) * args.decoder_layers, 0.0625
            #         )
            #         / 1.15
            #     )
            # else:
            init_scale = math.pow(8.0 * args.encoder_layers, 0.25)
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.div_(init_scale)

    def build_encoder_layer(self, args: "MERTConfig", **kwargs):

        if args.layer_type == "transformer":
            if (
                args.deepnorm
                or args.subln
                or args.attention_relax > 0.0
                or self.self_attn_type != "standard"
                or self.use_rope
            ):
                residual_alpha = 1.0
                if args.deepnorm:
                    residual_alpha = math.pow(2.0 * args.encoder_layers, 0.25)

                layer = TransformerSentenceEncoderLayerExtend(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                    residual_alpha=residual_alpha,
                    attention_relax=args.attention_relax,
                    self_attn_type=self.self_attn_type,
                    mwmha_window_sizes=self.mwmha_window_sizes,
                    use_rope=self.use_rope,
                    rope_base=self.rope_base,
                )
            elif self.causal:
                layer = TransformerSentenceEncoderLayerCausal(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                )
            else:
                layer = TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                )

        elif args.layer_type == "conformer":
            layer = ConformerWav2Vec2EncoderLayer(
                embed_dim=self.embedding_dim,
                ffn_embed_dim=args.encoder_ffn_embed_dim,
                attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,
                activation_fn="swish",
                attn_type=args.attn_type,
                use_fp16=args.fp16,
                pos_enc_type="abs",
            )
        from fairseq.distributed import fsdp_wrap
        from fairseq.modules.checkpoint_activations import checkpoint_wrapper

        layer = fsdp_wrap(layer)
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        return layer

    def forward(
        self,
        x,
        padding_mask=None,
        layer=None,
        corpus_key=None,
        position_ids=None,
    ):
        if not self.causal and not self.use_rope:
            return super().forward(
                x, padding_mask=padding_mask, layer=layer, corpus_key=corpus_key
            )

        x, layer_results = self.extract_features(
            x,
            padding_mask=padding_mask,
            tgt_layer=layer,
            corpus_key=corpus_key,
            position_ids=position_ids,
        )

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def buffered_future_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]

    def extract_features(
        self,
        x,
        padding_mask=None,
        tgt_layer=None,
        min_layer=0,
        corpus_key=None,
        position_ids=None,
    ):
        if not self.causal and not self.use_rope:
            return super().extract_features(
                x,
                padding_mask=padding_mask,
                tgt_layer=tgt_layer,
                min_layer=min_layer,
                corpus_key=corpus_key,
            )

        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        if self.pos_conv is not None:
            x_conv = self.pos_conv(x.transpose(1, 2))
            x_conv = x_conv.transpose(1, 2)
            x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x, pad_length = pad_to_multiple(
            x, self.required_seq_len_multiple, dim=-2, value=0
        )
        if pad_length > 0 and padding_mask is None:
            padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            padding_mask[:, -pad_length:] = True
        else:
            padding_mask, _ = pad_to_multiple(
                padding_mask, self.required_seq_len_multiple, dim=-1, value=True
            )
        if self.use_rope:
            position_ids = _normalize_position_ids(
                position_ids, x.size(0), x.size(1) - pad_length, x.device
            )
            if pad_length > 0:
                position_ids = F.pad(position_ids, (0, pad_length))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.transpose(0, 1)

        self_attn_mask = self.buffered_future_mask(x) if self.causal else None

        layer_results = []
        r = None

        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random() if self.layerdrop > 0 else 1
            if not self.training or (dropout_probability > self.layerdrop):
                layer_check = layer
                if isinstance(layer, FullyShardedDataParallel):
                    layer_check = layer.unwrapped_module
                if (corpus_key is None) or (
                    not isinstance(
                        layer_check, (TransformerSentenceEncoderWithAdapterLayer,)
                    )
                ):
                    x, (z, lr) = layer(
                        x,
                        self_attn_mask=self_attn_mask,
                        self_attn_padding_mask=padding_mask,
                        need_weights=False,
                        **({"position_ids": position_ids} if self.use_rope else {}),
                    )
                else:
                    x, (z, lr) = layer(
                        x,
                        self_attn_mask=self_attn_mask,
                        self_attn_padding_mask=padding_mask,
                        need_weights=False,
                        corpus_key=corpus_key,
                    )
                if i >= min_layer:
                    layer_results.append((x, z, lr))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        x = x.transpose(0, 1)

        if pad_length > 0:
            x = x[:, :-pad_length]

            def undo_pad(a, b, c):
                return (
                    a[:-pad_length],
                    b[:-pad_length] if b is not None else b,
                    c[:-pad_length],
                )

            layer_results = [undo_pad(*u) for u in layer_results]

        return x, layer_results


class TransformerSentenceEncoderLayerCausal(TransformerSentenceEncoderLayer):
    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=need_weights,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, (attn, layer_result)


class TransformerSentenceEncoderLayerExtend(TransformerSentenceEncoderLayer):
    """
    Extend the Transformer Encoder Layer to support DeepNorm.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        residual_alpha: float = 1.0,
        subln: bool = False,
        attention_relax: float = -1.0,
        self_attn_type: str = "standard",
        mwmha_window_sizes: Optional[List[int]] = None,
        use_rope: bool = False,
        rope_base: float = 10000.0,
    ) -> None:

        super().__init__()
        # nn.Module().__init__(self)

        self.residual_alpha = residual_alpha
        self.use_rope = use_rope
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)

        if self_attn_type == "mwmha":
            if not mwmha_window_sizes:
                raise ValueError("MW-MHA requires non-empty mwmha_window_sizes")
            self.self_attn = MultiWindowMultiheadAttention(
                self.embedding_dim,
                num_attention_heads,
                dropout=attention_dropout,
                window_sizes=mwmha_window_sizes,
                use_rope=use_rope,
                rope_base=rope_base,
            )
        elif attention_relax > 0 or use_rope:
            logger.info(
                f"creating custom attention layer with relaxation scale: {attention_relax}"
            )
            self.self_attn = MultiheadAttention_extend(
                self.embedding_dim,
                num_attention_heads,
                dropout=attention_dropout,
                self_attention=True,
                attention_relax=attention_relax,
                use_rope=use_rope,
                rope_base=rope_base,
            )
        else:
            self.self_attn = MultiheadAttention(
                self.embedding_dim,
                num_attention_heads,
                dropout=attention_dropout,
                self_attention=True,
            )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        self.ffn_layernorm = LayerNorm(ffn_embedding_dim) if subln else None

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def residual_connection(self, x, residual):
        return residual * self.residual_alpha + x

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            attn_kwargs = {"position_ids": position_ids} if self.use_rope else {}
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
                **attn_kwargs,
            )
            x = self.dropout1(x)
            # x = residual + x
            x = self.residual_connection(x, residual)

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)

            # for subln
            if self.ffn_layernorm is not None:
                x = self.ffn_layernorm(x)

            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            # x = residual + x
            x = self.residual_connection(x, residual)

        else:
            attn_kwargs = {"position_ids": position_ids} if self.use_rope else {}
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
                **attn_kwargs,
            )

            x = self.dropout1(x)
            # x = residual + x
            x = self.residual_connection(x, residual)

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            # x = residual + x
            x = self.residual_connection(x, residual)
            x = self.final_layer_norm(x)

        return x, (attn, layer_result)


class MultiheadAttention_extend(MultiheadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        # dictionary=None,
        q_noise=0.0,
        qn_block_size=8,
        attention_relax=-1.0,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        # TODO: pass in config rather than string.
        # config defined in xformers.components.attention.AttentionConfig
        xformers_att_config: Optional[str] = None,
        xformers_blocksparse_layout: Optional[
            torch.Tensor
        ] = None,  # This should be part of the config
        xformers_blocksparse_blocksize: Optional[
            int
        ] = 16,  # This should be part of the config
    ):
        # nn.Module.__init__(self)
        # super().__init__()
        # initialize the instance with the father class method
        # MultiheadAttention.__init__(self,
        # super(MultiheadAttention_extend, self).__init__(
        # super(self).__init__(
        super().__init__(
            embed_dim,
            num_heads,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=self_attention,
            encoder_decoder_attention=encoder_decoder_attention,
            # dictionary=dictionary,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            xformers_att_config=xformers_att_config,
            xformers_blocksparse_layout=xformers_blocksparse_layout,
            xformers_blocksparse_blocksize=xformers_blocksparse_blocksize,
        )

        self.attention_relax = attention_relax
        self.use_rope = use_rope
        self.rope_base = rope_base

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[
            Dict[str, Dict[str, Optional[torch.Tensor]]]
        ] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        key_position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        if not self.skip_embed_dim_check:
            assert (
                embed_dim == self.embed_dim
            ), f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert value is not None
                assert src_len, key_bsz == value.shape[:2]

        if (
            not self.onnx_trace
            and not is_tpu  # don't use PyTorch version on TPUs
            and incremental_state is None
            and not static_kv
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
            and not torch.jit.is_scripting()
            # The Multihead attention implemented in pytorch forces strong dimension check
            # for input embedding dimention and K,Q,V projection dimension.
            # Since pruning will break the dimension check and it is not easy to modify the pytorch API,
            # it is preferred to bypass the pytorch MHA when we need to skip embed_dim_check
            and not self.skip_embed_dim_check
            and not self.use_rope
        ):
            assert key is not None and value is not None

            if self.use_xformers:
                return self._xformers_attn_forward(
                    query, key, value, key_padding_mask, need_weights, attn_mask
                )

            else:
                return F.multi_head_attention_forward(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    torch.empty([0]),
                    torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                    self.bias_k,
                    self.bias_v,
                    self.add_zero_attn,
                    self.dropout_module.p,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    self.training or self.dropout_module.apply_during_inference,
                    key_padding_mask.bool() if key_padding_mask is not None else None,
                    need_weights,
                    attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj.weight,
                    k_proj_weight=self.k_proj.weight,
                    v_proj_weight=self.v_proj.weight,
                )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                if self.beam_size > 1 and bsz == key.size(1):
                    # key is [T, bsz*beam_size, C], reduce to [T, bsz, C]
                    key = key.view(key.size(0), -1, self.beam_size, key.size(2))[
                        :, :, 0, :
                    ]
                    if key_padding_mask is not None:
                        key_padding_mask = key_padding_mask.view(
                            -1, self.beam_size, key_padding_mask.size(1)
                        )[:, 0, :]
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.use_rope:
            if k is None:
                raise ValueError("RoPE requires key projections")
            query_positions = _normalize_position_ids(
                position_ids, bsz, tgt_len, query.device
            )
            key_positions = _normalize_position_ids(
                key_position_ids if key_position_ids is not None else position_ids,
                k.size(1),
                k.size(0),
                query.device,
            )
            q = _apply_rope(
                q.view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3),
                query_positions,
                self.rope_base,
            ).permute(2, 0, 1, 3).reshape(tgt_len, bsz, embed_dim)
            k = _apply_rope(
                k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).permute(1, 2, 0, 3),
                key_positions,
                self.rope_base,
            ).permute(2, 0, 1, 3).reshape(k.size(0), k.size(1), embed_dim)

        if self.bias_k is not None:
            assert self.bias_v is not None
            k, v, attn_mask, key_padding_mask = self._add_bias(
                k, v, attn_mask, key_padding_mask, bsz
            )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        kv_bsz = bsz  # need default value for scripting
        if k is not None:
            kv_bsz = k.size(1)
            k = (
                k.contiguous()
                .view(-1, kv_bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, kv_bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                kv_bsz = _prev_key.size(0)
                prev_key = _prev_key.view(kv_bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                assert kv_bsz == _prev_value.size(0)
                prev_value = _prev_value.view(
                    kv_bsz * self.num_heads, -1, self.head_dim
                )
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[torch.Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=kv_bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(kv_bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(
                kv_bsz, self.num_heads, -1, self.head_dim
            )
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == kv_bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k, v, key_padding_mask, attn_mask = self._append_zero_attn(
                k=k, v=v, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )

        if self.encoder_decoder_attention and bsz != kv_bsz:
            attn_weights = torch.einsum(
                "bxhtd,bhsd->bxhts",
                q.view((kv_bsz, -1, self.num_heads) + q.size()[1:]),
                k.view((kv_bsz, self.num_heads) + k.size()[1:]),
            )
            attn_weights = attn_weights.reshape((-1,) + attn_weights.size()[-2:])
        else:
            attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.view(
                    kv_bsz, -1, self.num_heads, tgt_len, src_len
                )
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v
        if self.attention_relax > 0:
            # tgt_len == src_len

            # => (bsz, self.num_heads, tgt_len, src_len)
            # attn_weights_relax = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)/self.attention_relax

            # => (bsz * self.num_heads, tgt_len, src_len)
            attn_weights_relax = attn_weights / self.attention_relax

            # # => (bsz, self.num_heads, 1, src_len)
            # attn_max_relax = torch.max(attn_weights_relax, dim=-2, keepdim=False).unsqueeze(2)

            # find max according to K_j' => (bsz* self.num_heads, tgt_len, 1)
            attn_max_relax = torch.max(
                attn_weights_relax, dim=-1, keepdim=False
            ).unsqueeze(2)

            # => (bsz * self.num_heads, tgt_len, src_len)
            attn_weights = (attn_weights_relax - attn_max_relax) * self.attention_relax
            # attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn: Optional[torch.Tensor] = None
        if self.encoder_decoder_attention and bsz != kv_bsz:
            attn = torch.einsum(
                "bxhts,bhsd->bxhtd",
                attn_probs.view(
                    (
                        kv_bsz,
                        -1,
                        self.num_heads,
                    )
                    + attn_probs.size()[1:]
                ),
                v.view(
                    (
                        kv_bsz,
                        self.num_heads,
                    )
                    + v.size()[1:]
                ),
            )
            attn = attn.reshape((-1,) + attn.size()[-2:])
        else:
            attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, self.embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[torch.Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights


class MultiWindowMultiheadAttention(MultiheadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        window_sizes: List[int],
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        q_noise=0.0,
        qn_block_size=8,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        xformers_att_config: Optional[str] = None,
        xformers_blocksparse_layout: Optional[torch.Tensor] = None,
        xformers_blocksparse_blocksize: Optional[int] = 16,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
            encoder_decoder_attention=False,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            xformers_att_config=xformers_att_config,
            xformers_blocksparse_layout=xformers_blocksparse_layout,
            xformers_blocksparse_blocksize=xformers_blocksparse_blocksize,
        )
        self.window_sizes = tuple(window_sizes)
        self.use_rope = use_rope
        self.rope_base = rope_base

    @staticmethod
    def build_paper_window_sizes(num_patches: int, num_global_heads: int = 2) -> List[int]:
        wins = [w for w in range(2, num_patches) if num_patches % w == 0]
        wins.extend([num_patches] * num_global_heads)
        return wins

    def _resolve_window_sizes(self, seq_len: int) -> List[int]:
        if len(self.window_sizes) == 1:
            return [min(self.window_sizes[0], seq_len)] * self.num_heads

        if len(self.window_sizes) != self.num_heads:
            raise ValueError(
                f"window_sizes must have length 1 or {self.num_heads}, "
                f"got {len(self.window_sizes)}"
            )

        return [min(int(w), seq_len) for w in self.window_sizes]

    def _windowed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        window_size: int,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, head_dim = q.shape

        if key_padding_mask is None:
            key_padding_mask = q.new_zeros((bsz, seq_len), dtype=torch.bool)

        if window_size >= seq_len:
            attn_weights = torch.matmul(q, k.transpose(-2, -1))
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask.unsqueeze(0).to(attn_weights.dtype)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1), float("-inf")
            )
            attn_weights = utils.softmax(
                attn_weights, dim=-1, onnx_trace=self.onnx_trace
            )
            attn_probs = self.dropout_module(attn_weights).type_as(v)
            attn = torch.matmul(attn_probs, v)
            attn = attn.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            return attn, attn_weights if need_weights else None

        pad_len = (window_size - seq_len % window_size) % window_size
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            key_padding_mask = F.pad(key_padding_mask, (0, pad_len), value=True)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, pad_len, 0, pad_len), value=float("-inf"))

        seq_len_pad = q.size(1)
        num_windows = seq_len_pad // window_size

        q = q.view(bsz, num_windows, window_size, head_dim)
        k = k.view(bsz, num_windows, window_size, head_dim)
        v = v.view(bsz, num_windows, window_size, head_dim)

        attn_weights = torch.matmul(q, k.transpose(-2, -1))

        if attn_mask is not None:
            local_attn_mask = attn_mask.view(
                num_windows, window_size, num_windows, window_size
            )
            index = torch.arange(num_windows, device=attn_weights.device)
            local_attn_mask = local_attn_mask[index, :, index, :]
            attn_weights = attn_weights + local_attn_mask.unsqueeze(0).to(attn_weights.dtype)

        local_padding_mask = key_padding_mask.view(bsz, num_windows, window_size)
        attn_weights = attn_weights.masked_fill(
            local_padding_mask.unsqueeze(-2), float("-inf")
        )
        attn_weights = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_probs = self.dropout_module(attn_weights).type_as(v)

        attn = torch.matmul(attn_probs, v)
        attn = attn.masked_fill(local_padding_mask.unsqueeze(-1), 0.0)
        attn = attn.reshape(bsz, seq_len_pad, head_dim)[:, :seq_len]

        if not need_weights:
            return attn, None

        dense_weights = attn_probs.new_zeros((bsz, seq_len_pad, seq_len_pad))
        attn_weights = attn_weights.view(bsz, num_windows, window_size, window_size)
        window_indices = torch.arange(
            seq_len_pad, device=attn_weights.device
        ).view(
            num_windows, window_size
        )
        dense_weights[
            :,
            window_indices[:, :, None],
            window_indices[:, None, :],
        ] = attn_weights

        return attn, dense_weights[:, :seq_len, :seq_len]

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[
            Dict[str, Dict[str, Optional[torch.Tensor]]]
        ] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if before_softmax:
            raise NotImplementedError("before_softmax is not supported for MW-MHA")
        if incremental_state is not None or static_kv:
            raise NotImplementedError("incremental decoding is not supported for MW-MHA")
        if key is not None and key is not query:
            raise NotImplementedError("MW-MHA only supports self-attention")
        if value is not None and value is not query:
            raise NotImplementedError("MW-MHA only supports self-attention")

        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        if not self.skip_embed_dim_check:
            assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.contiguous().view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.contiguous().view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        if self.use_rope:
            position_ids = _normalize_position_ids(
                position_ids, bsz, tgt_len, query.device
            )
            q = _apply_rope(q, position_ids, self.rope_base)
            k = _apply_rope(k, position_ids, self.rope_base)

        window_sizes = self._resolve_window_sizes(tgt_len)
        head_outputs = []
        head_weights = [] if need_weights else None

        for head_idx, window_size in enumerate(window_sizes):
            head_output, head_weight = self._windowed_attention(
                q[:, head_idx],
                k[:, head_idx],
                v[:, head_idx],
                window_size=window_size,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=need_weights,
            )
            head_outputs.append(head_output.unsqueeze(1))
            if need_weights:
                head_weights.append(head_weight.unsqueeze(1))

        attn = torch.cat(head_outputs, dim=1)
        attn = attn.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)

        attn_weights = None
        if need_weights:
            attn_weights = torch.cat(head_weights, dim=1)
            if not need_head_weights:
                attn_weights = attn_weights.mean(dim=1)

        return attn, attn_weights
