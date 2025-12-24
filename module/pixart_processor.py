import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from typing import Optional
import torch.nn.functional as F

import torch.nn as nn
import math

from .sparse_mask import generate_adaptive_mask

def get_attention_scores(
    query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""
    Compute the attention scores.

    Args:
        query (`torch.Tensor`): The query tensor.
        key (`torch.Tensor`): The key tensor.
        attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

    Returns:
        `torch.Tensor`: The attention probabilities/scores.
    """
    dtype = query.dtype
    scale = 1 / math.sqrt(query.shape[-1])

    if attention_mask is None:
        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0
    else:
        baddbmm_input = attention_mask
        beta = 1
    attention_scores = torch.baddbmm(
        baddbmm_input,
        query,
        key.transpose(-1, -2),
        beta=beta,
        alpha=scale,
    )
    del baddbmm_input

    attention_probs = attention_scores.softmax(dim=-1)
    del attention_scores

    attention_probs = attention_probs.to(dtype)

    return attention_probs

def get_attention_scores_and_denominator(
    query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""
    Compute the attention scores.

    Args:
        query (`torch.Tensor`): The query tensor.
        key (`torch.Tensor`): The key tensor.
        attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

    Returns:
        `torch.Tensor`: The attention probabilities/scores.
    """
    dtype = query.dtype
    scale = 1 / math.sqrt(query.shape[-1])

    if attention_mask is None:
        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0
    else:
        baddbmm_input = attention_mask
        beta = 1
    attention_scores = torch.baddbmm(
        baddbmm_input,
        query,
        key.transpose(-1, -2),
        beta=beta,
        alpha=scale,
    )
    del baddbmm_input

    #attention_probs = attention_scores.softmax(dim=-1)
    #del attention_scores
    # Must have max to avoid overflow
    # Choice: just cache denominator? Or cache the max as well?
    max_scores, _ = attention_scores.max(dim=-1, keepdim=True)
    attention_scores_stable = attention_scores - max_scores
    del attention_scores
    exp_scores = torch.exp(attention_scores_stable)
    del attention_scores_stable
    denominator = exp_scores.sum(dim=-1, keepdim=True)
    attention_probs = exp_scores / denominator

    attention_probs = attention_probs.to(dtype)

    return attention_probs, denominator

def get_attention_scores_and_ratio(
    query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""
    Compute the attention scores.

    Args:
        query (`torch.Tensor`): The query tensor.
        key (`torch.Tensor`): The key tensor.
        attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

    Returns:
        `torch.Tensor`: The attention probabilities/scores.
    """
    dtype = query.dtype
    scale = 1 / math.sqrt(query.shape[-1])

    if attention_mask is None:
        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0
    else:
        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0
    attention_scores = torch.baddbmm(
        baddbmm_input,
        query,
        key.transpose(-1, -2),
        beta=beta,
        alpha=scale,
    )
    del baddbmm_input
    max_scores, _ = attention_scores.max(dim=-1, keepdim=True)
    attention_scores_stable = attention_scores - max_scores
    del attention_scores
    exp_scores = torch.exp(attention_scores_stable)
    del attention_scores_stable
    denominator = exp_scores.sum(dim=-1, keepdim=True)
    attention_probs = exp_scores / denominator

    if attention_mask is not None:
        mask = attention_mask.to(dtype=exp_scores.dtype)
        masked_denominator = (exp_scores * mask).sum(dim=-1, keepdim=True)
        ratio = masked_denominator / denominator
    else:
        ratio = None

    attention_probs = attention_probs.to(dtype)

    return attention_probs, ratio

def get_attention_scores_with_denominator_ratio(
    query: torch.Tensor, key: torch.Tensor, ratio: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""
    Compute the attention scores.

    Args:
        query (`torch.Tensor`): The query tensor.
        key (`torch.Tensor`): The key tensor.
        attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

    Returns:
        `torch.Tensor`: The attention probabilities/scores.
    """
    dtype = query.dtype
    scale = 1 / math.sqrt(query.shape[-1])

    if attention_mask is None:
        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0
    else:
        baddbmm_input = attention_mask
        beta = 1
    attention_scores = torch.baddbmm(
        baddbmm_input,
        query,
        key.transpose(-1, -2),
        beta=beta,
        alpha=scale,
    )
    del baddbmm_input
    max_scores, _ = attention_scores.max(dim=-1, keepdim=True)
    attention_scores_stable = attention_scores - max_scores
    del attention_scores
    exp_scores = torch.exp(attention_scores_stable)
    del attention_scores_stable
    denominator = exp_scores.sum(dim=-1, keepdim=True)
    scaled_denominator = denominator / ratio
    attention_probs = exp_scores / scaled_denominator
    attention_probs = attention_probs.to(dtype)

    return attention_probs

def convert_binary_mask_to_additive(mask: torch.Tensor, masked_value: float = -1e5):
    """
    Convert a binary attention mask from [0, 1] to [-1e5, 0] for additive masking.

    Args:
        mask (torch.Tensor): Input binary mask of shape [..., seq, seq], values 0 or 1.
        masked_value (float): Value to use for masked positions (default: -1e5).

    Returns:
        torch.Tensor: Additive attention mask of the same shape.
    """
    additive_mask = (1.0 - mask) * masked_value
    return additive_mask

def create_local_attention_mask_old(height, window_size, batch_size, dtype):
    mask = torch.zeros((height, height), device="cuda", dtype=dtype)
    half_window = int(window_size // 2)
    for i in range(height):
        # Centered window
        start_i = i - half_window
        end_i = start_i + window_size

        # Clamp to stay within [0, height]
        if start_i < 0:
            start_i = 0
            end_i = window_size
        if end_i > height:
            end_i = height
            start_i = height - window_size


        mask[i, start_i:end_i] = 1
    #Wmask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
    return mask

def create_local_attention_mask(height, window_size, batch_size, dtype):
    device = "cuda"
    half_window = window_size // 2

    row_idx = torch.arange(height, device=device).view(-1, 1)
    col_idx = torch.arange(height, device=device).view(1, -1)

    # Build the boolean mask with clamped range
    mask = ((col_idx >= (row_idx - half_window)) & (col_idx <= (row_idx + half_window))).to(dtype)
    return mask

def full_attention(attn: Attention, query, key, value, attention_mask, residual):
    input_ndim = residual.ndim
    attention_probs, attn.cached_ratio = get_attention_scores_and_ratio(query, key, attention_mask)
    hidden_states = torch.bmm(attention_probs, value)


    hidden_states = attn.batch_to_head_dim(hidden_states)
    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor
    return hidden_states

def window_attention(attn: Attention, query, key, value, attention_mask, residual):
    input_ndim = residual.ndim
    if attn.use_ratio == True:
        attention_mask = convert_binary_mask_to_additive(attention_mask)
        attention_probs = get_attention_scores_with_denominator_ratio(query, key, attn.cached_ratio, attention_mask)
    else:
        attention_mask = convert_binary_mask_to_additive(attention_mask)
        attention_probs = get_attention_scores(query, key, attention_mask)

    hidden_states = torch.bmm(attention_probs, value)

    # For caching the masked QKV
    #if attn.use_ratio == True:
    #    hidden_states = hidden_states + attn.cached_qkv

    hidden_states = attn.batch_to_head_dim(hidden_states)
    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor
    return hidden_states

class Rettention_AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        # Create Attention mask
        #attention_mask = create_local_attention_mask(sequence_length, int(sequence_length/4*1), batch_size, hidden_states.dtype)

        #attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)


        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)


        attention_mask = create_local_attention_mask(sequence_length, int(sequence_length * attn.sparsity), batch_size, hidden_states.dtype)
        
        if attn.stepi < 5: # Full attention
            hidden_states = full_attention(attn, query, key, value, attention_mask, residual)
        else:

            if attn.stepi in [5, 10, 15]:
                hidden_states = full_attention(attn, query, key, value, attention_mask, residual)
                w = window_attention(attn, query, key, value, attention_mask, residual)
                attn.cached_residual = hidden_states - w
            else:
                hidden_states = window_attention(attn, query, key, value, attention_mask, residual)
                hidden_states = hidden_states + attn.cached_residual
                attn.cached_ratio += 0.04
        attn.stepi += 1
        return hidden_states