import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from typing import Optional
import torch.nn.functional as F
import torch.nn as nn
import math

from .sparse_mask import generate_adaptive_mask

import matplotlib.pyplot as plt
def visualize_attention_mask(attention_mask: torch.Tensor):
    """
    Visualize a large attention mask without downsampling.

    Args:
        attention_mask (torch.Tensor): Attention mask tensor of shape [H, W].

    Returns:
        None
    """
    assert attention_mask.ndim == 2, "Attention mask must be 2D"
    H, W = attention_mask.shape
    print(f"Mask shape: {H} x {W}")

    # Move to CPU and convert to numpy
    mask_np = attention_mask.detach().cpu().numpy()

    # Plot
    fig = plt.figure(figsize=(12, 12))  # You can adjust figure size
    plt.imshow(mask_np, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Attention Mask (Full Size)")
    plt.axis('off')
    fig.savefig('mask.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Important to free memory
    exit()

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

    #attention_probs = attention_scores.softmax(dim=-1)
    #del attention_scores


    max_scores, _ = attention_scores.max(dim=-1, keepdim=True)
    attention_scores -= max_scores
    attention_scores.exp_()
    attention_scores /= attention_scores.sum(dim=-1, keepdim=True)
    attention_probs = attention_scores  # reuse memory

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
    attention_scores -= max_scores
    exp_scores = torch.exp(attention_scores)
    del attention_scores
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
        # Even there is a mask, also do full attention
        # Because we need ratio
        #baddbmm_input = attention_mask
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
    attention_scores -= max_scores
    
    attention_scores.exp_()
    exp_scores = attention_scores  # reuse tensor
    denominator = exp_scores.sum(dim=-1, keepdim=True)

    if attention_mask is not None:
        mask = attention_mask.to(dtype=exp_scores.dtype)
        masked_denominator = (exp_scores * mask).sum(dim=-1, keepdim=True)
        ratio = masked_denominator / denominator
    else:
        ratio = None

    attention_probs = exp_scores / denominator
    del exp_scores
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
    attention_scores -= max_scores

    attention_scores.exp_()
    exp_scores = attention_scores  # reuse tensor
    denominator = exp_scores.sum(dim=-1, keepdim=True)

    scaled_denominator = denominator / ratio
    exp_scores /= scaled_denominator
    attention_probs = exp_scores
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
    return mask

'''
def create_local_attention_mask(height, window_size, batch_size, dtype):
    device = "cuda"
    half_window = window_size // 2

    row_idx = torch.arange(height, device=device).view(-1, 1)
    col_idx = torch.arange(height, device=device).view(1, -1)

    # Build the boolean mask with clamped range
    mask = ((col_idx >= (row_idx - half_window)) & (col_idx <= (row_idx + half_window))).to(dtype)
    return mask
'''
def create_local_attention_mask(height, window_size, batch_size, dtype, N=0):
    device = "cuda"
    half_window = window_size // 2

    # Create the base mask [height, height]
    row_idx = torch.arange(height, device=device).view(-1, 1)
    col_idx = torch.arange(height, device=device).view(1, -1)
    base_mask = ((col_idx >= (row_idx - half_window)) & (col_idx <= (row_idx + half_window))).to(dtype)

    if N == 0:
        return base_mask

    # Allocate the full mask directly: [height + N, height + N]
    full_size = height + N
    full_mask = torch.ones((full_size, full_size), dtype=dtype, device=device)

    # Insert the local attention mask into the bottom-right block
    full_mask[N:, N:] = base_mask

    return full_mask

def generate_sliding_window_attention_mask(seq_length=17776, always_attended=226, frames=13, window_size=5, dtype=torch.bool):
    frame_size = (seq_length - always_attended) // frames  # Number of tokens per frame
    attention_mask = torch.zeros((seq_length, seq_length), dtype=dtype)

    # Always attended tokens can attend to everything
    attention_mask[:always_attended, :] = 1
    attention_mask[:, :always_attended] = 1

    # Apply sliding window attention within each frame, then duplicate across frames
    for j in range(frames):  # Iterate over frames
        frame_start = always_attended + j * frame_size

        for i in range(frame_size):  # Iterate over tokens in the frame
            token_index = frame_start + i

            # Determine window start and end, ensuring it fits within the frame boundaries
            half_window = window_size // 2
            window_start = max(0, i - half_window)  # Shift left if window extends outside frame start
            window_end = min(frame_size, i + half_window + 1)  # Shift right if window extends outside frame end

            # Convert to absolute positions in the sequence
            absolute_window_start = frame_start + window_start
            absolute_window_end = frame_start + window_end

            # Allow the token to attend to its window in its frame
            attention_mask[token_index, absolute_window_start:absolute_window_end] = 1

            # Duplicate the window for corresponding positions in all frames
            for other_frame in range(frames):
                if other_frame != j:  # Avoid self-attention outside frame
                    other_frame_start = always_attended + other_frame * frame_size
                    other_window_start = other_frame_start + window_start
                    other_window_end = other_frame_start + window_end
                    attention_mask[token_index, other_window_start:other_window_end] = 1

    return attention_mask

def full_attention_no_cache(attn: Attention, query, key, value, attention_mask, text_seq_length):
    attention_probs = get_attention_scores(query, key, None)
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)
    
    #hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    encoder_hidden_states, hidden_states = hidden_states.split(
        [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
    )
    return encoder_hidden_states, hidden_states

def full_attention_cache(attn: Attention, query, key, value, attention_mask, text_seq_length):
    attention_probs, attn.cached_ratio = get_attention_scores_and_ratio(query, key, attention_mask)
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)
    
    #hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    encoder_hidden_states, hidden_states = hidden_states.split(
        [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
    )
    return encoder_hidden_states, hidden_states

def window_attention(attn: Attention, query, key, value, attention_mask, text_seq_length):
    if attn.use_ratio == True:
        attention_mask = convert_binary_mask_to_additive(attention_mask)
        attention_probs = get_attention_scores_with_denominator_ratio(query, key, attn.cached_ratio, attention_mask)
    else:
        attention_mask = convert_binary_mask_to_additive(attention_mask)
        attention_probs = get_attention_scores(query, key, attention_mask)
        #attention_probs = get_attention_scores_with_denominator_ratio(query, key, 1, attention_mask)

    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)
    
    #hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    encoder_hidden_states, hidden_states = hidden_states.split(
        [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
    )
    return encoder_hidden_states, hidden_states

class Rettention_AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        residual = hidden_states
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)


        batch_size, sequence_length, _ = (
            hidden_states.shape #if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        sequence_length = sequence_length - text_seq_length

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from .embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        # Note: Add these lines for 1.0
        query = query.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        key = key.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        value = value.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        device = query.device
        attention_mask = attn.mask
        if attention_mask.device != device:
            attention_mask = attention_mask.to(device)

        if attn.stepi < 15: # Full attention
            encoder_hidden_states, hidden_states = full_attention_no_cache(attn, query, key, value, attention_mask, text_seq_length)
        else:

            if attn.stepi % 5 == 0:
                encoder_hidden_states, hidden_states = full_attention_cache(attn, query, key, value, attention_mask, text_seq_length)
                encoder_hidden_states_w, hidden_states_w = window_attention(attn, query, key, value, attention_mask, text_seq_length)
                attn.cached_residual = (hidden_states - hidden_states_w, encoder_hidden_states - encoder_hidden_states_w)

            else:
                encoder_hidden_states, hidden_states = window_attention(attn, query, key, value, attention_mask, text_seq_length)
                hidden_states = hidden_states + attn.cached_residual[0]
                encoder_hidden_states = encoder_hidden_states + attn.cached_residual[1]
                attn.cached_ratio += 0.04
                
        attn.stepi += 1
        return hidden_states, encoder_hidden_states