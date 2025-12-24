import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from module.Cog_processor import Rettention_AttnProcessor
import argparse
from tqdm import tqdm, trange
import os

def generate_sliding_window_attention_mask(seq_length=17776, always_attended=226, frames=13, window_size=5):
    frame_size = (seq_length - always_attended) // frames  # Number of tokens per frame
    attention_mask = torch.zeros((seq_length, seq_length), dtype=torch.float16).to('cpu')

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

def transform_attn_processor(pipe, transfer_attn2=False,  use_ratio=True):
    blocks = pipe.transformer.transformer_blocks
    sparsity = 0.03125 # Hardcode this to change the sparsity
    shared_mask = generate_sliding_window_attention_mask(window_size= int(1350 * sparsity))
    for blocki, block in enumerate(blocks):
        block.attn1.processor = Rettention_AttnProcessor()
        if transfer_attn2:
            block.attn2.processor = Rettention_AttnProcessor()
        for layer in block.children():
            layer.stepi = 0
            layer.cached_ratio = None
            layer.cached_qkv = None
            layer.index = blocki
            layer.use_ratio = use_ratio
            layer.mask = shared_mask
    return pipe

def clear_stepi(pipe):
    blocks = pipe.transformer.transformer_blocks
    for blocki, block in enumerate(blocks):
        for layer in block.children():
            layer.stepi = 0
    return pipe

with open("captions/all_category.txt", 'r') as f:
    prompt_list = [item.strip() for item in f.readlines()]

sample_path = "VBench/Animal/Rettention_0.03125/videos"
os.makedirs(sample_path, exist_ok=True)

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.float16
).to("cuda")

pipe = transform_attn_processor(pipe, False, use_ratio=True)

pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()

for i in tqdm(range(0, len(prompt_list)), desc="data"):
    prompt = prompt_list[i]
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]
    path = os.path.join(sample_path, f"{i}.mp4")
    export_to_video(video, path, fps=8)
    clear_stepi(pipe)
