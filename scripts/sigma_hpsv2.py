import torch
from diffusers import PixArtSigmaPipeline
from module.pixart_processor import Rettention_AttnProcessor
import argparse
from tqdm import tqdm, trange
import os

def transform_attn_processor(pipe, transfer_attn2=False):
    blocks = pipe.transformer.transformer_blocks
    for blocki, block in enumerate(blocks):
        block.attn1.processor = Rettention_AttnProcessor()
        if transfer_attn2:
            block.attn2.processor = Rettention_AttnProcessor()
        for layer in block.children():
            layer.stepi = 0
            layer.cached_ratio = None
            layer.cached_qkv = None
            layer.index = blocki
            layer.use_ratio = True # Hard code this to switch from normal mode and ratio mode
            layer.sparsity = 0.03125
    return pipe

def clear_stepi(pipe):
    blocks = pipe.transformer.transformer_blocks
    for blocki, block in enumerate(blocks):
        for layer in block.children():
            layer.stepi = 0
    return pipe

#For path setting
sample_path = "output_cache_5_step/Rettention_3.125_HPSv2/images" # Hard code this for path

os.makedirs(sample_path, exist_ok=True)
with open("captions/hpsv2.txt", 'r') as f:
    prompt_list = [item.strip() for item in f.readlines()]

# You can replace the checkpoint id with "PixArt-alpha/PixArt-Sigma-XL-2-512-MS" too.
pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=torch.float16, generator=torch.manual_seed(42),
)

pipe = pipe.to("cuda")

pipe = transform_attn_processor(pipe)

for i in tqdm(range(0, len(prompt_list)), desc="data"):
    prompt = prompt_list[i]
    image = pipe(prompt, num_inference_steps=20).images[0]
    image.save(os.path.join(sample_path, f"{i}.png"))
    clear_stepi(pipe)