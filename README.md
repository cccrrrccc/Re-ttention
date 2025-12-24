# Re-ttention: Ultra Sparse Visual Generation via Attention Statistical Reshape

<p align="center">
    <a href="https://www.python.org/" alt="Python">
        <img src="https://img.shields.io/badge/Python-3.10-yellow" /></a>
    <a href="https://pytorch.org/" alt="PyTorch">
        <img src="https://img.shields.io/badge/PyTorch-2.5.1-orange" /></a>
<p/>

# Overview
This is the official PyTorch implementation of the paper: "Re-ttention: Ultra Sparse Visual Generation via Attention Statistical Reshape", accepted at NeurIPS 2025.

# Installation

```
# Clone the repository
git clone https://github.com/cccrrrccc/Re-ttention.git
cd Re-ttention

# Create environment
conda create -n rettention python=3.10
conda activate rettention

# Install dependencies
pip install -r requirements.txt
```

# Quick Start
You can apply Re-ttention to existing DiT pipeline with just a few lines of code:
```
import torch
from diffusers import PixArtSigmaPipeline, PixArtAlphaPipeline
from module.pixart_processor import Rettention_AttnProcessor
import argparse
from tqdm import tqdm, trange
import os

def transform_attn_processor(pipe, transfer_attn2=False):
    blocks = pipe.transformer.transformer_blocks
    for blocki, block in enumerate(blocks):
        #print(f"substitude attn1 for layer {blocki}")
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

sample_path = "output"
os.makedirs(sample_path, exist_ok=True)

# You can replace the checkpoint id with "PixArt-alpha/PixArt-Sigma-XL-2-512-MS" too.
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16, generator=torch.manual_seed(42),
)

pipe = pipe.to("cuda")
pipe = transform_attn_processor(pipe)

prompt = "Input Prompt"
image = pipe(prompt, num_inference_steps=20).images[0]
image.save(os.path.join(sample_path, f"{i}.png"))
```

We also provide scripts to run Re-ttention on MS-COCO. HPSv2 and VBench in the [`scripts/`](./scripts/) directory.

# Cite
If you find this work useful for your research, please cite:

```
@inproceedings{chen2025re,
  title={Re-ttention: Ultra Sparse Visual Generation via Attention Statistical Reshape},
  author={Chen, Ruichen and Mills, Keith G and Jiang, Liyao and Gao, Chao and Niu, Di},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```
