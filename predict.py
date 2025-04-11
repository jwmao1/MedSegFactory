import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download
# Caching function setup
def cached_download(*args, **kwargs):
    print("Warning: cached_download is deprecated, using hf_hub_download instead.")
    return hf_hub_download(*args, **kwargs)

import sys
sys.modules["huggingface_hub.cached_download"] = cached_download

from diffusers import AutoencoderKL, DDPMScheduler
from StableDiffusion.Our_UNet import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from medical_pipeline import MedicalPipeline
from diffusers import DDIMScheduler
from StableDiffusion.Our_Pipe import StableDiffusionPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--model_repo_id",
        type=str,
        default=r"runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--medsegfactory_ckpt",
        type=str,
        default="./ckpt",
        help="medsegfactory checkpoint path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="key",
        choices=["key","prompt"],
        help="predict mode",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="BUSI",
        choices=['AMOS2022', 'BUSI', 'ACDC', 'CVC-ClinicDB', 'LiTS2017', 'KiTS2019'],
        help="keys for medsegfactory",
    )
    parser.add_argument(
        "--organ",
        type=str,
        default="polyp colonoscopy",
        choices=['abdomen CT scans', 'breast ultrasound', 'cardiovascular ventricle mri', 'polyp colonoscopy'],
        help="organ type for medsegfactory",
    )
    parser.add_argument(
        "--kind",
        type=str,
        default="polyp",
        help='''AMOS2022: [organ:abdomen CT scans, kind:{liver, right kidney, spleen, pancreas, aorta, inferior vena cava, 
        right adrenal gland, left adrenal gland, gall bladder, esophagus, stomach, duodenum, left kidney, bladder, prostate}],
        ACDC: [organ:cardiovascular ventricle mri, kind:{right ventricle, myocardium, left ventricle}],
        BUSI: [organ:breast ultrasound, kind:{normal, breast tumor}],
        CVC-ClinicDB: [organ:polyp colonoscopy, kind:{polyp}],
        LiTS2017: [organ:abdomen CT scans, kind:{liver, liver tumor}],
        KiTS2019: [organ:abdomen CT scans, kind:{kidney, kidney tumor'}]
        Follow the tips to use the organ and kind combinations correctly.'''

    )
    args = parser.parse_args()

    return args


args = parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
text_encoder = CLIPTextModel.from_pretrained(args.model_repo_id, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(args.model_repo_id, subfolder="vae")
unet = UNet2DConditionModel.from_config(args.model_repo_id, subfolder="unet")
unet.load_state_dict(torch.load(args.medsegfactory_ckpt, map_location='cpu'))
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

weight_dtype = torch.float16
unet.to(device, dtype=weight_dtype)
vae.to(device, dtype=weight_dtype)
text_encoder.to(device, dtype=weight_dtype)

sd_noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

# load SD pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    args.model_repo_id,
    torch_dtype=torch.float16,
    unet=unet,
    scheduler=sd_noise_scheduler,
    feature_extractor=None,
    safety_checker=None
)

pipeline = MedicalPipeline(pipe, device)

# keys ['AMOS2022', 'BUSI', 'ACDC', 'CVC-ClinicDB', 'LiTS2017', 'KiTS2019']

if args.mode == 'key':
    image, label = pipeline.generate(args.key)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(label)
    plt.axis('off')
    plt.savefig('pred.png', bbox_inches='tight', pad_inches=0)
    plt.show()
elif args.mode == 'prompt':
    prompt = {'organ':args.organ, 'kind':args.kind}
    image, label = pipeline.generate(prompt)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(label)
    plt.axis('off')
    plt.savefig('pred.png', bbox_inches='tight', pad_inches=0)
    plt.show()
else:
    raise RuntimeError('undefined mode')






