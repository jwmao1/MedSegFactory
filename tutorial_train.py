import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
def cached_download(*args, **kwargs):
    print("Warning: cached_download is deprecated, using hf_hub_download instead.")
    return hf_hub_download(*args, **kwargs)
import sys
sys.modules["huggingface_hub.cached_download"] = cached_download
from diffusers import AutoencoderKL, DDPMScheduler
from StableDiffusion.Our_UNet import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from dataset.dataset import MyDataset
from medical_pipeline import MedicalPipeline
from diffusers import DDIMScheduler
from StableDiffusion.Our_Pipe import StableDiffusionPipeline
from tqdm import tqdm

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def collate_fn(data):
    
    aspect_ratios = [
            (16, 9),  # 16:9
            (4, 3),  # 4:3
            (3, 2),  # 3:2
            (1, 1),  # 1:1
            (2, 1),  # 2:1
            (9, 16),  # 9:16
            (5, 4),  # 5:4
            (3, 4),  # 3:4
            (2, 3)  # 2:3
        ]

    def get_target_size(aspect_ratio, max_size=256):
        h_ratio, w_ratio = aspect_ratio
        if h_ratio > w_ratio:
            height = max_size
            # print(w_ratio, h_ratio)
            width = int(max_size * w_ratio / h_ratio)
        else:
            width = max_size
            height = int(max_size * h_ratio / w_ratio)

        return (height, width)
    
    aspect = aspect_ratios[random.randint(0, len(aspect_ratios) - 1)]
    shape = get_target_size(aspect, 256)
	
    # shape = (256, 256)
    images = torch.stack([transforms.Resize(size=shape)(example["image"]) for example in data])
    masks = torch.stack([transforms.Resize(size=shape)(example["mask"]) for example in data])
    img_text_input_ids = torch.cat([example["img_text_input_ids"] for example in data], dim=0)
    mask_text_input_ids = torch.cat([example["mask_text_input_ids"] for example in data], dim=0)

    return {
        "images": images,
        "masks": masks,
        "img_text_input_ids": img_text_input_ids,
        "mask_text_input_ids": mask_text_input_ids,
    }
    
def dummy(images, **kwargs): 
	return images, False

    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=r"runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="./train",
        help="Training data root path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./save",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="strach",
        choices=["strach","finetune"],
        help="trainging mode",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=10000)
    parser.add_argument(
        "--train_batch_size", type=int, default=20, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    if args.mode == 'finetune':
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", low_cpu_mem_usage=False, device_map=None)
    elif args.mode == 'strach':
        unet = UNet2DConditionModel.from_config(args.pretrained_model_name_or_path, subfolder="unet")
    else:
        raise RuntimeError('incorrect mode')
    # freeze parameters of models to save more memory
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

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
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        unet=unet,
        scheduler=sd_noise_scheduler,
        feature_extractor=None,
        safety_checker=None
    )
    
    pipe.safety_checker = dummy

    pipeline = MedicalPipeline(pipe, accelerator.device)

    # for name, param in unet.named_parameters():
    #     if "attn3" in name:
    #         param.requires_grad = True

    # optimizer
    # for name, param in unet.named_parameters():
    #     if param.requires_grad == True:
    #         print(name)
    
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(filter(lambda p : p.requires_grad, unet.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MyDataset(args.data_root_path, tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
    
    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(unet):
                images = batch["images"].to(accelerator.device, dtype=weight_dtype)
                masks = batch["masks"].to(accelerator.device, dtype=weight_dtype)
                # print(images.shape, masks.shape)
                inputs = torch.cat([images, masks], dim=0).to(accelerator.device, dtype=weight_dtype)
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(inputs).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
                with torch.no_grad():
                    img_encoder_output = text_encoder(batch['img_text_input_ids'].to(accelerator.device))[0]
                    mask_encoder_output = text_encoder(batch['mask_text_input_ids'].to(accelerator.device))[0]
                    encoder_hidden_states = torch.cat([img_encoder_output, mask_encoder_output], dim=0)

                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))
            
            global_step += 1
            begin = time.perf_counter()  
        
        if epoch % 10 == 0:
            
            # if accelerator.is_main_process:
            keys = ['AMOS2022','BUSI','ACDC','CVC-ClinicDB','kvasir-seg','LiTS2017','KiTS2019']

            for key in keys:
                image, label, name = pipeline.generate(key, height=args.resolution, width=args.resolution)
                np.savez(os.path.join(args.output_dir, 'imgs', f'{epoch}_{key}'), image=image, label=label, txt=name)

            
        if epoch % 100 == 0:
            save_path = os.path.join(args.output_dir, 'ckpts', f"checkpoint-{epoch}.pth")
            # accelerator.save_state(save_path)
            torch.save(unet.module.state_dict(),save_path)
                
if __name__ == "__main__":
    main()    
