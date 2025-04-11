import os
import random
from typing import List
import torch.nn.functional as F
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from skimage.color import rgb2gray

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")


def get_generator(seed, device):
    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator


class MedicalPipeline:
    def __init__(self, sd_pipe, device):
        self.device = device
        self.pipe = sd_pipe.to(self.device)
        self.AMOS2022 = {1: 'liver', 2: 'right kidney', 3: 'spleen', 4: 'pancreas', 5: 'aorta', 6: 'inferior vena cava',
                         7: 'right adrenal gland', 8: 'left adrenal gland',
                         9: 'gall bladder', 10: 'esophagus', 11: 'stomach', 12: 'duodenum', 13: 'left kidney',
                         14: 'bladder', 15: 'prostate'}
        self.ACDC = {1: 'right ventricle', 2: 'myocardium', 3: 'left ventricle'}

        self.BUSI = {0: 'normal', 1: 'breast tumor'}
        self.CVC_ClinicDB = {1: 'polyp'}
        self.kvasir_seg = {1: 'polyp'}

        self.LiTS2017 = {1: 'liver', 2: 'liver tumor'}
        self.KiTS2019 = {1: 'kidney', 2: 'kidney tumor'}

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        return images

    def map_to_classes(self, label_array, max_pixel):
        """将标签值映射到 [0, 1, 2, ..., num_classes-1] 范围"""
        return np.clip(np.round(label_array * (max_pixel)), 0, max_pixel).astype(np.uint8)

    def get_random_values(self, my_dict):
        values_list = list(my_dict.values()) 
        num_choices = random.randint(1, len(values_list))  
        kinds = random.sample(values_list, num_choices)
        kind = ''

        for k in kinds:
            if kind == '':
                kind = k
            else:
                kind = kind + ',' + k

        return kind

    def generate(
            self,
            keys=None,
            prompts=None,
            negative_prompt=None,
            height=256,
            width=256,
            num_samples=1,
            seed=None,
            guidance_scale=7.5,
            num_inference_steps=50,
            **kwargs,
    ):
        if prompts is None:
            if not isinstance(keys, List):
                if keys is None:
                    idx = random.randint(1, 7)
                    if idx == 1:
                        keys = 'AMOS2022'
                        organ = 'abdomen CT scans'
                        kind = self.get_random_values(self.AMOS2022)
                        img_prompt = [f'a photo of {organ} image, with {kind}.'] * num_samples
                        mask_prompt = [f'a photo of {organ} label, with {kind}.'] * num_samples
                    elif idx == 2:
                        keys = 'BUSI'
                        organ = 'breast ultrasound'
                        choice = random.random()
                        if choice < 0.5:
                            kind = 'normal'
                        else:
                            kind = 'breast tumor'
                        img_prompt = [f'a photo of {organ} image, with {kind}.'] * num_samples
                        mask_prompt = [f'a photo of {organ} label, with {kind}.'] * num_samples
                    elif idx == 3:
                        keys = 'ACDC'
                        organ = 'cardiovascular ventricle mri'
                        kind = self.get_random_values(self.ACDC)
                        img_prompt = [f'a photo of {organ} image, with {kind}.'] * num_samples
                        mask_prompt = [f'a photo of {organ} label, with {kind}.'] * num_samples
                    elif idx == 4:
                        keys = 'CVC-ClinicDB'
                        organ = 'polyp colonoscopy'
                        kind = 'polyp'
                        img_prompt = [f'a photo of {organ} image, with {kind}.'] * num_samples
                        mask_prompt = [f'a photo of {organ} label, with {kind}.'] * num_samples
                    elif idx == 5:
                        keys = 'kvasir-seg'
                        organ = 'polyp colonoscopy'
                        kind = 'polyp'
                        img_prompt = [f'a photo of {organ} image, with {kind}.'] * num_samples
                        mask_prompt = [f'a photo of {organ} label, with {kind}.'] * num_samples
                    elif idx == 6:
                        keys = 'LiTS2017'
                        organ = 'abdomen CT scans'
                        kind = self.get_random_values(self.LiTS2017)
                        img_prompt = [f'a photo of {organ} image, with {kind}.'] * num_samples
                        mask_prompt = [f'a photo of {organ} label, with {kind}.'] * num_samples
                    elif idx == 7:
                        keys = 'KiTS2019'
                        organ = 'abdomen CT scans'
                        kind = self.get_random_values(self.KiTS2019)
                        img_prompt = [f'a photo of {organ} image, with {kind}.'] * num_samples
                        mask_prompt = [f'a photo of {organ} label, with {kind}.'] * num_samples
                    else:
                        raise RuntimeError('no mode')

                    print(img_prompt, mask_prompt)

                else:
                    if keys == 'AMOS2022':
                        organ = 'abdomen CT scans'
                        kind = self.get_random_values(self.AMOS2022)
                    elif keys == 'BUSI':
                        organ = 'breast ultrasound'
                        choice = random.random()
                        if choice < 0.5:
                            kind = 'normal'
                        else:
                            kind = 'breast tumor'
                    elif keys == 'ACDC':
                        organ = 'cardiovascular ventricle mri'
                        kind = self.get_random_values(self.ACDC)
                    elif keys == 'CVC-ClinicDB':
                        organ = 'polyp colonoscopy'
                        kind = 'polyp'
                    elif keys == 'kvasir-seg':
                        organ = 'polyp colonoscopy'
                        kind = 'polyp'
                    elif keys == 'LiTS2017':
                        organ = 'abdomen CT scans'
                        kind = self.get_random_values(self.LiTS2017)
                    elif keys == 'KiTS2019':
                        organ = 'abdomen CT scans'
                        kind = self.get_random_values(self.KiTS2019)
                    else:
                        raise RuntimeError('undefined keys')

                    img_prompt = [f'a photo of {organ} image, with {kind}.'] * num_samples
                    mask_prompt = [f'a photo of {organ} label, with {kind}.'] * num_samples
                    keys = keys
                    print(img_prompt, mask_prompt)
            else:
                # img_prompt, mask_prompt = [], []
                # for key in keys:
                #     img_prompt.append(f'a photo of {organ} image, with {kind}.')
                #     mask_prompt.append(f'a photo of {organ} label, with {kind}.')
                # img_prompt = img_prompt * num_samples
                # mask_prompt = mask_prompt * num_samples
                img_prompt = [f'a photo of {organ} image, with {kind}.'] * num_samples
                mask_prompt = [f'a photo of {organ} label, with {kind}.'] * num_samples

        else:
            organ, kind = prompts['organ'], prompts['kind']
            img_prompt = [f'a photo of {organ} image, with {kind}.'] * num_samples
            mask_prompt = [f'a photo of {organ} label, with {kind}.'] * num_samples
            keys = prompts['key']

        with torch.inference_mode():
            img_prompt_embeds_, img_negative_prompt_embeds_ = self.pipe.encode_prompt(
                img_prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            mask_prompt_embeds_, mask_negative_prompt_embeds_ = self.pipe.encode_prompt(
                mask_prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([img_prompt_embeds_, img_negative_prompt_embeds_], dim=0)
            negative_prompt_embeds = torch.cat([mask_prompt_embeds_, mask_negative_prompt_embeds_], dim=0)

        generator = get_generator(seed, self.device)

        data = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type='np',
            **kwargs,
        ).images

        # num = data.shape[0]
        # index = int(num // 2)

        image, label = data[0], data[1]

        label = rgb2gray(label)
        image = self.numpy_to_pil(image).squeeze()
        
        
        if keys == 'AMOS2022':
            label = self.map_to_classes(label, 15)
        elif keys == 'ACDC':
            label = self.map_to_classes(label, 3)
        elif keys == 'BUSI':
            label = self.map_to_classes(label, 1)
        elif keys == 'CVC-ClinicDB':
            label = self.map_to_classes(label, 1)
        elif keys == 'kvasir-seg':
            label = self.map_to_classes(label, 1)
        elif keys == 'LiTS2017':
            label = self.map_to_classes(label, 2)
        elif keys == 'KiTS2019':
            label = self.map_to_classes(label, 2)

        return image, label


