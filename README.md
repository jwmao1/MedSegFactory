<p align="center">
  <img src="./docs/logo.png" height=150>
</p>

# MedSegFactory: Text-Guided Generation of Medical Image-Mask Pairs
<span>
<a href="https://arxiv.org/abs/2504.06897"><img src="https://img.shields.io/badge/arXiv-2504.06897-b31b1b.svg" height=22.5></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a>  
<a href="https://jwmao1.github.io/MedSegFactory_web/"><img src="https://img.shields.io/badge/project-MedSegFactory-purple.svg" height=22.5></a>
<a href="https://huggingface.co/spaces/JohnWeck/medsegfactory"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" height=22.5></a>
</span>

Code for the paper [MedSegFactory: Text-Guided Generation of Medical Image-Mask Pairs](https://arxiv.org/abs/2504.06897) will be available soon. 

### About this repo:

The repository contains the official implementation of "MedSegFactory".

## 🏥 Introduction 

> This paper presents **MedSegFactory**, a versatile medical synthesis framework that generates high-quality paired medical images and segmentation masks across modalities and tasks. It aims to serve as an unlimited data repository, supplying image-mask pairs to enhance existing segmentation tools. The core of MedSegFactory is a dual-stream diffusion model, where one stream synthesizes medical images and the other generates corresponding segmentation masks. To ensure precise alignment between image-mask pairs, we introduce Joint Cross-Attention (JCA), enabling a collaborative denoising paradigm by dynamic cross-conditioning between streams. This bidirectional interaction allows both representations to guide each other's generation, enhancing consistency between generated pairs.
MedSegFactory unlocks on-demand generation of paired medical images and segmentation masks through user-defined prompts that specify the target labels, imaging modalities, anatomical regions, and pathological conditions, facilitating scalable and high-quality data generation. This new paradigm of medical image synthesis enables seamless integration into diverse medical imaging workflows, enhancing both efficiency and accuracy. Extensive experiments show that MedSegFactory generates data of superior quality and usability, achieving competitive or state-of-the-art performance in 2D and 3D segmentation tasks while addressing data scarcity and regulatory constraints.
<br>

<img src="./docs/images/teaser.jpg" width="950"/>

## 🔥 News/TODO
- [x] [Paper](https://arxiv.org/abs/2504.06897) is released on ArXiv.
- [x] Source code of [gradio demo](https://huggingface.co/spaces/JohnWeck/medsegfactory/tree/main).
- [x] Code released.
- [x] Pretrained weight of [MedSegFactory](https://huggingface.co/JohnWeck/StableDiffusion/resolve/main/checkpoint-300.pth).
- [x] Training code.
- [ ] Pretrained weights of nnUNet for MedSegFactory.

## 🧑‍⚕️ Framework 

> **Overview of the MedSegFactory Training Pipeline.** _Left:_ A dual-stream generation framework built upon the Stable Diffusion model, where each paired image and segmentation mask $\{x_{i}, m_{i}\}$ are first encoded into latent representations and noised. These noised representations are then denoised by the Diffusion U-Net, producing the image and segmentation mask latent $\{\hat z_{0},\hat y_{0}\}$, which are ultimately decoded into a paired image and mask $\{\hat x_{i},\hat m_{i}\}$. To improve their semantic consistency, we introduce **Joint Cross-Attention (JCA)**, which allows the denoised latent representations of both streams to act as dynamic conditions, enhancing the mutual generation of images and masks. _Right:_ A detailed illustration of JCA, which applies two cross-attention functions to the intermediate denoising features $\{z_{t}, y_{t}\}$, enabling the cross-conditioning of the medical image and segmentation label for improved consistency in the final outputs.
<br>

<img src="./docs/images/framework.jpg" width="950"/>

## 💊 Checkpoints
you can download MedSegFactory checkpoint from [here](https://huggingface.co/JohnWeck/StableDiffusion/resolve/main/checkpoint-300.pth). Additionally, we provide the following nnUNet checkpoints:

|       Model       | Dataset | DSC | IoU |   Huggingface Download URL    |
|:-----------------:|:---:|:----:|:----:|:------------------------------------------------------------------------------------------------------:|
| MedSegFactory+nnUNet | AMOS | 0.7679 | 0.7042 |[AMOS](https://huggingface.co/JohnWeck/MedSegFactory_nnUNet/tree/main/Dataset001_AMOS)        |
| MedSegFactory+nnUNet | KiTS19 | 0.5803 | 0.5339 |[KiTS19](https://huggingface.co/JohnWeck/MedSegFactory_nnUNet/tree/main/Dataset005_KiTS)         |
| MedSegFactory+nnUNet | LiTS17 | 0.6237 | 0.5690 |[LiTS17](https://huggingface.co/JohnWeck/MedSegFactory_nnUNet/tree/main/Dataset006_LiTS)           |
| MedSegFactory+nnUNet | ACDC | 0.8802 | 0.8150 |[ACDC](https://huggingface.co/JohnWeck/MedSegFactory_nnUNet/tree/main/Dataset003_ACDC) |
| MedSegFactory+nnUNet | BUSI | 0.8338 | 0.7875 |[BUSI](https://huggingface.co/JohnWeck/MedSegFactory_nnUNet/tree/main/Dataset002_BUSI)          |
| MedSegFactory+nnUNet |  CVC-ClinicDB | 0.9085 | 0.8661 |[CVC-ClinicDB](https://huggingface.co/JohnWeck/MedSegFactory_nnUNet/tree/main/Dataset004_CVC)     |

## 💉 Quick Start 
**</h2>Quickly Run</h2>**
- Install these libraries: 
  ```shell
      # create new anaconda env
          conda create -n MedSeg python=3.10
          conda activate MedSeg 

      # install packages
          pip install -r requirements.txt

-  running the demo from the following choice: 
    ```shell
        # 1. key: AMOS2022; organ: abdomen CT scans; kind:
        # {liver, right kidney, spleen, pancreas, aorta, inferior vena cava,
        # right adrenal gland, left adrenal gland, gall bladder, esophagus,
        # stomach, duodenum, left kidney, bladder, prostate}.
        # 2. key: ACDC; organ: cardiovascular ventricle MRI; kind:
        # {right ventricle, myocardium, left ventricle}.
        # 3. key: BUSI; organ: breast ultrasound; kind:
        # {normal, breast tumor}.
        # 4. key: CVC-ClinicDB; organ: polyp colonoscopy; kind:
        # {polyp}.
        # 5. key: LiTS2017; organ: abdomen CT scans; kind:
        # {liver, liver tumor}.
        # 6. key: KiTS2019; organ: abdomen CT scans; kind:
        # {kidney, kidney tumor}.
        # using random prompt with specific key via key mode
        python predict.py --medsegfactory_ckpt [medsegfactory ckpt] --mode key --key [customized key]
        # For example:
        python predict.py --medsegfactory_ckpt [medsegfactory ckpt] --mode key --key BUSI
    
        # using customized prompt via prompt mode
        python predict.py --medsegfactory_ckpt [medsegfactory ckpt] --mode prompt --key [customized key] --organ [customized organ] --kind [customized kind]
        # For example:
        python predict.py --medsegfactory_ckpt [medsegfactory ckpt] --mode prompt --key ACDC --organ "cardiovascular ventricle MRI" --kind "right ventricle,myocardium,left ventricle"
    
-  training code:
    ```shell
        python tutorial_train.py --data_root_path [your data path] --output_dir [your output path]
    
## 🚑 Performance 

### Examples 

<img src="./docs/images/examples.jpg" width="1080"/>

## 🙏 Acknowledgement

Deeply appreciate these wonderful open source projects: [stablediffusion](https://github.com/Stability-AI/StableDiffusion), [clip](https://github.com/openai/CLIP), [controlnet](https://github.com/lllyasviel/ControlNet), [diffboost](https://github.com/NUBagciLab/DiffBoost), [maisi](https://github.com/Project-MONAI/tutorials/tree/main/generation/maisi), [nnunet](https://github.com/MIC-DKFZ/nnUNet).

## 🩺 Citation 

If you find this repository useful, please consider giving a star ⭐ and citation 💓:

```
@misc{mao2025medsegfactorytextguidedgenerationmedical,
      title={MedSegFactory: Text-Guided Generation of Medical Image-Mask Pairs}, 
      author={Jiawei Mao and Yuhan Wang and Yucheng Tang and Daguang Xu and Kang Wang and Yang Yang and Zongwei Zhou and Yuyin Zhou},
      year={2025},
      eprint={2504.06897},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.06897}, 
}
```
