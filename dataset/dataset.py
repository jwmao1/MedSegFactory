import torch
from torchvision import transforms
import numpy as np
import random
import os
import glob
import torch.nn.functional as F
from PIL import Image
from skimage.color import rgb2gray


def map_to_classes(label_array, max_pixel):
    return np.clip(np.round(label_array * (max_pixel)), 0, max_pixel).astype(np.uint8)


def map_to_classes_isic(label_array):
    image = np.where(label_array >= 0.5, 1, 0)
    image = (image * 255.0).astype('uint8')
    return image


def map_to_classes2(label_array):
    image = np.where(label_array >= 0.5, 1, 0).astype('uint8')
    return image


def center_crop(image, crop_size):
    height, width = image.shape[:2]
    crop_height, crop_width = crop_size
    start_y = (height - crop_height) // 2
    start_x = (width - crop_width) // 2

    cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]

    return cropped_image


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, root, tokenizer, size=256, center_crop=True, t_drop_rate=0.05,
                 i_drop_rate=0.05, ti_drop_rate=0.05):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate

        self.data = glob.glob(os.path.join(root, '*', '*.npz'))

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.max_pixels = {
            'AMOS2022': 15,
            'ACDC': 3,
            'BUSI': 1,
            'CVC-ClinicDB': 1,
            'kvasir-seg': 1,
            'LiTS2017': 2,
            'KiTS2019': 2,
        }

        self.AMOS2022 = {1: 'liver', 2: 'right kidney', 3: 'spleen', 4: 'pancreas', 5: 'aorta', 6: 'inferior vena cava',
                         7: 'right adrenal gland', 8: 'left adrenal gland',
                         9: 'gall bladder', 10: 'esophagus', 11: 'stomach', 12: 'duodenum', 13: 'left kidney',
                         14: 'bladder', 15: 'prostate'}
        self.ACDC = {1: 'right ventricle', 2: 'myocardium', 3: 'left ventricle'}
        self.LiTS2017 = {1: 'liver', 2: 'liver tumor'}
        self.KiTS2019 = {1: 'kidney', 2: 'kidney tumor'}

        self.aspect_ratios = [
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

    def get_target_size(self, aspect_ratio, max_size=512):
        h_ratio, w_ratio = aspect_ratio
        if h_ratio > w_ratio:
            height = max_size
            # print(w_ratio, h_ratio)
            width = int(max_size * w_ratio / h_ratio)
        else:
            width = max_size
            height = int(max_size * h_ratio / w_ratio)

        return (height, width)

    def convert_to_rgb(self, image):
        if len(image.shape) == 2:
            rgb_img = np.stack((image, image, image), axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            rgb_img = image
        else:
            raise ValueError("不支持的图像格式")

        return rgb_img

    def __getitem__(self, idx):
        path = self.data[idx]
        name = path.split('/')[-2]

        # read image

        raw_image, ori_raw_mask = np.load(path)['image'], np.load(path)['label']
        kinds = np.unique(ori_raw_mask)
        raw_image, raw_mask = self.convert_to_rgb(raw_image), self.convert_to_rgb(ori_raw_mask)

        # original size
        # aspect = self.aspect_ratios[random.randint(0, len(self.aspect_ratios) - 1)]
        # shape = self.get_target_size(aspect, self.size)

        image_tensor = self.img_transform(raw_image)
        raw_mask = raw_mask / self.max_pixels[name]
        raw_mask = torch.from_numpy(raw_mask.transpose((2, 0, 1))).contiguous()
        mask_tensor = self.mask_transform(raw_mask)
        # image_tensor = transforms.Resize(size=shape)(image_tensor)
        # mask_tensor = transforms.Resize(size=shape)(mask_tensor)

        image = image_tensor.squeeze(dim=0)
        mask = mask_tensor.squeeze(dim=0)

        organ, kind = '', ''
        tips = []
        if name == 'AMOS2022':
            organ = 'abdomen CT scans'
            for k in kinds:
                if k == 0:
                    pass
                else:
                    tips.append(self.AMOS2022[k])

            if len(tips) != 0:
                random.shuffle(tips)
                for tip in tips:
                    if kind == '':
                        kind = tip
                    else:
                        kind = kind + ',' + tip

        elif name == 'ACDC':
            organ = 'cardiovascular ventricle mri'
            for k in kinds:
                if k == 0:
                    pass
                else:
                    tips.append(self.ACDC[k])

            if len(tips) != 0:
                random.shuffle(tips)
                for tip in tips:
                    if kind == '':
                        kind = tip
                    else:
                        kind = kind + ',' + tip

        elif name == 'BUSI':
            organ = 'breast ultrasound'
            if not kinds.any():
                kind = 'normal'
            else:
                kind = 'breast tumor'

        elif name == 'CVC-ClinicDB':
            organ = 'polyp colonoscopy'
            if not kinds.any():
                kind = 'normal'
            else:
                kind = 'polyp'

        elif name == 'kvasir-seg':
            organ = 'polyp colonoscopy'
            if not kinds.any():
                kind = 'normal'
            else:
                kind = 'polyp'

        elif name == 'LiTS2017':
            organ = 'abdomen CT scans'
            for k in kinds:
                if k == 0:
                    pass
                else:
                    tips.append(self.LiTS2017[k])

            if len(tips) != 0:
                random.shuffle(tips)
                for tip in tips:
                    if kind == '':
                        kind = tip
                    else:
                        kind = kind + ',' + tip

        elif name == 'KiTS2019':
            organ = 'abdomen CT scans'
            for k in kinds:
                if k == 0:
                    pass
                else:
                    tips.append(self.KiTS2019[k])

            if len(tips) != 0:
                random.shuffle(tips)
                for tip in tips:
                    if kind == '':
                        kind = tip
                    else:
                        kind = kind + ',' + tip

        if kind == '':
            kind = 'no found'

        img_text = f'a photo of {organ} image, with {kind}.'
        mask_text = f'a photo of {organ} label, with {kind}.'

        # if name == 'LiTS2017':
        #     print(kinds, img_text)

        # drop
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            img_text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            mask_text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            img_text = ""
            mask_text = ""

        # get text and tokenize
        img_text_input_ids = self.tokenizer(
            img_text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        mask_text_input_ids = self.tokenizer(
            mask_text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return {
            "image": image,
            "mask": mask,
            "img_text_input_ids": img_text_input_ids,
            "mask_text_input_ids": mask_text_input_ids,
            "raw_mask": ori_raw_mask,
            "kind": kind
        }

    def __len__(self):
        return len(self.data)


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
    shape = get_target_size(aspect, 512)

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
