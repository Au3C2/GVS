from os.path import splitext
import os.path
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }

class LiverDataset(Dataset):
    def __init__(self, imgs_dir, other_dir=''):
        self.imgs_dir = imgs_dir
        self.other_dir = other_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((-99.5946,-99.5946), (126.5396,126.5396))
        ])

        with open(self.imgs_dir,'r') as fp:
             img_list = fp.readlines()
        for i in range(len(img_list)):
            img_list[i] = img_list[i].strip()
        if self.other_dir != '':
            with open(self.other_dir,'r') as fp:
                other_list = fp.readlines()
            for i in range(len(other_list)):
                other_list[i] = other_list[i].strip()
            self.img_list =  img_list + other_list
        else:
            self.img_list =  img_list
        # self.idx = 0

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,idx):
        data = np.load(self.img_list[idx]).item()
        case_name = (self.img_list[idx]).split('/')[-2]
        slice_idx = os.path.basename(self.img_list[idx])[:6]
        img = data['liver']
        seg = data['seg_label']
        seg_tumor = np.where(seg==2, 1, 0) #只留肿瘤标签
        img = torch.tensor(img)
        seg = torch.tensor(seg)
        seg_tumor = torch.tensor(seg_tumor)

        return img, seg, seg_tumor, [case_name,slice_idx]

class BrainDataset(Dataset):
    def __init__(self, imgs_dir, other_dir='',has_mean=True):
        self.has_mean = has_mean
        self.imgs_dir = imgs_dir
        self.other_dir = other_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((-99.5946,-99.5946), (126.5396,126.5396))
        ])

        with open(self.imgs_dir,'r') as fp:
             img_list = fp.readlines()
        for i in range(len(img_list)):
            img_list[i] = img_list[i].strip()
        if self.other_dir != '':
            with open(self.other_dir,'r') as fp:
                other_list = fp.readlines()
            for i in range(len(other_list)):
                other_list[i] = other_list[i].strip()
            self.img_list =  img_list + other_list
        else:
            self.img_list =  img_list
        # self.idx = 0

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,idx):
        data = np.load(self.img_list[idx]).item()
        case_name = (self.img_list[idx]).split('/')[-2]
        slice_idx = os.path.basename(self.img_list[idx])[:3]
        img = data['brain']
        seg = data['seg_label']
        seg_tumor = np.where(seg>0, 1, 0) #只留肿瘤标签
        brain_mask = np.where(img!=0,1,0)

        img = torch.tensor(img)
        seg = torch.tensor(seg)
        seg_tumor = torch.tensor(seg_tumor)

        if self.has_mean:
            std = data['std']
            mean = data['mean']
            mean = mean*brain_mask 
            std = std*brain_mask 
            mean = torch.tensor(mean)
            std = torch.tensor(std)
            return img, seg, seg_tumor, [case_name,slice_idx], [mean,std]
        else:
            return img, seg, seg_tumor, [case_name,slice_idx]


if __name__ == "__main__":
    train = BrainDataset('data/train_brats.txt')
    train_loader = DataLoader(train, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    for i,(img, seg, seg_tumor, [name,slice_idx], [mean,std]) in enumerate(train_loader):
        img = img.numpy().astype(np.int32)
        img_min = img.min(-1).min(-1)
        img_max = img.max(-1).max(-1)
        # ones = torch.ones((512,512),dtype=torch.uint8)
        for i in range(len(img_min)):
            tmp = (img[i] + abs(img_min[i])) / (img_max[i] + abs(img_min[i])) * 255
            tmp = tmp.astype(np.uint8)
            tmp = cv2.cvtColor(tmp,cv2.COLOR_GRAY2RGB)
            cv2.circle(tmp, tuple(center[i]), 1, (0,0,255), 4)
            for j in range(36):
                x = (center[i][0] + dists[i][-j] * np.cos((j * 10) * np.pi/180)).astype(np.int16)
                y = (center[i][1] - dists[i][-j] * np.sin((j * 10) * np.pi/180)).astype(np.int16)
                cv2.circle(tmp, (x,y), 1, (0,0,255), 4)
            cv2.imwrite('test/test%d.png'%(i),tmp)

            # cv2.circle(img[i], tuple(center[i]), 1, 0, 4)
        pass

