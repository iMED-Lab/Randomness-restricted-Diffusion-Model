import math
import random
import glob
import os
from PIL import Image
from mpi4py import MPI
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import blobfile as bf
import torch
import torchvision.transforms.functional as TF

def random_crop(image,mask, crop_size = 0.8):
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(int(crop_size*1200),int(crop_size*1600)))
    data = TF.crop(image, i, j, h, w)
    for a in range(len(mask)):
        mask[a] = TF.crop(mask[a], i, j, h, w)

    return data,mask

def load_data(
    *,
    batch_size,image_size,deterministic=False,color_adjust = True,random_rotate=False,random_H_flip=True,random_V_flip=True
        ,train_val_test = 'train',data_name = 'Pterygium'
):
    if data_name == 'Pterygium':
        root = os.getcwd()[:-7]+'/DATA/nurou/'+train_val_test+'/'
        all_files = glob.glob(root + '*' + 'image.png')
        dataset = Pterygium_dataset(image_size,all_files,shard=MPI.COMM_WORLD.Get_rank(),num_shards=MPI.COMM_WORLD.Get_size(),
                                color_adjust=color_adjust,random_rotate=random_rotate,random_H_flip=random_H_flip,random_V_flip=random_V_flip
                                ,train_val_test = train_val_test)
    if data_name == 'MG':
        root = os.getcwd()[:-9]+'/DATA/MG/'+train_val_test+'/'
        all_files = os.listdir(root + 'image')
        dataset = MG_dataset(image_size,all_files,root,shard=MPI.COMM_WORLD.Get_rank(),num_shards=MPI.COMM_WORLD.Get_size(),
                                color_adjust=color_adjust,random_rotate=True,random_H_flip=random_H_flip,random_V_flip=random_V_flip
                                ,train_val_test = train_val_test)
    if data_name == 'Tear':
        root = os.getcwd()[:-9]+'/DATA/Tear/'+train_val_test+'/'
        all_files = os.listdir(root + 'image')
        dataset = Tear_dataset(image_size,all_files,root,shard=MPI.COMM_WORLD.Get_rank(),num_shards=MPI.COMM_WORLD.Get_size(),
                                color_adjust=color_adjust,random_rotate=random_rotate,random_H_flip=random_H_flip,random_V_flip=random_V_flip
                                ,train_val_test = train_val_test) 
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    while True:
        yield from loader

def data_loader(
    *,
    batch_size,image_size,color_adjust=False,random_rotate=False,random_H_flip=False,random_V_flip=False
        ,train_val_test = 'test',data_name = 'cup_disc_650'
):
    if data_name == 'Pterygium':
        root = os.getcwd()[:-7] + '/DATA/nurou/' + train_val_test + '/'
        all_files = glob.glob(root + '*' + 'image.png')
        dataset = Pterygium_dataset(image_size, all_files, shard=MPI.COMM_WORLD.Get_rank(),
                                    num_shards=MPI.COMM_WORLD.Get_size(),
                                    color_adjust=color_adjust, random_rotate=random_rotate, random_H_flip=random_H_flip,
                                    random_V_flip=random_V_flip
                                    , train_val_test=train_val_test)
    if data_name == 'MG':
        root = os.getcwd()[:-9] + '/DATA/MG/' + train_val_test + '/'
        all_files = os.listdir(root + 'image')
        dataset = MG_dataset(image_size, all_files, root, shard=MPI.COMM_WORLD.Get_rank(),
                             num_shards=MPI.COMM_WORLD.Get_size(),
                             color_adjust=color_adjust, random_rotate=True, random_H_flip=random_H_flip,
                             random_V_flip=random_V_flip
                             , train_val_test=train_val_test)
    if data_name == 'Tear':
        root = os.getcwd()[:-9] + '/DATA/Tear/' + train_val_test + '/'
        all_files = os.listdir(root + 'image')
        dataset = Tear_dataset(image_size, all_files, root, shard=MPI.COMM_WORLD.Get_rank(),
                               num_shards=MPI.COMM_WORLD.Get_size(),
                               color_adjust=color_adjust, random_rotate=random_rotate, random_H_flip=random_H_flip,
                               random_V_flip=random_V_flip
                               , train_val_test=train_val_test)

    if train_val_test == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
        )
    return loader

class Pterygium_dataset(Dataset):
    def __init__(
        self,
        resolution,image_paths,shard=0,num_shards=1,color_adjust = False,random_rotate=False,random_H_flip=True,random_V_flip=False
        ,train_val_test = 'train',
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]  #从第shard个开始，并且每次间隔num——shards
        self.color_adjust = color_adjust
        self.random_rotate = random_rotate
        self.random_H_flip = random_H_flip
        self.random_V_flip = random_V_flip
        self.train_val_test = train_val_test
    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]  
        name = path[-37:-9]
        mask_path = path[:-9]+'mask.png'
        
        with bf.BlobFile(path, "rb") as f:
            Pil_image = Image.open(f)
            Pil_image.load()
        Pil_image = Pil_image.convert("RGB")

        with bf.BlobFile(mask_path, "rb") as f:
            mask = Image.open(f)
            mask.load()
        mask = mask.convert("RGB")

        Pil_image = Pil_image.resize((self.resolution,self.resolution), resample=Image.NEAREST)
        mask = mask.resize((self.resolution,self.resolution), resample=Image.NEAREST)

        if self.train_val_test == 'train' and self.color_adjust and random.random() < 0.5:
            Pil_image = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)(Pil_image)
        
        Pil_image = np.array(Pil_image)
        mask = np.array(mask)

        if self.train_val_test == 'train':
            if self.random_rotate:
                Pil_image,mask = random_ShiftScaleRotate(image=Pil_image,mask=mask)
            if self.random_H_flip and random.random() < 0.5:
                Pil_image = Pil_image[:,::-1]
                mask = mask[:,::-1]
            if self.random_V_flip and random.random() < 0.5:
                Pil_image = Pil_image[::-1,:]
                mask = mask[::-1,:]
            
        if self.train_val_test == 'test' or self.train_val_test == 'val':
            segpath =  path[:-9] +  'segment.png'
            with bf.BlobFile(segpath, "rb") as f:
                pre_seg = Image.open(f)
                pre_seg.load()

            pre_seg = pre_seg.convert("RGB")
            pre_seg = pre_seg.resize((self.resolution,self.resolution), resample=Image.NEAREST)
            pre_seg = np.array(pre_seg)

            pre_seg = pre_seg.astype(np.float32) / 127.5 - 1  #归一化
            pre_seg = np.transpose(pre_seg, [2, 0, 1])
            pre_seg = pre_seg[0:1,:,:]


        Pil_image = Pil_image.astype(np.float32) 
        mask = mask.astype(np.float32)

        Pil_image = np.transpose(Pil_image, [2, 0, 1])
        mask = np.transpose(mask, [2, 0, 1])

        mask = mask[0:1,:,:]

        Pil_image = Pil_image / 127.5 - 1
        mask = mask / 127.5 - 1
        
        if self.train_val_test == 'test':
            return Pil_image,mask,mask,name,pre_seg
        return Pil_image,mask,mask,name

class MG_dataset(Dataset):
    def __init__(
        self,
        resolution,image_paths,root,shard=0,num_shards=1,color_adjust = False,random_rotate=False,random_H_flip=True,random_V_flip=False
        ,train_val_test = 'train',
    ):
        super().__init__()
        self.root = root
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]  #从第shard个开始，并且每次间隔num——shards
        self.color_adjust = color_adjust
        self.random_rotate = random_rotate
        self.random_H_flip = random_H_flip
        self.random_V_flip = random_V_flip
        self.train_val_test = train_val_test
    def __len__(self):
        return len(self.local_images)

    def get_mask(self, mask,mask_1,mask_2):
        label = np.ones(mask.shape).astype(np.float32)*255
        label[mask > 0] = 127
        label[mask_1 > 0] = 0
        # label[mask_2 > 0] = 0
        return label


    def __getitem__(self, idx):
        name = self.local_images[idx][:-4]

        image_path = self.root + 'image/' + name + '.jpg'
        mask_path = self.root + 'mask/' + name + '.png'
        mask_path_1 = self.root + 'mask/' + name + '-1.png'
        mask_path_2 = self.root + 'mask/' + name + '-2.png'
        
        with bf.BlobFile(image_path, "rb") as f:
            Pil_image = Image.open(f)
            Pil_image.load()
        Pil_image = Pil_image.convert("RGB")

        with bf.BlobFile(mask_path, "rb") as f:
            mask = Image.open(f)
            mask.load()
        mask = mask.convert("RGB")

        with bf.BlobFile(mask_path_1, "rb") as f:
            mask_1 = Image.open(f)
            mask_1.load()
        mask_1 = mask_1.convert("RGB")

        with bf.BlobFile(mask_path_2, "rb") as f:
            mask_2 = Image.open(f)
            mask_2.load()
        mask_2 = mask_2.convert("RGB")

        Pil_image = Pil_image.resize((self.resolution,self.resolution), resample=Image.NEAREST)
        mask = mask.resize((self.resolution,self.resolution), resample=Image.NEAREST)
        mask_1 = mask_1.resize((self.resolution,self.resolution), resample=Image.NEAREST)
        mask_2 = mask_2.resize((self.resolution,self.resolution), resample=Image.NEAREST)

        if self.train_val_test == 'train' and self.color_adjust and random.random() < 0.6:
            gamma_v = round(np.random.uniform(0.7,1.7),2)
            Pil_image = TF.adjust_gamma(img=Pil_image, gamma = gamma_v)
            Pil_image = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)(Pil_image)

        Pil_image = np.array(Pil_image)
        mask = np.array(mask)
        mask_1 = np.array(mask_1)
        mask_2 = np.array(mask_2)

        mask = self.get_mask(mask,mask_1,mask_2)

        if self.train_val_test == 'train':
            if self.random_rotate:
                Pil_image,mask = random_ShiftScaleRotate(image=Pil_image,mask=mask)
            if self.random_H_flip and random.random() < 0.5:
                Pil_image = Pil_image[:,::-1]
                mask = mask[:,::-1]
            if self.random_V_flip and random.random() < 0.5:
                Pil_image = Pil_image[::-1,:]
                mask = mask[::-1,:]
            
        if self.train_val_test == 'test' or self.train_val_test == 'val':
            segpath =  './Pred_Save/MG/' + name + '.png'
            with bf.BlobFile(segpath, "rb") as f:
                pre_seg = Image.open(f)
                pre_seg.load()

            pre_seg = pre_seg.convert("RGB")
            pre_seg = pre_seg.resize((self.resolution,self.resolution), resample=Image.NEAREST)
            pre_seg = np.array(pre_seg)

            pre_seg = pre_seg.astype(np.float32) / 127.5 - 1  #归一化
            pre_seg = np.transpose(pre_seg, [2, 0, 1])
            pre_seg = pre_seg[0:1,:,:]

        Pil_image = Pil_image.astype(np.float32) 
        mask = mask.astype(np.float32)

        Pil_image = np.transpose(Pil_image, [2, 0, 1])
        mask = np.transpose(mask, [2, 0, 1])

        mask = mask[0:1,:,:]

        Pil_image = Pil_image / 127.5 - 1
        mask = mask / 127.5 - 1
        
        if self.train_val_test == 'test':
            return Pil_image,mask,mask,name,pre_seg

        return Pil_image,mask,mask,name
    

class Tear_dataset(Dataset):
    def __init__(
        self,
        resolution,image_paths,root,shard=0,num_shards=1,color_adjust = False,random_rotate=False,random_H_flip=True,random_V_flip=False
        ,train_val_test = 'train',
    ):
        super().__init__()
        self.root = root
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]  #从第shard个开始，并且每次间隔num——shards
        self.color_adjust = color_adjust
        self.random_rotate = random_rotate
        self.random_H_flip = random_H_flip
        self.random_V_flip = random_V_flip
        self.train_val_test = train_val_test
    def __len__(self):
        return len(self.local_images)

    def get_mask(self, mask,mask_1):
        label = np.ones(mask.shape).astype(np.float32)*255
        label[mask > 0] = 0
        return label


    def __getitem__(self, idx):
        name = self.local_images[idx][:-4]

        image_path = self.root + 'image/' + name + '.jpg'
        mask_path = self.root + 'mask/' + name + '.png'
        mask_path_1 = self.root + 'mask/' + name + '-1.png'

        with bf.BlobFile(image_path, "rb") as f:
            Pil_image = Image.open(f)
            Pil_image.load()
        Pil_image = Pil_image.convert("RGB")

        with bf.BlobFile(mask_path, "rb") as f:
            mask = Image.open(f)
            mask.load()
        mask = mask.convert("RGB")

        with bf.BlobFile(mask_path_1, "rb") as f:
            mask_1 = Image.open(f)
            mask_1.load()
        mask_1 = mask_1.convert("RGB")


        Pil_image = Pil_image.resize((self.resolution,self.resolution), resample=Image.NEAREST)
        mask = mask.resize((self.resolution,self.resolution), resample=Image.NEAREST)
        mask_1 = mask_1.resize((self.resolution,self.resolution), resample=Image.NEAREST)

        if self.train_val_test == 'train' and self.color_adjust and random.random() < 0.5:
            Pil_image = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)(Pil_image)

        Pil_image = np.array(Pil_image)
        mask = np.array(mask)
        mask_1 = np.array(mask_1)

        mask = self.get_mask(mask,mask_1)

        if self.train_val_test == 'train':
            if self.random_rotate:
                Pil_image,mask = random_ShiftScaleRotate(image=Pil_image,mask=mask)
            if self.random_H_flip and random.random() < 0.5:
                Pil_image = Pil_image[:,::-1]
                mask = mask[:,::-1]
            # if self.random_V_flip and random.random() < 0.5:
            #     Pil_image = Pil_image[::-1,:]
            #     mask = mask[::-1,:]
            
        if self.train_val_test == 'test' or self.train_val_test == 'val':
            segpath =  './Pred_Save/Tear/' + name + '.png'
            with bf.BlobFile(segpath, "rb") as f:
                pre_seg = Image.open(f)
                pre_seg.load()

            pre_seg = pre_seg.convert("RGB")
            pre_seg = pre_seg.resize((self.resolution,self.resolution), resample=Image.NEAREST)
            pre_seg = np.array(pre_seg)

            pre_seg = pre_seg.astype(np.float32) / 127.5 - 1  #归一化
            pre_seg = np.transpose(pre_seg, [2, 0, 1])
            pre_seg = pre_seg[0:1,:,:]

        Pil_image = Pil_image.astype(np.float32) 
        mask = mask.astype(np.float32)

        Pil_image = np.transpose(Pil_image, [2, 0, 1])
        mask = np.transpose(mask, [2, 0, 1])

        mask = mask[0:1,:,:]

        Pil_image = Pil_image / 127.5 - 1
        mask = mask / 127.5 - 1
        
        if self.train_val_test == 'test':
            return Pil_image,mask,mask,name,pre_seg

        return Pil_image,mask,mask,name

def random_ShiftScaleRotate(image, mask, shift_limit=(-0.0, 0.0),
                            scale_limit=(-0.0, 0.0), rotate_limit=(-15, 15),
                            aspect_limit=(-0.0, 0.0), borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])

        sx = scale * aspect / (aspect ** 0.5)
        sy = scale * (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height]])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(255, 255, 255))

    return image, mask