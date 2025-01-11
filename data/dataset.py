import os
import json
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize

from PIL import Image
from collections import OrderedDict

from configs import get_task_info
from .transforms import random_crop
from utils import mprint

class RVSD(Dataset):
    '''
    Realistic Video Desnowing Dataset single task
    '''
    def __init__(self, dataset_root: str, task: str = 'desnow', split: str = 'train',
                 transforms=None, random_crop_ratio: tuple = (1.0, 1.0)):
        super().__init__()
        self.dataset_root = dataset_root
        self.snow_frame_path_list = []
        self.clean_frame_path_list = []
        self.reference_frame_path_list = []
        self.task_id_list = []
        
        if split == 'train':
            split_name = split
        else:
            split_name = 'test'
        
        assert task == 'desnow', "RVSD must be [desnow] task, got {}".format(task)

        clean_path = os.path.join(dataset_root, split_name, 'clean')
        video_name_list = os.listdir(clean_path)
        video_name_list = sorted(video_name_list)
        for video_name in video_name_list:
            clean_video_path = os.path.join(clean_path, video_name)
            frame_name_list = os.listdir(clean_video_path)
            frame_name_list = sorted(frame_name_list)
            for idx, frame_name in enumerate(frame_name_list):
                clean_frame_path = os.path.join(clean_video_path, frame_name)
                snow_frame_path = clean_frame_path.replace('clean', 'snow')
                if (idx == 0):
                    reference_frame_path = None
                else:
                    reference_frame_name = frame_name_list[idx - 1]
                    reference_frame_path = os.path.join(clean_video_path, reference_frame_name)
                
                self.snow_frame_path_list.append(snow_frame_path)
                self.clean_frame_path_list.append(clean_frame_path)
                self.reference_frame_path_list.append(reference_frame_path)
                task_id = get_task_info(dataset='snow100k', type='idx', task=task)
                self.task_id_list.append(task_id)
        
        self.length = len(self.snow_frame_path_list)
        self.split = split
        self.transforms = transforms
        self.random_crop_ratio = random_crop_ratio

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        snow = Image.open(self.snow_frame_path_list[idx]).resize((720, 480))
        clean = Image.open(self.clean_frame_path_list[idx]).resize((720, 480))
        snow_name = f"{self.snow_frame_path_list[idx].split('/')[-2]}-{self.snow_frame_path_list[idx].split('/')[-1]}"
        
        if (self.reference_frame_path_list[idx] != None):
            reference = Image.open(self.reference_frame_path_list[idx]).resize((720, 480))
        else:
            reference = torch.zeros((3, 480, 720))
        
        to_tensor = ToTensor()
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)     
        if isinstance(reference, torch.Tensor) == False:
            reference = to_tensor(reference)
        
        
        snow = normalize(to_tensor(snow))
        clean =to_tensor(clean)
        reference = normalize(reference)
        mix = torch.cat((snow, reference))
        
        task_id = torch.tensor(self.task_id_list[idx])
        
        if self.split == 'train':
            mix, clean = random_crop(mix, clean, self.random_crop_ratio)

        if self.transforms is not None:
            mode = random.randint(a=0, b=7)  # include a and b
            mix = self.transforms(input=mix, mode=mode)
            clean = self.transforms(input=clean, mode=mode)

        if 'infer' in self.split:
            return (mix, clean, snow_name)
        else:
            return (mix, clean, task_id)

class AllWeather(Dataset):
    '''
    LowLevel single task
    '''
    def __init__(self, dataset_root: str, task: str = 'derain', split: str = 'train',
                 transforms=None, random_crop_ratio: tuple = (1.0, 1.0)):
        self.dataset_root = dataset_root
        self.lq_list = []  # low-quality input
        self.gt_list = []  # output
        self.task_id_list = []

        if split == 'test':
            split_name = '{}_{}'.format(split, task.replace('de', ''))
        elif 'infer' in split:
            split_name = '{}_{}'.format(split.replace('infer+', ''), task.replace('de', ''))
        else:  # train, val
            split_name = split

        lq_dir_path = os.path.join(dataset_root, split_name, 'input')
        lq_name_list = os.listdir(lq_dir_path)
        for idx, lq_name in enumerate(lq_name_list):
            if idx > 5000 and (split == 'test' or 'infer' in split):
                break
            lq_path = os.path.join(lq_dir_path, lq_name)
            gt_path = lq_path.replace('input', 'gt')
            if task == 'deraindrop':
                gt_path = gt_path.replace('_rain.png', '_clean.png')
            self.lq_list.append(lq_path)
            self.gt_list.append(gt_path)
            task_id = get_task_info(dataset='allweather', type='idx', task=task)
            self.task_id_list.append(task_id)

        self.length = len(self.lq_list)
        self.split = split
        self.transforms = transforms
        self.random_crop_ratio = random_crop_ratio  # 调ratio即可

    def __len__(self):

        return self.length

    def __getitem__(self, idx):

        lq = Image.open(self.lq_list[idx]).resize((720, 480))  # w, h
        gt = Image.open(self.gt_list[idx]).resize((720, 480))  # pil: w, h
        lq_name = self.lq_list[idx].split('/')[-1]

        to_tensor = ToTensor()
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        lq = normalize(to_tensor(lq))  # [3, h, w]
        gt = to_tensor(gt)
        task_id = torch.tensor(self.task_id_list[idx])

        if self.split == 'train':  # only crop in train
            lq, gt = random_crop(lq, gt, self.random_crop_ratio)
            
        # transforms, default: 8 similar spatial strategy
        if self.transforms is not None:
            mode = random.randint(a=0, b=7)  # include a and b
            lq = self.transforms(input=lq, mode=mode)
            gt = self.transforms(input=gt, mode=mode)

        if 'infer' in self.split:
            return (lq, gt, lq_name)

        else:
            return (lq, gt, task_id)
        

class LowLevel_Cityscapes(Dataset):
    '''
    LowLevel single task
    '''
    def __init__(self, dataset_root: str, task: str = 'deblur', split: str = 'train',
                 transforms=None, random_crop_ratio: tuple = (1.0, 1.0)):
        self.dataset_root = dataset_root
        self.lq_list = []  # low-quality input
        self.gt_list = []  # output
        self.task_id_list = []

        if split == 'test':
            split_name = '{}_{}'.format(split, task.replace('de', ''))
        elif 'infer' in split:
            split_name = '{}_{}'.format(split.replace('infer+', ''), task.replace('de', ''))
        else:  # train, val
            split_name = split

        lq_dir_path = os.path.join(dataset_root, split_name, 'input')
        if split == 'train':
            weather_type = os.listdir(lq_dir_path)
            for weather in weather_type:
                lq_weather_dir_path = os.path.join(lq_dir_path, weather)
                scene_type = os.listdir(lq_weather_dir_path)
                for scene in scene_type:
                    lq_weather_scene_dir_path = os.path.join(lq_weather_dir_path, scene)
                    lq_name_list = os.listdir(lq_weather_scene_dir_path)
                    for lq_name in lq_name_list:
                        # if task.replace('de', '') not in lq_name:
                        #     continue
                        lq_path = os.path.join(lq_weather_scene_dir_path, lq_name)
                        self.lq_list.append(lq_path)

                        lq_name = lq_name.split("_leftImg8bit")[0]
                        lq_name = "%s_leftImg8bit.png"%(lq_name)
                        
                        gt_path = os.path.join(lq_weather_scene_dir_path, lq_name)
                        gt_path = gt_path.replace('/%s'%(weather), '')
                        gt_path = gt_path.replace('input', 'gt')
                        
                        self.gt_list.append(gt_path)
                        task_id = get_task_info(dataset='cityscapes', type='idx', task=task)
                        self.task_id_list.append(task_id)
        else:
            scene_type = os.listdir(lq_dir_path)
            for scene in scene_type:
                lq_scene_dir_path = os.path.join(lq_dir_path, scene)
                lq_name_list = os.listdir(lq_scene_dir_path)
                for lq_name in lq_name_list:
                    # if task.replace('de', '') not in lq_name:
                    #     continue
                    lq_path = os.path.join(lq_scene_dir_path, lq_name)
                    self.lq_list.append(lq_path)

                    lq_name = lq_name.split("_leftImg8bit")[0]
                    lq_name = "%s_leftImg8bit.png"%(lq_name)
                    
                    gt_path = os.path.join(lq_scene_dir_path, lq_name)
                    gt_path = gt_path.replace('input', 'gt')
                    
                    self.gt_list.append(gt_path)
                    task_id = get_task_info(dataset='cityscapes', type='idx', task=task)
                    self.task_id_list.append(task_id)

        self.length = len(self.lq_list)
        self.split = split
        self.transforms = transforms
        self.random_crop_ratio = random_crop_ratio

    def __len__(self):

        return self.length

    def __getitem__(self, idx):

        lq = Image.open(self.lq_list[idx])  # w, h
        gt = Image.open(self.gt_list[idx])  # pil: w, h
        lq_name = self.lq_list[idx].split('/')[-1]

        to_tensor = ToTensor()
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        lq = normalize(to_tensor(lq))  # [3, h, w]
        gt = to_tensor(gt)
        task_id = torch.tensor(self.task_id_list[idx])

        if self.split == 'train':  # only crop in train
            lq, gt = random_crop(lq, gt, self.random_crop_ratio)

        # transforms, default: 8 similar spatial strategy
        if self.transforms is not None:
            mode = random.randint(a=0, b=7)  # include a and b
            lq = self.transforms(input=lq, mode=mode)
            gt = self.transforms(input=gt, mode=mode)

        if 'infer' in self.split:
            return (lq, gt, lq_name)
        else:
            return (lq, gt, task_id)
        

class Raindrop(Dataset):
    '''
    deraindrop single task
    '''
    def __init__(self, dataset_root: str, task: str = 'deraindrop', split: str = 'train',
                 transforms=None, random_crop_ratio: tuple = (1.0, 1.0)):
        self.dataset_root = dataset_root
        self.lq_list = []  # low-quality input
        self.gt_list = []  # output
        self.task_id_list = []

        assert task == 'deraindrop', "Raindrop must be [deraindrop] task, got {}".format(task)
        if 'test' in split or 'infer' in split:
            split_name = 'test_b' if 'b' in split else 'test_a'  # default: test_a
            mprint(split_name)
        else:  # train
            split_name = split

        lq_dir_path = os.path.join(dataset_root, split_name, 'input')
        lq_name_list = os.listdir(lq_dir_path)
        for lq_name in lq_name_list:
            lq_path = os.path.join(lq_dir_path, lq_name)
            gt_path = lq_path.replace('input', 'gt')
            gt_path = gt_path.replace('_rain.', '_clean.')
            self.lq_list.append(lq_path)
            self.gt_list.append(gt_path)
            task_id = get_task_info(dataset='raindrop', type='idx', task=task)
            self.task_id_list.append(task_id)

        self.length = len(self.lq_list)
        self.split = split
        self.transforms = transforms
        self.random_crop_ratio = random_crop_ratio

    def __len__(self):

        return self.length

    def __getitem__(self, idx):

        lq = Image.open(self.lq_list[idx]).resize((720, 480))  # w, h
        gt = Image.open(self.gt_list[idx]).resize((720, 480))  # pil: w, h
        lq_name = self.lq_list[idx].split('/')[-1]

        to_tensor = ToTensor()
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        lq = normalize(to_tensor(lq))  # [3, h, w]
        gt = to_tensor(gt)
        task_id = torch.tensor(self.task_id_list[idx])

        if self.split == 'train':  # only crop in train
            lq, gt = random_crop(lq, gt, self.random_crop_ratio)
            
        # transforms, default: 8 similar spatial strategy
        if self.transforms is not None:
            mode = random.randint(a=0, b=7)  # include a and b
            lq = self.transforms(input=lq, mode=mode)
            gt = self.transforms(input=gt, mode=mode)

        if 'infer' in self.split:
            return (lq, gt, lq_name)

        else:
            return (lq, gt, task_id)
        

class Snow100K(Dataset):
    '''
    snow100k single task
    '''
    def __init__(self, dataset_root: str, task: str = 'desnow', split: str = 'train',
                 transforms=None, random_crop_ratio: tuple = (1.0, 1.0)):
        self.dataset_root = dataset_root
        self.lq_list = []  # low-quality input
        self.gt_list = []  # output
        self.task_id_list = []

        assert task == 'desnow', "Snow100K must be [desnow] task, got {}".format(task)
        if 'test' in split or 'infer' in split:
            if '+s' in split:
                split_name = os.path.join('test', 'Snow100K-S')
                mprint('Snow100K-S')
            elif '+m' in split:
                split_name = os.path.join('test', 'Snow100K-M')
                mprint('Snow100K-M')
            else:  # default: L
                split_name = os.path.join('test', 'Snow100K-L')
                mprint('Snow100K-L')
        else:  # train
            split_name = split

        lq_dir_path = os.path.join(dataset_root, split_name, 'synthetic')
        lq_name_list = os.listdir(lq_dir_path)
        for lq_name in lq_name_list:
            lq_path = os.path.join(lq_dir_path, lq_name)
            gt_path = lq_path.replace('synthetic', 'gt')
            self.lq_list.append(lq_path)
            self.gt_list.append(gt_path)
            task_id = get_task_info(dataset='snow100k', type='idx', task=task)
            self.task_id_list.append(task_id)

        self.length = len(self.lq_list)
        self.split = split
        self.transforms = transforms
        self.random_crop_ratio = random_crop_ratio

    def __len__(self):

        return self.length

    def __getitem__(self, idx):
        lq = Image.open(self.lq_list[idx]).resize((720, 480))  # w, h
        gt = Image.open(self.gt_list[idx]).resize((720, 480))  # pil: w, h
        lq_name = self.lq_list[idx].split('/')[-1]

        to_tensor = ToTensor()
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        lq = normalize(to_tensor(lq))  # [3, h, w]
        gt = to_tensor(gt)
        task_id = torch.tensor(self.task_id_list[idx])

        if self.split == 'train':  # # only crop in train
            lq, gt = random_crop(lq, gt, self.random_crop_ratio)
            
        # transforms, default: 8 similar spatial strategy
        if self.transforms is not None:
            mode = random.randint(a=0, b=7)  # include a and b
            lq = self.transforms(input=lq, mode=mode)
            gt = self.transforms(input=gt, mode=mode)

        if 'infer' in self.split:
            return (lq, gt, lq_name)

        else:
            return (lq, gt, task_id)
        

class Synthetic_Rain(Dataset):
    '''
    derain single task
    '''
    def __init__(self, dataset_root: str, task: str = 'derain', split: str = 'train',
                 transforms=None, random_crop_ratio: tuple = (1.0, 1.0)):
        self.dataset_root = dataset_root
        self.lq_list = []  # low-quality input
        self.gt_list = []  # output
        self.task_id_list = []

        assert task == 'derain', "Raindrop must be [derain] task, got {}".format(task)
        if 'test' in split or 'infer' in split:
            if '+100h' in split:
                split_name = os.path.join('test', 'Rain100H')
                mprint('Rain100H')
            elif '+100l' in split:
                split_name = os.path.join('test', 'Rain100L')
                mprint('Rain100L')
            elif '+100' in split:
                split_name = os.path.join('test', 'Test100')
                mprint('Test100')
            elif '+1200' in split:
                split_name = os.path.join('test', 'Test1200')
                mprint('Test1200')
            elif '+2800' in split:
                split_name = os.path.join('test', 'Test2800')
                mprint('Test2800')
            else:  # default
                split_name = os.path.join('test', 'Rain100L')
                mprint('Rain100L')
        else:  # train
            split_name = split

        lq_dir_path = os.path.join(dataset_root, split_name, 'input')
        lq_name_list = os.listdir(lq_dir_path)
        for lq_name in lq_name_list:
            lq_path = os.path.join(lq_dir_path, lq_name)
            gt_path = lq_path.replace('input', 'target')
            self.lq_list.append(lq_path)
            self.gt_list.append(gt_path)
            task_id = get_task_info(dataset='synthetic_rain', type='idx', task=task)
            self.task_id_list.append(task_id)

        self.length = len(self.lq_list)
        self.split = split
        self.transforms = transforms
        self.random_crop_ratio = random_crop_ratio

    def __len__(self):

        return self.length

    def __getitem__(self, idx):

        lq = Image.open(self.lq_list[idx]).resize((480, 320))  # w, h
        gt = Image.open(self.gt_list[idx]).resize((480, 320))  # pil: w, h
        lq_name = self.lq_list[idx].split('/')[-1]

        to_tensor = ToTensor()
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        lq = normalize(to_tensor(lq))  # [3, h, w]
        gt = to_tensor(gt)
        task_id = torch.tensor(self.task_id_list[idx])

        if self.split == 'train':  # only crop in train
            lq, gt = random_crop(lq, gt, self.random_crop_ratio)
            
        # transforms, default: 8 similar spatial strategy
        if self.transforms is not None:
            mode = random.randint(a=0, b=7)  # include a and b
            lq = self.transforms(input=lq, mode=mode)
            gt = self.transforms(input=gt, mode=mode)

        if 'infer' in self.split:
            return (lq, gt, lq_name)

        else:
            return (lq, gt, task_id)
        

class Outdoor_Rain(Dataset):
    '''
    derain single task
    '''
    def __init__(self, dataset_root: str, task: str = 'derain', split: str = 'train',
                 transforms=None, random_crop_ratio: tuple = (1.0, 1.0)):
        self.dataset_root = dataset_root
        self.lq_list = []  # low-quality input
        self.gt_list = []  # output
        self.task_id_list = []

        assert task == 'derain', "Outdoor_Rain must be [derain] task, got {}".format(task)
        if 'test' in split or 'infer' in split:
            split_name = 'test'
        else:  # train
            split_name = split

        lq_dir_path = os.path.join(dataset_root, split_name, 'input')
        lq_name_list = os.listdir(lq_dir_path)
        for lq_name in lq_name_list:
            lq_path = os.path.join(lq_dir_path, lq_name)
            gt_path = lq_path.replace('input', 'gt')
            self.lq_list.append(lq_path)
            self.gt_list.append(gt_path)
            task_id = get_task_info(dataset='outdoor_rain', type='idx', task=task)
            self.task_id_list.append(task_id)

        self.length = len(self.lq_list)
        self.split = split
        self.transforms = transforms
        self.random_crop_ratio = random_crop_ratio

    def __len__(self):

        return self.length

    def __getitem__(self, idx):

        lq = Image.open(self.lq_list[idx]).resize((720, 480))  # w, h
        gt = Image.open(self.gt_list[idx]).resize((720, 480))  # pil: w, h
        lq_name = self.lq_list[idx].split('/')[-1]

        to_tensor = ToTensor()
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        lq = normalize(to_tensor(lq))  # [3, h, w]
        gt = to_tensor(gt)
        task_id = torch.tensor(self.task_id_list[idx])

        if self.split == 'train':  # only crop in train
            lq, gt = random_crop(lq, gt, self.random_crop_ratio)
            
        # transforms, default: 8 similar spatial strategy
        if self.transforms is not None:
            mode = random.randint(a=0, b=7)  # include a and b
            lq = self.transforms(input=lq, mode=mode)
            gt = self.transforms(input=gt, mode=mode)

        if 'infer' in self.split:
            return (lq, gt, lq_name)

        else:
            return (lq, gt, task_id)
        


class OTS(Dataset):
    '''
    dehaze single task
    '''
    def __init__(self, dataset_root: str, task: str = 'dehaze', split: str = 'train',
                 transforms=None, random_crop_ratio: tuple = (1.0, 1.0)):
        self.dataset_root = dataset_root
        self.lq_list = []  # low-quality input
        self.gt_list = []  # output
        self.task_id_list = []

        assert task == 'dehaze', "OTS must be [dehaze] task, got {}".format(task)
        print(split)
        if 'test' in split or 'infer' in split:
            split_name = 'test'
        else:  # train
            split_name = split

        lq_dir_path = os.path.join(dataset_root, split_name, 'input')
        lq_name_list = os.listdir(lq_dir_path)
        for lq_name in lq_name_list:
            lq_path = os.path.join(lq_dir_path, lq_name)
            gt_path = lq_path.replace('input', 'gt').split('_')[0]+'.jpg'
            self.lq_list.append(lq_path)
            if split != 'train':
                self.gt_list.append(lq_path)
            else:
                self.gt_list.append(gt_path)
            task_id = get_task_info(dataset='ots', type='idx', task=task)
            self.task_id_list.append(task_id)

        self.length = len(self.lq_list)
        self.split = split
        self.transforms = transforms
        self.random_crop_ratio = random_crop_ratio

    def __len__(self):

        return self.length

    def __getitem__(self, idx):

        lq = Image.open(self.lq_list[idx]).convert('RGB').resize((640, 640))  # w, h
        gt = Image.open(self.gt_list[idx]).convert('RGB').resize((640, 640))  # pil: w, h
        lq_name = self.lq_list[idx].split('/')[-1]

        to_tensor = ToTensor()
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        lq = normalize(to_tensor(lq))  # [3, h, w]
        gt = to_tensor(gt)
        task_id = torch.tensor(self.task_id_list[idx])

        if self.split == 'train':  # only crop in train
            lq, gt = random_crop(lq, gt, self.random_crop_ratio)
            
        # transforms, default: 8 similar spatial strategy
        if self.transforms is not None:
            mode = random.randint(a=0, b=7)  # include a and b
            lq = self.transforms(input=lq, mode=mode)
            gt = self.transforms(input=gt, mode=mode)

        if 'infer' in self.split:
            return (lq, gt, lq_name)

        else:
            return (lq, gt, task_id)
        
