import argparse
import glob
import os
import cv2
import random
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms.functional as T
import torchvision.transforms as transforms
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop,ToTensor
from numpy.random import RandomState

def parse_args():
    desc = 'Pytorch Implementation of \'Restormer: Efficient Transformer for High-Resolution Image Restoration\''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path', type=str, default='allweather')
    parser.add_argument('--test_data_path', type=str, default='AllWeather_Testing')
    # parser.add_argument('--data_name', type=str, default='Rain100L', choices=['dehaze','BSD68', 'urban100','Rain100L'])
    parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')
    parser.add_argument('--save_path', type=str, default='result_Allweather_pretrain')
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[1, 1, 1, 1],
                        help='number of transformer blocks for each level')
    parser.add_argument('--num_heads', nargs='+', type=int, default=[1, 2, 4, 8],
                        help='number of attention heads for each level')
    parser.add_argument('--channels', nargs='+', type=int, default=[48, 96, 192, 384],
                        help='number of channels for each level')
    parser.add_argument('--expansion_factor', type=float, default=2.66, help='factor of channel expansion for GDFN')
    parser.add_argument('--num_refinement', type=int, default=4, help='number of channels for refinement stage')
    parser.add_argument('--num_iter', type=int, default=300000, help='iterations of training') #300000
    parser.add_argument('--batch_size', nargs='+', type=int, default=[16, 16, 4, 4, 2, 2]),
    parser.add_argument('--patch_size', nargs='+', type=int, default=[64, 64, 128, 128, 256, 256])

    parser.add_argument('--lr', type=float, default=0.0003, help='initial learning rate') #0.0003
    parser.add_argument('--milestone', nargs='+', type=int, default=[92000, 156000, 204000, 240000, 276000],
                        help='when to change patch size and batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for no manual seed)')
    parser.add_argument('--stage', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--val_iter', type=int, default=10000)
    parser.add_argument('--model_file', type=str, default=None, help='path of pre-trained model file')

    return init_args(parser.parse_args())


class Config(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.save_path = args.save_path
        self.num_blocks = args.num_blocks
        self.num_heads = args.num_heads
        self.channels = args.channels
        self.expansion_factor = args.expansion_factor
        self.num_refinement = args.num_refinement
        self.num_iter = args.num_iter
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.lr = args.lr
        self.milestone = args.milestone
        self.workers = args.workers
        self.model_file = args.model_file
        self.stage = args.stage
        self.val_iter = args.val_iter
        self.de_type = args.de_type


def init_args(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


    return Config(args)


class OHCeLoss(nn.Module):
    def __init__(self):
        super(OHCeLoss,self).__init__()
    def forward(self,pred,onehot_label):
        pred = pred.squeeze()
        onehot_label = onehot_label.squeeze()
        N = pred.size(0)
        # log_prob = F.log_softmax(pred, dim=1)
        log_prob = torch.log(pred)
        loss = -torch.sum(log_prob * onehot_label) / N
        return loss

def pad_image_needed(img, size):
    # width, height = T.get_image_size(img)
    width, height =  img.shape[1:]
    if width < size[1]:
        img = T.pad(img, [size[1] - width, 0], padding_mode='reflect')
    if height < size[0]:
        img = T.pad(img, [0, size[0] - height], padding_mode='reflect')
    return img

def extend_data(data_list, extend_num):
    extend_list = []
    tempH = []
    tempS = []
    for img in data_list:
        if ("_rain" in img):
            extend_list.append(img)
        elif ("im_" in img):
            tempH.append(img)
        else:
            tempS.append(img)
    print(f'RainDrop :{len(extend_list)}')
    print(f'Haze :{len(tempH)}')
    print(f'Snow :{len(tempS)}')
    extend_lists = []
    for _ in range(extend_num):
        extend_lists += copy.deepcopy(extend_list)

    return data_list + extend_lists


class RainDataset(Dataset):
    def __init__(self, args,data_type=None, patch_size=None, length=None , test_path = None, dataset_name='AllWeather'):
        super().__init__()
        self.args = args
        self.data_type, self.patch_size = data_type, patch_size
        self.toTensor = ToTensor()
        self.rand_state = RandomState(66)
        self.dataset_name = dataset_name

        if dataset_name == 'WeatherStream':
            if data_type == 'train':
                data_path = self.args.data_path + '/input'
                data_folders = glob.glob(str(data_path) + "/*")
                self.rain_images = []
                for folder in data_folders:
                    self.rain_images += glob.glob(str(folder) + "/*")
            else:
                data_path = test_path + '/input'
                data_folders = glob.glob(str(data_path) + "/*")
                self.rain_images = []
                for folder in data_folders:
                    self.rain_images += glob.glob(str(folder) + "/*")
        else:
            if data_type == 'train':
                data_path = self.args.data_path + '/input'
                rain_images = glob.glob(str(data_path) + "/*")
                self.rain_images = extend_data(rain_images, 9)
            else:
                data_path = str(test_path) + '/input'
                self.rain_images = glob.glob(str(data_path) + "/*")
            
        # make sure the length of training and testing different
        self.num = len(self.rain_images)
        self.sample_num = length if data_type == 'train' else self.num
        print(f'Num of Data : {self.sample_num}, {dataset_name}')

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        image_name = os.path.basename(self.rain_images[idx % self.num])
        rain = T.to_tensor(Image.open(self.rain_images[idx % self.num]).convert('RGB'))
        if self.dataset_name == 'WeatherStream':
            change_name = self.rain_images[idx % self.num].split('/')[-1]
            gt_path = self.rain_images[idx % self.num].replace('input','target').replace(change_name,'gt.png')
            norain =  T.to_tensor(Image.open(gt_path).convert('RGB'))
        else:
            norain =  T.to_tensor(Image.open(self.rain_images[idx % self.num].replace('input','gt')).convert('RGB'))

        h, w = rain.shape[1:]
        if self.data_type == 'train':
            # make sure the image could be cropped
            try:
                rain = pad_image_needed(rain, (self.patch_size, self.patch_size))
                norain = pad_image_needed(norain, (self.patch_size, self.patch_size))
                i, j, th, tw = RandomCrop.get_params(rain, (self.patch_size, self.patch_size))
                rain = T.crop(rain, i, j, th, tw)
                norain = T.crop(norain, i, j, th, tw)
            except:
                print(f"!! Use Resize : {image_name}, SHAPE:{rain.shape}")
                resize_img = transforms.Resize([self.patch_size, self.patch_size])
                rain, norain = resize_img(rain), resize_img(norain)
                rain = pad_image_needed(rain, (self.patch_size, self.patch_size))
                norain = pad_image_needed(norain, (self.patch_size, self.patch_size))
                i, j, th, tw = RandomCrop.get_params(rain, (self.patch_size, self.patch_size))
                rain = T.crop(rain, i, j, th, tw)
                norain = T.crop(norain, i, j, th, tw)
            
            if torch.rand(1) < 0.5:
                rain = T.hflip(rain)
                norain = T.hflip(norain)
            if torch.rand(1) < 0.5:
                rain = T.vflip(rain)
                norain = T.vflip(norain)
        else:
            # padding in case images are not multiples of 8
            new_h, new_w = ((h + 8) // 8) * 8, ((w + 8) // 8) * 8
            pad_h = new_h - h if h % 8 != 0 else 0
            pad_w = new_w - w if w % 8 != 0 else 0
            rain = F.pad(rain, (0, pad_w, 0, pad_h), 'reflect')
            norain = F.pad(norain, (0, pad_w, 0, pad_h), 'reflect')

        clip_img = self.get_clip_img(rain)
        return rain, norain, image_name ,clip_img

        
    def get_clip_img(self,img):
        img_resize = transforms.Resize((224,224))
        clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        img=img_resize(img)
        img=clip_normalizer((img.reshape(3,224,224)))
        return img



def rgb_to_y(x):
    rgb_to_grey = torch.tensor([0.256789, 0.504129, 0.097906], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return torch.sum(x * rgb_to_grey, dim=1, keepdim=True).add(16.0)


def psnr(x, y, data_range=255.0):

    x = x.cpu().numpy()
    y = y.cpu().numpy()
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mse = np.mean((x - y)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(x, y, kernel_size=11, kernel_sigma=1.5, data_range=255.0, k1=0.01, k2=0.03):
    x, y = x / data_range, y / data_range
    # average pool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if f > 1:
        x, y = F.avg_pool2d(x, kernel_size=f), F.avg_pool2d(y, kernel_size=f)

    # gaussian filter
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
    coords -= (kernel_size - 1) / 2.0
    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * kernel_sigma ** 2)).exp()
    g /= g.sum()
    kernel = g.unsqueeze(0).repeat(x.size(1), 1, 1, 1)

    # compute
    c1, c2 = k1 ** 2, k2 ** 2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx, mu_yy, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy

    # contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
    # structural similarity (SSIM)
    ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs
    return ss.mean()

def select_gt_prompt(img_names, device='cuda'):
    gt_prompt = []
    for name in img_names:
        if ("_rain" in name):
            gt_prompt.append(0)
        elif name.startswith("im_"):
            gt_prompt.append(1)
        else:
            gt_prompt.append(2)
    gt_prompt = torch.tensor(gt_prompt).to(device)
    return gt_prompt