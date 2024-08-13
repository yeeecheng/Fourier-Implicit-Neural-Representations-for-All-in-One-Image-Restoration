import os
import pdb
import random
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torchvision.transforms as transforms

import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.INR_Restormer import Restormer
# from models.Restormer import Restormer
from utils import parse_args, rgb_to_y, psnr, ssim ,RainDataset

from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


########################################################

factor = 8
def test_loop(net, data_loader, num_iter ,data_name):
    net.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    psnr_rgbs = 0
    ssim_rgbs = 0
    with torch.no_grad():
        test_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)

        for rain, norain, name, clip_features in test_bar:
            rain, norain = rain.cuda(), norain.cuda()

            # Padding in case images are not multiples of 8
            if len(rain.shape)==3:
                rain = rain.unsqueeze(0)
                norain = norain.unsqueeze(0)
            h,w = rain.shape[2], rain.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            rain = F.pad(rain, (0,padw,0,padh), 'reflect')
            norain = F.pad(norain, (0, padw, 0, padh), 'reflect')

            # Model
            out, INR_feat = model(rain, None)
            out = torch.clamp((torch.clamp(out[:, :, :h, :w], 0, 1).mul(255)), 0, 255)#.byte()

            norain = torch.clamp(norain[:, :, :h, :w].mul(255), 0, 255)#.byte()
            
            # computer the metrics with Y channel and double precision
            y, gt = rgb_to_y(out.double()), rgb_to_y(norain.double())
            current_psnr, current_ssim = psnr(y, gt), ssim(y, gt)
            total_psnr += current_psnr#.item()
            total_ssim += current_ssim.item()
            count += 1
            save_path = '{}/{}/{}'.format(args.save_path, data_name, name[0])
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
                
            ## Save #######################################
            # save the image
            # Image.fromarray(out.squeeze(dim=0).permute(1, 2, 0).byte().contiguous().cpu().numpy()).save(save_path) 
            
            test_bar.set_description(str(data_name)+' Test Iter: [{}/{}] PSNR: {:.2f} SSIM: {:.3f}'
                                     .format(num_iter, 1 if args.model_file else args.num_iter,
                                             total_psnr / count, total_ssim / count))

            recoverd = out.squeeze(dim=0).permute(1, 2, 0).contiguous().cpu().numpy()
            clean = norain.squeeze(dim=0).permute(1, 2, 0).contiguous().cpu().numpy()
            psnr_rgb = peak_signal_noise_ratio(recoverd, clean,data_range=255)
            ssim_rgb = structural_similarity(recoverd,clean, data_range=255, channel_axis=-1)
            psnr_rgbs += psnr_rgb#.item()
            ssim_rgbs += ssim_rgb
            
        print(f'Final : Avg PSNR_y {total_psnr / count}, Avg SSIM_y {total_ssim / count}')
        print(f'avg psnr_rgb : {psnr_rgbs / count} , avg ssim_rgb : {ssim_rgbs / count} ')
    return total_psnr / count, total_ssim / count , psnr_rgbs / count, ssim_rgbs / count


def save_loop(net, data_loader, num_iter, data_name , save=True,mode=None, multi_loader=False):
    global best_psnr_derain, best_ssim_derain, best_psnr_deRainDrop, best_ssim_deRainDrop ,best_psnr_desnow, best_ssim_desnow,best_psnr_all, best_ssim_all
    # val_psnr_y, val_ssim_y , val_psnr_rgb, val_ssim_rgb = test_loop(net, data_loader, num_iter ,data_name)
    val_psnr_y, val_ssim_y , val_psnr_rgb, val_ssim_rgb = 0.0, 0.0, 0.0, 0.0
    if not multi_loader:
        val_psnr_y, val_ssim_y , val_psnr_rgb, val_ssim_rgb = test_loop(net, data_loader, num_iter ,data_name)
    else:
        for loader in data_loader:
            vpy, vsy, vpr, vsr = test_loop(net, loader, num_iter ,data_name)
            val_psnr_y += vpy
            val_ssim_y += vsy
            val_psnr_rgb +=  vpr
            val_ssim_rgb += vsr
        val_psnr_y = val_psnr_y/3.0
        val_ssim_y = val_ssim_y/3.0
        val_psnr_rgb = val_psnr_rgb/3.0
        val_ssim_rgb = val_ssim_rgb/3.0
        print(f'TOTAL AVG PSNR : {val_psnr_y} \t TOTAL AVG SSIM {val_ssim_y}')
    
    
    if mode !='test':
        writer.add_scalars('Test PSNR_y', {str(data_name)+"_PSNR": val_psnr_y}, num_iter)
        writer.add_scalars('Test SSIM_y', {str(data_name)+"_SSIM" :val_ssim_y}, num_iter)
        writer.add_scalars('Test PSNR_rgb', {str(data_name)+"_PSNR": val_psnr_rgb}, num_iter)
        writer.add_scalars('Test SSIM_rgb', {str(data_name)+"_SSIM" :val_ssim_rgb}, num_iter)


        # if val_psnr > best_psnr_derain and val_ssim > best_ssim_derain:
        if val_psnr_y > best_psnr_all :
            best_psnr_all, best_ssim_all = val_psnr_y, val_ssim_y
            with open('{}/{}.txt'.format(args.save_path, data_name), 'w') as f:
                f.write('Iter: {} PSNR:{:.2f} SSIM:{:.3f}'.format(num_iter, best_psnr_all, best_ssim_all))
            if save:
                # 保存模型狀態
                checkpoint = {
                    'epoch': num_iter + 1,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint, '{}/{}.pth'.format(args.save_path, data_name))

        if save:
                # 保存模型狀態
                checkpoint = {
                    'epoch': num_iter + 1,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint, '{}/{}.pth'.format(args.save_path, 'last'))

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    
    # Testing Datasets
    writer = SummaryWriter('./'+args.save_path+"/"+'tensorboard')
    test_dataset_RainDrop = RainDataset(args, test_path =args.test_data_path+'/raindrop')
    test_dataset_Rain = RainDataset(args,test_path =args.test_data_path+'/rain')
    test_dataset_Snow_sample = RainDataset(args,test_path =args.test_data_path+'/Snow100K')
    test_dataset_Snow = RainDataset(args,test_path =args.test_data_path+'/Snow100K-L')
    # test_dataset_test = RainDataset(args,test_path ='/home/u3732345/multi_weather/test')

    # Testing Dataloaders
    test_loader_RainDrop = DataLoader(test_dataset_RainDrop, batch_size=1, shuffle=False, num_workers=args.workers,pin_memory=True)
    test_loader_Snow_sample = DataLoader(test_dataset_Snow_sample, batch_size=1, shuffle=False, num_workers=args.workers,pin_memory=True)
    test_loader_Snow = DataLoader(test_dataset_Snow, batch_size=1, shuffle=False, num_workers=args.workers,pin_memory=True)
    test_loader_Rain = DataLoader(test_dataset_Rain, batch_size=1, shuffle=False, num_workers=args.workers,pin_memory=True)
    # test_loader_test = DataLoader(test_dataset_test, batch_size=1, shuffle=False, num_workers=args.workers,pin_memory=True)

    results, best_psnr_derain, best_ssim_derain, best_psnr_deRainDrop, best_ssim_deRainDrop ,best_psnr_desnow, best_ssim_desnow ,best_psnr_all, best_ssim_all = {'PSNR': [], 'SSIM': []}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    model = Restormer(args.num_blocks, args.num_heads, args.channels, args.num_refinement, args.expansion_factor).cuda()

    Test_Loaders = [test_loader_RainDrop, test_loader_Rain, test_loader_Snow]
    Test_Sample_Loaders = [test_loader_RainDrop, test_loader_Rain, test_loader_Snow_sample]
    

    if args.stage == 'test':
        print("Testing ...")
        model.load_state_dict(torch.load(args.model_file)['model_state_dict'], strict=True)
        # model.load_state_dict(torch.load(args.model_file), strict=True)
        save_loop(model, Test_Loaders, 1,'All_Test', multi_loader=True, save=False ,mode=args.stage)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        last_epoch = 1
        if args.model_file:
            print(f'Load model ckpt from : {args.model_file} ....')
            checkpoint = torch.load(args.model_file)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            last_epoch = checkpoint['epoch']

        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iter, eta_min=1e-6)
        total_loss, total_num, results['Loss'], i = 0.0, 0, [], 0
        total_loss_INR = 0.0
        total_loss_mse = 0.0
        total_class_mse = 0.0
        train_bar = tqdm(range(last_epoch, args.num_iter + 1), initial=1, dynamic_ncols=True)
        
        for n_iter in train_bar:
            # progressive learning
            if n_iter == last_epoch or n_iter - 1 in args.milestone:
                end_iter = args.milestone[i] if i < len(args.milestone) else args.num_iter
                start_iter = args.milestone[i - 1] if i > 0 else 0
                length = args.batch_size[i] * (end_iter - start_iter)
                train_dataset = RainDataset(args,'train',args.patch_size[i],length)
                train_loader = iter(DataLoader(train_dataset, args.batch_size[i], shuffle=True, num_workers=args.workers,pin_memory=True))
                i += 1

            # train
            model.train()
            rain, norain, name, clip_features = next(train_loader)
            rain, norain = rain.cuda(), norain.cuda()

            # output & loss
            out , loss_INR, INR_feat = model(rain, norain)
            loss = F.l1_loss(out, norain) + loss_INR

            # optim
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_num += rain.size(0)
            total_loss += loss.item() * rain.size(0)
            total_loss_INR += loss_INR.item()* rain.size(0)
            # total_loss_INR += loss_INR* rain.size(0)

            train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f} loss_INR {:.3f} Loss_class {:.3f}'
                                      .format(n_iter, args.num_iter, total_loss / total_num , total_loss_INR / total_num, total_class_mse/total_num))
            
            lr_scheduler.step()
            if n_iter % args.val_iter == 0 :
                writer.add_scalars('Train', {"loss": (total_loss / total_num),
                                             "loss_INR": (total_loss_INR / total_num),
                                             "loss_class": (total_class_mse / total_num)}, n_iter)
                save_loop(model, Test_Sample_Loaders, n_iter,'All', multi_loader=True)
                #if n_iter < 250000:
                #    print('='*100)
                #    save_loop(model, test_loader_test, n_iter,'All', multi_loader=False)
                #    print('='*100)
                #else:
                #    print('='*100)
                #    save_loop(model, Test_Sample_Loaders, n_iter,'All', multi_loader=True)
                #    print('='*100)



'''
python train.py --stage 'test' --model_file './result_Allweather_RN50_CA/All.pth' --save_path 'result_Allweather_RN50_CA'
python train.py --model_file './result_AllinOne/dehaze.pth' --save_path 'result_AllinOne_sk' --num_iter 150
python train.py --num_iter 100000 --val_iter 2000 --save_path 'result_Allweather_minus'
python train.py --save_path 'result_Allweather_RN50_CA'
python train.py --save_path ./results/result_lrca3_rn
'''
