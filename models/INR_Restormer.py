import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize 
# from einops import rearrange
# import math
from models.mlp import INR

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        
        x_norm = self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w)
        x = x + self.attn(x_norm)
        
        x_norm = self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w)
        x = x + self.ffn(x_norm)
        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

######################################################################################################################################

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_num_in_millions = total_num / 1000000
    trainable_num_in_millions = trainable_num / 1000000
    print('{:<30}  {:.2f}M'.format('Number of parameters: ', total_num_in_millions))
    print('{:<30}  {:.2f}M'.format('Number of Trainable parameters: ', trainable_num_in_millions))
    # return {'Total': total_num, 'Trainable': trainable_num}
    
######################################################################################################################################

class Restormer(nn.Module):
    def __init__(self, num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8], channels=[48, 96, 192, 384], num_refinement=4,
                 expansion_factor=2.66):
        super(Restormer, self).__init__()
        # num_blocks = [1, 2, 2, 4]

        # restormer
        self.embed_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in zip(num_blocks, num_heads, channels)])
        
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])

        self.ups1 = UpSample(channels[3])
        self.ups2 = UpSample(channels[2])
        self.ups3 = UpSample(channels[1])
        # # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList([nn.Conv2d(channels[i]+3, channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(2, len(channels)))])
        # # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))
        # # the channel of last one is not change
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1]+3, num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[1]+6, num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        # self.output = nn.Conv2d(channels[1], 3, kernel_size=3, padding=1, bias=False)
        self.output = nn.Sequential(
            nn.Conv2d(channels[1]+6, 3, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(3, 3, kernel_size=1, bias=False)
        )
        
        self.INR_s1 = INR(channels[2])
        self.INR_s2 = INR(channels[1])
        self.INR_s3 = INR(channels[0])
        self.cri_pix = nn.L1Loss().cuda()
        # self.output = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x, gt=None):
        B,C,H,W = x.shape

        # Encoder
        fo = self.embed_conv(x)
        
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))
        
        # INR
        inr_feat_s1 = self.ups1(out_enc4)
        inr_feat_s2 = self.ups2(inr_feat_s1)
        inr_feat_s3 = self.ups3(inr_feat_s2)
        
        inr_out_s1 = self.INR_s1(inr_feat_s1)
        inr_out_s2 = self.INR_s2(inr_feat_s2)
        inr_out_s3 = self.INR_s3(inr_feat_s3)

        # Decoder
        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3, inr_out_s1], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2, inr_out_s2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1, inr_out_s3], dim=1))
        
        fr = self.refinement(torch.cat([fd, x], dim=1))
        out = self.output(fr)

        # INR Loss
        torch_resize1 = Resize([out_dec3.shape[2],out_dec3.shape[3]])
        torch_resize2 = Resize([out_dec2.shape[2],out_dec2.shape[3]])
        torch_resize3 = Resize([fd.shape[2],fd.shape[3]])

        if gt is not None:
            gt_img1 = torch_resize1(gt)
            gt_img2 = torch_resize2(gt)
            gt_img3 = torch_resize3(gt)
            output_img = [inr_out_s1, inr_out_s2, inr_out_s3] 
            gt_img = [gt_img1,gt_img2,gt_img3] 
            loss = torch.stack([self.cri_pix(output_img[j],gt_img[j]) for j in range(len(output_img))])
            loss_sum = torch.sum(loss)
            return [out, loss_sum, [inr_out_s1, inr_out_s2, inr_out_s3]]
        else:
            # return out
            return out, [inr_out_s1, inr_out_s2, inr_out_s3]