import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from GmSa.grassmann import FRMap, QRComposition,QR, Projmap, Orthmap,Orthmap2, OrthmapFunction, ProjPoolLayer, ProjPoolLayer_A
from GmSa.GSMA import E2R,E2R2 ,AttentionManifold
import fusion_strategy
from args_fusion import args
import matplotlib.pyplot as plt
import cv2
from scipy import stats
from torch.nn.functional import unfold
from einops.layers.torch import Rearrange
from einops import rearrange

from timm.models.layers import DropPath
from t2t_vit import Channel, Spatial
from torch import einsum
import math
import seaborn as sns
import pandas as pd


def get_patches(input, kernel_size, stride):

    patches = F.unfold(input, kernel_size, stride=stride,padding=8)

    # image = patches.squeeze().detach().cpu().numpy()
    #
    # image = (image - image.min()) / (image.max() - image.min())
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    # plt.show()
    #

    patches = patches.transpose(1, 2).contiguous().view(patches.size(0)*patches.size(-1), -1)

    return patches

def average(num1, num2):
    return (num1 + num2) / 2

def fold_with_overlap_average(patches):
    patches = patches.transpose(1, 2).contiguous()
    # 对 patches 做 fold 操作
    output = F.fold(patches, (256,256), kernel_size=16, stride=8,padding=8)



    # torch.set_printoptions(threshold=float('inf'))
    # # patches = patches.view(1,-1,256).contiguous()
    # patches = patches.transpose(1, 2).contiguous()
    # patches = patches.view(-1).contiguous()


    # for k in range(17):                               #行遍历18次
    #     for j in range(18):                           #列遍历16次
    #         for i in range(16):                       #上下两个patch之间的距离
    #             output[0, 0, i + j * 14, 14+k*14] = average(patches[14 + 256 * j * 18 + 256 * k + i * 16],patches[256 + 256 * j * 18 + 256 * k + i * 16])
    #             output[0, 0, i + j * 14, 15+k*14] = average(patches[15 + 256 * j * 18 + 256 * k + i * 16], patches[257 + 256 * j * 18  + 256 * k + i * 16])
    #
    # for k in range(17):
    #     for j in range(18):
    #         for i in range(16):
    #             output[0, 0, 14+k*14, i + 14 * j] = average(patches[224 + i + 256 * j + 18 * 256 * k], patches[256 * 18 + i + 256 * j + 18 * 256 * k])
    #             output[0, 0, 15+k*14, i + 14 * j] = average(patches[224 + 16 + i + 256 * j + 18 * 256 * k], patches[256 * 18 + 16 + i + 256 * j + 18 * 256 * k])





    # image = output.squeeze().detach().cpu().numpy()
    #
    # image = (image - image.min()) / (image.max() - image.min())
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    # plt.show()


    return output

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out


# Dense convolution unit
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out



class PatchFlattener1(nn.Module):
    def __init__(self, patch_size):
        super(PatchFlattener1, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        assert height % self.patch_size == 0 and width % self.patch_size == 0, "Invalid patch size"
        num_patches = (height // self.patch_size) * (width // self.patch_size)
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        # Rearrange patches into columns
        x = x.view(batch_size, channels, patch_height, self.patch_size, patch_width, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(batch_size, num_patches, -1)
        return x

class PatchFlattener2(nn.Module):
    def __init__(self, patch_size):
        super(PatchFlattener2, self).__init__()
        self.patch_size = patch_size
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        assert height % self.patch_size == 0 and width % self.patch_size == 0, "Invalid patch size"
        num_patches = (height // self.patch_size) * (width // self.patch_size)
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        # Rearrange patches into columns
        x = x.view(batch_size, channels, patch_height, self.patch_size, patch_width, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(batch_size, num_patches, -1)
        return x



class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        b, c, h, w = X.shape
        if c == 1:
            X = X.repeat(1, 3, 1, 1)

        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        relu4_3 = h
        # [relu1_2, relu2_2, relu3_3, relu4_3]
        return [relu1_2, relu2_2, relu3_3, relu4_3]


class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 is 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2

# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out


class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock_light, self).__init__()
        out_channels_def = int(in_channels / 2)
        # out_channels_def = out_channels
        denseblock = []

        denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvLayer(out_channels_def, out_channels, 1, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):

        x = self.proj(x)
        return x

class AttentionChannel(nn.Module):
    def __init__(self, dim, num_heads, bias, frnum):
        super(AttentionChannel, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out2 = nn.Conv2d(dim, 16, kernel_size=1, bias=bias)

        # self.frnum = 5 - frnum
        self.frnum = frnum+1

        self.orthmap = Orthmap(6)

        self.frmap = FRMap(6, self.frnum)

        self.frmap1 = FRMap(6, 5)
        self.frmap2 = FRMap(6, 4)
        self.frmap3 = FRMap(6, 3)
        self.frmap4 = FRMap(6, 2)
        # self.frmap5 = FRMap(6, 1)

        self.reorth = QRComposition()
        self.projmap = Projmap()
        self.projpooling = ProjPoolLayer()

        self.E2R = E2R(1)
        self.E2R2 = E2R2(3)
        self.att1 = AttentionManifold(48, 4)

        in_size=8
        out_size=4
        self.q_trans = FRMap(in_size, out_size).cpu()
        self.k_trans = FRMap(in_size, out_size).cpu()
        self.v_trans = FRMap(in_size, out_size).cpu()


        self.qr = QRComposition()
        self.proj = Projmap()

        self.E2R = E2R(1)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = q.double()
        k = k.double()
        v = v.double()



        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)


        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)


        attn = self.E2R(attn)
        attn = self.orthmap(attn)


        attn = attn.permute(0,1,3,2)
        attn = attn.to(args.device)

        attn1 = self.frmap1(attn)
        attn1 = attn1.permute(0, 1, 3, 2)
        # attn1 = self.qr(attn1)
        attn1 = self.proj(attn1)

        attn2 = self.frmap2(attn)
        attn2 = attn2.permute(0, 1, 3, 2)
        # attn2 = self.qr(attn2)
        attn2 = self.proj(attn2)

        attn3 = self.frmap3(attn)
        attn3 = attn3.permute(0, 1, 3, 2)
        # attn3 = self.qr(attn3)
        attn3 = self.proj(attn3)

        attn4 = self.frmap4(attn)
        attn4 = attn4.permute(0, 1, 3, 2)
        # attn4 = self.qr(attn4)
        attn4 = self.proj(attn4)



        attn = self.frmap(attn)
        attn = attn.permute(0, 1, 3, 2)
        # attn = self.qr(attn)
        attn = self.proj(attn)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)


        out = out1+out2+out3+out4
        out = out / 4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = out.float()
        out = self.project_out(out)

        return out

class AttentionSpatial(nn.Module):
    def __init__(self, dim, num_heads, bias, frnum):
        super(AttentionSpatial, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out2 = nn.Conv2d(dim, 16, kernel_size=1, bias=bias)

        self.frnum = frnum+1

        self.orthmap = Orthmap2(100)
        self.frmap = FRMap(100,100)
        self.reorth = QRComposition()
        self.projmap = Projmap()
        self.projpooling = ProjPoolLayer()


        self.att1 = AttentionManifold(48, 4)

        in_size=8
        out_size=4
        self.q_trans = FRMap(in_size, out_size).cpu()
        self.k_trans = FRMap(in_size, out_size).cpu()
        self.v_trans = FRMap(in_size, out_size).cpu()


        self.qr = QRComposition()
        self.proj = Projmap()

        self.E2R = E2R(1)
        self.E2R2 = E2R2(1)

        self.patch = PatchFlattener1(16)
        self.conv64to1 = ConvLayer(64,1,1,1)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        processed_channels1 = []
        processed_channels2 = []
        processed_channels3 = []
        for i in range(c):
            channel1 = q[:, i, :, :]  # 提取第i个通道
            channel1 = channel1.unsqueeze(1)
            processed_channel1 = self.patch(channel1)
            processed_channels1.append(processed_channel1.unsqueeze(1))


            channel2 = k[:, i, :, :]  # 提取第i个通道
            channel2 = channel2.unsqueeze(1)
            processed_channel2 = self.patch(channel2)
            processed_channels2.append(processed_channel2.unsqueeze(1))

            channel3 = v[:, i, :, :]  # 提取第i个通道
            channel3 = channel3.unsqueeze(1)
            processed_channel3 = self.patch(channel3)
            processed_channels3.append(processed_channel3.unsqueeze(1))

        q = torch.cat(processed_channels1, dim=1)

        k = torch.cat(processed_channels2, dim=1)
        v = torch.cat(processed_channels3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c h w',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c h w',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c h w',
                      head=self.num_heads)

        q = q.double()
        k = k.double()
        v = v.double()

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)


        attn = self.E2R2(attn)
        attn = self.orthmap(attn)
        attn = attn.permute(0,1,2,4,3)
        attn = attn.to(args.device)
        attn = self.frmap(attn)
        attn = attn.permute(0, 1, 2,4,3)

        attn = self.proj(attn)

        out = (attn @ v)

        out = rearrange(out, 'b head c h w -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        processed_channels4 = []
        for i in range(c):
            channel4 = out[:, i, :, :]  # 提取第i个通道
            channel4 = channel4.unsqueeze(1)
            processed_channel4 = self.patch(channel4)  # 应用函数
            processed_channels4.append(processed_channel4.unsqueeze(1))

        out = torch.cat(processed_channels4, dim=1)
        out = out.float()
        out = self.project_out(out)
        return out

class AttentionCrossChannel(nn.Module):
    def __init__(self, dim, num_heads, bias, frnum):
        super(AttentionCrossChannel, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.frnum = frnum+1

        self.orthmap = Orthmap(6)

        self.frmap = FRMap(6, self.frnum)

        self.frmap1 = FRMap(6, 5)
        self.frmap2 = FRMap(6, 4)
        self.frmap3 = FRMap(6, 3)
        self.frmap4 = FRMap(6, 2)
        self.reorth = QRComposition()
        self.projmap = Projmap()
        self.projpooling = ProjPoolLayer()

        self.E2R = E2R(1)
        self.E2R2 = E2R2(3)
        self.att1 = AttentionManifold(48, 4)

        in_size=8
        out_size=4
        self.q_trans = FRMap(in_size, out_size).cpu()
        self.k_trans = FRMap(in_size, out_size).cpu()
        self.v_trans = FRMap(in_size, out_size).cpu()


        self.qr = QRComposition()
        self.proj = Projmap()

        self.E2R = E2R(1)

    def forward(self, xir,xvi):
        b, c, h, w = xir.shape


        qkv1 = self.qkv_dwconv(self.qkv(xir))
        qkv2 = self.qkv_dwconv(self.qkv(xvi))

        q1, k1, v1 = qkv1.chunk(3, dim=1)
        q2, k2, v2 = qkv2.chunk(3, dim=1)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q1 = q1.double()
        k1 = k1.double()
        v1 = v1.double()

        q2 = q2.double()
        k2 = k2.double()
        v2 = v2.double()

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)

        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        attn1 = (q1 @ k2.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)

        attn1 = self.E2R(attn1)
        if attn1.ndim == 5:
            attn1 = attn1.squeeze(0)

        attn1 = self.orthmap(attn1)
        attn1 = attn1.permute(0,1,3,2)
        attn1 = attn1.to(args.device)

        attn11 = self.frmap1(attn1)
        attn11 = attn11.permute(0, 1, 3, 2)
        attn11 = self.proj(attn11)

        attn12 = self.frmap2(attn1)
        attn12 = attn12.permute(0, 1, 3, 2)
        attn12 = self.proj(attn12)

        attn13 = self.frmap3(attn1)
        attn13 = attn13.permute(0, 1, 3, 2)
        attn13 = self.proj(attn13)

        attn14 = self.frmap4(attn1)
        attn14 = attn14.permute(0, 1, 3, 2)
        attn14 = self.proj(attn14)

        attn2 = (q2 @ k1.transpose(-2, -1)) * self.temperature
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.E2R(attn2)
        if attn2.ndim == 5:
            attn2 = attn2.squeeze(0)

        attn2 = self.orthmap(attn2)
        # weight2 = torch.randn(8, 6, dtype=torch.float64)
        # attn2 = torch.matmul(attn2, weight2)
        attn2 = attn2.permute(0, 1, 3, 2)
        attn2 = attn2.to(args.device)

        attn21 = self.frmap1(attn2)
        attn21 = attn21.permute(0, 1, 3, 2)
        attn21 = self.proj(attn21)

        attn22 = self.frmap2(attn2)
        attn22 = attn22.permute(0, 1, 3, 2)
        attn22 = self.proj(attn22)

        attn23 = self.frmap3(attn2)
        attn23 = attn23.permute(0, 1, 3, 2)
        attn23 = self.proj(attn23)

        attn24 = self.frmap4(attn2)
        attn24 = attn24.permute(0, 1, 3, 2)
        attn24 = self.proj(attn24)


        mask = -torch.ones(8, 8)
        for i in range(min(8, 8)):
            mask[i, i] = 1.0

        # 把掩码扩展到原始张量的形状，这里利用了广播机制
        mask = mask.expand(1, 8, 8, 8)



        # 应用掩码到原始张量
        attn11 = attn11 * mask.to(args.device)
        attn12 = attn12 * mask.to(args.device)
        attn13 = attn13 * mask.to(args.device)
        attn14 = attn14 * mask.to(args.device)


        attn21 = attn21 * mask.to(args.device)
        attn22 = attn22 * mask.to(args.device)
        attn23 = attn23 * mask.to(args.device)
        attn24 = attn24 * mask.to(args.device)



        out1 = (attn1 @ v1)

        out11 = (attn11 @ v1)
        out12 = (attn12 @ v1)
        out13 = (attn13 @ v1)
        out14 = (attn14 @ v1)
        out1 = out11+out12+out13+out14
        out1 = out1 / 4


        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w',
                         head=self.num_heads, h=h, w=w)




        out21 = (attn21 @ v2)
        out22 = (attn22 @ v2)
        out23 = (attn23 @ v2)
        out24 = (attn24 @ v2)
        out2 = out21 + out22 + out23 + out24
        out2 = out2 / 4
        # out2 = (attn2 @ v2)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w',
                         head=self.num_heads, h=h, w=w)

        out1 = out1.float()
        out1 = self.project_out(out1)

        out2 = out2.float()
        out2 = self.project_out(out2)


        return [out1,out2]

class AttentionCrossSpatial(nn.Module):
    def __init__(self, dim, num_heads, bias, frnum):
        super(AttentionCrossSpatial, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.frnum = frnum+1

        self.orthmap = Orthmap2(100)
        self.frmap = FRMap(100 ,100)
        self.reorth = QRComposition()
        self.projmap = Projmap()
        self.projpooling = ProjPoolLayer()


        self.att1 = AttentionManifold(48, 4)

        in_size=8
        out_size=4
        self.q_trans = FRMap(in_size, out_size).cpu()
        self.k_trans = FRMap(in_size, out_size).cpu()
        self.v_trans = FRMap(in_size, out_size).cpu()


        self.qr = QRComposition()
        self.proj = Projmap()

        self.E2R = E2R(1)
        self.E2R2 = E2R2(1)

        self.patch = PatchFlattener1(16)
        self.conv64to1 = ConvLayer(64,1,1,1)

    def forward(self, xir,xvi):
        b, c, h, w = xir.shape

        qkv1 = self.qkv_dwconv(self.qkv(xir))
        qkv2 = self.qkv_dwconv(self.qkv(xvi))

        q1, k1, v1 = qkv1.chunk(3, dim=1)
        q2, k2, v2 = qkv2.chunk(3, dim=1)

        processed_channels11 = []
        processed_channels12 = []
        processed_channels13 = []

        processed_channels21 = []
        processed_channels22 = []
        processed_channels23 = []
        for i in range(c):
            channel11 = q1[:, i, :, :]  # 提取第i个通道
            channel11 = channel11.unsqueeze(1)
            processed_channel11 = self.patch(channel11)  # 应用函数
            processed_channels11.append(processed_channel11.unsqueeze(1))

            channel12 = k1[:, i, :, :]  # 提取第i个通道
            channel12 = channel12.unsqueeze(1)
            processed_channel12 = self.patch(channel12)  # 应用函数
            processed_channels12.append(processed_channel12.unsqueeze(1))

            channel13 = v1[:, i, :, :]  # 提取第i个通道
            channel13 = channel13.unsqueeze(1)
            processed_channel13 = self.patch(channel13)  # 应用函数
            processed_channels13.append(processed_channel13.unsqueeze(1))

            channel21 = q2[:, i, :, :]  # 提取第i个通道
            channel21 = channel21.unsqueeze(1)
            processed_channel21 = self.patch(channel21)  # 应用函数
            processed_channels21.append(processed_channel21.unsqueeze(1))

            channel22 = k2[:, i, :, :]  # 提取第i个通道
            channel22 = channel22.unsqueeze(1)
            processed_channel22 = self.patch(channel22)  # 应用函数
            processed_channels22.append(processed_channel22.unsqueeze(1))

            channel23 = v2[:, i, :, :]  # 提取第i个通道
            channel23 = channel23.unsqueeze(1)
            processed_channel23 = self.patch(channel23)  # 应用函数
            processed_channels23.append(processed_channel23.unsqueeze(1))

        q1 = torch.cat(processed_channels11, dim=1)
        k1 = torch.cat(processed_channels12, dim=1)
        v1 = torch.cat(processed_channels13, dim=1)

        q2 = torch.cat(processed_channels21, dim=1)
        k2 = torch.cat(processed_channels22, dim=1)
        v2 = torch.cat(processed_channels23, dim=1)

        q1 = rearrange(q1, 'b (head c) h w -> b head c h w',
                      head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head c h w',
                      head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head c h w',
                      head=self.num_heads)

        q2 = rearrange(q2, 'b (head c) h w -> b head c h w',
                       head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c h w',
                       head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h w -> b head c h w',
                       head=self.num_heads)

        q1 = q1.double()
        k1 = k1.double()
        v1 = v1.double()

        q2 = q2.double()
        k2 = k2.double()
        v2 = v2.double()



        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)

        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)


        attn1 = (q1 @ k2.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)


        attn1 = self.E2R2(attn1)
        attn1 = self.orthmap(attn1)
        attn1 = attn1.permute(0,1,2,4,3)
        attn1 = attn1.to(args.device)
        attn1 = self.frmap(attn1)
        attn1 = attn1.permute(0, 1, 2,4,3)
        attn1 = self.proj(attn1)

        out1 = (attn1 @ v1)

        attn2 = (q2 @ k1.transpose(-2, -1)) * self.temperature
        attn2 = attn2.softmax(dim=-1)

        attn2 = self.E2R2(attn2)
        # attn = attn.to(args.device)

        attn2 = self.orthmap(attn2)
        # weight2 = torch.randn(256, 100, dtype=torch.float64)
        # attn2 = torch.matmul(attn2, weight2)

        attn2 = attn2.permute(0, 1, 2, 4, 3)

        attn2 = attn2.to(args.device)
        attn2 = self.frmap(attn2)

        attn2 = attn2.permute(0, 1, 2, 4, 3)
        attn2 = self.proj(attn2)

        mask2 = -torch.ones(256, 256)
        for i in range(min(256, 256)):
            mask2[i, i] = 1.0

        # 把掩码扩展到原始张量的形状，这里利用了广播机制
        mask2 = mask2.expand(1, 8, 8, 256, 256)
        # 应用掩码到原始张量
        attn1 = attn1 * mask2.to(args.device)
        attn2 = attn2 * mask2.to(args.device)
        out1 = (attn1 @ v1)
        out2 = (attn2 @ v2)

        out1 = rearrange(out1, 'b head c h w -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out2 = rearrange(out2, 'b head c h w -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        processed_channels41 = []
        for i in range(c):
            channel41 = out1[:, i, :, :]  # 提取第i个通道
            channel41 = channel41.unsqueeze(1)
            processed_channel41 = self.patch(channel41)  # 应用函数
            processed_channels41.append(processed_channel41.unsqueeze(1))

        out1 = torch.cat(processed_channels41, dim=1)
        processed_channels42 = []
        for i in range(c):
            channel42 = out2[:, i, :, :]  # 提取第i个通道
            channel42 = channel42.unsqueeze(1)
            processed_channel42 = self.patch(channel42)  # 应用函数
            processed_channels42.append(processed_channel42.unsqueeze(1))

        out2 = torch.cat(processed_channels42, dim=1)
        out1 = out1.float()
        out2 = out2.float()
        out1 = self.project_out(out1)
        out2 = self.project_out(out2)
        return [out1, out2]



class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class TransformerBlock1(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, frnum):
        super(TransformerBlock1, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = AttentionChannel(dim, num_heads, bias, frnum)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.project_out2 = nn.Conv2d(dim, 16, kernel_size=1, bias=bias)

    def forward(self, x):

        xattn =  self.attn(self.norm1(x))
        x = x + xattn
        x = x + self.ffn(self.norm2(x))

        return x

class TransformerBlock2(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, frnum):
        super(TransformerBlock2, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = AttentionSpatial(dim, num_heads, bias, frnum)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        xattn = self.attn(self.norm1(x))
        x = x + xattn
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerBlock3(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, frnum):
        super(TransformerBlock3, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = AttentionCrossChannel(dim, num_heads, bias, frnum)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self,x):

        xir, xvi = x.chunk(2, dim=1)
        xattn1 = self.attn(self.norm1(xir),self.norm1(xvi))[0]
        xattn2 =  self.attn(self.norm1(xir),self.norm1(xvi))[1]
        xir = xir + xattn1
        xvi = xvi + xattn2
        xir = xir + self.ffn(self.norm2(xir))
        xvi = xvi + self.ffn(self.norm2(xvi))
        x = xir+xvi
        return x

class TransformerBlock4(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, frnum):
        super(TransformerBlock4, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = AttentionCrossSpatial(dim, num_heads, bias, frnum)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self,x):
        xir, xvi = x.chunk(2, dim=1)
        xattn1 =  self.attn(self.norm1(xir),self.norm1(xvi))[0]
        xattn2 =  self.attn(self.norm1(xir),self.norm1(xvi))[1]
        xir = xir + xattn1
        xvi = xvi + xattn2
        xir = xir + self.ffn(self.norm2(xir))
        xvi = xvi + self.ffn(self.norm2(xvi))
        x = xir+xvi
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = AttentionBase(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

import numbers
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        image = torch.sum(out, dim=1, keepdim=True)
        image = (image).squeeze().detach().cpu().numpy()

        image = (image - image.min()) / (image.max() - image.min())
        plt.imshow(image, cmap='bone')
        plt.axis('off')
        plt.show()

        out = self.proj(out)
        return out

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,
                 qkv_bias=False,):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x




# NestFuse network - light, no desnse
class GrFormer(nn.Module):
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=False):
        super(GrFormer, self).__init__()
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
        output_filter = 16
        kernel_size = 3
        stride = 1



        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2)
        # self.up_eval = UpsampleReshape_eval()


        self.conve1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU());
        self.conve2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU());
        self.conve3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU());
        self.conve4 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU());

        self.convd1 = ConvLayer(1, 16, 3, stride)
        self.convd2 = ConvLayer(16, 32, kernel_size, stride)
        self.convd3 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.convd4 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.convd5 = ConvLayer(64, output_nc, 3, stride)

        self.linear1 = nn.Linear(256, 128)
        self.pos_drop = nn.Dropout(p=0.)
        self.linear2 = nn.Linear(128, 256)
        out_channels_def = 16
        in_channels = 1

        nb_filter = [16, 64, 32, 16]
        denseblock = DenseBlock
        self.DB1 = denseblock(nb_filter[0], kernel_size, stride)



        self.convd6 = ConvLayer(256, 192, 3, stride)
        self.convd7 = ConvLayer(192, 128, 3, stride)
        self.convd8 = ConvLayer(128, 64, 3, stride)
        self.convd9 = ConvLayer(64, 1, 3, stride, is_last=True)


        inp_channels = 1
        out_channels = 1
        dim = 64
        num_blocks = [4, 4]
        heads = [8, 8, 8]
        ffn_expansion_factor = 2
        bias = False
        LayerNorm_type = 'WithBias'

        self.patch_embed1 = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed2 = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed31 = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed32 = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed41 = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed42 = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock1(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type, frnum=i) for i in range(1)])
        self.encoder_level2 = nn.Sequential(*[TransformerBlock2(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type, frnum=i) for i in range(1)])
        self.encoder_level3 = nn.Sequential(*[TransformerBlock3(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type, frnum=i) for i in range(1)])
        self.encoder_level4 = nn.Sequential(*[TransformerBlock4(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type, frnum=i) for i in range(1)])

        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])

        self.conve1 = ConvLayer(1, 128, 3, stride)
        self.conve2 = ConvLayer(1, 128, 3, stride)


    def forward(self, spdd11, spdd12):

        x1 = self.patch_embed1(spdd11)
        x1 = self.encoder_level1(x1)
        # x1 = self.encoder_level2(x1)
        # x1 = self.baseFeature(x1)

        x2 = self.patch_embed2(spdd12)
        x2 = self.encoder_level2(x2)
        # x2 = self.encoder_level2(x2)
        # x2 = self.baseFeature(x2)


        x3ir = self.patch_embed31(spdd11)
        x3vi = self.patch_embed32(spdd12)
        x3 = torch.cat([x3ir, x3vi], dim=1)
        x3 = self.encoder_level3(x3)
        # x3 = self.baseFeature(x3)


        x4ir = self.patch_embed41(spdd11)
        x4vi = self.patch_embed42(spdd12)
        x4 = torch.cat([x4ir, x4vi], dim=1)
        x4 = self.encoder_level4(x4)

        x = torch.cat([x1,x2,x3,x4], dim=1)
        # x = torch.cat([x1, x2], dim=1)

        x = self.convd6(x)
        x = self.convd7(x)
        x = self.convd8(x)
        x = self.convd9(x)


        return [x]