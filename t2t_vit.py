import torch
import torch.nn as nn

import numpy as np
# from token_transformer import Token_transformer
# from transformer_block import Block, get_sinusoid_encoding, PositionEmbs, DePatchEmbed
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from timm.models.layers import DropPath


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        # x = x.cuda()

        # out = self.conv2d(x)
        # out = self.bn(out)
        out = self.reflection_pad(x)

        out = self.conv2d(out)

        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class C_DePatch(nn.Module):
    def __init__(self, channel=3, embed_dim=128, patch_size=16):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, patch_size**2),
        )
   
    def forward(self, x, ori):
        b, c, h, w = ori
        h_ = h // self.patch_size
        w_ = w // self.patch_size
        x = self.projection(x)
        x = rearrange(x, '(b h w) c (p1 p2) -> b c (h p1) (w p2)', h=h_, w=w_, p1=self.patch_size, p2=self.patch_size)
        return x

#class C_DePatch(nn.Module):
#    def __init__(self, channel=3, embed_dim=128, patch_size=16):
#        self.patch_size = patch_size
#        super().__init__()
#        self.projection = nn.Sequential(
#            nn.Linear(embed_dim, patch_size**2),
#        )
#        self.f = nn.Linear(channel, 1)
#
#    def forward(self, x, ori):
#        b, c, h, w = ori
#        h_ = h // self.patch_size
#        w_ = w // self.patch_size
#        x = self.projection(x)
#        x = rearrange(x, '(b h w) c (p1 p2) -> (b h w) (p1 p2) c', h=h_, w=w_, p1=self.patch_size, p2=self.patch_size)
#        x = self.f(x)
#        x = rearrange(x, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', h=h_, w=w_, p1=self.patch_size, p2=self.patch_size)
#        return x

class S_DePatch(nn.Module):
    def __init__(self, channel=16, embed_dim=128, patch_size=16):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, patch_size**2),
        )

    def forward(self, x, ori):
        b, c, h, w = ori
        h_ = h // self.patch_size
        w_ = w // self.patch_size
        x = self.projection(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h_, w=w_, p1=self.patch_size, p2=self.patch_size)
        return x



class encoder(nn.Module):
    def __init__(self, embed_dim=256, depth=4,
                 num_heads=4, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):

            # x = self.tokens_to_token(x)

            # x = self.pos_embedding(x, ori)

            for blk in self.blocks:
                x = blk(x)

            x = self.norm(x)
            return x



class Channel(nn.Module):
    def __init__(self, size=224,embed_dim=128, depth=4, channel=16,
                 num_heads=4, mlp_ratio=2., patch_size=16,qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size**2, embed_dim),
        )
        # self.embedding = nn.Sequential(
        #     nn.Conv2d(channel, channel, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size)),
        #     Rearrange('b c h w -> b c (h w)'),
        # )
        # self.linear = nn.Linear(embed_dim, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.en = encoder(embed_dim, depth,
                 num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                 drop_path_rate, norm_layer)

        self.depatch = C_DePatch(channel=channel, embed_dim=embed_dim, patch_size=patch_size)

    def forward(self, x):
        ori = x.shape
        x2_t = self.embedding(x)                    #尺寸[256]->[128]
        x2_t = self.pos_drop(x2_t)
        x2_t = self.en(x2_t)
        out = self.depatch(x2_t, ori)               ##尺寸[128]->[256]
        return out


class Spatial(nn.Module):
    def __init__(self, size=256, embed_dim=128, depth=4, channel=16,
                 num_heads=4, mlp_ratio=2., patch_size=16, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size ** 2 * channel, embed_dim),
        )
        # self.embedding = nn.Conv2d(in_chans, embed_dim*in_chans, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        # self.linear = nn.Linear(embed_dim, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.en = encoder(embed_dim, depth,
                          num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                          drop_path_rate, norm_layer)

        self.depatch = S_DePatch(channel=channel, embed_dim=embed_dim, patch_size=patch_size)

    def forward(self, x):
        ori = x.shape
        x2_t = self.embedding(x)
        x2_t = self.pos_drop(x2_t)
        x2_t = self.en(x2_t)
        out = self.depatch(x2_t, ori)
        return out



