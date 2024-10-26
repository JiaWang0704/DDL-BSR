import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class Encoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.encoder(x)
        return out


class DDL(nn.Module):
    def __init__(self, in_ch=3, blur_k_sz=21, n_res=12):
        super().__init__()
        md_ch = 64
        conv_k_sz = 3

        self.enc1 = EncBlock(in_ch, md_ch, n_res)
        self.enc2 = EncBlock(md_ch, md_ch*2, n_res)

        self.Bottlenec = BottlenecResBlcok(md_ch*2, md_ch*4)

        self.dec1 = DecBlock(md_ch*4, md_ch*2, n_res)
        self.dec2 = DecBlock(md_ch*2, md_ch, n_res)
        self.dec3 = nn.Sequential(nn.Conv2d(md_ch, md_ch, kernel_size=conv_k_sz, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(md_ch, blur_k_sz ** 2,
                                            kernel_size=conv_k_sz, padding=1)
                                  )

    def forward(self, x):
        skips = {}
        h, skips[0] = self.enc1(x)
        # print('1', h.shape, skips[0].shape)
        h, skips[1] = self.enc2(h)
        # print('2', h.shape, skips[1].shape)
        # bottleneck
        h = self.Bottlenec(h)
        # print('3', h.shape, skips[1].shape)
        # decoder
        h = self.dec1(h, skips[1])
        # print('4', h.shape)
        h = self.dec2(h, skips[0])
        # print('5', h.shape)
        # downsampling kernel branch
        k2d = self.dec3(h)
        return k2d

# --------------------------------------------
# MAConv and MABlock for MANet
# --------------------------------------------


class MAConv(nn.Module):
    ''' Mutual Affine Convolution (MAConv) layer '''

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, split=2, reduction=2):
        super(MAConv, self).__init__()
        assert split >= 2, 'Num of splits should be larger than one'

        self.num_split = split
        splits = [1 / split] * split
        self.in_split, self.in_split_rest, self.out_split = [], [], []

        for i in range(self.num_split):
            in_split = round(
                in_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.in_split)
            in_split_rest = in_channels - in_split
            out_split = round(
                out_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.out_split)

            self.in_split.append(in_split)
            self.in_split_rest.append(in_split_rest)
            self.out_split.append(out_split)

            setattr(self, 'fc{}'.format(i), nn.Sequential(*[
                nn.Conv2d(in_channels=in_split_rest, out_channels=int(in_split_rest // reduction),
                          kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(in_split_rest // reduction), out_channels=in_split * 2,
                          kernel_size=1, stride=1, padding=0, bias=True),
            ]))
            setattr(self, 'conv{}'.format(i), nn.Conv2d(in_channels=in_split, out_channels=out_split,
                                                        kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))

    def forward(self, input):
        input = torch.split(input, self.in_split, dim=1)
        output = []

        for i in range(self.num_split):
            scale, translation = torch.split(getattr(self, 'fc{}'.format(i))(torch.cat(input[:i] + input[i + 1:], 1)),
                                             (self.in_split[i], self.in_split[i]), dim=1)
            output.append(getattr(self, 'conv{}'.format(i))(
                input[i] * torch.sigmoid(scale) + translation))

        return torch.cat(output, 1)


class MABlock(nn.Module):
    ''' Residual block based on MAConv '''

    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True,
                 split=2, reduction=2):
        super(MABlock, self).__init__()

        self.res = nn.Sequential(*[
            MAConv(in_channels, in_channels, kernel_size,
                   stride, padding, bias, split, reduction),
            nn.ReLU(inplace=True),
            MAConv(in_channels, out_channels, kernel_size,
                   stride, padding, bias, split, reduction),
        ])

    def forward(self, x):
        return x + self.res(x)


class MANet(nn.Module):
    ''' Network of MANet'''

    def __init__(self, in_nc=3, kernel_size=21, nc=[128, 256], nb=1, split=2):
        super(MANet, self).__init__()
        self.kernel_size = kernel_size

        self.m_head = nn.Conv2d(
            in_channels=in_nc, out_channels=nc[0], kernel_size=3, padding=1, bias=True)
        self.m_down1 = sequential(*[MABlock(nc[0], nc[0], bias=True, split=split) for _ in range(nb)],
                                  nn.Conv2d(in_channels=nc[0], out_channels=nc[1], kernel_size=2, stride=2, padding=0,
                                            bias=True))

        self.m_body = sequential(
            *[MABlock(nc[1], nc[1], bias=True, split=split) for _ in range(nb)])

        self.m_up1 = sequential(nn.ConvTranspose2d(in_channels=nc[1], out_channels=nc[0],
                                                   kernel_size=2, stride=2, padding=0, bias=True),
                                *[MABlock(nc[0], nc[0], bias=True, split=split) for _ in range(nb)])
        self.m_tail = nn.Conv2d(
            in_channels=nc[0], out_channels=kernel_size ** 2, kernel_size=3, padding=1, bias=True)

        self.softmax = nn.Softmax(1)

    def forward(self, x):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x = self.m_body(x2)
        x = self.m_up1(x + x2) #[1,128,64,64] 
        # print('0',x.shape)
        x = self.m_tail(x + x1) #[1,441,64,64]
        # print('1',x.shape)
        x = x[..., :h, :w]
        # print('2',x.shape)
        x = self.softmax(x) #[1,441,64,64]
        # print('3',x.shape)
        # zzz

        return x


class D2SR_s1(nn.Module):
    '''stage1, Difficulty Prediction subnetwork'''

    def __init__(self, ):
        super().__init__()
        self.dpnet = Encoder()

    def forward(self, x):
        fea = self.dpnet(x)
        return fea


class D2SR_s2(nn.Module):
    '''stage2, Difficulty-guided degradation learning'''

    def __init__(self, in_ch=3, ksize=21, n_res=12, scale=4, nf=64, k_size=3):
        super().__init__()
        md_ch = 64
        conv_k_sz = 3
        self.scale = scale
        self.kernel_size = ksize
        self.dpnet = Encoder()
        self.ddlnet = MANet(kernel_size=ksize)
        self.linear = nn.Linear(ksize**2, ksize)
        self.linear2 = nn.Linear(ksize, 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, real_k):
        with torch.no_grad():
            d_predic = self.dpnet(x)
        
        kernel_diff = self.ddlnet(x)
        kernel_diff = kernel_diff.flatten(2).permute(0, 2, 1)

        kernel = kernel_diff.view(-1, kernel_diff.size(1),
                             self.kernel_size, self.kernel_size)

        kernel_diff = kernel_diff.flatten(start_dim=0, end_dim=1) # [b*h*w, df] df表示为退化核的特征
        kernel_soft = self.relu(self.linear(kernel_diff))
        kernel_soft = self.linear2(kernel_soft)
        with torch.no_grad():
            out = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        return out, [kernel_diff,F.log_softmax(kernel_soft, dim=1)], kernel, d_predic


class EncBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_res):
        super(EncBlock, self).__init__()
        k_sz = 3
        res_blocks = [ResBlock(out_ch, out_ch, k_sz) for i in range(n_res)]
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                  *res_blocks,
                  nn.ReLU(inplace=True),
                  ]
        self.encode = nn.Sequential(*layers)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # print('enc', x.shape)
        skip = self.encode(x)
        # print('enc2', skip.shape)
        h = self.max_pool(skip)
        # print('enc3', h.shape)
        return h, skip


class BottlenecResBlcok(nn.Module):
    def __init__(self, in_ch, out_ch):
        k_sz = 3
        super(BottlenecResBlcok, self).__init__()
        self.f = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                               ResBlock(out_ch, out_ch, k_sz),
                               nn.ReLU(inplace=True),
                               )

    def forward(self, x):
        return self.f(x)


class DecBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_res):
        super(DecBlock, self).__init__()
        # print(in_ch,out_ch,n_res)
        k_sz = 4
        self.deconv = nn.ConvTranspose2d(
            in_ch, out_ch, k_sz, padding=1, stride=2)  # Scale factor x2
        res_blocks = [ResBlock(out_ch, out_ch, 3) for i in range(n_res)]
        layers = [nn.Conv2d(out_ch*2, out_ch, kernel_size=3, padding=1),
                  *res_blocks,
                  nn.ReLU(inplace=True),
                  ]
        self.conv_res = nn.Sequential(*layers)

    def forward(self, x, skip):
        # print("11", x.shape)
        h = self.deconv(x)
        # print('22', h.shape, skip.shape)
        h = torch.cat((h, skip), 1)
        # print('33',h.shape)
        y = self.conv_res(h)
        return y


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_sz):
        super(ResBlock, self).__init__()
        self.f = nn.Sequential(nn.ReLU(inplace=True),
                               nn.Conv2d(in_ch, out_ch,
                                         kernel_size=k_sz, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(out_ch, out_ch,
                                         kernel_size=k_sz, padding=1)
                               )

    def forward(self, x):
        y = self.f(x)
        return y+x


class LocalConvDs(nn.Module):
    def __init__(self, ch, k_sz):
        super(LocalConvDs, self).__init__()
        self.ch = ch
        self.k_sz = k_sz
        self.image_patches = ExtractSplitStackImagePatches(k_sz, k_sz)

    def forward(self, img, kernel_2d):
        # local filtering operation for features
        # img: [B, C, H, W]
        # kernel_2d: [B, kernel*kernel, H, W]
        img = self.image_patches(img)  # [B, C*kh*kw, H, W]
        img = torch.split(img, self.k_sz**2, dim=1)  # kh*kw of [B, C, H, W]
        img = torch.stack(img, dim=2)  # [B, C, kh*kw, H, W]

        k_dim = kernel_2d.size()
        kernel_2d = kernel_2d.unsqueeze(1).expand(
            k_dim[0], self.ch, *k_dim[1:]).contiguous()  # [B, C, kh*kw, H, W]

        # [B, C, kh*kw, H, W] -> [B, C, H, W]
        y = torch.sum(img * kernel_2d, dim=2)
        return y


class KOALAModlule(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, covn_k_sz=3, lc_k_sz=7):
        super(KOALAModlule, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=covn_k_sz, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch,
                                            kernel_size=covn_k_sz, padding=1),
                                  nn.ReLU(inplace=True),
                                  )
        self.mult_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=covn_k_sz, padding=1),
                                       nn.ReLU(inplace=False),
                                       nn.Conv2d(
                                           out_ch, out_ch, kernel_size=covn_k_sz, padding=1),
                                       )
        self.loc_filter_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0),
                                             nn.ReLU(inplace=False),
                                             nn.Conv2d(
                                                 in_ch, lc_k_sz**2, kernel_size=1, padding=0),
                                             KernelNormalize(lc_k_sz)
                                             )
        self.local_conv = LocalConvFeat(out_ch, lc_k_sz)

    def forward(self, x, kernel):
        h = self.conv(x)
        m = self.mult_conv(kernel) #[2,64,100,100]
        # print('m', m.shape, kernel.shape)
        h = h * m
        k = self.loc_filter_conv(kernel)
        # print('k', k.shape) #[2,49,100,100]
        h = self.local_conv(h, k)
        # print('x h',h.shape, x.shape)
        return x+h

# difficulty-aware


class CAAMSFT(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.SFT_scale_conv0 = nn.Conv2d(n_feat, n_feat, 1)
        self.SFT_scale_conv1 = nn.Conv2d(n_feat, n_feat, 1)
        self.SFT_shift_conv0 = nn.Conv2d(n_feat, n_feat, 1)
        self.SFT_shift_conv1 = nn.Conv2d(n_feat, n_feat, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        # print(x[0].shape, x[1].shape)
        scale = self.SFT_scale_conv1(F.leaky_relu(
            self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        # print
        shift = self.SFT_shift_conv1(F.leaky_relu(
            self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        # print(x[0].shape, scale.shape, shift.shape)
        # zz
        return x[0] * (scale + 1) + shift


class CAAMB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size):
        super().__init__()
        self.daam1 = CAAMSFT(n_feat)
        self.daam2 = CAAMSFT(n_feat)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.daam1(x))
        out = self.relu(self.conv1(out))
        out = self.relu(self.daam2([out, x[1]]))
        out = self.conv2(out)+x[0]
        return out


class Gather(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super().__init__()
        self.dab = KOALAModlule()
        self.caamb = CAAMB(conv, n_feat, kernel_size)

    def forward(self, x):
        # print('gather', x[0].shape, x[1].shape)
        out = self.dab(x[0], x[1])
        # print(out.shape)
        out = self.caamb([out, x[2]])
        return out

# 将困难程度mask和退化信息同时融入到主分支中


class FUSION(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
        super().__init__()
        self.n_blocks = n_blocks
        # stage1 将退化信息融入
        # stage2 将困难程度mask融入
        modules_body = [Gather(conv, n_feat, kernel_size, reduction)
                        for _ in range(n_blocks)]
        self.body = nn.Sequential(*modules_body)
        self.conv = conv(n_feat, n_feat, kernel_size)

    def forward(self, x):
        res = x[0]
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1], x[2]])
        res = self.conv(res)
        res = res+x[0]
        return res


class SR(nn.Module):
    def __init__(self, scale=4, conv=default_conv):
        super().__init__()
        self.n_groups = 5
        n_feats = 64
        kernel_size = 3
        reduction = 8
        n_blocks = 5
        scale = int(scale)

        # RGB mean for DIV2K
        # compress

        self.conv1 = conv(3, n_feats, kernel_size)

        modules_body = [
            FUSION(conv, n_feats, kernel_size, reduction, n_blocks)
            for _ in range(self.n_groups)
        ]
        self.body = nn.Sequential(*modules_body)

        self.conv2 = conv(n_feats, n_feats, kernel_size)
        modules_tail = [Upsampler(conv, scale, n_feats, act=False),
                        conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, k_v, diff):
        # k_v = self.compress(k_v)
        # x = self.sub_mean(x)
        # head
        x = self.conv1(x)
        # body
        res = x
        # print('DPSR', diff.shape)
        for i in range(self.n_groups):
            res = self.body[i]([res, k_v, diff])
        res = self.conv2(res)
        res = res + x
        # end
        x = self.tail(res)
        return x


class KernelNormalize(nn.Module):
    def __init__(self, k_sz):
        super(KernelNormalize, self).__init__()
        self.k_sz = k_sz

    def forward(self, kernel_2d, dim=1):
        kernel_2d = kernel_2d - torch.mean(kernel_2d, dim, True)
        kernel_2d = kernel_2d + 1.0 / (self.k_sz ** 2)
        return kernel_2d


class LocalConvFeat(nn.Module):
    def __init__(self, ch, k_sz):
        super(LocalConvFeat, self).__init__()
        self.ch = ch
        self.k_sz = k_sz
        self.image_patches = ExtractSplitStackImagePatches(k_sz, k_sz)

    def forward(self, img, kernel_2d):
        # local filtering operation for features
        # img: [B, C, H, W]
        # kernel_2d: [B, kernel*kernel, H, W]
        # print('local feat', img.shape, kernel_2d.shape)
        # print('0', img.shape)
        img = self.image_patches(img)  # [B, C*kh*kw, H, W]
        # print('1', img.shape)
        img = torch.split(img, self.k_sz**2, dim=1)  # kh*kw of [B, C, H, W]
        # print('2', img[0].shape, len(img))
        img = torch.stack(img, dim=1)  # [B, C, kh*kw, H, W]
        # print('3', img.shape)
        # print('ker 2d',kernel_2d.shape)
        k_dim = kernel_2d.size()
        kernel_2d = kernel_2d.unsqueeze(1).expand(
            k_dim[0], self.ch, *k_dim[1:]).contiguous()  # [B, C, kh*kw, H, W]
        # print('img kernel', img.shape, kernel_2d.shape)
        # [B, C, kh*kw, H, W] -> [B, C, H, W]
        y = torch.sum(img * kernel_2d, dim=2)
        return y


class CorrectKernelBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks, k_sz=3):
        super(CorrectKernelBlock, self).__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k_sz, padding=1),
                  nn.ReLU(inplace=True)]
        for i in range(num_blocks-1):
            layers += [nn.Conv2d(out_ch, out_ch, kernel_size=k_sz, padding=1),
                       nn.ReLU(inplace=True)]
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class ExtractSplitStackImagePatches(nn.Module):
    def __init__(self, kh, kw, padding="same"):
        super(ExtractSplitStackImagePatches, self).__init__()
        # stride = 1
        self.k_sz = [kh, kw]
        if padding == 'same':
            self.pad = [(int(kw - 1)//2) for i in range(2)] + \
                [int((kh - 1)/2) for i in range(2)]
        else:
            self.pad = [0, 0]
        # print(self.pad)
        self.stride = [1, 1]

    def forward(self, x):
        # https://discuss.pytorch.org/t/tf-extract-image-patches-in-pytorch/43837/8
        x = F.pad(x, self.pad)
        # print('patch 0', x.shape)
        patches = x.unfold(2, self.k_sz[0], self.stride[0]).unfold(
            3, self.k_sz[1], self.stride[1])
        # print('patch 1', patches.shape)
        patches = patches.permute(
            0, 1, 4, 5, 2, 3).contiguous()  # [B, C, kh, kw, H, W]
        # print(patches.size()[0], patches.size()[4:])
        # print('patch 2', patches.shape)
        # zzz
        patches = patches.view(patches.size()[
                               0], -1, patches.size()[4], patches.size()[5])  # [B, C*kh*kw, H, W]
        # print('patch 3', patches.shape)

        return patches


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False


class DownsamplingNetwork(nn.Module):
    def __init__(self, in_ch=3, blur_k_sz=20, n_res=12):
        super(DownsamplingNetwork, self).__init__()
        md_ch = 64
        conv_k_sz = 3

        self.enc1 = EncBlock(in_ch, md_ch, n_res)
        self.enc2 = EncBlock(md_ch, md_ch*2, n_res)

        self.Bottlenec = BottlenecResBlcok(md_ch*2, md_ch*4)

        self.dec1 = DecBlock(md_ch*4, md_ch*2, n_res)
        self.dec2 = DecBlock(md_ch*2, md_ch, n_res)
        self.dec3 = nn.Sequential(nn.Conv2d(md_ch, md_ch, kernel_size=conv_k_sz, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(md_ch, blur_k_sz ** 2,
                                            kernel_size=conv_k_sz, padding=1)
                                  )

    def forward(self, x):
        # encoder

        skips = {}
        h, skips[0] = self.enc1(x)
        # print('1', h.shape, skips[0].shape)
        h, skips[1] = self.enc2(h)
        # print('2', h.shape, skips[1].shape)
        # bottleneck
        h = self.Bottlenec(h)
        # print('3', h.shape, skips[1].shape)
        # decoder
        h = self.dec1(h, skips[1])
        # print('4', h.shape)
        h = self.dec2(h, skips[0])
        # print('5', h.shape)
        # downsampling kernel branch
        k2d = self.dec3(h)

        return k2d


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if act:
                m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class DAAM(nn.Module):
    def __init__(self):
        super(DAAM, self).__init__()
        self.ecoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        fea = self.ecoder(x)
        return fea


class D2SR_s3(nn.Module):
    '''stage 3：是重构子网和预训练好的困难预测子网进行训练，其中，只需要优化重构子网'''

    def __init__(self, nf=64, scale=4, ksize=21):
        super(D2SR_s3, self).__init__()
        self.scale = scale
        self.kernel_size = ksize
        # s1
        self.dpnet = Encoder()
        self.c_conv = CorrectKernelBlock(self.kernel_size**2, nf, 3)
        self.bottle = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1,1), nn.LeakyReLU(0.1, True), nn.Conv2d(64, 128, 3,1,1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 3,1,1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 64, 3,1,1)
        )
        # s2
        self.ddlnet = MANet(kernel_size=ksize)
        # s3
        self.G = SR(scale=scale)
        self.linear = nn.Linear(ksize**2, ksize)
        self.linear2 = nn.Linear(ksize, 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, real_K):
        '''
        x[]
        '''
        d_predic = self.dpnet(x)
        kernel_for = self.ddlnet(x)
        kernel_diff = kernel_for.flatten(2).permute(0, 2, 1)

        kernel = kernel_diff.view(-1, kernel_diff.size(1),
                             self.kernel_size, self.kernel_size)

        kernel_diff = kernel_diff.flatten(start_dim=0, end_dim=1) # [b*h*w, df] df表示为退化核的特征
        kernel_soft = self.relu(self.linear(kernel_diff))
        kernel_soft = self.linear2(kernel_soft)
        fea_ker = self.c_conv(kernel_for)
        fea2 = self.bottle(d_predic)
        sr = self.G(x, fea_ker, fea2)
        return sr, [kernel_diff,F.log_softmax(kernel_soft, dim=1)], kernel, d_predic


if __name__ == '__main__':
    model = D2SR_s3()
    print(model)

    x = torch.randn((2, 3, 100, 100))
    x = model(x, None)
    print(x[1][0].shape, x[1][1].shape)
    zzz
    # print(x.shape)