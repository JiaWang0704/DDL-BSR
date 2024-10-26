import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict

from thop import profile



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class KernelNormalize(nn.Module):
    def __init__(self, k_sz):
        super(KernelNormalize, self).__init__()
        self.k_sz = k_sz

    def forward(self, kernel_2d, dim=1):
        kernel_2d = kernel_2d - torch.mean(kernel_2d, dim, True)
        kernel_2d = kernel_2d + 1.0 / (self.k_sz ** 2)
        return kernel_2d
    

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
        h = h * m
        k = self.loc_filter_conv(kernel)
        # print('k', k.shape) #[2,49,100,100]
        h = self.local_conv(h, k)
        # print('x h',h.shape, x.shape)
        return x+h

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


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_sz):
        super(ResBlock, self).__init__()
        self.f = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch,
                                kernel_size=k_sz, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch,
                                kernel_size=k_sz, padding=1),
                    nn.ReLU(inplace=True),
                    )

    def forward(self, x):
        y = self.f(x)
        return y + x
    
class CascadeResBlock(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, k_sz=3):
        super().__init__()
        self.f = nn.Sequential(
                ResBlock(in_ch, out_ch,k_sz),
                nn.ReLU(inplace=True),
                ResBlock(in_ch, out_ch,k_sz),
            )
    
    def forward(self, x):
        y = self.f(x)
        return y + x

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

class DDLNet(nn.Module):
    '''Difficulty-guided degradation learning'''
    def __init__(self, nf=64, k_size=21, num_crblocks=5,conv_k_sz=3):
        super().__init__()
        self.conv_head = nn.Conv2d(nf, nf, 3,1,1)
        blocks = []
        for _ in range(num_crblocks):
            block = CascadeResBlock(in_ch=nf, out_ch=nf, k_sz=conv_k_sz)
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)
        self.conv_tail=nn.Conv2d(nf, k_size**2, 3,1,1)
        self.conv_tail2 = nn.Conv2d(k_size**2,k_size**2,3,1,1)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        y = self.conv_head(x)
        y = self.blocks(y)
        y = self.conv_tail(y)
        y = self.conv_tail2(y)
        y = self.softmax(y)
        return y
    
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

        

class D2SR_s1(nn.Module):
    '''stage1, Difficulty Prediction subnetwork'''

    def __init__(self, ):
        super().__init__()
        self.dpnet = Encoder()

    def forward(self, x):
        fea = self.dpnet(x)
        return fea
    
class D2SR_s2(nn.Module):
    def __init__(self, nf=64, k_size=21, scale=4):
        super().__init__()
        self.kernel_size = k_size
        self.scale = scale
        self.dpnet = Encoder()
        self.feature_extra = nn.Conv2d(3,nf,3,1,1)
        self.ddlnet = DDLNet(nf=nf)
        self.linear = nn.Linear(k_size**2, k_size)
        self.linear2 = nn.Linear(k_size, 4)
        self.relu=nn.ReLU(inplace=True)
        
    def forward(self, x, real_k):
        with torch.no_grad():
            d_predic = self.dpnet(x)
        y = self.feature_extra(x)
        kernel_diff = self.ddlnet(y)
        # print('kernel diff', kernel_diff.shape)
        kernel_diff = kernel_diff.flatten(2).permute(0, 2, 1)

        kernel = kernel_diff.view(-1, kernel_diff.size(1),
                             self.kernel_size, self.kernel_size)

        kernel_diff = kernel_diff.flatten(start_dim=0, end_dim=1) # [b*h*w, df] df表示为退化核的特征
        kernel_soft = self.relu(self.linear(kernel_diff))
        kernel_soft = self.linear2(kernel_soft)
        with torch.no_grad():
            out = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        return out, [kernel_diff,F.log_softmax(kernel_soft, dim=1)], kernel, d_predic
    

class D2SR_s3(nn.Module):
    '''stage 3：是重构子网和预训练好的困难预测子网进行训练，其中，只需要优化重构子网'''

    def __init__(self, nf=64, scale=4, ksize=21, n_resblock=3):
        super(D2SR_s3, self).__init__()
        self.scale = scale
        self.kernel_size = ksize
        # feature extraction 特征提取
        self.fea_extra = nn.Conv2d(3,nf,3,1,1)
        # stage1 困难程度图预测
        self.dpnet = Encoder()
        # self.c_conv = CorrectKernelBlock(self.kernel_size**2, nf, 3)
        self.bottle = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1,1), nn.LeakyReLU(0.1, True), nn.Conv2d(64, 128, 3,1,1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 3,1,1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 64, 3,1,1)
        )
        # stage2 退化核预测网络
        self.ddlnet = DDLNet(k_size=ksize)
        # stage3 重构网络
        self.c_conv = CorrectKernelBlock(self.kernel_size**2, nf, 3)
        self.G = SR(scale=scale)
        self.linear = nn.Linear(ksize**2, ksize)
        self.linear2 = nn.Linear(ksize, 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, real_K=None):
        '''
        x[]
        '''
        # print('input', x.shape)
        kernel_for = self.fea_extra(x)
        # with torch.no_grad():
        d_predic = self.dpnet(x)
        kernel_for = self.ddlnet(kernel_for)
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



    x = torch.randn((1,3, 64, 64)).float().cpu()
    # stat(model, (1, 3, 64, 64))
    # x = model(x)
    # print(x[0].shape, x[1][0].shape, x[1][1].shape)
    # # print(x.shape)
    import thop
    flops, params = thop.profile(model,inputs=(x,))
    print(flops, params)
    print('model size:{:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))