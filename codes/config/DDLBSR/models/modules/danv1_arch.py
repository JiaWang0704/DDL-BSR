import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import PCAEncoder


class CALayer(nn.Module):
    def __init__(self, nf, reduction=16):
        super(CALayer, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf // reduction, nf, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        y = self.avg(x)
        y = self.body(y)
        return torch.mul(x, y)


class CRB_Layer(nn.Module):
    def __init__(self, nf1, nf2):
        super(CRB_Layer, self).__init__()

        body = [
            nn.Conv2d(nf1 + nf2, nf1 + nf2, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf1 + nf2, nf1, 3, 1, 1),
            CALayer(nf1),
        ]

        self.body = nn.Sequential(*body)

    def forward(self, x):
        f1, f2 = x
        f1 = self.body(torch.cat(x, 1)) + f1
        return [f1, f2]


class Estimator(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_blocks=5, scale=4, kernel_size=4):
        super(Estimator, self).__init__()
        self.ksize = kernel_size

        self.head_LR = nn.Conv2d(in_nc, nf // 2, 1, 1, 0)
        self.head_HR = nn.Conv2d(in_nc, nf // 2, 9, scale, 4)

        body = [CRB_Layer(nf // 2, nf // 2) for _ in range(num_blocks)]
        self.body = nn.Sequential(*body)

        self.out = nn.Conv2d(nf // 2, 10, 3, 1, 1)
        self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, GT, LR):

        lrf = self.head_LR(LR)
        hrf = self.head_HR(GT)

        f = [lrf, hrf]
        f, _ = self.body(f)
        f = self.out(f)
        f = self.globalPooling(f)
        f = f.view(f.size()[:2])

        return f


class Restorer(nn.Module):
    def __init__(
        self, in_nc=3, out_nc=3, nf=64, nb=8, scale=4, input_para=10, min=0.0, max=1.0
    ):
        super(Restorer, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.num_blocks = nb

        self.head = nn.Conv2d(in_nc, nf, 3, stride=1, padding=1)

        body = [CRB_Layer(nf, input_para) for _ in range(nb)]
        self.body = nn.Sequential(*body)

        self.fusion = nn.Conv2d(nf, nf, 3, 1, 1)

        if scale == 4:  # x4
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(nf, 3, 3, 1, 1),
            )
        else:  # x2, x3
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale ** 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale),
                nn.Conv2d(nf, 3, 3, 1, 1),
            )

    def forward(self, input, ker_code):
        B, C, H, W = input.size()  # I_LR batch
        B_h, C_h = ker_code.size()  # Batch, Len=10
        ker_code_exp = ker_code.view((B_h, C_h, 1, 1)).expand(
            (B_h, C_h, H, W)
        )  # kernel_map stretch

        f = self.head(input)
        inputs = [f, ker_code_exp]
        f, _ = self.body(inputs)
        f = self.fusion(f)
        out = self.upscale(f)

        return out  # torch.clamp(out, min=self.min, max=self.max)


class DANv1(nn.Module):
    def __init__(
        self,
        nf=64,
        nb=16,
        upscale=2,
        input_para=10,
        kernel_size=11,
        loop=8,
        pca_matrix_path='/userHome/guest/wangjia/blindsr/DDL-BSR/pca_matrix/pca_aniso_matrix_x2.pth',
    ):
        super(DANv1, self).__init__()

        self.ksize = kernel_size
        self.loop = loop
        self.scale = upscale

        self.Restorer = Restorer(nf=nf, nb=nb, scale=self.scale, input_para=input_para)
        self.Estimator = Estimator(kernel_size=kernel_size, scale=self.scale)
        # print(pca_matrix_path, kernel_size)
        # zz
        self.register_buffer("encoder", torch.load(pca_matrix_path)[None])

        kernel = torch.zeros(1, self.ksize, self.ksize)
        kernel[:, self.ksize // 2, self.ksize // 2] = 1

        self.register_buffer("init_kernel", kernel)
        init_ker_map = self.init_kernel.view(1, 1, self.ksize ** 2).matmul(
            self.encoder
        )[:, 0]
        self.register_buffer("init_ker_map", init_ker_map)

    def forward(self, lr):
        # print(lr.shape)
        srs = []
        ker_maps = []

        B, C, H, W = lr.shape
        ker_map = self.init_ker_map.repeat([B, 1])

        for i in range(self.loop):

            sr = self.Restorer(lr, ker_map.detach())
            ker_map = self.Estimator(sr.detach(), lr)

            srs.append(sr)
            ker_maps.append(ker_map)
            # print(ker_map.shape)
            # zzz
        return [srs, ker_maps]
