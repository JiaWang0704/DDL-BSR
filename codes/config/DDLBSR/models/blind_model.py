import logging
import os
from collections import OrderedDict

import torchvision.utils as tvutils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.modules.loss import CharbonnierLoss, CorrectionLoss,CenterLoss

from .base_model import BaseModel
from utils import BatchBlur
from data.util import bicubic

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger("base")


class B_Model(BaseModel):
    def __init__(self, opt, netg2=None):
        super(B_Model, self).__init__(opt)
        self.scale = opt['scale']

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        params = list(self.netG.parameters())
        k = 0
        # print(self.netG.named_parameters())
        # for i in params:
        #     l = 1
        #     # print(i)
        #     print("该层的结构：" + str(list(i.size())))
        #     for j in i.size():
        #         l *= j
        #     print("该层参数和：" + str(l))
        #     k = k + l
        #     zz
        # print("总参数数量和：" + str(k))
        # zz

        if netg2 is not None:
            self.netG2 = netg2
        if opt["dist"]:
            self.netG = DistributedDataParallel(
                self.netG, device_ids=[torch.cuda.current_device()]
            )
        else:
            self.netG = DataParallel(self.netG)
        self.bicubic = bicubic()
        self.scale = opt['scale']
        # print network
        self.print_network()
        self.load()


        if self.is_train:
            train_opt = opt["train"]
            # self.init_model() # Not use init is OK, since Pytorch has its owen init (by default)
            self.netG.train()

            # loss
            loss_type = train_opt["pixel_criterion"]
            if loss_type == "l1":
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == "l2":
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == "cb":
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type is None:
                self.cri_pix = None
            else:
                raise NotImplementedError(
                    "Loss type [{:s}] is not recognized.".format(loss_type)
                )
            self.l_pix_w = train_opt["pixel_weight"]

            loss_type = train_opt["diff_criterion"]
            if loss_type == "l1":
                self.cri_diff = nn.L1Loss().to(self.device)
            elif loss_type == "l2":
                self.cri_diff = nn.MSELoss().to(self.device)
            elif loss_type == "cb":
                self.cri_diff = CharbonnierLoss().to(self.device)
            elif loss_type is None:
                self.cri_diff = None
            else:
                raise NotImplementedError(
                    "Loss type [{:s}] is not recognized.".format(loss_type)
                )
            self.l_diff_w = train_opt["diff_weight"]

            loss_type = train_opt['']
            self.l_center = CenterLoss(
                4, opt['degradation']['ksize']**2, size_average=True).to(self.device)
            loss_type = train_opt['kernel_criterion']
            if loss_type == 'l1':
                self.cri_ker = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_ker = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_ker = CharbonnierLoss().to(self.device)
            elif loss_type is None:
                self.cri_ker = None
            else:
                raise NotImplementedError(
                    'Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_ker_w = train_opt['kernel_weight']
            self.l_nl_w = train_opt['diff_weight']
            self.nllloss = nn.NLLLoss().to(self.device)
            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_diff = []
            optim_estimator = []
            optim_sr = []
            cent_params = []
            for (
                k,
                v,
            ) in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    if "ddlnet" in k:
                        optim_estimator.append(v)
                    elif "dpnet" in k:
                        optim_diff.append(v)
                    else:
                        optim_sr.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))
            # print(self.netG.named_parameters())
            # zz
            for (
                k,
                v,
            ) in self.l_center.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    cent_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning(
                            "Params [{:s}] will not optimize.".format(k))
            self.optimizer_G = torch.optim.Adam(
                [
                    {"params": optim_sr, "lr": train_opt["lr_G"]},
                    {"params": optim_estimator, "lr": train_opt["lr_E"]},
                    {"params": optim_diff, "lr": train_opt["lr_D"]},
                ],
                weight_decay=wd_G,
                betas=(train_opt["beta1"], train_opt["beta2"]),
            )

            self.optimizer_C = torch.optim.Adam(
                cent_params,
                lr=train_opt["lr_E"],
                weight_decay=wd_G,
                betas=(train_opt["beta1"], train_opt["beta2"]),
            )
            # print(self.optimizer_G,train_opt["lr_D"])
            # print(self.optimizer_C)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_C)
            # self.optimizer_G = torch.optim.SGD(optim_params, lr=train_opt['lr_G'], momentum=0.9)
            # schedulers
            # print(self.optimizers)
            # zz
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "CosineAnnealingLR_Restart":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer,
                            train_opt["T_period"],
                            eta_min=train_opt["eta_min"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                        )
                    )
            else:
                print("MultiStepLR learning rate scheme is enough.")

            self.log_dict = OrderedDict()

    def init_model(self, scale=0.1):
        # Common practise for initialization.
        for layer in self.netG.modules():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode="fan_in")
                layer.weight.data *= scale  # for residual block
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, a=0, mode="fan_in")
                layer.weight.data *= scale
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                init.constant_(layer.weight, 1)
                init.constant_(layer.bias.data, 0.0)

    def feed_data(self, LR_img, GT_img=None, kernel=None, ker_map=None, lr_blured=None, lr=None):
        self.var_L = LR_img.to(self.device)
        if not (GT_img is None):
            self.real_H = GT_img.to(self.device)
        if not (ker_map is None):
            self.real_ker_map = ker_map.to(self.device)
        if not (kernel is None):
            self.real_kernel = kernel.to(self.device)
        if not (lr_blured is None):
            self.lr_blured = lr_blured.to(self.device)
        if not (lr is None):
            self.lr = lr.to(self.device)


    def optimize_parameters(self, step):
        import time
        self.optimizer_G.zero_grad()
        if self.cri_ker is not None:
            self.optimizer_C.zero_grad()
        sr, kernels, kernel, diff  = self.netG(self.var_L, None)
        kernel_diff = kernels[0]
        kernel_soft = kernels[1]
        sr2, kernels = self.netG2(self.var_L)
        self.fake_SR = sr
        self.fake_ker = kernel
        self.diif = diff
        total_loss = 0
        
        # 对RDP参数进行优化
        diff_label = torch.abs(self.real_H-sr2[-1])
        diff_label = self.bicubic(diff_label, scale=1/self.scale)
        diff_label = self.normalization(diff_label)

        l_diff = self.l_diff_w*self.cri_diff(diff, diff_label)
        total_loss=total_loss+l_diff
        self.log_dict["l_diff"] = l_diff.item()

        if self.cri_pix is not None:
            d_sr = self.cri_pix(sr, self.real_H)
            total_loss=total_loss+d_sr
            self.log_dict["l_pix"] = d_sr.item()
        if self.cri_ker is not None:
            cri2 = self.real_kernel.unsqueeze(1).expand(-1, kernel.size(1), -1,-1) * 10000
            l_ker = self.l_ker_w * self.cri_ker(kernel * 10000,
                                                    cri2)/self.fake_ker.size(1)
            total_loss += l_ker
            # threshold
            label = self.get_label(diff)
            label = label.view(-1)
            d_ker = self.l_center(label, kernel_diff)/label.size(0)
            total_loss += self.l_ker_w * d_ker[0]
            l_soft = self.nllloss(kernel_soft, label.long())/label.size(0)
            total_loss=total_loss+self.l_nl_w*l_soft
            total_loss += self.l_ker_w * d_ker[0]
            self.log_dict['l_ker'] = l_ker.item()
            self.log_dict['d_ker'] = d_ker.item()  
            self.log_dict["l_total"] = total_loss.item()
            total_loss.backward()
            self.optimizer_G.step()
            self.optimizer_C.step()

    def visualize(self, feat, labels, epoch):
        plt.ion()
        c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff']
        plt.clf()
        for i in range(4):
            plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
        plt.legend(['0', '1', '2', '3'], loc = 'upper right')
        plt.xlim(xmin=-8,xmax=8)
        plt.ylim(ymin=-8,ymax=8)
        plt.text(-7.8,7.3,"epoch=%d" % epoch)
        plt.savefig('./images/step=%d.jpg' % epoch)
        plt.draw()
        plt.pause(0.001)

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            sr, kernel_diff, kernel, diff  = self.netG(self.var_L, None)
            self.fake_SR = sr
            self.fake_ker = kernel
            self.diff= diff
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["LQ"] = self.var_L.detach()[0].float().cpu()
        out_dict["SR"] = self.fake_SR.detach()[0].float().cpu()
        out_dict["GT"] = self.real_H.detach()[0].float().cpu()
        out_dict["ker"] = self.fake_ker.detach()[0].float().cpu()
        out_dict["diff"] = self.diff.detach()[0].float().cpu()
        out_dict["Batch_SR"] = (
            self.fake_SR.detach().float().cpu()
        )  # Batch SR, for train
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(
            self.netG, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.netG.__class__.__name__, self.netG.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path = self.opt["path"]["pretrain_model"]
        print(load_path)
        if load_path is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path))
            self.load_network(load_path, self.netG, self.opt["path"]["strict_load"])
        load_path_G = self.opt["path"]["pretrain_model_G"]
        print(load_path_G)
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            # self.load_network(load_path_G, self.netG, self.opt["path"]["strict_load"])
            self.load_network(load_path_G, self.netG, False)

    def save(self, iter_label):
        self.save_network(self.netG, "G", iter_label)

    def normalization(self, data):
        b, c, h, w = data.size()
        data_n = data.contiguous().view(b, c, -1)
        _range = torch.max(data_n, dim=2).values - \
            torch.min(data_n, dim=2).values
        # print(_range.shape, torch.max(data_n, dim=2).values.shape, data_n.shape)
        data_min = torch.min(data_n, dim=2).values
        # print(data_min.unsqueeze(2).shape)
        # zz
        # print(data_n - .unsqueeze(3))

        data_n = (data_n - data_min.unsqueeze(2)) / _range.unsqueeze(2)
        # print('2', data.shape, data_n.shape)
        data_n = data_n.view(b, c, h, w)
        return data_n
    
    def get_label(self, mask):
        # print(mask.shape)
        mask = mask.sum(dim=1)
        # print(mask.shape)
        zero = torch.zeros_like(mask)
        one = torch.ones_like(mask)
        two = torch.full_like(mask, 2)
        three = torch.full_like(mask, 3)
         
        ####   4 class
        mask = torch.where(mask >= 0.5, three, mask)
        mask = torch.where(torch.lt(mask, 0.5) &
                           torch.gt(mask, 0.2), two, mask)
        mask = torch.where(torch.lt(mask, 0.2) &
                           torch.gt(mask, 0.05), one, mask)
        mask = torch.where(torch.lt(mask, 0.05) &
                           torch.gt(mask, 0), zero, mask) # lt 是最高限，gt是最低限
        # print(mask)
        return mask

    def save_scatter(self, data):
        import matplotlib.pyplot as plt
        print(data.shape)
        matr = data.detach()[0].float().cpu()
        print(matr.shape)
        ax = plt.matshow(matr.numpy(), cmap=plt.cm.Blues)
        plt.colorbar(ax.colorbar, fraction=0.15)
        plt.savefig('/userHome/guest/wangjia/blindsr/DCLS-SR/diff.png')