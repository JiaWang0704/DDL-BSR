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
import matplotlib.pyplot as plt



logger = logging.getLogger("base")


class S2_Model(BaseModel):
    def __init__(self, opt, netg2=None):
        super(S2_Model, self).__init__(opt)
        self.scale = opt['scale']

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
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
            self.nllloss = nn.NLLLoss().to(self.device)
            self.l_nl_w = train_opt['nl_weight']
            self.l_c_w = train_opt['c_weight']
            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            cent_params = []
            for (
                k,
                v,
            ) in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
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
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_E'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizer_C = torch.optim.Adam(
                cent_params,
                lr=train_opt["lr_E"],
                weight_decay=wd_G,
                betas=(train_opt["beta1"], train_opt["beta2"]),
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_C)
            # self.optimizer_G = torch.optim.SGD(optim_params, lr=train_opt['lr_G'], momentum=0.9)
            # schedulers
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
        torch.autograd.set_detect_anomaly = True
        self.optimizer_G.zero_grad()
        if self.cri_ker is not None:
            self.optimizer_C.zero_grad()
        sr, kernels, kernel, diff  = self.netG(self.var_L, None)
        print(diff.shape)
        zzz
        # sr2, kernel2 = self.netG2(self.var_L)
        # print('net2', self.netG2)
        kernel_diff = kernels[0]
        kernel_soft = kernels[1]
        # print(kernel_soft.shape)
        self.fake_SR = sr
        self.fake_ker = kernel
        self.diff = diff
        total_loss = 0

        
        if self.cri_ker is not None:
            l_ker = self.l_ker_w * self.cri_ker(self.fake_ker * 10000,
                                                    self.real_kernel.unsqueeze(1).expand(-1, self.fake_ker.size(1), -1,
                                                                                    -1) * 10000) / self.fake_ker.size(1)
            total_loss = total_loss + l_ker
            # label = self.get_label(diff)
            # label = label.view(-1)
            # # torch.set_printoptions(profile="full")
            # # print(label)
            # # zzz
            # d_ker = self.l_center(label, kernel_diff)/label.size(0)
            # #softmax
            # # print(kernel_soft.shape, label.shape)
            # l_soft = self.nllloss(kernel_soft, label.long())/label.size(0)
            # print('l_soft, loss kernel',self.l_nl_w, l_soft, d_ker)
            # total_loss=total_loss+self.l_nl_w*l_soft
            # if step%5000 ==0 :      
            #     self.visualize(kernel_diff.detach().cpu().numpy(), label.detach().cpu().numpy(), step)
            # # print(d_ker.shape, l_all.shape, l_all)
            # total_loss = total_loss +  d_ker[0]
            # self.log_dict['l_ker'] = l_ker.item()
            # self.log_dict['d_ker'] = d_ker.item()  
            # self.log_dict["l_total"] = total_loss.item()
            total_loss.backward(retain_graph=True)
            self.optimizer_G.step()
            self.optimizer_C.step()

    def visualize(self, feat, labels, epoch):
            plt.ion()
            c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff']
            plt.clf()
            for i in range(4):
                plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
            plt.legend(['0', '1', '2', '3'], loc = 'upper right')
            plt.xlim(xmin=0,xmax=1)
            plt.ylim(ymin=0,ymax=1)
            plt.text(-7.8,7.3,"epoch=%d" % epoch)
            plt.savefig('/userHome/guest/wangjia/blindsr/DCLS-SR/images/step=%d.jpg' % epoch)
            plt.draw()
            plt.pause(0.001)

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            sr, kernel_diff, kernel, diff  = self.netG(self.var_L, None)

            self.fake_SR = sr
            self.fake_ker = kernel
            self.diff= diff
            # print(self.fake_ker)
        self.netG.train()

    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == "v":
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == "h":
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == "t":
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.var_L]
        for tf in "v", "h", "t":
            lr_list.extend([_transform(t, tf) for t in lr_list])
        with torch.no_grad():
            sr_list = [self.netG(aug)[0] for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], "t")
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], "h")
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], "v")

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_H = output_cat.mean(dim=0, keepdim=True)
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
    
    def get_label(self, mask):
        # print(mask.shape)
        mask = mask.sum(dim=1)
        # print(mask.shape)
        zero = torch.zeros_like(mask)
        one = torch.ones_like(mask)
        two = torch.full_like(mask, 2)
        three = torch.full_like(mask, 3)
        mask = torch.where(mask >= 0.5, three, mask)
        mask = torch.where(torch.lt(mask, 0.5) &
                           torch.gt(mask, 0.2), two, mask)
        mask = torch.where(torch.lt(mask, 0.2) &
                           torch.gt(mask, 0.1), one, mask)
        mask = torch.where(torch.lt(mask, 0.1) &
                           torch.gt(mask, 0), zero, mask)
        # print(mask)
        return mask

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
            self.load_network(load_path_G, self.netG, self.opt["path"]["strict_load"])

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