import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import BatchBlur, b_Bicubic, normkernel_to_downkernel, zeroize_negligible_val
from torch.autograd.function import Function


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim, feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(
            batch_size if self.size_average else 1)
        loss = self.centerlossfunc(
            feat, label, self.centers, batch_size_tensor)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(
            1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == "gan" or self.gan_type == "ragan":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif self.gan_type == "wgan-gp":

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError(
                "GAN type [{:s}] is not found".format(self.gan_type)
            )

    def get_target_label(self, input, target_is_real):
        if self.gan_type == "wgan-gp":
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer("grad_outputs", torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(
            outputs=interp_crit,
            inputs=interp,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss

class CorrectionLoss(nn.Module):
    def __init__(self, scale=4.0, eps=1e-6):
        super(CorrectionLoss, self).__init__()
        self.scale = scale
        self.eps = eps
        self.cri_pix = nn.L1Loss()

    def forward(self, k_pred, lr_blured, lr):

        ks = []
        mask = torch.ones_like(k_pred).cuda()
        for c in range(lr_blured.shape[1]):
            k_correct = normkernel_to_downkernel(lr_blured[:, c:c+1, ...], lr[:, c:c+1, ...], k_pred.size(), self.eps)
            ks.append(k_correct.clone())
            mask *= k_correct
        ks = torch.cat(ks, dim=1)
        k_correct = torch.mean(ks, dim=1, keepdim=True) * (mask>0)
        k_correct = zeroize_negligible_val(k_correct, n=40)

        return self.cri_pix(k_pred, k_correct), k_correct


