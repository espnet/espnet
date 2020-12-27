import torch
import torch.nn.functional as F

from torch.autograd import grad as torch_grad

import math

EPSILON = 1e-6
PI = math.pi
LOG_2PI = math.log( 2.0 * PI)


class Conditions(torch.nn.Module):
    def __init__(self, cond_num, cond_dim, normalize=False):
        super(Conditions, self).__init__()
        self._embedding = torch.nn.Embedding(   cond_num,
                                                cond_dim, 
                                                padding_idx=None, 
                                                max_norm=None, 
                                                norm_type=2.0, 
                                                scale_grad_by_freq=False, 
                                                sparse=False, 
                                                _weight=None)
        if normalize:
            self.target_norm = 1.0
        else:
            self.target_norm = None
        self.embed_norm()

    def embed_norm(self):
        if self.target_norm:
            with torch.no_grad():
                self._embedding.weight.mul_(
                    self.target_norm / self._embedding.weight.norm(dim=1, keepdim=True)
                )

    def forward(self, input, pre_norm=True):
        if self.target_norm:
            if pre_norm:
                self.embed_norm()
            embedding = self.target_norm * self._embedding.weight / self._embedding.weight.norm(dim=1, keepdim=True)
            return F.embedding( input, embedding, 
                                padding_idx=None, 
                                max_norm=None, 
                                norm_type=2.0, 
                                scale_grad_by_freq=False, 
                                sparse=False)
        else:
            return self._embedding(input)

    def sparsity(self):
        sparsity = torch.mm(self._embedding.weight,self._embedding.weight.t())
        sparsity_target = torch.arange(sparsity.size(0),device=sparsity.device)
        sparsity = F.cross_entropy(sparsity,sparsity_target)
        return sparsity


class Conv2d_Layernorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=[1,1], dilation=[1,1], use_our_affine=False):
        super(Conv2d_Layernorm, self).__init__()

        padding = [ int((kernel_size[i]*dilation[i] - dilation[i])/2) 
                    for i in range(len(k))]
        self.conv = torch.nn.Conv2d( in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation,
                                     groups=1,
                                     bias=True)

        self.layernorm = torch.nn.GroupNorm( 1, 
                                             out_channels, 
                                             eps=1e-05, 
                                             affine=affine)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.02)

    def forward(self, x):
        x = self.conv(x)
        x = self.layernorm(x)
        x = self.lrelu(x)

        return x


class Conv1d_Layernorm_LRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(Conv1d_Layernorm_LRelu, self).__init__()

        padding = int((kernel_size*dilation - dilation)/2)
        self.conv = torch.nn.Conv1d( in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation,
                                     groups=1,
                                     bias=True)
 
        self.layernorm = torch.nn.GroupNorm( 1, 
                                             out_channels, 
                                             eps=1e-05, 
                                             affine=True)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.02)

        self.padding = (padding, padding)

    def forward(self, x):
        x = self.conv(x)
        x = self.layernorm(x)
        x = self.lrelu(x)

        return x


class DeConv1d_Layernorm_GLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(DeConv1d_Layernorm_GLU, self).__init__()

        padding = int((kernel_size*dilation - dilation)/2)
        self.deconv = torch.nn.ConvTranspose1d( in_channels=in_channels,
                                                out_channels=out_channels*2,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=padding,
                                                dilation=dilation,
                                                groups=1,
                                                bias=True)
 
        self.layernorm = torch.nn.GroupNorm( 2,
                                             out_channels*2, 
                                             eps=1e-05, 
                                             affine=True)

        self.half_channel = out_channels

    def forward(self, x):
        x = self.deconv(x)
        x = self.layernorm(x)
        x_tanh = torch.tanh(x[:,:self.half_channel])
        x_sigmoid = torch.sigmoid(x[:,self.half_channel:])
        x = x_tanh * x_sigmoid

        return x


def GaussianSampler(z_mu, z_lv):
    z = torch.randn_like(z_mu)
    z_std = torch.exp(0.5 * z_lv)
    z = z * z_std + z_mu
    return z


def GaussianKLD(mu1, lv1, mu2, lv2, dim=-1):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    mu_diff_sq = (mu1 - mu2).pow(2)
    element_kld = 0.5 * ((lv2 - lv1) + ( v1 + mu_diff_sq ) / ( v2 + EPSILON ) - 1.0 )
    return element_kld.sum(dim=dim)


def GaussianLogDensity(x, mu, log_var, dim=-1):
    var = torch.exp(log_var)
    mu_diff2_over_var = (x - mu).pow(2) / (var + EPSILON)
    log_prob = -0.5 * ( LOG_2PI + log_var + mu_diff2_over_var )
    return log_prob.sum(dim=dim)


def kl_loss(mu, lv):
    # Simplified from GaussianKLD
    return 0.5 * (torch.exp(lv) + mu.pow(2) - lv - 1.0).sum()

def skl_loss(mu1, lv1, mu2, lv2):
    # Symmetric GaussianKLD
    v1, v2 = torch.exp(lv1), torch.exp(lv2)
    return 0.5 * ( v2/v1 + v1/v2 - 2 + (mu1 - mu2).pow(2) / ( 1/v1 + 1/v2) ).sum()

def log_loss(x, mu):
    # Simplified from GaussianLogDensity
    return 0.5 * (LOG_2PI + (x - mu).pow(2)).sum()


def gradient_penalty_loss( x_real, x_fake, discriminator):
    assert x_real.shape == x_fake.shape
    batch_size = x_real.size(0)
    device = x_real.device

    alpha_size = [1 for i in range(x_real.ndim)]
    alpha_size[0] = batch_size
    alpha = torch.rand(alpha_size, device=device)

    interpolated = alpha * x_real.data + (1 - alpha) * x_fake.data
    interpolated.requires_grad = True

    inte_logit = discriminator(interpolated)

    gradients = torch_grad(outputs=inte_logit, inputs=interpolated,
                           grad_outputs=torch.ones_like(inte_logit, device=device),
                           create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(batch_size, -1)
    grad_l2 = torch.sqrt(torch.sum(gradients ** 2, dim=-1) + 1e-12)
    gradient_penalty = ((grad_l2 - 1) ** 2).mean()

    return gradient_penalty


def gradient_penalty_loss_S2S( x_real, x_fake, mask, discriminator):
    assert x_real.shape == x_fake.shape
    batch_size = x_real.size(0)
    device = x_real.device

    alpha_size = [1 for i in range(x_real.ndim)]
    alpha_size[0] = batch_size
    alpha = torch.rand(alpha_size, device=device)

    interpolated = alpha * x_real.data + (1 - alpha) * x_fake.data
    interpolated = interpolated.masked_fill(mask.eq(0).unsqueeze(-1), 0.0)
    interpolated.requires_grad = True

    # For Seq-to-Seq model
    inte_logit = discriminator(interpolated, mask, reduction=True)

    gradients = torch_grad(outputs=inte_logit, inputs=interpolated,
                           grad_outputs=torch.ones_like(inte_logit, device=device),
                           create_graph=True, retain_graph=True)[0]

    gradients = gradients.masked_fill(mask.eq(0).unsqueeze(-1), 0.0)
    gradients = gradients.contiguous().view(batch_size, -1)
    grad_l2 = torch.sqrt(torch.sum(gradients ** 2, dim=-1) + 1e-12)
    gradient_penalty = ((grad_l2 - 1) ** 2).mean()

    return gradient_penalty
