import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict 

class affine(nn.Module):

    def __init__(self, num_features, ncf):
        super().__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(ncf, ncf)),
            ("relu1", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(ncf, num_features)),
        ]))
        
        self.fc_beta = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(ncf, ncf)),
            ("relu1", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(ncf, num_features)),
        ]))

        self._initialize()


    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.ones_(self.fc_beta.linear2.bias.data)


    def forward(self, x, c):
        weight = self.fc_gamma(c)
        bias = self.fc_beta(c)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias


class G_Block(nn.Module):
    def __init__(self, in_ch, out_ch, ncf):
        super().__init__()
        self.learnable_sc = (in_ch != out_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.affine0 = affine(in_ch, ncf)
        self.affine1 = affine(in_ch, ncf)
        self.affine2 = affine(out_ch, ncf)
        self.affine3 = affine(out_ch, ncf)
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, 1, 0)


    def forward(self, x, c):
        return self.shortcut(x) + self.gamma * self.residual(x, c)


    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x


    def residual(self, x, c):
        h = self.affine0(x, c)
        F.leaky_relu_(h, 0.2)
        h = self.affine1(x, c)
        F.leaky_relu_(h, 0.2)
        h = self.c1(h)

        h = self.affine2(h, c)
        F.leaky_relu_(h, 0.2)
        h = self.affine3(h, c)
        F.leaky_relu_(h, 0.2)
        return self.c2(h)


class Generator(nn.Module):
    def __init__(self, ngf, nz, ncf):
        super().__init__()

        self.ngf = ngf #_C.GENERATOR.FEATURE_SIZE
        self.nz = nz #_C.GENERATOR.NOISE_SIZE
        self.ncf = ncf #_C.TEXT_ENCODER.EMBEDDING_SIZE

        self._define_modules()


    def _define_modules(self):
        self.fc = nn.Linear(self.nz , 4 * 4 * self.ngf * 8)
        self.block0 = G_Block(self.ngf * 8, self.ngf * 8, ncf = self.ncf) # 4x4 
        self.block1 = G_Block(self.ngf * 8, self.ngf * 8, ncf = self.ncf) # 8x8
        self.block2 = G_Block(self.ngf * 8, self.ngf * 8, ncf = self.ncf) # 16x16
        self.block3 = G_Block(self.ngf * 8, self.ngf * 8, ncf = self.ncf) # 32x32
        self.block4 = G_Block(self.ngf * 8, self.ngf * 4, ncf = self.ncf) # 64x64
        self.block5 = G_Block(self.ngf * 4, self.ngf * 2, ncf = self.ncf) # 128x128
        self.block6 = G_Block(self.ngf * 2, self.ngf * 1, ncf = self.ncf) # 256x256

        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ngf, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x, c):
        out = self.fc(x)
        out = out.view(x.size(0), self.ngf * 8, 4, 4)
        out = self.block0(out, c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block1(out, c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block2(out, c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block3(out, c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block4(out, c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block5(out, c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block6(out, c)

        return self.conv_img(out) 



if __name__ == "__main__":
    import os 
    import sys 
    sys.path.append(os.getcwd())
    
    from core.utils.common import count_params

    netG = Generator(32, 100, 256).cuda()
    print("netG param:", count_params(netG)) # 12M

    c = torch.randn((4, 256), device="cuda")
    x = torch.randn((4, 100), device="cuda")

    out = netG(x, c)

    print(out.size())
        