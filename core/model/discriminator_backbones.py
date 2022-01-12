from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        logitor: nn.Module,
        decoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.encoder = encoder 
        self.logitor = logitor
        if decoder is not None:
            self.decoder = decoder

    def forward(self, x, **kwargs):
        out, dec = self.encoder(x)
        return out, dec


class resD(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_s = nn.Conv2d(fin,fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x)+self.gamma*self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)


class Encoder(nn.Module):
    def __init__(self, ndf, ncf):
        super().__init__()

        self.ndf = ndf 
        self.ncf = ncf

        self._define_modules()

    def _define_modules(self):
        self.conv_img = nn.Conv2d(3, self.ndf, 3, 1 ,1)
        self.block0 = resD(self.ndf * 1, self.ndf * 2)
        self.block1 = resD(self.ndf * 2, self.ndf * 4)
        self.block2 = resD(self.ndf * 4, self.ndf * 8)
        self.block3 = resD(self.ndf * 8, self.ndf * 16)
        self.block4 = resD(self.ndf * 16, self.ndf * 16)
        self.block5 = resD(self.ndf * 16, self.ndf * 16)

    def forward(self, x):
        out = self.conv_img(x) # 256x256
        out = self.block0(out) # 128x128
        out = self.block1(out) # 64x64
        out = self.block2(out) # 32x32
        out = self.block3(out) # 16x16
        dec = out
        out = self.block4(out) # 8x8
        out = self.block5(out) # 4x4

        return out, dec


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, "channels dont divide 2!"
        nc //= 2
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


def upBlock(in_planes, out_planes):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_planes, out_planes * 2, 3, 1, 1),
        nn.BatchNorm2d(out_planes * 2),
        GLU())


class Decoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.in_ch = in_ch

        self._define_modules()

    def _define_modules(self):
        self.block0 = upBlock(self.in_ch, self.in_ch//8)
        self.block1 = upBlock(self.in_ch//8, self.in_ch//16)
        self.block2 = upBlock(self.in_ch//16, self.in_ch//32)
        self.block3 = upBlock(self.in_ch//32, self.in_ch//64)
        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.in_ch//64, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.block0(x) 
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.conv_img(out)
        return out


class Logitor(nn.Module):
    def __init__(self, ndf, ncf):
        super().__init__()

        self.ndf = ndf 
        self.ncf = ncf

        self._define_modules()

    def _define_modules(self):
        self.conv_joint = nn.Sequential(
            nn.Conv2d(self.ndf * 16 + self.ncf, self.ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2 ,inplace=True),
            nn.Conv2d(self.ndf * 2, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x, c):
        c = c.view(c.size(0), -1, 1, 1)
        c = c.repeat(1, 1, x.size(-2), x.size(-1))
        x_c_code = torch.cat((x, c), dim=1)
        out = self.conv_joint(x_c_code)
        return out


if __name__ == "__main__":
    ndf = 32
    ncf = 256
    encoder = Encoder(ndf, ncf)
    decoder = Decoder(ndf * 16)
    logitor = Logitor(ndf, ncf)

    netD = Discriminator(encoder, logitor, decoder)

    x = torch.randn(4, 3, 256, 256)
    c = torch.randn(4, 256)

    out, dec = netD(x)
    logits = netD.logitor(out, c)
    rec = netD.decoder(dec)

    print(out.size(), dec.size(), rec.size())