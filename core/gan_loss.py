import torch 
import torch.nn as nn
import torch.nn.functional as F

import lpips

def magp(img, sent, netD):
    img_interp = (img.data).requires_grad_()
    sent_interp = (sent.data).requires_grad_()
    out, _ = netD(img_interp)
    out = netD.logitor(out, sent_interp)
    grads = torch.autograd.grad(
        outputs=out,
        inputs=(img_interp, sent_interp),
        grad_outputs=torch.ones(out.size()).cuda(),
        retain_graph=True,
        create_graph=True,
        only_inputs=True,
    )
    grad0 = grads[0].view(grads[0].size(0), -1)
    grad1 = grads[1].view(grads[1].size(0), -1)
    grad = torch.cat((grad0, grad1), dim=1)
    grad_l2norm = torch.sqrt(torch.sum(grad**2, dim=1))
    gp = torch.mean((grad_l2norm)**6)
    return gp


class GANLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_loss_component = cfg.D_LOSS_COMPONENT.split(",")
        self.g_loss_component = cfg.G_LOSS_COMPONENT.split(",")

        self.fa_coeff = cfg.FA_COEFF

        if "img_rec" in self.d_loss_component:
            self.rec_fn = lpips.LPIPS(net="vgg").eval().requires_grad_(False)

    def get_sent_embs(self, batch, text_encoder):
        with torch.no_grad():
            tokens, tok_lens = batch["damsm_tokens"], batch["damsm_length"]
            hidden = text_encoder.init_hidden(tokens.size(0))
            _, sent_embs = text_encoder(tokens, tok_lens, hidden)
        return sent_embs

    def compute_gp(self, image, sent_embs, netD):
        loss = {}
        errD_reg = magp(image, sent_embs, netD)
        loss.update(errD_reg=errD_reg)
        return loss

    def compute_d_loss(self, image, sent_embs, fakes, netD):
        loss = {}
        rec = None

        real_feat, dec_feat = netD(image)
        fake_feat, _ = netD(fakes.detach())

        if "cond_logit" in self.d_loss_component:
            real_output = netD.logitor(real_feat, sent_embs)
            mis_output = netD.logitor(real_feat[:-1], sent_embs[1:])
            fake_output = netD.logitor(fake_feat, sent_embs)

            errD_real = F.relu(1.0 - real_output).mean()
            errD_mis = F.relu(1.0 + mis_output).mean()
            errD_fake = F.relu(1.0 + fake_output).mean()

            loss.update(
                errD_real=errD_real,
                errD_mis=0.5 * errD_mis,
                errD_fake=0.5 * errD_fake,
            )

        if "img_rec" in self.d_loss_component:
            rec = netD.decoder(dec_feat)
            errD_rec = self.rec_fn(rec, image.detach()).mean()
            loss.update(errD_rec=errD_rec)

        return loss, rec

    def compute_g_loss(self, image, sent_embs, fakes, netD):
        loss = {}

        fake_feat, fake_dec = netD(fakes)

        if "cond_logit" in self.g_loss_component:
            fake_output = netD.logitor(fake_feat, sent_embs)
            errG_fake = - fake_output.mean()
            loss.update(errG_fake=errG_fake)

        if "img_fa" in self.g_loss_component:
            with torch.no_grad():
                _, real_dec = netD(image)
            errG_fa = F.l1_loss(fake_dec, real_dec)
            loss.update(errG_fa=errG_fa)

        return loss

    def accumulate_loss(self, loss_dict):
        loss = 0.
        for val in loss_dict.values():
            loss += val
        return loss