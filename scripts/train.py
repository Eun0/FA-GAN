import os
import sys


sys.path.append(os.path.abspath(__file__).split("scripts")[0])

import argparse
from typing import Any
from loguru import logger
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision import transforms

# fmt: off
from core.config import _C, cfg_from_file
from core.utils.common import count_params, set_seed
from core.data.damsm import DAMSMDataset
from core.model.embedding import RNN_ENCODER
from core.model.generator_backbones import Generator
from core.model.discriminator_backbones import Discriminator, Encoder, Decoder, Logitor
from core.utils.checkpointing import CheckpointManager, update_average
from core.gan_loss import GANLoss, magp 

def parse_args():
    parser = argparse.ArgumentParser(description="Train FA-GAN")
    parser.add_argument("--config", type=str, default="configs/debug.yml",
                        help="Path to a config file"
    )
    parser.add_argument("--save-dir", type=str, default="exps/debug",
                        help="Path to save a checkpoint"
    )
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a checkpoint to resume training"
    )
    parser.add_argument("--checkpoint-epoch", type=int, default=1,
                        help="Save checkpoint after every these many epochs"
    )
    parser.add_argument("--log-iter", type=int, default=20,
                        help="Log training loss after every these many iterations"
    )
    parser.add_argument("--logging", type=str, default="tb",
                        help="Where to log"
    )
    parser.add_argument("--seed", type=int, default=100, help="manual seed")
    args = parser.parse_args()
    return args


def main(_A:argparse.Namespace):

    device = torch.cuda.current_device()

    if _A.logging == "tb":
        from torch.utils.tensorboard import SummaryWriter
    else:
        raise NotImplementedError
        logger.add(os.path.join(_A.serialization_dir, "metric.log"), filter=lambda record: "metric" in record["extra"])
        logger.add(os.path.join(_A.serialization_dir, "loss.log"), filter=lambda record: "loss" in record["extra"])

    if _A.config is not None:
        cfg_from_file(_A.config)

    logger.info("Using config")
    logger.info(_C)

    # -------------------------------------------------------------------------
    #   INSTANTIATE DATALOADER, MODEL, OPTIMIZER, SCHEDULER
    # -------------------------------------------------------------------------
    img_size = _C.DATA.IMAGE_CROP_SIZE
    train_trans = transforms.Compose([
        transforms.Resize(int(img_size*76/64)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = DAMSMDataset(data_root = _C.DATA.ROOT, 
                                split="train", 
                                image_transform=train_trans, 
                                max_caption_length=_C.DATA.MAX_CAPTION_LENGTH) 
    #test_dataset = DAMSMDataset(data_root=, split="test", image_transform=, max_caption_length=18) 
    logger.info(f"Dataset size: {len(train_dataset)}")

    logger.info(f"Seed now is: {_A.seed}") 
    set_seed(_A.seed)

    batch_size = _C.TRAIN.BATCH_SIZE
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    )

    ngf = _C.GENERATOR.FEATURE_SIZE
    nz = _C.GENERATOR.NOISE_SIZE
    ncf = _C.TEXT_ENCODER.EMBEDDING_SIZE 
    netG = Generator(ngf, nz, ncf).to(device)
    netG_ema = deepcopy(netG).eval().requires_grad_(False)

    ndf = _C.DISCRIMINATOR.FEATURE_SIZE
    encoder = Encoder(ndf, ncf)
    logitor = Logitor(ndf, ncf)
    decoder = Decoder(ndf * 16) if _C.DISCRIMINATOR.DECODER else None
    netD = Discriminator(encoder, logitor, decoder).to(device)

    # For text encoder
    if _C.TEXT_ENCODER.NAME == "damsm":
        text_encoder = RNN_ENCODER(n_steps=_C.DATA.MAX_CAPTION_LENGTH).to(device)
        text_encoder.requires_grad_(False)
        text_encoder.eval()
    else:
        raise NotImplementedError

    gan_loss = GANLoss(_C.GAN_LOSS).to(device)

    optG = torch.optim.Adam(netG.parameters(), lr=_C.OPTIM.G.LR, betas=_C.OPTIM.G.BETAS)
    optD = torch.optim.Adam(netD.parameters(), lr=_C.OPTIM.D.LR, betas=_C.OPTIM.D.BETAS)
    # -------------------------------------------------------------------------
    #   BEFORE TRAINING STARTS
    # -------------------------------------------------------------------------

    # Load checkpoint to resume training if specified.
    if _A.resume_from is not None:
        start_epoch = CheckpointManager(
            _A.save_dir,
            netG=netG, netD=netD, optG=optG, optD=optD, netG_ema=netG_ema,
        ).load(_A.resume_from)
    else:
        start_epoch = 0

    logger.info(f"netG param: {count_params(netG)}")
    logger.info(f"netD param: {count_params(netD)}")
    if _A.logging == "tb":
        tensorboard_writer = SummaryWriter(log_dir=_A.save_dir)
        tensorboard_writer.add_text("config", f"```\n{_C}\n```")
    else:
        raise NotImplementedError

    checkpoint_manager = CheckpointManager(
        _A.save_dir,
        netG=netG,
        netD=netD,
        optG=optG,
        optD=optD,
        netG_ema = netG_ema,
    )

    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------

    for epoch in range(start_epoch + 1, _C.TRAIN.MAX_EPOCH + 1):
        for step, batch in enumerate(train_dataloader):
            netG.train(), netD.train()

            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Train Discriminator
            z = torch.randn(batch_size, nz).to(device).detach()
            sent_embs = gan_loss.get_sent_embs(batch, text_encoder)
            fakes = netG(z, sent_embs)

            netD.requires_grad_(True)
            d_loss_dict, rec = gan_loss.compute_d_loss(batch["image"], sent_embs, fakes, netD) 
            errD = gan_loss.accumulate_loss(d_loss_dict)

            optD.zero_grad(), optG.zero_grad()
            errD.backward()
            optD.step()

            if _C.GAN_LOSS.GP == "magp":
                errD_reg = magp(batch["image"], sent_embs, netD) 
                optD.zero_grad(), optG.zero_grad()
                errD_reg.backward()
                optD.step()

            # Train Generator
            netD.requires_grad_(False)
            g_loss_dict = gan_loss.compute_g_loss(batch["image"], sent_embs, fakes, netD)
            errG = gan_loss.accumulate_loss(g_loss_dict) 

            optD.zero_grad(), optG.zero_grad()
            errG.backward()
            optG.step()
            update_average(netG, netG_ema)
        
            # ---------------------------------------------------------------------
            #   LOGGING
            # ---------------------------------------------------------------------
            if step % _A.log_iter == 0:
                log = f"[errD {errD.detach():.3f} errG {errG.detach():.3f}]\n"
                for key in d_loss_dict:
                    log += f'{key}: {d_loss_dict[key].detach():.3f} '
                for key in g_loss_dict:
                    log += f'{key}: {g_loss_dict[key].detach():.3f} '


                if _A.config == "configs/debug.yml":
                    vutils.save_image(fakes.data, f'fake.png', normalize=True, scale_each=True)
                    vutils.save_image(batch["image"].data, f"real.png", normalize=True, scale_each=True)
                    if rec is not None:
                        vutils.save_image(rec.data, f'rec.png', normalize=True, scale_each=True)
                if _A.logging == "tb":
                    logger.info(log)
                    #tensorboard_writer.add_scalars("D", d_loss_dict, step)
                    #tensorboard_writer.add_scalars("G", g_loss_dict, step)
                else:
                    logger.bind(loss=True).info(log)

        # ---------------------------------------------------------------------
        #  Checkpointing
        # ---------------------------------------------------------------------
        if epoch % _A.checkpoint_epoch == 0 or epoch == _C.TRAIN.MAX_EPOCH:
            checkpoint_manager.step(epoch)
            netG_ema.eval(), netD.eval()
            vutils.save_image(fakes.data, os.path.join(_A.save_dir, f'{epoch}.png'), normalize=True, scale_each=True, nrow=8)
            if rec is not None:
                vutils.save_image(rec.data, f'rec.png', normalize=True, scale_each=True)

if __name__ == "__main__":
    _A = parse_args()
    main(_A)
