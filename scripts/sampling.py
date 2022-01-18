import os
import sys


sys.path.append(os.path.abspath(__file__).split("scripts")[0])

import argparse
from loguru import logger
from copy import deepcopy
from PIL import Image

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision import transforms

# fmt: off
from core.config import _C, cfg_from_file
from core.utils.common import count_params, set_seed
from core.data.damsm import DAMSMDataset, id2word
from core.model.embedding import RNN_ENCODER
from core.model.generator_backbones import Generator
from core.model.discriminator_backbones import Discriminator, Encoder, Decoder, Logitor
from core.utils.checkpointing import CheckpointManager, update_average
from core.gan_loss import GANLoss, magp 

def parse_args():
    parser = argparse.ArgumentParser(description="Train FA-GAN")
    parser.add_argument("--config", type=str, default="configs/base.yml",
                        help="Path to a config file"
    )
    parser.add_argument("--save-dir", type=str, default="exps/fa_gan",
                        help="Path to save a checkpoint"
    )
    parser.add_argument("--resume-from", type=str, default="exps/fa_gan/checkpoint_120.pth",
                        help="Path to a checkpoint to resume training"
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
    test_trans = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = DAMSMDataset(data_root = _C.DATA.ROOT, 
                                split="train", 
                                image_transform=test_trans, 
                                max_caption_length=_C.DATA.MAX_CAPTION_LENGTH) 
    #test_dataset = DAMSMDataset(data_root=, split="test", image_transform=, max_caption_length=18) 
    logger.info(f"Dataset size: {len(test_dataset)}")

    logger.info(f"Seed now is: {_A.seed}") 
    set_seed(_A.seed)

    batch_size = _C.TRAIN.BATCH_SIZE
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=test_dataset.collate_fn,
    )

    ngf = _C.GENERATOR.FEATURE_SIZE
    nz = _C.GENERATOR.NOISE_SIZE
    ncf = _C.TEXT_ENCODER.EMBEDDING_SIZE 
    netG_ema = Generator(ngf, nz, ncf).to(device)

    gan_loss = GANLoss(_C.GAN_LOSS)

    # For text encoder
    if _C.TEXT_ENCODER.NAME == "damsm":
        text_encoder = RNN_ENCODER(n_steps=_C.DATA.MAX_CAPTION_LENGTH).to(device)
        text_encoder.requires_grad_(False)
        text_encoder.eval()
    else:
        raise NotImplementedError

    # Load checkpoint to resume training if specified.
    if _A.resume_from is not None:
        start_epoch = CheckpointManager(
            _A.save_dir,
            netG_ema=netG_ema,
        ).load(_A.resume_from)
        netG_ema.requires_grad_(False)
        netG_ema.eval()
    else:
        start_epoch = 0

    logger.info(f"netG param: {count_params(netG_ema)}")
    name = _A.resume_from.split("/")[1]
    save_dir = f"eval/{name}"
    os.makedirs(save_dir, exist_ok=True)
    cnt = 0
    cond = False

    for i, batch in enumerate(test_dataloader):
        with torch.no_grad():
            batch["damsm_tokens"], batch["damsm_length"] = batch["damsm_tokens"].to(device), batch["damsm_length"].to(device)
            z = torch.randn(batch_size, nz).to(device)
            sent_embs = gan_loss.get_sent_embs(batch, text_encoder)
            fakes = netG_ema(z, sent_embs)

            for j in range(_C.TRAIN.BATCH_SIZE):
                im = fakes[j].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1,2,0))
                im = Image.fromarray(im)
                fullpath = os.path.join(save_dir, f"{id2word(test_dataset.i2w, batch['damsm_tokens'][j][:batch['damsm_length'][j]])}_{i}_{j}.jpg")
                im.save(fullpath)
                cnt += 1
                if cnt >= 30000:
                    cond = True
                    break

            if cond:
                break

if __name__ == "__main__":
    _A = parse_args()
    main(_A)
