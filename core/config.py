import os.path as osp
import numpy as np
from easydict import EasyDict as edict

_C = edict()

_C.DATA = edict()
_C.DATA.ROOT = "datasets/coco"
_C.DATA.IMAGE_CROP_SIZE = 256
_C.DATA.MAX_CAPTION_LENGTH = 18

_C.TRAIN = edict()
_C.TRAIN.BATCH_SIZE = 24
_C.TRAIN.MAX_EPOCH = 120

_C.TEXT_ENCODER = edict()
_C.TEXT_ENCODER.NAME = "damsm"
_C.TEXT_ENCODER.DIR= "datasets/DAMSMencoders/text_encoder100.pth"
_C.TEXT_ENCODER.EMBEDDING_SIZE = 256
_C.TEXT_ENCODER.FROZEN = True

_C.GENERATOR = edict()
_C.GENERATOR.NAME = "df"
_C.GENERATOR.NOISE_SIZE = 100
_C.GENERATOR.FEATURE_SIZE = 32

_C.DISCRIMINATOR = edict()
_C.DISCRIMINATOR.NAME = "df"
_C.DISCRIMINATOR.FEATURE_SIZE = 32
_C.DISCRIMINATOR.DECODER = True

_C.GAN_LOSS = edict()
_C.GAN_LOSS.D_LOSS_COMPONENT = "cond_logit,img_rec"
_C.GAN_LOSS.G_LOSS_COMPONENT = "cond_logit,img_fa"
_C.GAN_LOSS.GP = "magp"
_C.GAN_LOSS.FA_COEFF = 1.

_C.OPTIM = edict()
_C.OPTIM.G = edict()
_C.OPTIM.D = edict()

_C.OPTIM.G.LR = 0.0001
_C.OPTIM.G.BETAS = [0.0 ,0.9]

_C.OPTIM.D.LR= 0.0004
_C.OPTIM.D.BETAS = [0.0, 0.9]


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, _C)