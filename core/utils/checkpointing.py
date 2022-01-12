import pathlib
from typing import Any, Dict, List
import copy 

from loguru import logger
import torch

def update_average(netG, netG_ema, m=0.999):
    netG = netG.module if hasattr(netG, "module") else netG
    for p, p_ema in zip(netG.parameters(), netG_ema.parameters()):
        p_ema.data.mul_(m).add_((1.0-m)*p.detach().data)


class CheckpointManager():
    def __init__(
        self,
        save_dir: str,
        **checkpointables: Any,
    ):
        self.save_dir = pathlib.Path(save_dir) 
        self.checkpointables = copy.copy(checkpointables)

    def step(self, epoch: int):
        checkpointable_state_dict: Dict[str, Any] = self._state_dict()

        checkpointable_state_dict["epoch"] = epoch
        torch.save(
            checkpointable_state_dict,
            self.save_dir / f"checkpoint_{epoch:03d}.pth",
        )
    
    def _state_dict(self):
        __state_dict: Dict[str, Any] = {}
        for key in self.checkpointables:
            if self.checkpointables[key] is None:
                continue
            __state_dict[key] = self.checkpointables[key].state_dict()
        return __state_dict

    def load(self, checkpoint_path: str):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        epoch = checkpoint.pop("epoch", -1)

        is_loaded = {key: False for key in self.checkpointables}

        for key in checkpoint:
            if key in self.checkpointables:
                if self.checkpointables[key] is None:
                    logger.info(f"{key} is None")
                    continue
                logger.info(f"Loading {key} from {checkpoint_path}")
                self.checkpointables[key].load_state_dict[checkpoint[key]]
                is_loaded[key] = True
            else:
                logger.info(f"{key} not found in 'checkpointables'.")

        not_loaded: List[str] = [key for key in is_loaded if not is_loaded[key]]
        if len(not_loaded) > 0:
            logger.info(f"Checkpointables not found in file {not_loaded}")
        return epoch

