import os
import pickle
from typing import Dict, List, Any
from PIL import Image

import torch
from torch.utils.data import Dataset
import numpy as np

from torchvision import transforms

class DAMSMDataset(Dataset):
    r"""
    A PyTorch dataset to read MS-COCO14 dataset 
    Args:
        data_root: Path to the COCO dataset root directory.
        split: Name of COCO 2014 split to read. One of ``{"train", "test"}``.
    """

    def __init__(
        self, 
        data_root: str, 
        split: str,
        image_transform: transforms,
        max_caption_length: int = 18):

        self.image_transform = image_transform
        self.max_caption_length = max_caption_length

        # Load preprocessed data.
        x = pickle.load(open(os.path.join(data_root, "captions.pickle"),"rb"))
        self.image_dir = os.path.join(data_root, f"images")
        self.filenames = pickle.load(open(os.path.join(data_root,f"{split}/filenames.pickle"),"rb"))
        self.i2w = x[2]
        self.w2i = x[3]
        self.damsm_tokens = x[0] if split == "train" else x[1]
        del x

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        key = self.filenames[idx] 
        image_path = os.path.join(self.image_dir, f"{key}.jpg")
        choice_idx = np.random.choice(5)

        # shape: (height, width, channels), dtype: uint8
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)
        sent_idx = 5 * idx + choice_idx
        damsm_tokens = self.damsm_tokens[sent_idx]
        damsm_length = len(damsm_tokens)
        if damsm_length > self.max_caption_length:
            damsm_length = self.max_caption_length
            damsm_tokens = damsm_tokens[:damsm_length]

        return {
            "key": key,
            "image": image, 
            "damsm_tokens": torch.tensor(damsm_tokens, dtype=torch.long),
            "damsm_length": torch.tensor(damsm_length, dtype=torch.long),
        }

    
    def collate_fn(
        self, data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        damsm_tokens = torch.nn.utils.rnn.pad_sequence(
            [d["damsm_tokens"] for d in data],
            batch_first=True,
            padding_value = 0
        )

        return {
            "key": [d["key"] for d in data],
            "image": torch.stack([d["image"] for d in data], dim=0),
            "damsm_tokens": damsm_tokens,
            "damsm_length": torch.stack([d["damsm_length"] for d in data], dim=0),
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # Same as DF-GAN
    img_size = 256
    train_trans = transforms.Compose([
        transforms.Resize(int(img_size*76/64)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_trans = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = DAMSMDataset(
        data_root="datasets/coco", 
        split="train", 
        image_transform=train_trans, 
        max_caption_length=18
    )
    test_dataset= DAMSMDataset(
        data_root="datasets/coco", 
        split="test", 
        image_transform=test_trans, 
        max_caption_length=18
    )

    print("train_sample", train_dataset[0])
    print("test_sample", test_dataset[0])

    train_loader = DataLoader(train_dataset, collate_fn=train_dataset.collate_fn,
                            batch_size = 4, shuffle=False,  drop_last=True)

    test_loader = DataLoader(test_dataset, collate_fn=test_dataset.collate_fn,
                            batch_size = 4, shuffle=False,  drop_last=True)


    it = iter(train_loader)
    batch = next(it)
    print("train_batch_damsm_tokens", batch["damsm_tokens"])

    it = iter(test_loader)
    batch = next(it)
    print("test_batch_damsm_tokens", batch["damsm_tokens"])
