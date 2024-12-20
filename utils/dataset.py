import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset


def load_image(filename):
    ext = os.path.splitext(filename)[1]
    if ext == ".npy":
        return Image.fromarray(np.load(filename))
    elif ext in [".pt", ".pth"]:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


class CIIDataset(Dataset):
    def __init__(
        self,
        pre_folder: str,
        post_folder: str,
        transform=None,
    ):
        self.pre_folder = Path(pre_folder)
        self.post_folder = Path(post_folder)
        self.pre_files = list(self.pre_folder.glob("*.jpg"))
        self.post_files = list(self.post_folder.glob("*.jpg"))
        self.pre_files.sort()
        self.post_files.sort()
        if len(self.pre_files) != len(self.post_files):
            raise ValueError("Number of pre and post images must be equal")
        self.transform = transform

    def __len__(self):
        return len(self.pre_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = load_image(self.pre_files[idx])
        y = load_image(self.post_files[idx])
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y
