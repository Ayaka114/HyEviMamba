import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class MedMnistDataset(Dataset):
    def __init__(self, npz_path, split, transform=None, as_rgb=True):
        data = np.load(npz_path)
        self.X = data[f"{split}_images"]
        self.y = data[f"{split}_labels"]
        self.transform = transform
        self.as_rgb = as_rgb

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        img = self.X[idx]
        if img.ndim == 2:                   # 灰度
            img = Image.fromarray(img)
            if self.as_rgb:
                img = img.convert("RGB")
        else:                               # (H,W,3)
            img = Image.fromarray(img)
        if self.transform: img = self.transform(img)
        label = torch.tensor(self.y[idx]).long().squeeze()
        return img, label
