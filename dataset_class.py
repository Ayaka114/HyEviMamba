# dataset_class.py
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, random_split, Subset
from sklearn.model_selection import train_test_split

# ---------------------------
# 通用工具
# ---------------------------

def infer_num_classes(ds):
    """兼容 ImageFolder/OdirDataset/Subset 的类别数推断"""
    if hasattr(ds, "num_classes"):
        return int(ds.num_classes)
    if hasattr(ds, "classes"):
        return len(ds.classes)
    if isinstance(ds, Subset):
        return infer_num_classes(ds.dataset)
    # 兜底：从一条样本的标签形状推断
    x, y = ds[0]
    if torch.is_tensor(y):
        return int(y.numel()) if y.ndim > 0 else int(max(int(y), 0) + 1)
    raise ValueError("Cannot infer num_classes from dataset")

def set_transform_for_subset(subset: Subset, transform):
    """给 Subset 底层数据集设置 transform"""
    if isinstance(subset, Subset):
        subset.dataset.transform = transform

# ---------------------------
# 1) Messidor2：688维向量 → 灰度图
# ---------------------------

class Messidor2VectorAsImage(Dataset):
    """
    将每条 688 维向量 reshape 为灰度图 (H×W)，默认 H=16, W=43（16*43=688）
    返回 (PIL.Image 或 Tensor, label)
    """
    def __init__(self, mat_path, H=16, W=43, transform=None, to_rgb=True, per_sample_minmax=True):
        self.transform = transform
        self.H, self.W = H, W
        self.to_rgb = to_rgb
        self.per_sample_minmax = per_sample_minmax

        import scipy.io as sio
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        data = mat["data"]  # (N_bags,2)

        self.samples = []
        for i in range(data.shape[0]):
            feats = data[i, 0]                      # (n_i, 688)
            lab = int(np.array(data[i, 1]).ravel()[0])
            for k in range(feats.shape[0]):
                vec = feats[k].astype(np.float32)

                if self.per_sample_minmax:
                    vmin, vmax = float(vec.min()), float(vec.max())
                    if vmax > vmin:
                        vec = (vec - vmin) / (vmax - vmin)
                else:
                    vec = (vec - vec.mean()) / (vec.std() + 1e-6)
                    vec = (vec - vec.min()) / (vec.max() - vec.min() + 1e-6)

                img = (vec * 255.0).reshape(self.H, self.W).clip(0, 255).astype(np.uint8)
                self.samples.append((img, lab))

        self.classes = [0, 1]           # 二分类（按原 bag 标签）
        self.num_classes = 2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr, label = self.samples[idx]
        img = Image.fromarray(arr, mode="L")
        if self.to_rgb:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

def load_messidor2_as_image(mat_path, transform_train=None, transform_val=None,
                            val_ratio=0.2, seed=42):
    """
    Messidor2VectorAsImage 的分层采样划分：
    - 保持类别分布一致（stratify）
    - 训练/验证使用不同 transform
    - 为 Subset 挂上 classes，方便后续 len(...classes)
    """
    # 先构造一个无增强的数据集，用于拿标签和确定样本顺序
    full_plain = Messidor2VectorAsImage(mat_path, transform=None)

    # 提取每个样本的标签（用于分层）
    y = np.array([lab for _, lab in full_plain.samples], dtype=np.int64)

    # 分层划分索引
    idx_all = np.arange(len(full_plain))
    train_idx, val_idx = train_test_split(
        idx_all,
        test_size=val_ratio,
        random_state=seed,
        stratify=y
    )

    # 分别构造带各自 transform 的底层数据集（样本顺序与 full_plain 一致）
    ds_train_full = Messidor2VectorAsImage(mat_path, transform=transform_train)
    ds_val_full   = Messidor2VectorAsImage(mat_path, transform=transform_val)

    # 用相同索引切子集
    train_set = Subset(ds_train_full, train_idx.tolist())
    val_set   = Subset(ds_val_full,   val_idx.tolist())

    # 方便后续使用：给 Subset 挂上 classes / num_classes
    train_set.classes = full_plain.classes
    val_set.classes   = full_plain.classes
    train_set.num_classes = getattr(full_plain, "num_classes", len(full_plain.classes))
    val_set.num_classes   = getattr(full_plain, "num_classes", len(full_plain.classes))

    return train_set, val_set

# ---------------------------
# 2) MedMNIST：npz 格式
# ---------------------------

class MedMnistDataset(Dataset):
    """
    兼容 medmnist 导出的 *.npz：
    - {split}_images: (N, H, W) 或 (N, H, W, 3)
    - {split}_labels: (N, 1) 或 (N,)；也可能是 one-hot
    """
    def __init__(self, npz_path, split, transform=None, as_rgb=True, labels_are_multilabel=False):
        data = np.load(npz_path)
        self.X = data[f"{split}_images"]
        self.y = data[f"{split}_labels"]
        self.transform = transform
        self.as_rgb = as_rgb
        self.labels_are_multilabel = labels_are_multilabel

        # 统一标签形状
        self.y = np.array(self.y)
        if self.y.ndim > 1 and self.y.shape[-1] == 1:
            self.y = self.y.squeeze(-1)

        # 推断类别数
        if labels_are_multilabel:
            self.num_classes = int(self.y.shape[-1])
            self.classes = list(range(self.num_classes))
        else:
            self.num_classes = int(np.max(self.y)) + 1
            self.classes = list(range(self.num_classes))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        img = self.X[idx]
        if img.ndim == 2:  # 灰度
            img = Image.fromarray(img)
            if self.as_rgb:
                img = img.convert("RGB")
        else:              # (H,W,3)
            img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)

        if self.labels_are_multilabel:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
        else:
            label = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return img, label

# ---------------------------
# 3) ODIR-5K：Excel + 图像目录（多标签）
# ---------------------------

class OdirDataset(Dataset):
    """
    默认读取 Training Images（带标签，8维多标签：N,D,G,C,A,H,M,O）
    支持 patient-wise 划分时的下游使用（本类不做划分，仅负责读取）
    """
    LABEL_COLS = ["N", "D", "G", "C", "A", "H", "M", "O"]

    def __init__(self, root_dir, img_dir="Training Images", excel_file="data.xlsx",
                 use_eyes=("Left-Fundus", "Right-Fundus"),
                 transform=None, as_rgb=True, skip_missing=True, return_path=False):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, img_dir)
        self.excel_path = os.path.join(root_dir, excel_file)
        self.use_eyes = use_eyes
        self.transform = transform
        self.as_rgb = as_rgb
        self.skip_missing = skip_missing
        self.return_path = return_path

        df = pd.read_excel(self.excel_path)
        # 兼容不同大小写/空格的列名
        df.columns = [str(c).strip() for c in df.columns]

        samples = []
        for _, row in df.iterrows():
            labels = row[self.LABEL_COLS].values.astype("float32")
            for col in self.use_eyes:
                img_name = str(row[col]).strip()
                img_path = os.path.join(self.img_dir, img_name)
                if not os.path.exists(img_path):
                    if self.skip_missing:
                        continue
                    else:
                        raise FileNotFoundError(img_path)
                item = (img_path, torch.tensor(labels, dtype=torch.float32))
                if self.return_path:
                    item = (img_path, torch.tensor(labels, dtype=torch.float32), img_path)
                samples.append(item)

        self.samples = samples
        self.classes = self.LABEL_COLS[:]
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if self.return_path:
            img_path, label, path = item
        else:
            img_path, label = item
        img = Image.open(img_path)
        if self.as_rgb and img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return (img, label, img_path) if self.return_path else (img, label)

def load_odir5k(root_dir, transform_train=None, transform_val=None,
                val_ratio=0.2, seed=42, patient_wise=True, id_col="ID"):
    """
    加载 ODIR-5K 并划分 train/val
    - patient_wise=True：按病人 ID 划分（避免左右眼泄漏）
    - patient_wise=False：按样本随机划分
    """
    import math
    df = pd.read_excel(os.path.join(root_dir, "data.xlsx"))
    df.columns = [str(c).strip() for c in df.columns]

    if patient_wise and id_col in df.columns:
        # 病人级划分：先分病人，再展开左右眼
        ids = df[id_col].astype(str).unique().tolist()
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(ids), generator=g).tolist()
        split = int(math.floor(len(ids) * (1 - val_ratio)))
        train_ids = set([ids[i] for i in perm[:split]])
        val_ids = set([ids[i] for i in perm[split:]])

        def _filter_by_ids(_ids):
            mask = df[id_col].astype(str).isin(_ids)
            sub = df.loc[mask].reset_index(drop=True)
            tmp_path = os.path.join(root_dir, "_temp_split.xlsx")
            sub.to_excel(tmp_path, index=False)  # 临时写一个 excel 供 Dataset 读取
            return tmp_path

        train_excel = _filter_by_ids(train_ids)
        val_excel = _filter_by_ids(val_ids)

        train_set = OdirDataset(root_dir, img_dir="Training Images", excel_file=os.path.basename(train_excel),
                                transform=transform_train)
        val_set = OdirDataset(root_dir, img_dir="Training Images", excel_file=os.path.basename(val_excel),
                              transform=transform_val)
        # 清理临时文件（可选：若训练中断也不影响）
        try:
            os.remove(train_excel)
            os.remove(val_excel)
        except Exception:
            pass
        return train_set, val_set

    else:
        # 简单样本随机划分
        full = OdirDataset(root_dir, img_dir="Training Images", transform=transform_train)
        val_len = int(len(full) * val_ratio)
        train_len = len(full) - val_len
        train_set, val_set = random_split(full, [train_len, val_len],
                                          generator=torch.Generator().manual_seed(seed))
        set_transform_for_subset(val_set, transform_val)
        return train_set, val_set

# ---------------------------
# 4)（可选）多标签类权重估计：用于 BCEWithLogitsLoss(pos_weight=...)
# ---------------------------

def multilabel_pos_weight(dataset_or_loader):
    """
    统计多标签数据的正样本频率并返回 pos_weight 张量： (C,)
    适用于 BCEWithLogitsLoss(pos_weight=...)
    """
    # 迭代 Dataset 更稳（避免 DataLoader shuffle）
    if isinstance(dataset_or_loader, Subset):
        ds = dataset_or_loader.dataset
    else:
        ds = dataset_or_loader

    counts = None
    for _, y in ds:
        y = y.float().view(-1)
        if counts is None:
            counts = torch.zeros_like(y)
        counts += y
    total = len(ds)
    pos = counts.clamp(min=1)                         # 防止除0
    neg = (total - counts).clamp(min=1)
    return (neg / pos)                                # 更少见的类别权重更大
