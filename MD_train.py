import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from loss_evidential import EvidentialLoss,compute_uncertainty
from torch.cuda.amp import autocast
from MedMamba_AS import VSSM as medmamba # import model
from args import get_args
from sklearn.model_selection import train_test_split

from dataset_class import *

def main():
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    print(f"dataset:{'ours' if args.dataset == None else args.dataset}")
    if args.dataset == None:
        train_dataset = datasets.ImageFolder(root="/home1/zhouhao/dataset/train",
                                             transform=data_transform["train"])
        flower_list = train_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in flower_list.items())
        # write dict into json file
        json_str = json.dumps(cla_dict, indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)
        validate_dataset = datasets.ImageFolder(root="/home1/zhouhao/dataset/val",
                                                transform=data_transform["val"])
    elif args.dataset == "octmnist":
        npz_path = os.path.join("./datasets/octmnist", "octmnist_224.npz")
        train_dataset = MedMnistDataset(npz_path, split="train", transform=data_transform["train"], as_rgb=True)
        validate_dataset = MedMnistDataset(npz_path, split="val", transform=data_transform["val"], as_rgb=True)

    elif args.dataset == "retinamnist":
        npz_path = os.path.join("./datasets/RetinaMNIST", "retinamnist.npz")
        train_dataset = MedMnistDataset(npz_path, split="train", transform=data_transform["train"], as_rgb=True)
        validate_dataset = MedMnistDataset(npz_path, split="val", transform=data_transform["val"], as_rgb=True)
    elif args.dataset == "odir-5k":
        root_dir = "./datasets/ODIR-5K"
        train_dataset, validate_dataset = load_odir5k(
            root_dir,
            transform_train=data_transform["train"],
            transform_val=data_transform["val"],
            val_ratio=0.2  # 20% 作为验证集
        )
    elif args.dataset == "messidor":
        mat_path = "/home1/zhouhao/MedMamba_ywj/datasets/Messidor‑2/messidor/messidor+.mat"
        train_dataset, validate_dataset = load_messidor2_as_image(
            mat_path,
            transform_train=data_transform["train"],
            transform_val=data_transform["val"],
            val_ratio=0.2
        )
    elif args.dataset == "JSIEC":

        root_dir = "./datasets/JSIEC Retinal39/1000images"

        # 先用一个 ImageFolder 构建索引与标签
        full_ds = datasets.ImageFolder(root=root_dir, transform=None)

        # 取出每个样本的类别索引（ImageFolder.samples: List[(path, class_idx)])
        targets = np.array([cls for _, cls in full_ds.samples], dtype=np.int64)

        # 分层划分：80% 训练 / 20% 验证
        train_idx, val_idx = train_test_split(
            np.arange(len(targets)),
            test_size=0.2,
            random_state=42,
            stratify=targets
        )

        # 分别建立带不同 transform 的两个底层数据集（顺序与 full_ds 一致）
        ds_train_full = datasets.ImageFolder(root=root_dir, transform=data_transform["train"])
        ds_val_full = datasets.ImageFolder(root=root_dir, transform=data_transform["val"])

        # 用相同的索引切子集
        from torch.utils.data import Subset
        train_dataset = Subset(ds_train_full, train_idx.tolist())
        validate_dataset = Subset(ds_val_full, val_idx.tolist())

        # 方便后续使用：给 Subset 挂上 classes 属性
        train_dataset.classes = full_ds.classes
        validate_dataset.classes = full_ds.classes


    num_classes = len(train_dataset.classes)
    train_num = len(train_dataset)
    print(train_num)
    # flower_list = train_dataset.class_to_idx
    # cla_dict = dict((val, key) for key, val in flower_list.items())
    # # write dict into json file
    # json_str = json.dumps(cla_dict, indent=4)
    # with open('class_indices.json', 'w') as json_file:
    #     json_file.write(json_str)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.{} total classes.".format(train_num,
                                                                           val_num,num_classes))

    model_name = args.save_name
    if args.hyper_ad:
        print(f"Using Hyper-Adaptive Mechanism with reduction_ratio: {args.reduction_ratio}; feature_dim: {args.had_feat_dim}")

    print(f"Model params: patch_size{args.patch_size}, in_chans({args.in_chans}), depths({args.depths}), dims({args.dims}")
    net = medmamba(num_classes=num_classes,hyper_ad=args.hyper_ad, EDL=args.EDL,
                   reduction_ratio=args.reduction_ratio,had_feature_dim=args.had_feat_dim,
                   patch_size=args.patch_size, in_chans=args.in_chans,depths=args.depths,dims=args.dims,
                   proj_dim=args.proj_dim,p_drop=args.p_drop)

    # net = medmamba(depths=[2, 2, 12, 2],dims=[128,256,512,1024],num_classes=num_classes,hyper_ad=args.hyper_ad) # medmamba_b

    net.to(device)
    if args.EDL:
        print("Using Evidential Loss")
        loss_function = EvidentialLoss(num_classes=num_classes,adaptive=args.adaptive,c=args.c,kl_coef=min(args.kl_coef, 1.0))
    else:
        loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    epochs = args.epochs
    best_acc = 0.0
    save_path = './models/{}Net.pth'.format(model_name)
    train_steps = len(train_loader)

    for epoch in range(epochs):
        if args.EDL:
            loss_function.kl_coef = min(args.kl_coef, 1.0)
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if args.EDL:
                train_bar.desc = f"Epoch[{epoch + 1}/{epochs}] loss:{loss:.2f} kl_coef:{loss_function.kl_coef:.6f} NLL: {loss_function.last_nll:.6f}, KL: {loss_function.last_kl:.6f}"
            else:
                train_bar.desc = f"Epoch[{epoch + 1}/{epochs}] loss:{loss:.2f}"
        with torch.no_grad():
            entropy, mutual_info, u_total = compute_uncertainty(outputs)
            print(f"[Epoch {epoch}] Evidence Mean: {outputs.mean().item():.2f}, "
                  f"Entropy: {entropy.mean():.2f}, MI: {mutual_info.mean():.2f}")
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                if args.EDL:
                    alpha = outputs + 1
                    prob = alpha / alpha.sum(dim=1, keepdim=True)
                    predict_y = torch.argmax(prob, dim=1)
                else:
                    predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()