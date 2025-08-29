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
from MedMamba import VSSM as medmamba # import model
from args import get_args

class NPZ2DDataset(Dataset):
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

    # npz_path = os.path.join("./datasets/octmnist", "octmnist_224.npz")
    npz_path = os.path.join("./datasets/RetinaMNIST", "retinamnist.npz")
    train_dataset = NPZ2DDataset(npz_path, split="train", transform=data_transform["train"], as_rgb=True)
    validate_dataset = NPZ2DDataset(npz_path, split="val", transform=data_transform["val"], as_rgb=True)

    num_classes = 5

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
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    model_name = args.save_name
    if args.hyper_ad:
        print("Using Hyper-Adaptive Mechanism with reduction_ratio: {}".format(args.reduction_ratio))
    net = medmamba(num_classes=num_classes, hyper_ad=args.hyper_ad, EDL=args.EDL, reduction_ratio=args.reduction_ratio)
    # net = medmamba(depths=[2, 2, 12, 2],dims=[128,256,512,1024],num_classes=num_classes,hyper_ad=args.hyper_ad) # medmamba_b

    net.to(device)
    if args.EDL:
        print("Using Evidential Loss")
        loss_function = EvidentialLoss(num_classes=num_classes)
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