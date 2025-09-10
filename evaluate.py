import numpy as np
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from loss_evidential import EvidentialLoss, compute_uncertainty
from torch.cuda.amp import autocast
from MedMamba_AS import VSSM as medmamba  # 引用预训练模型
from args import get_args
from sklearn.model_selection import train_test_split
from dataset_class import *  # 自定义数据集类
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def calculate_metrics(y_true, y_pred, y_prob):
    # Precision (micro 平均)
    precision_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)

    # Recall (Sensitivity) (micro 平均)
    recall_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    # F1-score (micro 平均)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Overall Accuracy (OA)
    accuracy = (y_true == y_pred).mean()

    # Specificity for each class (in multi-class, it is defined as 1 - FPR for each class)
    cm = confusion_matrix(y_true, y_pred)
    specificity_per_class = []
    n_classes = cm.shape[0]

    print(len(y_true))
    print(cm.shape)
    print(cm.sum())

    for i in range(n_classes):
        # 计算每个类别的真负类（TN）和假正类（FP）

        TN = cm.sum() - cm.sum(axis=0)[i] - cm.sum(axis=1)[i] + cm[i, i] # True negatives for class i
        FP = cm.sum(axis=0)[i] - cm[i, i] # False positives for class i

        # print(TN,FP)

        # 确保 TN 和 FP 都是非负的
        TN = max(TN, 0)
        FP = max(FP, 0)

        # Handle scalar values of TN + FP
        if (TN + FP) != 0:
            specificity_per_class.append(TN / (TN + FP))
        else:
            specificity_per_class.append(float('nan'))  # If no negatives, set specificity to NaN

    specificity_macro = np.nanmean(specificity_per_class)  # Macro average specificity

    # 计算加权特异性时，确保权重的维度与类别数一致
    bincount_weights = np.bincount(y_true, minlength=n_classes)  # 确保类别数对齐
    specificity_weighted = np.average(specificity_per_class, weights=bincount_weights)  # 加权平均特异性

    # AUC (One-vs-Rest for multi-class)
    auc_per_class = None
    auc_macro = auc_micro = auc_weighted = None
    if y_prob is not None:
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))  # 将真实标签进行二值化，适用于 AUC
        try:
            auc_per_class = []
            for c in range(n_classes):
                auc_per_class.append(roc_auc_score(y_true_bin[:, c], y_prob[:, c]))

            auc_macro = np.nanmean(auc_per_class)  # AUC-macro
            auc_micro = roc_auc_score(y_true_bin, y_prob, average='micro')  # AUC-micro
            auc_weighted = roc_auc_score(y_true_bin, y_prob, average='weighted')  # AUC-weighted
        except ValueError:
            auc_per_class = [np.nan] * n_classes
            auc_macro = auc_micro = auc_weighted = np.nan

    # 打印各项指标
    # print(f"Precision (Micro): {precision_micro * 100:.2f}%")
    # print(f"Precision (Macro): {precision_macro * 100:.2f}%")
    print(f"Precision (Weighted): {precision_weighted * 100:.2f}%")
    # print(f"Sensitivity (Recall) (Micro): {recall_micro * 100:.2f}%")
    # print(f"Sensitivity (Recall) (Macro): {recall_macro * 100:.2f}%")
    print(f"Sensitivity (Recall) (Weighted): {recall_weighted * 100:.2f}%")
    # print(f"Specificity (Macro): {specificity_macro * 100:.2f}%")
    print(f"Specificity (Weighted): {specificity_weighted * 100:.2f}%")
    # print(f"F1-score (Micro): {f1_micro * 100:.2f}%")
    # print(f"F1-score (Macro): {f1_macro * 100:.2f}%")
    print(f"F1-score (Weighted): {f1_weighted * 100:.2f}%")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    # print(f"AUC (Macro): {auc_macro:.4f}")
    # print(f"AUC (Micro): {auc_micro:.4f}")
    print(f"AUC (Weighted): {auc_weighted:.4f}")

    # 返回各项指标字典
    return {
        "Precision Micro": precision_micro,
        "Precision Macro": precision_macro,
        "Precision Weighted": precision_weighted,
        "Recall Micro": recall_micro,
        "Recall Macro": recall_macro,
        "Recall Weighted": recall_weighted,
        "Specificity Macro": specificity_macro,
        "Specificity Weighted": specificity_weighted,
        "F1 Micro": f1_micro,
        "F1 Macro": f1_macro,
        "F1 Weighted": f1_weighted,
        "Accuracy": accuracy,
        "AUC Macro": auc_macro,
        "AUC Micro": auc_micro,
        "AUC Weighted": auc_weighted
    }

# 加载预训练模型
def load_pretrained_model(model, pretrained_weights_path, device='cuda'):
    checkpoint = torch.load(pretrained_weights_path, map_location=device)
    # print("Checkpoint keys:", checkpoint.keys())
    # print("--------------------------------------------------")
    # print("Current model keys:", model.state_dict().keys())
    model.load_state_dict(checkpoint, strict=True)  # 使用 strict=False 忽略不匹配的层
    model.to(device)
    model.eval()
    print(f"Loaded pretrained model from {pretrained_weights_path}")
    return model
# 评估部分
def main():
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    # 数据预处理
    data_transform = {
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    # 数据集加载（读取验证集）
    print(f"dataset: {'ours' if args.dataset is None else args.dataset}")
    if args.dataset is None:
        validate_dataset = datasets.ImageFolder(root="../dataset/val", transform=data_transform["val"])
    else:
        # 其他数据集的处理
        pass

    num_classes = len(validate_dataset.classes)
    val_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 创建模型实例
    net = medmamba(num_classes=num_classes, hyper_ad=args.hyper_ad, EDL=args.EDL,
                   reduction_ratio=args.reduction_ratio, had_feature_dim=args.had_feat_dim,
                   patch_size=args.patch_size, in_chans=args.in_chans, depths=args.depths,
                   dims=args.dims, proj_dim=args.proj_dim, p_drop=args.p_drop)

    # 加载预训练模型
    if args.pretrained_weights:
        net = load_pretrained_model(net, args.pretrained_weights, device=device)

    # 推理部分
    all_preds = []
    all_tgts = []
    all_probs = []

    # 进行评估
    with torch.no_grad():
        for imgs, tgts in val_loader:
            imgs = imgs.to(device)
            tgts = tgts.to(device)
            outputs = net(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_tgts.append(tgts.cpu().numpy())

    # 将所有结果合并
    probs_np = np.concatenate(all_probs)
    preds_np = np.concatenate(all_preds)
    tgts_np = np.concatenate(all_tgts)

    # 计算评估指标
    metrics = calculate_metrics(tgts_np, preds_np, probs_np)


if __name__ == '__main__':
    main()
