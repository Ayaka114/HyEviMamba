import os
import re
import matplotlib.pyplot as plt

# 自动搜索 ./logs 下的 .log 文件
log_files = [os.path.join("./logs", f) for f in os.listdir("./logs") if f.endswith(".log")]
if not log_files:
    raise FileNotFoundError("当前目录下未找到 .log 文件")
print(log_files)

for log_path in log_files:
    print(f"找到日志文件：{log_path}")

    train_losses, val_accuracies = [], []

    # 正则提取 loss 和 accuracy
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"train_loss:\s*([\d.]+)\s*val_accuracy:\s*([\d.]+)", line)
            if m:
                train_losses.append(float(m.group(1)))
                val_accuracies.append(float(m.group(2)))

    # 数据完整性与轮数检查
    n_train, n_val = len(train_losses), len(val_accuracies)
    if n_train == 0 or n_val == 0:
        print(f"跳过（无有效数据）：{log_path}")
        continue
    if n_train != n_val:
        print(f"警告：轮数不一致（train={n_train}, val={n_val}），以较小者为准")
        cut = min(n_train, n_val)
        train_losses, val_accuracies = train_losses[:cut], val_accuracies[:cut]

    # ➜ 新机制：少于 100 轮直接跳过不画
    if len(train_losses) < 100:
        print(f"跳过（未训练完成 <100 epochs）：{log_path}，仅 {len(train_losses)} 轮")
        continue

    epochs = range(1, len(train_losses) + 1)

    # 绘图
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "o-", label="Train Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training Loss over Epochs"); plt.grid(True); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, "s-", color="green", label="Validation Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Validation Accuracy over Epochs"); plt.grid(True)

    # 标注最高点
    max_acc = max(val_accuracies)
    max_epoch = 1 + val_accuracies.index(max_acc)
    plt.scatter(max_epoch, max_acc, color="red", s=50, zorder=5, label="Best")
    plt.text(max_epoch, max_acc, f"{max_acc:.3f}", ha="left", va="bottom", fontsize=9, color="red")
    plt.legend()

    plt.tight_layout()
    output_png = log_path.replace(".log", ".png")
    plt.savefig(output_png, dpi=300)
    print(f"图像已保存为 {output_png}")
    plt.show()
