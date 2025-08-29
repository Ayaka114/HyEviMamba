import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialLoss(nn.Module):
    def __init__(self, num_classes, kl_coef=1e-3,
                 adaptive=True, c=1.2, kl_min=1e-6, kl_max=2e-2, ema=0.9):
        super().__init__()
        self.num_classes = num_classes
        self.kl_coef = kl_coef

        # 记录 batch 级诊断
        self.last_nll = None
        self.last_kl = None

        # 自适应相关
        self.adaptive = adaptive
        self.c = c
        self.kl_min = kl_min
        self.kl_max = kl_max
        self.ema = ema
        self.nll_ma = None
        self.kl_ma = None

    def forward(self, evidence, target):
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()

        # NLL（正项）
        nll = torch.sum(one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)

        # KL
        kl_div = self._kl_divergence(alpha)

        # 记录 batch 均值（诊断）
        nll_mean = nll.mean().item()
        kl_mean  = kl_div.mean().item()
        self.last_nll = nll_mean
        self.last_kl  = kl_mean

        # —— 自适应更新 kl_coef（按 EMA 的 NLL/KL 比例）——
        if self.adaptive:
            # 初始化 EMA
            if self.nll_ma is None:
                self.nll_ma = nll_mean
                self.kl_ma  = max(kl_mean, 1e-8)

            # 更新 EMA
            self.nll_ma = self.ema * self.nll_ma + (1.0 - self.ema) * nll_mean
            self.kl_ma  = self.ema * self.kl_ma  + (1.0 - self.ema) * max(kl_mean, 1e-8)

            ratio = self.nll_ma / (self.kl_ma + 1e-8)
            λ_new = max(self.kl_min, min(self.c * ratio, self.kl_max))
            # 对 λ 也做一次平滑
            self.kl_coef = self.ema * self.kl_coef + (1.0 - self.ema) * float(λ_new)

        loss = nll + self.kl_coef * kl_div
        return loss.mean()

    def _kl_divergence(self, alpha):
        K = alpha.shape[1]
        S = torch.sum(alpha, dim=1, keepdim=True)
        kl = (
            torch.lgamma(S) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
            + torch.sum((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S)), dim=1, keepdim=True)
        )
        return kl.squeeze(1)

def compute_uncertainty(evidence, eps=1e-8):
    """
    计算预测熵、领域互信息和联合不确定性（用于 Evidential Learning）
    """
    # 稳定性处理：避免 alpha 过小造成 digamma 爆炸
    alpha = torch.clamp(evidence + 1, min=1e-3)  # 防止 alpha < 1e-3
    S = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / S

    # 预测熵（Predictive Uncertainty）
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)

    # 互信息（Domain Uncertainty）
    digamma_alpha = torch.digamma(alpha)
    digamma_S = torch.digamma(S)
    mutual_info = torch.sum(probs * (digamma_alpha - digamma_S), dim=1)

    # 确保 MI 不为负（数值误差导致）
    mutual_info = torch.clamp(mutual_info, min=0.0)

    total_uncertainty = entropy + mutual_info
    return entropy, mutual_info, total_uncertainty
