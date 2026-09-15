"""
================================================================================
EA-RevIN (Entropy-Aware RevIN) — 移植自 EDformer/layers/RevIN.py
================================================================================
【迁移说明（TimeBridge 适配）】
  本文件原版位于 EDformer 工程，是在 ts-kim/RevIN 之上做的「熵感知动态 affine」增强。
  当前文件作为 TimeBridge 「并立目录 + 独立复现」实验线的归一化层使用：
    - 接口与 EDformer 版本完全一致：forward(x, mode='norm'|'denorm', mask=None)，输入形状 [B, L, N]；
    - 默认 revin_dyn_bound=0 时，行为与原始 RevIN 等价（用于公平对照与消融）；
    - 默认 affine=False 时，等价于纯 z-score（与 TimeBridge 原始内置归一化形态对齐）；
    - TimeBridge 原始模型走的是「(x-mean)/std + 末端 *std+mean」，等价于「affine=False, revin_dyn_bound=0」分支。

【EA-RevIN 创新点（保留 EDformer 已实现版本）】
  仅在 revin_dyn_bound > 0 且 affine=True 时启用，否则与原始 RevIN 数值路径对齐：
  1) 波动熵 Volatility Entropy：在窗口内将「各时间步绝对离差」归一化为概率分布，
     计算香农熵并除以 log(L) 得到 norm_entropy，刻画序列波动结构的复杂程度。
  2) EA-Router（轻量 MLP）：以 [norm_entropy, entropy_reliability] 为输入，预测对 affine
     的增量 delta_w / delta_b，经 tanh 与 dyn_bound 限制幅度，得到动态 dyn_w、dyn_b。
  3) 熵门控 entropy_gate：只有归一化熵高于阈值时才逐步放开动态调节，抑制「平稳段被乱拉」。
  4) 稳定性尺度 stability_scale：用 stdev 与中心量级之比（CV 思想）压低近平稳段的动态强度。
  5) denorm 阶段使用 norm 时缓存的 dyn_w、dyn_b，而不是仅用固定 affine_weight/bias，
     保证可逆性与训练一致性。

【相对 backup 的工程修正（与 EDformer 版本一致）】
  带 mask 时方差：对 centered 使用与均值一致的掩码数据，避免均值/方差不一致问题。

【完美热启动 Warm-start】
  entropy_router 末层零初始化 → tanh(0)=0 → multiplier=1.0, adder=0.0
  → 训练第一步与 vanilla RevIN 数值等价，便于公平对比与消融。
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np


class RevIN(nn.Module):
    """可逆实例归一化：vanilla RevIN 等价路径 + 可选 EA-RevIN（熵感知动态 affine）。"""

    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False,
                 revin_dyn_bound=0.0, entropy_gate=0.55):
        """
        :param num_features: the number of features or channels (即 enc_in / N)
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        :param subtract_last: if True, subtract last value instead of mean
        :param revin_dyn_bound: 动态 affine 增量相对幅度上界；0 表示关闭 EA 分支，与 vanilla RevIN 一致。
        :param entropy_gate: 归一化波动熵低于该阈值时减弱动态分支，减轻平稳序列上的扰动。
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.mask = None
        self.dyn_bound = revin_dyn_bound
        self.entropy_gate = entropy_gate

        self.dyn_w = None
        self.dyn_b = None

        if self.affine:
            self._init_params()

        if self.affine and self.dyn_bound > 0:
            self.entropy_router = nn.Sequential(
                nn.Linear(2, 16),
                nn.GELU(),
                nn.Linear(16, 2)
            )
            nn.init.constant_(self.entropy_router[-1].weight, 0.0)
            nn.init.constant_(self.entropy_router[-1].bias, 0.0)
            self.dynamic_scale = None

    def forward(self, x, mode: str, mask=None):
        if mode == 'norm':
            self._get_statistics(x, mask)
            x = self._normalize(x, mask)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, mask=None):
        self.mask = mask
        dim2reduce = tuple(range(1, x.ndim - 1))
        x_masked = x.masked_fill(mask, 0) if mask is not None else None

        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            if mask is None:
                self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            else:
                assert isinstance(mask, torch.Tensor)
                self.mean = (torch.sum(x_masked, dim=1) / torch.sum(~mask, dim=1)).unsqueeze(1).detach()
                self.mean = torch.nan_to_num(self.mean, nan=0.0, posinf=0.0, neginf=0.0)

        if mask is None:
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        else:
            assert isinstance(mask, torch.Tensor)
            centered = (x - self.last) if self.subtract_last else (x_masked - self.mean)
            centered = centered.masked_fill(mask, 0)
            self.stdev = (torch.sqrt(torch.sum(centered ** 2, dim=1) / torch.sum(~mask, dim=1) + self.eps)
                          .unsqueeze(1).detach())
            self.stdev = torch.nan_to_num(self.stdev, nan=0.0, posinf=None, neginf=None)

        if self.affine and self.dyn_bound > 0:
            L = x.shape[1]
            dev = torch.abs(x - (self.last if self.subtract_last else self.mean))
            if mask is not None:
                dev = dev.masked_fill(mask, 0)

            sum_dev = torch.sum(dev, dim=1, keepdim=True) + self.eps
            prob = dev / sum_dev

            entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1, keepdim=True)
            self.norm_entropy = (entropy / np.log(L)).detach()  # [B, 1, N]
            self.norm_entropy = torch.nan_to_num(self.norm_entropy, nan=1.0)

            peak_prob = prob.max(dim=1, keepdim=True).values
            uniform_prob = 1.0 / max(L, 1)
            self.entropy_reliability = ((peak_prob - uniform_prob) / (1.0 - uniform_prob + 1e-6)).detach()
            self.entropy_reliability = torch.nan_to_num(self.entropy_reliability, nan=0.0, posinf=0.0, neginf=0.0)
            self.entropy_reliability = self.entropy_reliability.clamp(0.0, 1.0)

    def _normalize(self, x, mask=None):
        self.mask = mask
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean

        x = x / self.stdev

        if mask is not None:
            x = x.masked_fill(mask, 0)

        if self.affine:
            if self.dyn_bound > 0:
                router_in = torch.stack([self.norm_entropy, self.entropy_reliability], dim=-1)
                meta_out = self.entropy_router(router_in)

                entropy_scale = torch.sigmoid((self.norm_entropy - self.entropy_gate) / 0.08)
                center_ref = self.last if self.subtract_last else self.mean
                cv = (self.stdev / (torch.abs(center_ref) + 1e-4)).clamp(min=0.0)
                stability_scale = (cv / (cv + 1.0)).clamp(0.0, 1.0)
                self.dynamic_scale = (entropy_scale * stability_scale * self.entropy_reliability).clamp(0.0, 1.0)

                eff_dyn_bound = self.dyn_bound * self.dynamic_scale
                delta_w = torch.tanh(meta_out[..., 0]) * eff_dyn_bound
                delta_b = torch.tanh(meta_out[..., 1]) * eff_dyn_bound

                weight_multiplier = 1.0 + delta_w
                bias_adder = delta_b

                self.dyn_w = self.affine_weight * weight_multiplier
                self.dyn_b = self.affine_bias + bias_adder
            else:
                self.dyn_w = self.affine_weight
                self.dyn_b = self.affine_bias
                self.dynamic_scale = None

            x = x * self.dyn_w
            x = x + self.dyn_b

        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.dyn_b
            x = x / (self.dyn_w + self.eps * self.eps)

        x = x * self.stdev

        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean

        return x
