import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False, attn_map=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.attn_map = attn_map
        self.alpha = nn.Parameter(torch.rand(1))
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, long_term=True):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn_map = torch.softmax(scale * scores, dim=-1)
        A = self.dropout(attn_map)
        if self.attn_map is True:
            heat_map = attn_map[:, ...].max(1)[0]
            heat_map = torch.clamp_max(heat_map, 0.15)
            # heat_map = torch.softmax(heat_map, -1)
            for b in range(heat_map.shape[0]):
                # for c in range(heat_map.shape[1]):
                h_map = heat_map[b, ...].detach().cpu().numpy()
                # plt.savefig(heat_map, f'{b} sample {c} channel')
                plt.figure(figsize=(10, 8), dpi=200)
                plt.imshow(h_map, cmap='Reds', interpolation='nearest')
                plt.colorbar()

                # 设置X轴和Y轴的标签为黑体文字
                plt.rcParams['font.family'] = 'serif'
                plt.rcParams['font.serif'] = ['Times New Roman']
                plt.xlabel('Key Channel', fontsize=14)
                plt.ylabel('Query Channel', fontsize=14)

                # 设置标题
                # plt.title('Long-Term Correlations', fontdict={'weight': 'bold'}, fontsize=16, color='green')

                plt.tight_layout()
                plt.savefig(f'./stable map/{b}_sample.png')
                # plt.savefig(f'./non_stable map/{b}_sample.png')
                plt.close()
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        if self.inner_attention is None:
            return self.out_projection(self.value_projection(values)), None
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class TSMixer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(TSMixer, self).__init__()

        self.attention = attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, q, k, v, res=False, attn=None):
        B, L, _ = q.shape
        _, S, _ = k.shape
        H = self.n_heads

        q = self.q_proj(q).reshape(B, L, H, -1)
        k = self.k_proj(k).reshape(B, S, H, -1)
        v = self.v_proj(v).reshape(B, S, H, -1)

        out, attn = self.attention(
            q, k, v,
            res=res, attn=attn
        )
        out = out.view(B, L, -1)

        return self.out(out), attn


class ResAttention(nn.Module):
    def __init__(self, attention_dropout=0.1, scale=None, attn_map=False, nst=False):
        super(ResAttention, self).__init__()

        self.nst = nst
        self.scale = scale
        self.attn_map = attn_map
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, res=False, attn=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn_map = torch.softmax(scale * scores, dim=-1)
        if self.attn_map is True:
            heat_map = attn_map.reshape(32, -1, H, L, S)
            for b in range(heat_map.shape[0]):
                for c in range(heat_map.shape[1]):
                    h_map = heat_map[b, c, 0, ...].detach().cpu().numpy()
                    # plt.savefig(heat_map, f'{b} sample {c} channel')

                    plt.figure(figsize=(10, 8), dpi=200)
                    plt.imshow(h_map, cmap='Reds', interpolation='nearest')
                    plt.colorbar()

                    # 设置X轴和Y轴的标签为黑体文字
                    plt.rcParams['font.family'] = 'serif'
                    plt.rcParams['font.serif'] = ['Times New Roman']
                    plt.xlabel('Key Time Patch', fontsize=14)
                    plt.ylabel('Query Time Patch', fontsize=14)
                    plt.tight_layout()
                    if self.nst is True:
                        plt.savefig(f'./time map/{b}_sample_{c}_channel.png')
                    else:
                        plt.savefig(f'./stable time map/{b}_sample_{c}_channel.png')
                    # 关闭当前图形窗口
                    plt.close()
        A = self.dropout(attn_map)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous(), A


# ==============================================================================
# ED-SRA — Entropy-Driven Sparse Residual Attention（移植自 EDformer）
# ==============================================================================
# 【相对 EDformer 版本的关键修正：解耦 SF 依赖】
#   EDformer 中 ED-SRA 与 SF (weight_mat 先验) 强绑定，要求 SF_mode=True 同步开启。
#   TimeBridge 没有 SF 路径，本文件中的 _ED_SRA_block 作为独立函数，仅依赖 softmax(A)，
#   可独立开关，不需要任何外部空间先验。
#
# 【设计动机】
#   在 softmax 概率 A 之上追加一层信息论启发的重分配：
#     1) 自适应阈值带：thre_w/thre_b 把 A 映射为标量阈值 t；γ_min/γ_max 双锚点（logit 参数化）
#        生成相对带 [t_min, ·]，单边稀疏门控只抑制低权重尾部，不压制 spike。
#     2) 归一化熵 + 行峰度：双特征 router 产生 dyn_steepness（门控陡峭度）与 dyn_alpha（残差权重）。
#     3) 稳定性门控：行内方差小（注意力均匀）时压低 effective_alpha，避免破坏已有结构。
#     4) 残差混合：normalize(A * gate_mask) 与 A 凸组合 → 重新归一化。
#     5) N 阈值自适应：通道数 S 大于 ed_sra_N_threshold 时对注入强度做指数衰减
#        n_scale = exp(-max(0, S - N_thr) / N_thr)，大 N 数据集（Traffic/ECL）近乎自动关闭。
#
# 【完美热启动 Warm-start】
#   router_steepness/alpha 末层零初始化 + bias 设计使 dyn_alpha=sigmoid(-1.386)≈0.2
#   再乘以 (1-stability_score)*reliability ≤ 1 + alpha_cap≤γ_max < 1，
#   且 base_alpha=0.03 极小，使训练第 0 步注入近乎为 0。
#
# 【接入 TimeBridge 时的注意事项】
#   - TimeBridge 的 attention IO：input [B,L,H,E]，softmax 在 bhls 的 s 维（与 EDformer 一致）。
#   - ResAttention（用于 IntAttention/PatchSampling）作用在 patch 维 n（小，不开启 N 阈值衰减）。
#   - FullAttention（用于 CointAttention）作用在通道维 c（最高 862，启用 N 阈值衰减）。
#   - dropout 应在 ED-SRA 之后施加，以避免破坏熵估计可靠性。
# ==============================================================================


class _EDSRAModule(nn.Module):
    """独立的 ED-SRA 模块：仅在 softmax 后的概率分布 A 上做熵驱动稀疏残差重分配。

    输入：
        A       : [B, H, L, S]  softmax 后的注意力概率（最后一维归一化）
        S_runtime: int         实际通道/位置数 S，用于 N 阈值自适应衰减
    输出：
        A_final : [B, H, L, S]  重分配后的注意力概率（仍然归一化）
    """

    def __init__(self, init_gamma_min=0.2, init_gamma_max=0.95,
                 ed_sra_N_threshold=300, base_alpha=0.03,
                 sigmoid_alpha=8.0, threshold_scale=1.0,
                 use_residual_mix=True, use_threshold_gate=True,
                 use_stability=True, use_reliability=True, use_alpha_cap=True,
                 enable_N_decay=True):
        super().__init__()
        self.thre_w = nn.Parameter(torch.tensor(1.0))
        self.thre_b = nn.Parameter(torch.tensor(0.0))
        self.logit_gamma_min = nn.Parameter(torch.logit(torch.tensor(float(init_gamma_min))))
        self.logit_gamma_max = nn.Parameter(torch.logit(torch.tensor(float(init_gamma_max))))
        self.router_steepness = nn.Linear(2, 1)
        self.router_alpha = nn.Linear(2, 1)
        nn.init.constant_(self.router_steepness.weight, 0.0)
        nn.init.constant_(self.router_steepness.bias, 0.0)
        nn.init.constant_(self.router_alpha.weight, 0.0)
        nn.init.constant_(self.router_alpha.bias, -1.386)
        self.base_alpha = base_alpha
        self.sigmoid_alpha = float(sigmoid_alpha)
        self.threshold_scale = float(threshold_scale)
        self.use_residual_mix = bool(use_residual_mix)
        self.use_threshold_gate = bool(use_threshold_gate)
        self.use_stability = bool(use_stability)
        self.use_reliability = bool(use_reliability)
        self.use_alpha_cap = bool(use_alpha_cap)
        self.ed_sra_N_threshold = float(ed_sra_N_threshold)
        self.enable_N_decay = bool(enable_N_decay)
        # Disabled during normal training/inference. Visualization tools can opt
        # in to capture the exact Softmax -> gate/redistribute -> residual-mix
        # states without changing this module's public return value.
        self.capture_intermediates = False
        self.last_intermediates = None

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        S = A.shape[-1]

        t = torch.sigmoid(A * self.thre_w + self.thre_b)
        entropy = -torch.sum(A * torch.log(A + 1e-8), dim=-1, keepdim=True)
        norm_entropy = entropy / np.log(max(S, 2))

        row_max = A.max(dim=-1, keepdim=True).values
        base_uniform = 1.0 / max(S, 1)
        peakness = ((row_max - base_uniform) / (1.0 - base_uniform + 1e-6)).clamp(0.0, 1.0)
        router_feat = torch.cat([norm_entropy, peakness], dim=-1)
        dyn_steepness = torch.exp(self.router_steepness(router_feat)) * max(self.sigmoid_alpha, 1e-6)
        dyn_alpha = torch.sigmoid(self.router_alpha(router_feat))

        attn_var = A.var(dim=-1, keepdim=True, unbiased=False)
        vol_score = attn_var / (attn_var + 1e-4)
        stability_score = (1.0 - vol_score).clamp(0.0, 1.0)

        if self.use_threshold_gate:
            gamma_min = torch.sigmoid(self.logit_gamma_min)
            t_min = gamma_min * self.threshold_scale * t
            gate_mask = torch.sigmoid(dyn_steepness * (A - t_min)).clamp(0.0, 1.0)
        else:
            gate_mask = torch.ones_like(A)

        attn_gated = A * gate_mask
        attn_gated = attn_gated / attn_gated.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        if self.use_reliability:
            reliability = ((1.0 - norm_entropy).clamp(0.0, 1.0) * peakness).clamp(0.0, 1.0)
        else:
            reliability = torch.ones_like(norm_entropy)
        if self.use_stability:
            stability_term = (1.0 - stability_score)
        else:
            stability_term = torch.ones_like(stability_score)

        effective_alpha = self.base_alpha + (dyn_alpha - self.base_alpha) * stability_term * reliability
        if self.use_alpha_cap:
            alpha_cap = torch.sigmoid(self.logit_gamma_max)
            effective_alpha = effective_alpha * alpha_cap

        if self.enable_N_decay:
            n_scale = float(np.exp(-max(0.0, S - self.ed_sra_N_threshold) / max(self.ed_sra_N_threshold, 1.0)))
            effective_alpha = (effective_alpha * n_scale).clamp(0.0, 1.0)
        else:
            effective_alpha = effective_alpha.clamp(0.0, 1.0)

        if self.use_residual_mix:
            A_final = (1.0 - effective_alpha) * A + effective_alpha * attn_gated
        else:
            A_final = attn_gated
        A_final = A_final / A_final.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        if self.capture_intermediates:
            self.last_intermediates = {
                "initial": A.detach().cpu(),
                "gate_mask": gate_mask.detach().cpu(),
                "sparse": attn_gated.detach().cpu(),
                "residual": A_final.detach().cpu(),
                "effective_alpha": effective_alpha.detach().cpu(),
                "normalized_entropy": norm_entropy.detach().cpu(),
                "peakness": peakness.detach().cpu(),
            }
        return A_final


class FullAttention_EDSRA(nn.Module):
    """FullAttention 的 ED-SRA 增强版：在 softmax 后插入熵驱动稀疏残差重分配。

    与原 FullAttention 相比，IO 完全一致；启用 N 阈值衰减（用于通道维 attention，S 可能很大）。
    """

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False, attn_map=False,
                 init_gamma_min=0.2, init_gamma_max=0.95, ed_sra_N_threshold=300,
                 base_alpha=0.03, sigmoid_alpha=8.0, threshold_scale=1.0,
                 use_residual_mix=True, use_threshold_gate=True,
                 use_stability=True, use_reliability=True, use_alpha_cap=True,
                 use_n_decay=True):
        super().__init__()
        self.scale = scale
        self.attn_map = attn_map
        self.alpha = nn.Parameter(torch.rand(1))
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        self.ed_sra = _EDSRAModule(
            init_gamma_min=init_gamma_min, init_gamma_max=init_gamma_max,
            ed_sra_N_threshold=ed_sra_N_threshold, base_alpha=base_alpha,
            sigmoid_alpha=sigmoid_alpha, threshold_scale=threshold_scale,
            use_residual_mix=use_residual_mix, use_threshold_gate=use_threshold_gate,
            use_stability=use_stability, use_reliability=use_reliability, use_alpha_cap=use_alpha_cap,
            enable_N_decay=use_n_decay,
        )

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, long_term=True):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn_map = torch.softmax(scale * scores, dim=-1)
        attn_map = self.ed_sra(attn_map)
        A = self.dropout(attn_map)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ResAttention_EDSRA(nn.Module):
    """ResAttention 的 ED-SRA 增强版：在 TimeBridge 中同时被用于
       - IntAttention/PatchSampling 内的 patch 维（n≈4~30，小 S）
       - CointAttention 内的通道维 c（7~862，可能极大）
    因此 N 阈值衰减默认启用：S ≤ N_thr 时 n_scale=1.0 不影响；S 远大于 N_thr 时自动近乎关闭。
    """

    def __init__(self, attention_dropout=0.1, scale=None, attn_map=False, nst=False,
                 init_gamma_min=0.2, init_gamma_max=0.95, ed_sra_N_threshold=300,
                 base_alpha=0.03, sigmoid_alpha=8.0, threshold_scale=1.0,
                 use_residual_mix=True, use_threshold_gate=True,
                 use_stability=True, use_reliability=True, use_alpha_cap=True,
                 use_n_decay=True):
        super().__init__()
        self.nst = nst
        self.scale = scale
        self.attn_map = attn_map
        self.dropout = nn.Dropout(attention_dropout)

        self.ed_sra = _EDSRAModule(
            init_gamma_min=init_gamma_min, init_gamma_max=init_gamma_max,
            ed_sra_N_threshold=ed_sra_N_threshold, base_alpha=base_alpha,
            sigmoid_alpha=sigmoid_alpha, threshold_scale=threshold_scale,
            use_residual_mix=use_residual_mix, use_threshold_gate=use_threshold_gate,
            use_stability=use_stability, use_reliability=use_reliability, use_alpha_cap=use_alpha_cap,
            enable_N_decay=use_n_decay,
        )

    def forward(self, queries, keys, values, res=False, attn=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn_map = torch.softmax(scale * scores, dim=-1)
        attn_map = self.ed_sra(attn_map)
        A = self.dropout(attn_map)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous(), A
