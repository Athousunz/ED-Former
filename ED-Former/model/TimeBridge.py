import torch
import torch.nn as nn
from layers.Embed import PatchEmbed
from layers.SelfAttention_Family import (
    TSMixer, ResAttention, ResAttention_EDSRA,
)
from layers.Transformer_EncDec import TSEncoder, IntAttention, PatchSampling, CointAttention
from layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.revin = configs.revin  # long-term with temporal

        self.c_in = configs.enc_in
        self.period = configs.period
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_p = self.seq_len // self.period
        if configs.num_p is None:
            configs.num_p = self.num_p

        # === EA-RevIN 接入（最小侵入） ===
        # 仅在 --use_ea_revin 时构建 RevIN 层；否则保持原 TimeBridge 内置 z-score 归一化路径不变。
        self.use_ea_revin = bool(getattr(configs, 'use_ea_revin', False))
        if self.use_ea_revin:
            self.revin_layer = RevIN(
                num_features=self.c_in,
                eps=1e-5,
                affine=bool(getattr(configs, 'revin_affine', False)),
                subtract_last=bool(getattr(configs, 'revin_subtract_last', False)),
                revin_dyn_bound=float(getattr(configs, 'revin_dyn_bound', 0.0)),
                entropy_gate=float(getattr(configs, 'revin_entropy_gate', 0.55)),
            )

        # === ED-SRA 接入（最小侵入） ===
        # 仅在 --use_ed_sra 时把所有 ResAttention 替换为 ResAttention_EDSRA；
        # 默认 False 时保持 TimeBridge 原始注意力路径，且 ResAttention_EDSRA 类不参与构建。
        self.use_ed_sra = bool(getattr(configs, 'use_ed_sra', False))

        self.embedding = PatchEmbed(configs, num_p=self.num_p)

        layers = self.layers_init(configs)
        self.encoder = TSEncoder(layers)

        out_p = self.num_p if configs.pd_layers == 0 else configs.num_p
        self.decoder = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(out_p * configs.d_model, configs.pred_len, bias=False)
        )

    def _make_inner_attention(self, configs):
        """根据 use_ed_sra 开关构造内层 attention：默认 ResAttention，启用时 ResAttention_EDSRA。"""
        if self.use_ed_sra:
            return ResAttention_EDSRA(
                attention_dropout=configs.attn_dropout,
                init_gamma_min=float(getattr(configs, 'ed_sra_init_gamma_min', 0.2)),
                init_gamma_max=float(getattr(configs, 'ed_sra_init_gamma_max', 0.95)),
                ed_sra_N_threshold=int(getattr(configs, 'ed_sra_N_threshold', 300)),
                base_alpha=float(getattr(configs, 'ed_sra_base_alpha', 0.03)),
                sigmoid_alpha=float(getattr(configs, 'ed_sra_sigmoid_alpha', 8.0)),
                threshold_scale=float(getattr(configs, 'ed_sra_threshold_scale', 1.0)),
                use_residual_mix=bool(getattr(configs, 'ed_sra_use_residual_mix', True)),
                use_threshold_gate=bool(getattr(configs, 'ed_sra_use_threshold_gate', True)),
                use_stability=bool(getattr(configs, 'ed_sra_use_stability', True)),
                use_reliability=bool(getattr(configs, 'ed_sra_use_reliability', True)),
                use_alpha_cap=bool(getattr(configs, 'ed_sra_use_alpha_cap', True)),
                use_n_decay=bool(getattr(configs, 'ed_sra_use_n_decay', True)),
            )
        return ResAttention(attention_dropout=configs.attn_dropout)

    def layers_init(self, configs):
        integrated_attention = [IntAttention(
            TSMixer(self._make_inner_attention(configs), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, dropout=configs.dropout, stable_len=configs.stable_len,
            activation=configs.activation, stable=True, enc_in=self.c_in
        ) for i in range(configs.ia_layers)]

        patch_sampling = [PatchSampling(
            TSMixer(self._make_inner_attention(configs), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, stable=False, stable_len=configs.stable_len,
            in_p=self.num_p if i == 0 else configs.num_p, out_p=configs.num_p,
            dropout=configs.dropout, activation=configs.activation
        ) for i in range(configs.pd_layers)]

        cointegrated_attention = [CointAttention(
            TSMixer(self._make_inner_attention(configs),
                    configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, dropout=configs.dropout,
            activation=configs.activation, stable=False, enc_in=self.c_in, stable_len=configs.stable_len,
        ) for i in range(configs.ca_layers)]

        return [*integrated_attention, *patch_sampling, *cointegrated_attention]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if x_mark_enc is None:
            x_mark_enc = torch.zeros((*x_enc.shape[:-1], 4), device=x_enc.device)

        if self.use_ea_revin:
            # EA-RevIN 路径：可逆实例归一化 + 可选熵感知动态 affine
            x_enc = self.revin_layer(x_enc, mode='norm')
            x_enc_emb = self.embedding(x_enc, x_mark_enc)
            enc_out = self.encoder(x_enc_emb)[0][:, :self.c_in, ...]
            dec_out = self.decoder(enc_out).transpose(-1, -2)
            dec_out = self.revin_layer(dec_out, mode='denorm')
            return dec_out
        else:
            # 原始 TimeBridge 内置 z-score 归一化（保留以确保零行为变化）
            mean, std = (x_enc.mean(1, keepdim=True).detach(),
                         x_enc.std(1, keepdim=True).detach())
            x_enc = (x_enc - mean) / (std + 1e-5)

            x_enc = self.embedding(x_enc, x_mark_enc)
            enc_out = self.encoder(x_enc)[0][:, :self.c_in, ...]
            dec_out = self.decoder(enc_out).transpose(-1, -2)

            return dec_out * std + mean

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
