"""
Paper-style visualizations for TimeBridge (Fig.7 / Fig.8 / Fig.9 style).

Figure types:
  1) cointegration  : multi-variate line plot + zoom inset (Fig.7)
  2) intra_attn     : Query/Key Time Patch heatmaps, Stationary vs Non-stationary (Fig.8)
  3) inter_attn     : Query/Key Channel heatmaps, Stationary vs Non-stationary (Fig.9)

Examples:
  # Fig.7-style cointegration plot on Solar
  python -u experiments/visualize_paper_figs.py --mode cointegration \
    --root_path ./dataset/Solar/ --data_path solar_AL.txt --data Solar \
    --channels 0,1,2,10,20,30 --out_dir ./_logs/vis_paper

  # Fig.8/9-style attention maps (needs a trained checkpoint setting folder)
  python -u experiments/visualize_paper_figs.py --mode attention \
    --checkpoint ./checkpoints/<setting>/checkpoint.pth \
    --root_path ./dataset/Solar/ --data_path solar_AL.txt --data Solar \
    --enc_in 137 --period 48 --num_p 12 --ia_layers 1 --pd_layers 1 --ca_layers 1 \
    --seq_len 720 --pred_len 336 --d_model 128 --d_ff 128 \
    --sample_idx 0 --out_dir ./_logs/vis_paper
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import ConnectionPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_factory import data_provider
from model.TimeBridge import Model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TimeBridge paper-style visualizations.")
    p.add_argument("--mode", type=str, default="all",
                   choices=["cointegration", "attention", "all"],
                   help="Which figure family to generate.")
    p.add_argument("--out_dir", type=str, default="./_logs/vis_paper")
    p.add_argument("--dpi", type=int, default=200)

    # data
    p.add_argument("--root_path", type=str, default="./dataset/Solar/")
    p.add_argument("--data_path", type=str, default="solar_AL.txt")
    p.add_argument("--data", type=str, default="Solar")
    p.add_argument("--features", type=str, default="M")
    p.add_argument("--target", type=str, default="OT")
    p.add_argument("--freq", type=str, default="h")
    p.add_argument("--embed", type=str, default="timeF")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=1)

    # cointegration plot
    p.add_argument("--channels", type=str, default="0,1,2,5,10,20",
                   help="Comma-separated channel indices for cointegration plot.")
    p.add_argument("--start", type=int, default=0, help="Start index on raw series.")
    p.add_argument("--length", type=int, default=2000, help="Number of points to plot.")
    p.add_argument("--zoom_start", type=float, default=0.72,
                   help="Relative start of zoom window in [0,1].")
    p.add_argument("--zoom_len", type=int, default=80, help="Zoom window length.")

    # model / attention
    p.add_argument("--checkpoint", type=str, default="",
                   help="Path to checkpoint.pth. If empty, use randomly initialized model.")
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=720)
    p.add_argument("--label_len", type=int, default=48)
    p.add_argument("--pred_len", type=int, default=336)
    p.add_argument("--enc_in", type=int, default=137)
    p.add_argument("--period", type=int, default=48)
    p.add_argument("--num_p", type=int, default=12)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--d_ff", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--ia_layers", type=int, default=1)
    p.add_argument("--pd_layers", type=int, default=1)
    p.add_argument("--ca_layers", type=int, default=1)
    p.add_argument("--stable_len", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--attn_dropout", type=float, default=0.15)
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--revin", action="store_true", default=True)
    p.add_argument("--use_ea_revin", action="store_true", default=False)
    p.add_argument("--use_ed_sra", action="store_true", default=False)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--max_channels_heatmap", type=int, default=64,
                   help="If C is large, subsample this many channels for inter-variate heatmap.")
    return p


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def choose_device(flag: str) -> torch.device:
    if flag == "cuda":
        return torch.device("cuda")
    if flag == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_channels(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


# ---------------------------------------------------------------------------
# Fig.7 style: cointegration + short-term zoom
# ---------------------------------------------------------------------------

def load_raw_matrix(args: argparse.Namespace) -> np.ndarray:
    """Return raw multivariate series as [T, C] without requiring a trained model."""
    path = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    if args.data == "Solar" or path.endswith(".txt"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append([float(x) for x in line.split(",")])
        return np.asarray(rows, dtype=np.float64)

    import pandas as pd
    df = pd.read_csv(path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    if args.features == "S" and args.target in df.columns:
        return df[[args.target]].values.astype(np.float64)
    return df.values.astype(np.float64)


def plot_cointegration(args: argparse.Namespace) -> str:
    data = load_raw_matrix(args)  # [T, C]
    t0 = max(0, args.start)
    t1 = min(data.shape[0], t0 + args.length)
    series = data[t0:t1]
    chs = parse_channels(args.channels)
    chs = [c for c in chs if 0 <= c < series.shape[1]]
    if not chs:
        raise ValueError("No valid channels selected for cointegration plot.")

    # group colors similar to paper (banks/NBFIs/retails style)
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]
    x = np.arange(series.shape[0])

    normalized = []
    fig, (ax, ax_zoom) = plt.subplots(
        1, 2,
        figsize=(12, 4.2),
        gridspec_kw={"width_ratios": [3.6, 1.25], "wspace": 0.12},
    )
    for i, c in enumerate(chs):
        y = series[:, c]
        # light normalize for visual comparison across channels
        y = (y - y.mean()) / (y.std() + 1e-5)
        normalized.append(y)
        ax.plot(x, y, color=palette[i % len(palette)], linewidth=1.1, alpha=0.9, label=f"ch{c}")

    # zoom window
    z0 = int(np.clip(args.zoom_start, 0.0, 0.95) * (len(x) - 1))
    z1 = min(len(x), z0 + max(10, args.zoom_len))
    zoom_values = np.concatenate([y[z0:z1] for y in normalized])
    padding = max(0.1, 0.08 * np.ptp(zoom_values))
    ymin, ymax = zoom_values.min() - padding, zoom_values.max() + padding
    ax.add_patch(Rectangle(
        (z0, ymin), z1 - z0, ymax - ymin,
        fill=False, edgecolor="black", linewidth=1.2,
    ))

    for i, y in enumerate(normalized):
        ax_zoom.plot(x[z0:z1], y[z0:z1], color=palette[i % len(palette)], linewidth=1.0)
    ax_zoom.set_xlim(z0, z1 - 1)
    ax_zoom.set_ylim(ymin, ymax)
    ax_zoom.set_title("Short-term detail", fontsize=10)
    ax_zoom.yaxis.tick_right()
    ax_zoom.tick_params(axis="y", labelleft=False, labelright=True)
    ax_zoom.tick_params(labelsize=8)

    for y_main, y_zoom in ((ymax, ymax), (ymin, ymin)):
        fig.add_artist(ConnectionPatch(
            xyA=(z1, y_main), coordsA=ax.transData,
            xyB=(z0, y_zoom), coordsB=ax_zoom.transData,
            color="0.35", linewidth=0.9,
        ))

    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized value")
    ax.set_title(f"Long-term cointegration & short-term fluctuations ({args.data})")
    ax.legend(ncol=min(6, len(chs)), fontsize=8, loc="upper left")
    fig.align_xlabels()

    ensure_dir(args.out_dir)
    out = os.path.join(args.out_dir, f"fig7_cointegration_{args.data}.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")
    return out


# ---------------------------------------------------------------------------
# Attention capture helpers (Fig.8 / Fig.9)
# ---------------------------------------------------------------------------

class AttentionCatcher:
    """Capture softmax attention maps from ResAttention / ResAttention_EDSRA."""

    def __init__(self):
        self.maps: List[torch.Tensor] = []
        self.hooks = []

    def clear(self):
        self.maps = []

    def _hook(self, module, inputs, output):
        # output is (V, A); A shape [B, H, L, S]
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            self.maps.append(output[1].detach().cpu())

    def register(self, model: torch.nn.Module):
        from layers.SelfAttention_Family import ResAttention, ResAttention_EDSRA
        for m in model.modules():
            if isinstance(m, (ResAttention, ResAttention_EDSRA)):
                self.hooks.append(m.register_forward_hook(self._hook))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def make_args_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        revin=True,
        enc_in=args.enc_in,
        period=args.period,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_p=args.num_p,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        attn_dropout=args.attn_dropout,
        dropout=args.dropout,
        ia_layers=args.ia_layers,
        pd_layers=args.pd_layers,
        ca_layers=args.ca_layers,
        stable_len=args.stable_len,
        activation=args.activation,
        use_ea_revin=args.use_ea_revin,
        revin_affine=False,
        revin_subtract_last=False,
        revin_dyn_bound=0.0,
        revin_entropy_gate=0.55,
        use_ed_sra=args.use_ed_sra,
        ed_sra_init_gamma_min=0.2,
        ed_sra_init_gamma_max=0.95,
        ed_sra_N_threshold=300,
        ed_sra_base_alpha=0.03,
        ed_sra_sigmoid_alpha=8.0,
        ed_sra_threshold_scale=1.0,
        ed_sra_use_residual_mix=True,
        ed_sra_use_threshold_gate=True,
        ed_sra_use_stability=True,
        ed_sra_use_reliability=True,
        ed_sra_use_alpha_cap=True,
        ed_sra_use_n_decay=True,
        # data_provider fields
        data=args.data,
        root_path=args.root_path,
        data_path=args.data_path,
        features=args.features,
        target=args.target,
        freq=args.freq,
        embed=args.embed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        label_len=args.label_len,
        seasonal_patterns="Monthly",
    )


def set_intattention_stable(model: torch.nn.Module, stable: bool) -> None:
    from layers.Transformer_EncDec import IntAttention
    for m in model.modules():
        if isinstance(m, IntAttention):
            m.stable = stable


def set_coint_full_attn(model: torch.nn.Module, full: bool = True) -> None:
    """Force CointAttention to use full channel attention for inter-variate heatmaps."""
    from layers.Transformer_EncDec import CointAttention
    for m in model.modules():
        if isinstance(m, CointAttention):
            m.axial_func = (not full)


def reduce_attn(attn: torch.Tensor) -> np.ndarray:
    """[B,H,L,S] -> [L,S] by mean over batch & heads."""
    if attn.ndim != 4:
        raise ValueError(f"Expected attn [B,H,L,S], got {tuple(attn.shape)}")
    return attn.mean(dim=(0, 1)).numpy()


def pick_intra_and_inter(maps: List[torch.Tensor], enc_in: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Heuristic:
      - intra (time-patch): L==S and L is small/moderate (patch count)
      - inter (channel): L==S and L close to enc_in (or padded square near enc_in)
    """
    intra, inter = None, None
    for a in maps:
        mat = reduce_attn(a)
        l, s = mat.shape
        if l != s:
            continue
        # channel attention usually larger / near enc_in
        if abs(l - enc_in) <= max(8, enc_in // 5) or l >= max(32, enc_in // 2):
            if inter is None or abs(l - enc_in) < abs(inter.shape[0] - enc_in):
                inter = mat
        else:
            if intra is None:
                intra = mat
            # prefer smaller patch maps as intra
            elif l < intra.shape[0]:
                intra = mat
    # fallback: first square as intra, last square as inter
    squares = [reduce_attn(a) for a in maps if a.shape[-1] == a.shape[-2]]
    if intra is None and squares:
        intra = min(squares, key=lambda m: m.shape[0])
    if inter is None and squares:
        inter = max(squares, key=lambda m: m.shape[0])
    return intra, inter


def subsample_square(mat: np.ndarray, max_n: int) -> np.ndarray:
    n = mat.shape[0]
    if n <= max_n:
        return mat
    idx = np.linspace(0, n - 1, max_n).astype(int)
    return mat[np.ix_(idx, idx)]


def plot_attn_pair(mats: List[Tuple[str, np.ndarray]], title: str, xlabel: str, ylabel: str,
                   out_path: str, dpi: int = 200) -> None:
    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.1), squeeze=False)
    for i, (name, mat) in enumerate(mats):
        ax = axes[0][i]
        # clamp like original TimeBridge heatmap style
        show = np.clip(mat, 0.0, np.quantile(mat, 0.98) if mat.size else 1.0)
        im = ax.imshow(show, cmap="Reds", aspect="auto", interpolation="nearest")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def get_one_batch(args: argparse.Namespace, device: torch.device):
    ns = make_args_namespace(args)
    # data_provider expects args-like object
    class _A:
        pass
    a = _A()
    for k, v in vars(ns).items():
        setattr(a, k, v)
    a.is_training = 0
    dataset, loader = data_provider(a, flag="test")
    for i, batch in enumerate(loader):
        if i == args.sample_idx:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if "PEMS" in args.data or "Solar" in args.data:
                batch_x_mark = None
                batch_y_mark = None
            else:
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            return batch_x, batch_x_mark, dec_inp, batch_y_mark
    raise IndexError(f"sample_idx={args.sample_idx} not found in test loader.")


def plot_attention_figs(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    cfg = make_args_namespace(args)
    model = Model(cfg).float().to(device).eval()

    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(args.checkpoint)
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"[load] {args.checkpoint}")
    else:
        print("[warn] no checkpoint provided; using randomly initialized weights for attention demo.")

    batch_x, batch_x_mark, dec_inp, batch_y_mark = get_one_batch(args, device)
    catcher = AttentionCatcher()
    catcher.register(model)
    # Prefer full channel attention so Fig.9 is true Query/Key Channel map.
    set_coint_full_attn(model, full=True)

    # Stationary: IntAttention.stable=True (PeriodNorm on QK)
    catcher.clear()
    set_intattention_stable(model, True)
    with torch.no_grad():
        _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    maps_stat = list(catcher.maps)
    intra_s, inter_s = pick_intra_and_inter(maps_stat, args.enc_in)

    # Non-stationary: disable PeriodNorm path
    catcher.clear()
    set_intattention_stable(model, False)
    with torch.no_grad():
        _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    maps_nst = list(catcher.maps)
    intra_n, inter_n = pick_intra_and_inter(maps_nst, args.enc_in)

    catcher.remove()
    ensure_dir(args.out_dir)

    # Fig.8 style: intra-variate (time patch)
    if intra_s is not None and intra_n is not None:
        plot_attn_pair(
            [("Stationary", intra_s), ("Non-stationary", intra_n)],
            title=f"Intra-variate Attention ({args.data})",
            xlabel="Key Time Patch",
            ylabel="Query Time Patch",
            out_path=os.path.join(args.out_dir, f"fig8_intra_attn_{args.data}.png"),
            dpi=args.dpi,
        )
    else:
        print("[warn] could not extract intra-variate attention maps.")

    # Fig.9 style: inter-variate (channel)
    if inter_s is not None and inter_n is not None:
        inter_s = subsample_square(inter_s, args.max_channels_heatmap)
        inter_n = subsample_square(inter_n, args.max_channels_heatmap)
        plot_attn_pair(
            [("Stationary", inter_s), ("Non-stationary", inter_n)],
            title=f"Inter-variate Attention ({args.data})",
            xlabel="Key Channel",
            ylabel="Query Channel",
            out_path=os.path.join(args.out_dir, f"fig9_inter_attn_{args.data}.png"),
            dpi=args.dpi,
        )
    else:
        print("[warn] could not extract inter-variate attention maps. "
              "Ensure --ca_layers >= 1 for channel attention.")


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.out_dir)
    if args.mode in ("cointegration", "all"):
        plot_cointegration(args)
    if args.mode in ("attention", "all"):
        plot_attention_figs(args)
    print(f"Done. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
