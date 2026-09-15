"""
Paper-style multi-model prediction visualization (2xN grid).

Expected result folder layout (after training/testing with updated exp_long_term_forecasting.py):
  results/<setting>/
    pred.npy    # [N, pred_len, C]
    true.npy    # [N, pred_len, C]
    input.npy   # [N, seq_len, 1] last-channel history (optional but preferred)

Example:
  python -u experiments/visualize_predictions.py \
    --dataset Solar \
    --sample_idx 0 \
    --channel -1 \
    --out_dir ./_logs/vis_solar \
    --models \
      TimeBridge:./results/Solar_720_336_D0_TimeBridge_... \
      TimeBridge+ED-SRA:./results/Solar_720_336_D2_TimeBridge_... \
      TimeBridge+EA-RevIN:./results/Solar_720_336_D1_TimeBridge_... \
      TimeBridge+Both:./results/Solar_720_336_D3_TimeBridge_...
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize multi-model forecasting predictions.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="List of 'Name:result_dir' pairs. Example: TimeBridge:./results/xxx",
    )
    parser.add_argument("--sample_idx", type=int, default=0, help="Which test sample to plot.")
    parser.add_argument("--channel", type=int, default=-1, help="Channel index; -1 means last channel.")
    parser.add_argument("--ncols", type=int, default=4, help="Number of subplot columns.")
    parser.add_argument("--dataset", type=str, default="Solar", help="Dataset name used in figure title.")
    parser.add_argument("--out_dir", type=str, default="./_logs/vis", help="Output directory.")
    parser.add_argument("--out_name", type=str, default="prediction_comparison", help="Output file stem.")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    parser.add_argument(
        "--history_len",
        type=int,
        default=-1,
        help="If input.npy missing, pad history with this length using NaN. -1 means no history.",
    )
    return parser


def parse_model_specs(specs: List[str]) -> List[Tuple[str, str]]:
    models = []
    for s in specs:
        if ":" not in s:
            raise ValueError(f"Invalid --models item '{s}'. Expected Name:result_dir")
        name, path = s.split(":", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --models item '{s}'.")
        models.append((name, path))
    return models


def load_series(result_dir: str, sample_idx: int, channel: int, history_len: int) -> Tuple[np.ndarray, np.ndarray]:
    pred_path = os.path.join(result_dir, "pred.npy")
    true_path = os.path.join(result_dir, "true.npy")
    input_path = os.path.join(result_dir, "input.npy")

    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        raise FileNotFoundError(
            f"Missing pred.npy/true.npy in {result_dir}. "
            "Re-run testing with the updated exp_long_term_forecasting.py first."
        )

    pred = np.load(pred_path)
    true = np.load(true_path)
    if pred.ndim != 3 or true.ndim != 3:
        raise ValueError(
            f"Expected pred/true with shape [N, T, C] in {result_dir}, "
            f"got pred={pred.shape}, true={true.shape}"
        )
    if pred.shape != true.shape:
        raise ValueError(f"pred/true shape mismatch in {result_dir}: {pred.shape} vs {true.shape}")
    if sample_idx < 0 or sample_idx >= pred.shape[0]:
        raise IndexError(f"sample_idx={sample_idx} out of range for {result_dir} (N={pred.shape[0]})")

    ch = channel if channel >= 0 else -1
    if not -pred.shape[-1] <= ch < pred.shape[-1]:
        raise IndexError(f"channel={channel} out of range for {result_dir} (C={pred.shape[-1]})")
    pred_y = pred[sample_idx, :, ch]
    true_y = true[sample_idx, :, ch]
    pred_len = pred_y.shape[0]

    if os.path.exists(input_path):
        inp = np.load(input_path)
        if inp.ndim != 3 or inp.shape[0] != pred.shape[0] or inp.shape[-1] not in (1, pred.shape[-1]):
            raise ValueError(
                f"input.npy is not aligned with pred.npy in {result_dir}: "
                f"input={inp.shape}, pred={pred.shape}"
            )
        if inp.shape[-1] == 1:
            if ch not in (-1, pred.shape[-1] - 1):
                raise ValueError(
                    f"{result_dir}/input.npy contains only the last-channel history; "
                    f"use --channel -1 (requested {channel})."
                )
            hist_ch = 0
        else:
            hist_ch = ch
        hist = inp[sample_idx, :, hist_ch]
        gt = np.concatenate([hist, true_y], axis=0)
        pd = np.full_like(gt, np.nan, dtype=float)
        pd[-pred_len:] = pred_y
        return gt, pd

    # Fallback: no input.npy -> only plot forecast window, optionally pad empty history.
    if history_len > 0:
        hist = np.full((history_len,), np.nan, dtype=float)
        gt = np.concatenate([hist, true_y], axis=0)
        pd = np.full_like(gt, np.nan, dtype=float)
        pd[-pred_len:] = pred_y
        return gt, pd

    gt = true_y
    pd = pred_y.astype(float)
    return gt, pd


def plot_comparison(
    model_series: List[Tuple[str, np.ndarray, np.ndarray]],
    dataset: str,
    out_path: str,
    ncols: int = 4,
    dpi: int = 200,
) -> None:
    n = len(model_series)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    letters = "abcdefghijklmnopqrstuvwxyz"

    for i, (name, gt, pd) in enumerate(model_series):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        ax.plot(gt, label="GroundTruth", color="#7EB6D9", linewidth=1.8)
        ax.plot(pd, label="Prediction", color="#D62728", linewidth=1.8)
        ax.set_title(f"({letters[i]}) {name}", fontsize=11)
        ax.legend(loc="upper left", fontsize=8, frameon=True)
        ax.grid(False)
        ax.set_facecolor("#F5F5F5")

    # Hide unused axes.
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.suptitle(f"Visualization of predictions from different models on the {dataset} dataset.", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path + ".pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path + ".png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}.pdf")
    print(f"Saved: {out_path}.png")


def auto_discover_results(root: str = "./results") -> Dict[str, str]:
    found = {}
    if not os.path.isdir(root):
        return found
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "pred.npy")):
            found[name] = path
    return found


def main() -> None:
    args = build_parser().parse_args()
    model_specs = parse_model_specs(args.models)

    series = []
    reference_gt = None
    reference_name = None
    for name, result_dir in model_specs:
        gt, pd = load_series(result_dir, args.sample_idx, args.channel, args.history_len)
        if reference_gt is None:
            reference_gt = gt
            reference_name = name
        elif gt.shape != reference_gt.shape or not np.allclose(gt, reference_gt, rtol=1e-6, atol=1e-7):
            raise ValueError(
                f"GroundTruth is not aligned: '{name}' differs from '{reference_name}'. "
                "Use result folders produced with the same dataset, split, seq_len, pred_len, "
                "sample_idx, channel, scaling and inverse settings."
            )

        # Draw one shared GroundTruth array in every panel. This guarantees that
        # only the prediction curve changes between model subplots.
        series.append((name, reference_gt, pd))
        print(f"[ok] {name}: gt_len={len(gt)}, pred_horizon={np.sum(~np.isnan(pd))}")

    out_path = os.path.join(args.out_dir, args.out_name)
    plot_comparison(series, args.dataset, out_path, ncols=args.ncols, dpi=args.dpi)


if __name__ == "__main__":
    main()
