"""
Generate mechanism-level ED-SRA attention evidence for a Solar experiment.

The script uses a test_results setting only to identify the experiment. Attention
is recomputed from the matching trained checkpoint because test_results contains
prediction PDFs, not attention tensors.

Outputs:
  - initial_attention.{png,pdf}                 (3 heads)
  - sparse_attention.{png,pdf}                  (3 heads)
  - sparse_residual_attention.{png,pdf}         (3 heads)
  - edsra_attention_3x3.{png,pdf}               (all nine heatmaps)
  - edsra_attention_raw.npz / metrics.csv / metadata.json

Style: YlOrRd, shared colorbar per stage, outer-only labels, Times New Roman.
Replot without a checkpoint via --from_npz; write the LaTeX asset via --paper_out.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


SETTING_RE = re.compile(
    r"^(?P<model_id>.+)_TimeBridge_(?P<data>[^_]+)"
    r"_bs(?P<batch_size>\d+)_ft(?P<features>[^_]+)"
    r"_sl(?P<seq_len>\d+)_ll(?P<label_len>\d+)_pl(?P<pred_len>\d+)"
    r"_dm(?P<d_model>\d+)_nh(?P<n_heads>\d+)"
    r"_ial(?P<ia_layers>\d+)_pdl(?P<pd_layers>\d+)"
    r"_cal(?P<ca_layers>\d+)_df(?P<d_ff>\d+)"
    r"_eb(?P<embed>[^_]+)_(?P<description>.+)_(?P<iteration>\d+)$"
)

STAGE_LABELS = {
    "initial": "Initial Attention",
    "sparse": "Gated Attention",
    "residual": "Redistributed Attention",
}

# Warm sequential map: near-zero stays tinted, peaks stay dark red.
# Matches the TimeBridge / intra-inter red family better than default Reds
# (which washes out when vmin=0) and avoids the black-hot palette of SRA.
PAPER_CMAP = "YlOrRd"
PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize Softmax -> sparse -> residual ED-SRA attention on Solar."
    )
    parser.add_argument(
        "--experiment_dir",
        type=Path,
        default=None,
        help="Solar directory under test_results. Auto-detected when omitted.",
    )
    parser.add_argument("--test_results_root", type=Path, default=REPO_ROOT / "test_results")
    parser.add_argument(
        "--experiment_glob",
        default="Solar_*_D3_*",
        help="Glob used for auto-detection (D3 = EA-RevIN + ED-SRA).",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--root_path", type=Path, default=None)
    parser.add_argument("--data_path", default="solar_AL.txt")
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "_logs" / "vis_edsra_solar")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index in the Solar test split.")
    parser.add_argument(
        "--channel_idx",
        type=int,
        default=0,
        help="Solar variable used for the intra-variate map.",
    )
    parser.add_argument("--heads", default="0,1,2")
    parser.add_argument(
        "--edsra_index",
        type=int,
        default=0,
        help="ED-SRA call in model order; 0 is the first intra-variate attention layer.",
    )
    parser.add_argument("--period", type=int, default=48)
    parser.add_argument("--num_p", type=int, default=12)
    parser.add_argument("--enc_in", type=int, default=137)
    parser.add_argument("--stable_len", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn_dropout", type=float, default=0.15)
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--from_npz",
        type=Path,
        default=None,
        help="Replot from a saved edsra_attention_raw.npz instead of a checkpoint.",
    )
    parser.add_argument(
        "--paper_out",
        type=Path,
        default=None,
        help="Also copy the 3x3 figure to this png/pdf path stem (LaTeX asset).",
    )
    parser.add_argument(
        "--no_title",
        action="store_true",
        help="Omit the figure suptitle so the LaTeX caption can stand alone.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a deterministic style preview without data or a checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--synthetic_queries", type=int, default=26)
    parser.add_argument("--synthetic_keys", type=int, default=96)
    return parser


def find_experiment(args: argparse.Namespace) -> Path:
    if args.experiment_dir is not None:
        path = args.experiment_dir.resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Experiment directory does not exist: {path}")
        return path

    candidates = [p for p in args.test_results_root.glob(args.experiment_glob) if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"No experiment matched {args.experiment_glob!r} in {args.test_results_root}. "
            "Pass --experiment_dir explicitly."
        )
    # Prefer a directory that actually contains prediction artifacts, then newest.
    candidates.sort(
        key=lambda p: (any(p.iterdir()), p.stat().st_mtime),
        reverse=True,
    )
    return candidates[0].resolve()


def parse_setting(experiment_dir: Path) -> Dict[str, object]:
    match = SETTING_RE.match(experiment_dir.name)
    if match is None:
        raise ValueError(
            "Cannot parse TimeBridge setting from directory name:\n"
            f"  {experiment_dir.name}"
        )
    values: Dict[str, object] = match.groupdict()
    for key in (
        "batch_size",
        "seq_len",
        "label_len",
        "pred_len",
        "d_model",
        "n_heads",
        "ia_layers",
        "pd_layers",
        "ca_layers",
        "d_ff",
        "iteration",
    ):
        values[key] = int(values[key])
    return values


def find_data_root(args: argparse.Namespace) -> Path:
    if args.root_path is not None:
        root = args.root_path.resolve()
        if not (root / args.data_path).is_file():
            raise FileNotFoundError(root / args.data_path)
        return root

    candidates = [
        REPO_ROOT / "dataset" / "Solar",
        REPO_ROOT.parent / "暂存" / "dataset" / "Solar",
        REPO_ROOT.parent / "AMD-main" / "dataset" / "Solar",
        REPO_ROOT.parent / "CMoS-main" / "dataset" / "Solar",
    ]
    for root in candidates:
        if (root / args.data_path).is_file():
            return root.resolve()
    raise FileNotFoundError(
        f"Could not locate {args.data_path}. Pass --root_path <Solar dataset directory>."
    )


def find_checkpoint(args: argparse.Namespace, experiment_dir: Path) -> Path:
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else REPO_ROOT / "checkpoints" / experiment_dir.name / "checkpoint.pth"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "The matching trained checkpoint is required for valid attention evidence.\n"
            f"Expected: {checkpoint}\n"
            "test_results prediction PDFs do not contain attention matrices."
        )
    return checkpoint


def make_config(
    setting: Dict[str, object], args: argparse.Namespace, data_root: Path
) -> SimpleNamespace:
    uses_ea_revin = "D3" in str(setting["model_id"])
    return SimpleNamespace(
        revin=True,
        use_ea_revin=uses_ea_revin,
        revin_affine=uses_ea_revin,
        revin_subtract_last=False,
        revin_dyn_bound=0.05 if uses_ea_revin else 0.0,
        revin_entropy_gate=0.55,
        use_ed_sra=True,
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
        enc_in=args.enc_in,
        period=args.period,
        num_p=args.num_p,
        stable_len=args.stable_len,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        activation=args.activation,
        data=setting["data"],
        root_path=str(data_root),
        data_path=args.data_path,
        features=setting["features"],
        target="OT",
        freq="h",
        seasonal_patterns="Monthly",
        num_workers=0,
        **{
            key: setting[key]
            for key in (
                "batch_size",
                "seq_len",
                "label_len",
                "pred_len",
                "d_model",
                "n_heads",
                "ia_layers",
                "pd_layers",
                "ca_layers",
                "d_ff",
                "embed",
            )
        },
    )


def _require_torch():
    import torch
    return torch


def _require_model_stack():
    from data_provider.data_factory import data_provider
    from layers.SelfAttention_Family import _EDSRAModule
    from model.TimeBridge import Model
    return data_provider, _EDSRAModule, Model


def choose_device(name: str):
    torch = _require_torch()
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model, path: Path, device) -> None:
    torch = _require_torch()
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(state).__name__}")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    if not any(".ed_sra." in str(key) for key in state):
        raise ValueError(
            "Checkpoint has no ED-SRA parameters. Use a D2/D3 checkpoint trained "
            "with --use_ed_sra."
        )
    model.load_state_dict(state, strict=True)


def get_test_sample(config: SimpleNamespace, index: int):
    torch = _require_torch()
    data_provider, _, _ = _require_model_stack()
    dataset, _ = data_provider(config, flag="test")
    if index < 0 or index >= len(dataset):
        raise IndexError(f"sample_idx={index}; valid range is [0, {len(dataset) - 1}].")
    batch_x = dataset[index][0]
    return torch.as_tensor(batch_x, dtype=torch.float32).unsqueeze(0)


def capture_stages(
    model, batch_x, edsra_index: int
) -> Tuple[str, Dict[str, object]]:
    torch = _require_torch()
    _, EDSRAModule, _ = _require_model_stack()
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, EDSRAModule)
    ]
    if edsra_index < 0 or edsra_index >= len(modules):
        raise IndexError(
            f"edsra_index={edsra_index}; model contains {len(modules)} ED-SRA modules."
        )
    for _, module in modules:
        module.capture_intermediates = True
        module.last_intermediates = None

    with torch.no_grad():
        model(batch_x, None, None, None)

    called = [(name, module) for name, module in modules if module.last_intermediates]
    if edsra_index >= len(called):
        names = ", ".join(name for name, _ in called)
        raise RuntimeError(
            f"Only {len(called)} ED-SRA modules ran; requested {edsra_index}. Called: {names}"
        )
    name, module = called[edsra_index]
    assert module.last_intermediates is not None
    return name, module.last_intermediates


def parse_heads(text: str, n_heads: int) -> List[int]:
    heads = [int(item.strip()) for item in text.split(",") if item.strip()]
    if len(heads) != 3 or len(set(heads)) != 3:
        raise ValueError("--heads must contain exactly three distinct head indices.")
    if min(heads) < 0 or max(heads) >= n_heads:
        raise IndexError(f"Head indices {heads} are invalid for n_heads={n_heads}.")
    return heads


def extract_maps(
    captured: Dict[str, object], channel_idx: int, heads: List[int]
) -> Dict[str, np.ndarray]:
    torch = _require_torch()
    initial = captured["initial"]
    if initial.ndim != 4:
        raise ValueError(f"Expected [B,H,L,S], got {tuple(initial.shape)}")
    if channel_idx < 0 or channel_idx >= initial.shape[0]:
        raise IndexError(
            f"channel_idx={channel_idx}; attention batch dimension is {initial.shape[0]}."
        )
    head_index = torch.tensor(heads, dtype=torch.long)
    return {
        stage: captured[stage][channel_idx].index_select(0, head_index).numpy()
        for stage in ("initial", "sparse", "residual")
    }


def apply_paper_style() -> None:
    plt.rcParams.update(PAPER_STYLE)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False


def stage_limit(maps: np.ndarray, quantile: float = 0.99) -> float:
    positive = maps[np.isfinite(maps) & (maps > 0)]
    if not positive.size:
        return 1.0
    vmax = float(np.quantile(positive, quantile))
    return vmax if vmax > 0 else 1.0


def _format_axis(axis, *, show_xlabel: bool, show_ylabel: bool) -> None:
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axis.tick_params(length=2.5, pad=1.5)
    axis.set_xlabel("Key Time Patch" if show_xlabel else "")
    axis.set_ylabel("Query Time Patch" if show_ylabel else "")


def _save_fig(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", pad_inches=0.03)


def draw_stage(
    stage: str,
    maps: np.ndarray,
    heads: List[int],
    path: Path,
    dpi: int,
    synthetic: bool = False,
    no_title: bool = False,
) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.15), sharex=True, sharey=True)
    vmax = stage_limit(maps)
    image = None
    for col, (axis, matrix, head) in enumerate(zip(axes, maps, heads)):
        image = axis.imshow(
            matrix,
            cmap=PAPER_CMAP,
            vmin=0.0,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
        )
        axis.set_title(f"Head {head}")
        _format_axis(axis, show_xlabel=True, show_ylabel=(col == 0))
    fig.colorbar(image, ax=list(axes), fraction=0.025, pad=0.02).ax.tick_params(labelsize=8)
    if not no_title:
        suffix = " — synthetic preview, not experimental evidence" if synthetic else ""
        fig.suptitle(f"{STAGE_LABELS[stage]}{suffix}", fontsize=13)
    fig.subplots_adjust(
        left=0.07,
        right=0.93,
        top=0.82 if not no_title else 0.90,
        bottom=0.18,
        wspace=0.06,
    )
    _save_fig(fig, path, dpi)
    plt.close(fig)


def draw_combined(
    maps: Dict[str, np.ndarray],
    heads: List[int],
    path: Path,
    dpi: int,
    synthetic: bool = False,
    no_title: bool = False,
) -> None:
    apply_paper_style()
    stages = ("initial", "sparse", "residual")
    fig, axes = plt.subplots(3, 3, figsize=(12.2, 7.2), sharex=True, sharey=True)
    for row, stage in enumerate(stages):
        vmax = stage_limit(maps[stage])
        image = None
        for col, head in enumerate(heads):
            axis = axes[row, col]
            image = axis.imshow(
                maps[stage][col],
                cmap=PAPER_CMAP,
                vmin=0.0,
                vmax=vmax,
                aspect="auto",
                interpolation="nearest",
                origin="upper",
            )
            axis.set_title(f"Head {head}" if row == 0 else "")
            _format_axis(axis, show_xlabel=(row == 2), show_ylabel=(col == 0))
            if col == 0:
                axis.text(
                    -0.42,
                    0.5,
                    STAGE_LABELS[stage],
                    rotation=90,
                    va="center",
                    ha="center",
                    transform=axis.transAxes,
                    fontsize=11,
                    fontweight="bold",
                )
        cbar = fig.colorbar(
            image,
            ax=list(axes[row, :]),
            fraction=0.028,
            pad=0.035,
            aspect=18,
        )
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    if not no_title:
        title = "ED-SRA: Softmax → Gated → Redistributed"
        if synthetic:
            title += " — synthetic preview, not experimental evidence"
        fig.suptitle(title, fontsize=13)
    fig.subplots_adjust(
        left=0.12,
        right=0.93,
        top=0.90 if not no_title else 0.96,
        bottom=0.09,
        wspace=0.07,
        hspace=0.16,
    )
    _save_fig(fig, path, dpi)
    plt.close(fig)


def entropy(matrix: np.ndarray) -> float:
    safe = np.clip(matrix, 1e-12, 1.0)
    return float(np.mean(-np.sum(safe * np.log(safe), axis=-1)))


def make_synthetic_maps(
    seed: int, queries: int, keys: int, n_heads: int = 3
) -> Dict[str, np.ndarray]:
    """Create structured attention solely for checking the visualization style."""
    if queries < 2 or keys < 4:
        raise ValueError("Synthetic maps require at least 2 queries and 4 keys.")
    rng = np.random.default_rng(seed)
    key_axis = np.arange(keys, dtype=np.float64)
    initial_maps = []
    sparse_maps = []
    residual_maps = []

    base_centers = np.asarray([0.36, 0.61, 0.86]) * (keys - 1)
    for head in range(n_heads):
        matrix = rng.uniform(0.0002, 0.002, size=(queries, keys))
        centers = base_centers + rng.normal(0.0, 1.8, size=base_centers.shape) + head * 0.7
        for row in range(queries):
            row_phase = 1.0 + 0.28 * np.sin(2.0 * np.pi * row / max(queries - 1, 1) + head)
            for peak, center in enumerate(centers):
                width = 1.4 + 0.45 * peak + 0.15 * head
                amplitude = row_phase * rng.uniform(0.8, 1.25)
                matrix[row] += amplitude * np.exp(
                    -0.5 * ((key_axis - center - 0.06 * row) / width) ** 2
                )
        # Add a few strong localized cells like learned sparse-attention examples.
        for _ in range(4):
            row = int(rng.integers(0, queries))
            col = int(rng.choice(np.round(centers).astype(int)))
            matrix[row, np.clip(col, 0, keys - 1)] *= rng.uniform(2.5, 4.0)

        initial = matrix / matrix.sum(axis=-1, keepdims=True)
        gated = np.power(initial, 10.0)
        sparse = gated / np.maximum(gated.sum(axis=-1, keepdims=True), 1e-12)
        alpha = 0.32
        residual = (1.0 - alpha) * initial + alpha * sparse
        residual /= residual.sum(axis=-1, keepdims=True)
        initial_maps.append(initial)
        sparse_maps.append(sparse)
        residual_maps.append(residual)

    return {
        "initial": np.stack(initial_maps),
        "sparse": np.stack(sparse_maps),
        "residual": np.stack(residual_maps),
    }


def load_maps_from_npz(path: Path) -> Tuple[Dict[str, np.ndarray], List[int]]:
    payload = np.load(path)
    missing = [key for key in ("initial", "sparse", "residual") if key not in payload]
    if missing:
        raise KeyError(f"{path} is missing {missing}")
    maps = {key: np.asarray(payload[key]) for key in ("initial", "sparse", "residual")}
    if "heads" in payload:
        heads = [int(head) for head in np.asarray(payload["heads"]).reshape(-1)]
    else:
        heads = list(range(int(maps["initial"].shape[0])))
    return maps, heads


def export_stage_and_combined(
    args: argparse.Namespace,
    maps: Dict[str, np.ndarray],
    heads: List[int],
    *,
    synthetic: bool,
    combined_name: str,
    stage_names: Dict[str, str],
) -> Path:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stage, file_name in stage_names.items():
        draw_stage(
            stage,
            maps[stage],
            heads,
            args.out_dir / file_name,
            args.dpi,
            synthetic=synthetic,
            no_title=args.no_title,
        )
    combined_path = args.out_dir / combined_name
    draw_combined(
        maps,
        heads,
        combined_path,
        args.dpi,
        synthetic=synthetic,
        no_title=args.no_title,
    )
    if args.paper_out is not None:
        paper_path = args.paper_out.with_suffix(".png")
        draw_combined(
            maps,
            heads,
            paper_path,
            args.dpi,
            synthetic=synthetic,
            no_title=True,
        )
        print(f"[paper] {paper_path.with_suffix('.pdf')}")
    return combined_path


def generate_synthetic_preview(args: argparse.Namespace) -> None:
    heads = [0, 1, 2]
    maps = make_synthetic_maps(
        args.seed, args.synthetic_queries, args.synthetic_keys, len(heads)
    )
    export_stage_and_combined(
        args,
        maps,
        heads,
        synthetic=True,
        combined_name="synthetic_edsra_attention_3x3.png",
        stage_names={
            "initial": "synthetic_initial_attention.png",
            "sparse": "synthetic_sparse_attention.png",
            "residual": "synthetic_sparse_residual_attention.png",
        },
    )
    np.savez_compressed(
        args.out_dir / "synthetic_edsra_attention_raw.npz",
        **maps,
        seed=args.seed,
        heads=np.asarray(heads),
    )
    print(f"[synthetic] seed={args.seed}; no checkpoint or experimental data used")
    print(f"[saved] {args.out_dir.resolve()}")


def write_metrics(
    path: Path,
    maps: Dict[str, np.ndarray],
    captured: Dict[str, object],
    channel_idx: int,
    heads: List[int],
) -> None:
    fields = [
        "head",
        "initial_entropy",
        "sparse_entropy",
        "residual_entropy",
        "initial_to_sparse_l1",
        "initial_to_residual_l1",
        "gate_suppression",
        "effective_alpha",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position, head in enumerate(heads):
            initial = maps["initial"][position]
            sparse = maps["sparse"][position]
            residual = maps["residual"][position]
            gate = captured["gate_mask"][channel_idx, head].numpy()
            alpha = captured["effective_alpha"][channel_idx, head].numpy()
            writer.writerow(
                {
                    "head": head,
                    "initial_entropy": entropy(initial),
                    "sparse_entropy": entropy(sparse),
                    "residual_entropy": entropy(residual),
                    "initial_to_sparse_l1": float(np.mean(np.abs(initial - sparse))),
                    "initial_to_residual_l1": float(np.mean(np.abs(initial - residual))),
                    "gate_suppression": float(np.mean(1.0 - gate)),
                    "effective_alpha": float(np.mean(alpha)),
                }
            )


def main() -> None:
    args = build_parser().parse_args()
    if args.synthetic:
        generate_synthetic_preview(args)
        return
    if args.from_npz is not None:
        maps, heads = load_maps_from_npz(args.from_npz)
        export_stage_and_combined(
            args,
            maps,
            heads,
            synthetic=False,
            combined_name="edsra_attention_3x3.png",
            stage_names={
                "initial": "initial_attention.png",
                "sparse": "sparse_attention.png",
                "residual": "sparse_residual_attention.png",
            },
        )
        print(f"[npz] {args.from_npz}")
        print(f"[saved] {args.out_dir.resolve()}")
        return
    experiment_dir = find_experiment(args)
    setting = parse_setting(experiment_dir)
    if setting["data"] != "Solar":
        raise ValueError(f"Expected a Solar experiment, got {setting['data']!r}.")
    data_root = find_data_root(args)
    checkpoint = find_checkpoint(args, experiment_dir)
    config = make_config(setting, args, data_root)
    heads = parse_heads(args.heads, int(setting["n_heads"]))
    device = choose_device(args.device)

    _, _, Model = _require_model_stack()
    model = Model(config).float().to(device).eval()
    load_checkpoint(model, checkpoint, device)
    batch_x = get_test_sample(config, args.sample_idx).to(device)
    module_name, captured = capture_stages(model, batch_x, args.edsra_index)
    maps = extract_maps(captured, args.channel_idx, heads)

    export_stage_and_combined(
        args,
        maps,
        heads,
        synthetic=False,
        combined_name="edsra_attention_3x3.png",
        stage_names={
            "initial": "initial_attention.png",
            "sparse": "sparse_attention.png",
            "residual": "sparse_residual_attention.png",
        },
    )

    np.savez_compressed(
        args.out_dir / "edsra_attention_raw.npz",
        initial=maps["initial"],
        sparse=maps["sparse"],
        residual=maps["residual"],
        heads=np.asarray(heads),
        channel_idx=args.channel_idx,
        sample_idx=args.sample_idx,
    )
    write_metrics(
        args.out_dir / "edsra_attention_metrics.csv",
        maps,
        captured,
        args.channel_idx,
        heads,
    )
    metadata = {
        "experiment_dir": str(experiment_dir),
        "checkpoint": str(checkpoint),
        "data_file": str(data_root / args.data_path),
        "edsra_module": module_name,
        "sample_idx": args.sample_idx,
        "channel_idx": args.channel_idx,
        "heads": heads,
        "attention_shape": list(captured["initial"].shape),
        "pipeline": [
            "Softmax A",
            "entropy/peakness routers",
            "soft threshold gate",
            "row-wise redistribution",
            "residual mix",
        ],
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[experiment] {experiment_dir.name}")
    print(f"[checkpoint] {checkpoint}")
    print(f"[captured] {module_name}, shape={tuple(captured['initial'].shape)}")
    print(f"[saved] {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
