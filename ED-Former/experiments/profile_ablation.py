import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch


def bootstrap_project_root() -> str:
    """Find project root that contains model/TimeBridge.py and prepend to sys.path."""
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.abspath(os.path.join(here, "..")),
        os.path.abspath(os.path.join(here, "..", "..")),
        os.path.abspath(os.path.join(here, "..", "..", "TimeBridge-main")),
    ]
    for root in candidates:
        if os.path.exists(os.path.join(root, "model", "TimeBridge.py")):
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    raise RuntimeError(
        "Cannot locate project root containing model/TimeBridge.py. "
        "Please run from TimeBridge project or check folder layout."
    )


PROJECT_ROOT = bootstrap_project_root()
from model.TimeBridge import Model


@dataclass
class ProfileResult:
    params: int
    params_m: float
    peak_mem_mb_mean: float
    peak_mem_mb_std: float
    forward_time_s_mean: float
    forward_time_s_std: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile TimeBridge variants on params / peak memory / forward time."
    )
    parser.add_argument(
        "--pred_lens",
        type=int,
        nargs="+",
        default=[96, 192, 336, 720],
        help="Prediction lengths to profile (table columns).",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Synthetic batch size.")
    parser.add_argument("--seq_len", type=int, default=720, help="Synthetic input sequence length.")
    parser.add_argument("--enc_in", type=int, default=137, help="Input channel count.")
    parser.add_argument("--period", type=int, default=48, help="Patch period.")
    parser.add_argument("--num_p", type=int, default=12, help="Downsampled patch count; None uses seq_len // period.")
    parser.add_argument("--d_model", type=int, default=128, help="Model width.")
    parser.add_argument("--d_ff", type=int, default=128, help="FFN hidden width.")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention head count.")
    parser.add_argument("--ia_layers", type=int, default=1, help="Integrated attention layers.")
    parser.add_argument("--pd_layers", type=int, default=1, help="Patch downsampling layers.")
    parser.add_argument("--ca_layers", type=int, default=1, help="Cointegrated attention layers.")
    parser.add_argument("--stable_len", type=int, default=6, help="Stable moving-average length.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout used in model.")
    parser.add_argument("--attn_dropout", type=float, default=0.15, help="Attention dropout used in model.")
    parser.add_argument("--warmup", type=int, default=15, help="Warmup forward iterations.")
    parser.add_argument("--iters", type=int, default=30, help="Timed forward iterations.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeat rounds for mean/std statistics.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Profiling device.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--csv_out", type=str, default="", help="Optional path to save raw CSV results.")
    parser.add_argument("--md_out", type=str, default="", help="Optional path to save markdown table.")
    parser.add_argument("--latex_out", type=str, default="", help="Optional path to save latex table.")
    return parser


def choose_device(device_flag: str) -> torch.device:
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_flag == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_base_config(args: argparse.Namespace, pred_len: int) -> SimpleNamespace:
    return SimpleNamespace(
        revin=True,
        enc_in=args.enc_in,
        period=args.period,
        seq_len=args.seq_len,
        pred_len=pred_len,
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
        activation="gelu",
        use_ea_revin=False,
        revin_affine=False,
        revin_subtract_last=False,
        revin_dyn_bound=0.0,
        revin_entropy_gate=0.55,
        use_ed_sra=False,
        ed_sra_init_gamma_min=0.2,
        ed_sra_init_gamma_max=0.95,
        ed_sra_N_threshold=300,
    )


def make_inputs(cfg: SimpleNamespace, batch_size: int, device: torch.device):
    x_enc = torch.randn(batch_size, cfg.seq_len, cfg.enc_in, device=device)
    x_mark_enc = torch.randn(batch_size, cfg.seq_len, 4, device=device)
    x_dec = torch.randn(batch_size, cfg.pred_len, cfg.enc_in, device=device)
    x_mark_dec = torch.randn(batch_size, cfg.pred_len, 4, device=device)
    return x_enc, x_mark_enc, x_dec, x_mark_dec


def apply_variant(cfg: SimpleNamespace, variant: str) -> None:
    if variant in ("ea_revin", "ea_revin_ed_sra"):
        cfg.use_ea_revin = True
        cfg.revin_affine = True
        cfg.revin_dyn_bound = 0.05
        cfg.revin_entropy_gate = 0.55
    if variant in ("ed_sra", "ea_revin_ed_sra"):
        cfg.use_ed_sra = True
        cfg.ed_sra_init_gamma_min = 0.2
        cfg.ed_sra_init_gamma_max = 0.95
        cfg.ed_sra_N_threshold = 300


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    t = torch.tensor(values, dtype=torch.float64)
    return float(t.mean().item()), float(t.std(unbiased=False).item())


def profile_one(args: argparse.Namespace, device: torch.device, pred_len: int, variant: str) -> ProfileResult:
    cfg = make_base_config(args, pred_len)
    apply_variant(cfg, variant)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    base_model = Model(cfg).to(device).eval()
    params = sum(p.numel() for p in base_model.parameters())
    params_m = params / 1e6
    del base_model

    peak_mem_values: List[float] = []
    forward_time_values: List[float] = []

    for rep in range(args.repeats):
        seed = args.seed + rep
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        model = Model(cfg).to(device).eval()
        inputs = make_inputs(cfg, args.batch_size, device)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        with torch.no_grad():
            for _ in range(args.warmup):
                _ = model(*inputs)
            synchronize_if_needed(device)

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(args.iters):
                _ = model(*inputs)
            synchronize_if_needed(device)
        elapsed = time.perf_counter() - start
        forward_time_s = elapsed / max(args.iters, 1)
        forward_time_values.append(forward_time_s)

        if device.type == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_mem_values.append(float(peak_mem_mb))
        else:
            peak_mem_values.append(float("nan"))

        del model

    peak_mean, peak_std = _mean_std(peak_mem_values)
    time_mean, time_std = _mean_std(forward_time_values)
    return ProfileResult(
        params=params,
        params_m=params_m,
        peak_mem_mb_mean=peak_mean,
        peak_mem_mb_std=peak_std,
        forward_time_s_mean=time_mean,
        forward_time_s_std=time_std,
    )


def format_value(value: float, digits: int = 3) -> str:
    if value != value:
        return "N/A"
    return f"{value:.{digits}f}"


def format_mean_std(mean: float, std: float, digits: int = 4, unit: str = "") -> str:
    if mean != mean:
        return "N/A"
    suffix = f" {unit}" if unit else ""
    return f"{mean:.{digits}f} ± {std:.{digits}f}{suffix}"


def print_variant_table(name: str, pred_lens: List[int], result_map: Dict[int, ProfileResult]) -> None:
    cols = " | ".join(str(p) for p in pred_lens)
    print(f"\n{name}")
    print(f"| Metric | {cols} |")
    print(f"|---|{'---|' * len(pred_lens)}")

    params_row = " | ".join(f"{result_map[p].params_m:.4f}" for p in pred_lens)
    params_abs_row = " | ".join(str(result_map[p].params) for p in pred_lens)
    peak_row = " | ".join(
        format_mean_std(result_map[p].peak_mem_mb_mean, result_map[p].peak_mem_mb_std, digits=1)
        for p in pred_lens
    )
    time_row = " | ".join(
        format_mean_std(result_map[p].forward_time_s_mean, result_map[p].forward_time_s_std, digits=4)
        for p in pred_lens
    )

    print(f"| Params (M, high-precision) | {params_row} |")
    print(f"| Params (absolute) | {params_abs_row} |")
    print(f"| Peak Mem (MB, mean±std) | {peak_row} |")
    print(f"| Forward time (s, mean±std) | {time_row} |")


def build_markdown_table(
    pred_lens: List[int], all_results: Dict[str, Tuple[str, Dict[int, ProfileResult]]]
) -> str:
    headers = ["Model Variant", "Metric"] + [str(p) for p in pred_lens]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, (variant_name, result_map) in all_results.items():
        lines.append(
            "| "
            + " | ".join(
                [variant_name, "Params (M)"] + [f"{result_map[p].params_m:.4f}" for p in pred_lens]
            )
            + " |"
        )
        lines.append(
            "| "
            + " | ".join(
                [variant_name, "Peak Mem (MB)"]
                + [format_mean_std(result_map[p].peak_mem_mb_mean, result_map[p].peak_mem_mb_std, 1) for p in pred_lens]
            )
            + " |"
        )
        lines.append(
            "| "
            + " | ".join(
                [variant_name, "Forward time (s)"]
                + [format_mean_std(result_map[p].forward_time_s_mean, result_map[p].forward_time_s_std, 4) for p in pred_lens]
            )
            + " |"
        )
    return "\n".join(lines)


def build_latex_table(
    pred_lens: List[int], all_results: Dict[str, Tuple[str, Dict[int, ProfileResult]]]
) -> str:
    col_spec = "ll" + "c" * len(pred_lens)
    header = " & ".join(["Model", "Metric"] + [str(p) for p in pred_lens]) + " \\\\"
    rows: List[str] = []
    for _, (variant_name, result_map) in all_results.items():
        rows.append(
            " & ".join([variant_name, "Params (M)"] + [f"{result_map[p].params_m:.4f}" for p in pred_lens]) + " \\\\"
        )
        rows.append(
            " & ".join(
                [variant_name, "Peak Mem (MB)"]
                + [f"{result_map[p].peak_mem_mb_mean:.1f} $\\pm$ {result_map[p].peak_mem_mb_std:.1f}" for p in pred_lens]
            ) + " \\\\"
        )
        rows.append(
            " & ".join(
                [variant_name, "Forward time (s)"]
                + [f"{result_map[p].forward_time_s_mean:.4f} $\\pm$ {result_map[p].forward_time_s_std:.4f}" for p in pred_lens]
            ) + " \\\\"
        )
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{TimeBridge ablation profiling with mean\\,$\\pm$\\,std across repeats.}\n"
        "\\label{tab:timebridge-profile}\n"
        "\\end{table}\n"
    )


def save_csv(path: str, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "variant",
        "pred_len",
        "params",
        "params_m",
        "peak_mem_mb_mean",
        "peak_mem_mb_std",
        "forward_time_s_mean",
        "forward_time_s_std",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = build_parser().parse_args()
    device = choose_device(args.device)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    print("=== TimeBridge Performance Profiling ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {device}")
    print(f"Pred lengths: {args.pred_lens}")
    print(f"Batch size: {args.batch_size}, warmup: {args.warmup}, iters: {args.iters}, repeats: {args.repeats}")

    variants = [
        ("baseline", "TimeBridge (baseline)"),
        ("ed_sra", "TimeBridge + ED-SRA"),
        ("ea_revin", "TimeBridge + EA-RevIN"),
        ("ea_revin_ed_sra", "TimeBridge + EA-RevIN + ED-SRA"),
    ]

    csv_rows: List[Dict[str, str]] = []
    all_results: Dict[str, Tuple[str, Dict[int, ProfileResult]]] = {}

    for variant_key, variant_name in variants:
        per_len_results: Dict[int, ProfileResult] = {}
        for pred_len in args.pred_lens:
            result = profile_one(args, device, pred_len, variant_key)
            per_len_results[pred_len] = result
            csv_rows.append(
                {
                    "variant": variant_key,
                    "pred_len": str(pred_len),
                    "params": str(result.params),
                    "params_m": f"{result.params_m:.6f}",
                    "peak_mem_mb_mean": "" if result.peak_mem_mb_mean != result.peak_mem_mb_mean else f"{result.peak_mem_mb_mean:.6f}",
                    "peak_mem_mb_std": "" if result.peak_mem_mb_std != result.peak_mem_mb_std else f"{result.peak_mem_mb_std:.6f}",
                    "forward_time_s_mean": f"{result.forward_time_s_mean:.6f}",
                    "forward_time_s_std": f"{result.forward_time_s_std:.6f}",
                }
            )
            print(
                f"[{variant_key}] pred_len={pred_len} | "
                f"params={result.params_m:.4f}M ({result.params}) | "
                f"peak_mem={format_mean_std(result.peak_mem_mb_mean, result.peak_mem_mb_std, 1)}MB | "
                f"forward={format_mean_std(result.forward_time_s_mean, result.forward_time_s_std, 4)}s"
            )
        all_results[variant_key] = (variant_name, per_len_results)
        print_variant_table(variant_name, args.pred_lens, per_len_results)

    if args.csv_out:
        save_csv(args.csv_out, csv_rows)
        print(f"\nCSV saved to: {args.csv_out}")

    md_table = build_markdown_table(args.pred_lens, all_results)
    latex_table = build_latex_table(args.pred_lens, all_results)
    print("\n=== Markdown Table ===\n")
    print(md_table)
    print("\n=== LaTeX Table ===\n")
    print(latex_table)
    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8") as f:
            f.write(md_table + "\n")
        print(f"Markdown table saved to: {args.md_out}")
    if args.latex_out:
        with open(args.latex_out, "w", encoding="utf-8") as f:
            f.write(latex_table + "\n")
        print(f"LaTeX table saved to: {args.latex_out}")


if __name__ == "__main__":
    main()
