import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Tuple


METRIC_RE = re.compile(r"mse:([0-9eE\.\+\-]+), mae:([0-9eE\.\+\-]+)")
RMSE_RE = re.compile(r"rmse:([0-9eE\.\+\-]+), mape:([0-9eE\.\+\-]+), mspe:([0-9eE\.\+\-]+)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-grained single-dataset ablation for ED-SRA and EA-RevIN in TimeBridge-main."
    )
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--root_path", type=str, default="./dataset/Solar/", help="Dataset root path.")
    parser.add_argument("--data_path", type=str, default="solar_AL.txt", help="Data file name.")
    parser.add_argument("--data", type=str, default="Solar", help="Dataset type argument for run.py.")
    parser.add_argument("--dataset_tag", type=str, default="Solar", help="Tag in output table/log filenames.")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--seq_len", type=int, default=720)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=336)
    parser.add_argument("--enc_in", type=int, default=137)
    parser.add_argument("--period", type=int, default=48)
    parser.add_argument("--num_p", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--ia_layers", type=int, default=1)
    parser.add_argument("--pd_layers", type=int, default=1)
    parser.add_argument("--ca_layers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./_logs/finegrained_ablation",
        help="Directory to store run logs and tables.",
    )
    return parser


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def parse_metrics(text: str) -> Tuple[float, float, float]:
    mse, mae, rmse = float("nan"), float("nan"), float("nan")
    mm = METRIC_RE.findall(text)
    rr = RMSE_RE.findall(text)
    if mm:
        mse, mae = map(float, mm[-1])
    if rr:
        rmse = float(rr[-1][0])
    return mse, mae, rmse


def run_one(args: argparse.Namespace, exp_name: str, flags: List[str], log_path: str) -> Dict[str, float]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    cmd = [
        sys.executable,
        "-u",
        "run.py",
        "--is_training", "1",
        "--model_id", exp_name,
        "--model", "TimeBridge",
        "--data", args.data,
        "--root_path", args.root_path,
        "--data_path", args.data_path,
        "--features", args.features,
        "--seq_len", str(args.seq_len),
        "--label_len", str(args.label_len),
        "--pred_len", str(args.pred_len),
        "--enc_in", str(args.enc_in),
        "--d_model", str(args.d_model),
        "--d_ff", str(args.d_ff),
        "--n_heads", str(args.n_heads),
        "--ia_layers", str(args.ia_layers),
        "--pd_layers", str(args.pd_layers),
        "--ca_layers", str(args.ca_layers),
        "--period", str(args.period),
        "--num_p", str(args.num_p),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--train_epochs", str(args.train_epochs),
        "--patience", str(args.patience),
        "--alpha", str(args.alpha),
        "--des", "FG_ABL",
        "--itr", str(args.itr),
    ] + flags

    print(f"\n[RUN] {exp_name}")
    print("CMD:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)

    mse, mae, rmse = parse_metrics(proc.stdout)
    status = "ok" if proc.returncode == 0 else f"fail({proc.returncode})"
    print(f"[{status}] mse={mse:.6f}, mae={mae:.6f}, rmse={rmse:.6f}, log={log_path}")
    return {"mse": mse, "mae": mae, "rmse": rmse, "returncode": proc.returncode}


def format_metric(x: float) -> str:
    return "nan" if x != x else f"{x:.4f}"


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    line1 = "| " + " | ".join(headers) + " |"
    line2 = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line1, line2] + body)


def write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ck(x: bool) -> str:
    return "✓" if x else "×"


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.log_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_md = os.path.join(args.log_dir, f"finegrained_ablation_{args.dataset_tag}_pl{args.pred_len}_{ts}.md")
    csv_out = os.path.join(args.log_dir, f"finegrained_ablation_{args.dataset_tag}_pl{args.pred_len}_{ts}.csv")

    experiments = [
        {
            "name": "Original TimeBridge",
            "flags": [],
            "ea": False, "ea_dyn": False, "ed": False, "ed_gate": False, "ed_mix": False, "n_decay": False
        },
        {
            "name": "EA-RevIN (vanilla)",
            "flags": ["--use_ea_revin", "--revin_affine", "--revin_dyn_bound", "0.0", "--revin_entropy_gate", "0.55"],
            "ea": True, "ea_dyn": False, "ed": False, "ed_gate": False, "ed_mix": False, "n_decay": False
        },
        {
            "name": "EA-RevIN (dynamic)",
            "flags": ["--use_ea_revin", "--revin_affine", "--revin_dyn_bound", "0.05", "--revin_entropy_gate", "0.55"],
            "ea": True, "ea_dyn": True, "ed": False, "ed_gate": False, "ed_mix": False, "n_decay": False
        },
        {
            "name": "ED-SRA (residual-only proxy)",
            "flags": ["--use_ed_sra", "--ed_sra_no_threshold_gate"],
            "ea": False, "ea_dyn": False, "ed": True, "ed_gate": False, "ed_mix": True, "n_decay": True
        },
        {
            "name": "ED-SRA (gated-only)",
            "flags": ["--use_ed_sra", "--ed_sra_no_residual_mix"],
            "ea": False, "ea_dyn": False, "ed": True, "ed_gate": True, "ed_mix": False, "n_decay": True
        },
        {
            "name": "ED-SRA (full default)",
            "flags": ["--use_ed_sra"],
            "ea": False, "ea_dyn": False, "ed": True, "ed_gate": True, "ed_mix": True, "n_decay": True
        },
        {
            "name": "EA-RevIN(dynamic) + ED-SRA(full)",
            "flags": ["--use_ea_revin", "--revin_affine", "--revin_dyn_bound", "0.05", "--revin_entropy_gate", "0.55", "--use_ed_sra"],
            "ea": True, "ea_dyn": True, "ed": True, "ed_gate": True, "ed_mix": True, "n_decay": True
        },
        {
            "name": "EA-RevIN + ED-SRA (no N-decay)",
            "flags": ["--use_ea_revin", "--revin_affine", "--revin_dyn_bound", "0.05", "--revin_entropy_gate", "0.55",
                      "--use_ed_sra", "--ed_sra_no_n_decay"],
            "ea": True, "ea_dyn": True, "ed": True, "ed_gate": True, "ed_mix": True, "n_decay": False
        },
        {
            "name": "EA-RevIN + ED-SRA (no stability)",
            "flags": ["--use_ea_revin", "--revin_affine", "--revin_dyn_bound", "0.05", "--revin_entropy_gate", "0.55",
                      "--use_ed_sra", "--ed_sra_no_stability"],
            "ea": True, "ea_dyn": True, "ed": True, "ed_gate": True, "ed_mix": True, "n_decay": True
        },
        {
            "name": "EA-RevIN + ED-SRA (no reliability)",
            "flags": ["--use_ea_revin", "--revin_affine", "--revin_dyn_bound", "0.05", "--revin_entropy_gate", "0.55",
                      "--use_ed_sra", "--ed_sra_no_reliability"],
            "ea": True, "ea_dyn": True, "ed": True, "ed_gate": True, "ed_mix": True, "n_decay": True
        },
    ]

    csv_rows: List[Dict[str, str]] = []
    md_rows: List[List[str]] = []
    for i, cfg in enumerate(experiments):
        exp_name = f"{args.dataset_tag}_pl{args.pred_len}_fg{i:02d}"
        log_path = os.path.join(args.log_dir, f"{exp_name}.log")
        result = run_one(args, exp_name, cfg["flags"], log_path)
        csv_rows.append(
            {
                "experiment": cfg["name"],
                "ea_revin": str(cfg["ea"]),
                "ea_dynamic": str(cfg["ea_dyn"]),
                "ed_sra": str(cfg["ed"]),
                "ed_threshold_gate": str(cfg["ed_gate"]),
                "ed_residual_mix": str(cfg["ed_mix"]),
                "ed_n_decay": str(cfg["n_decay"]),
                "mse": f"{result['mse']:.6f}" if result["mse"] == result["mse"] else "",
                "mae": f"{result['mae']:.6f}" if result["mae"] == result["mae"] else "",
                "rmse": f"{result['rmse']:.6f}" if result["rmse"] == result["rmse"] else "",
                "returncode": str(int(result["returncode"])),
                "log_path": log_path,
            }
        )
        md_rows.append(
            [
                cfg["name"],
                ck(cfg["ea"]),
                ck(cfg["ea_dyn"]),
                ck(cfg["ed"]),
                ck(cfg["ed_gate"]),
                ck(cfg["ed_mix"]),
                ck(cfg["n_decay"]),
                format_metric(result["mse"]),
                format_metric(result["mae"]),
                format_metric(result["rmse"]),
            ]
        )

    fieldnames = [
        "experiment",
        "ea_revin",
        "ea_dynamic",
        "ed_sra",
        "ed_threshold_gate",
        "ed_residual_mix",
        "ed_n_decay",
        "mse",
        "mae",
        "rmse",
        "returncode",
        "log_path",
    ]
    write_csv(csv_out, csv_rows, fieldnames)

    table = markdown_table(
        ["Experiment Version", "EA", "EA-Dyn", "ED-SRA", "ED-Gate", "ED-Mix", "N-Decay", "MSE", "MAE", "RMSE"],
        md_rows,
    )
    md = []
    md.append(f"# Fine-grained Ablation ({args.dataset_tag}, pred_len={args.pred_len})")
    md.append("")
    md.append("> Residual-only / gated-only are defined on current TimeBridge-main ED-SRA implementation.")
    md.append("")
    md.append(table)
    md.append("")
    md.append(f"- Raw CSV: `{csv_out}`")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("\n===== Fine-grained ablation summary saved =====")
    print(f"Markdown: {summary_md}")
    print(f"CSV:      {csv_out}")


if __name__ == "__main__":
    main()
