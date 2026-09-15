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
        description="ED-SRA hyperparameter sensitivity on a single dataset (TimeBridge-main)."
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
        "--sigmoid_alphas",
        type=float,
        nargs="+",
        default=[4, 8, 10, 20, 100, 200, 500, 1000],
        help="Sensitivity sweep on ED-SRA sigmoid(alpha) sharpness.",
    )
    parser.add_argument(
        "--threshold_scales",
        type=float,
        nargs="+",
        default=[0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="Sensitivity sweep on ED-SRA dual-threshold scaling factor.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./_logs/EDSRA_sensitivity",
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


def run_one(args: argparse.Namespace, exp_name: str, extra_flags: List[str], log_path: str) -> Dict[str, float]:
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
        "--des", "EDSRA_SEN",
        "--itr", str(args.itr),
    ] + extra_flags

    print(f"\n[RUN] {exp_name}")
    print("CMD:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)

    mse, mae, rmse = parse_metrics(proc.stdout)
    status = "ok" if proc.returncode == 0 else f"fail({proc.returncode})"
    print(f"[{status}] mse={mse:.6f}, mae={mae:.6f}, rmse={rmse:.6f}, log={log_path}")
    return {"mse": mse, "mae": mae, "rmse": rmse, "returncode": proc.returncode}


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    line1 = "| " + " | ".join(headers) + " |"
    line2 = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line1, line2] + body)


def format_metric(x: float) -> str:
    return "nan" if x != x else f"{x:.4f}"


def write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.log_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_md = os.path.join(args.log_dir, f"sensitivity_summary_{args.dataset_tag}_pl{args.pred_len}_{ts}.md")
    csv_sigmoid = os.path.join(args.log_dir, f"sensitivity_sigmoid_alpha_{args.dataset_tag}_pl{args.pred_len}_{ts}.csv")
    csv_gamma = os.path.join(args.log_dir, f"sensitivity_threshold_scale_{args.dataset_tag}_pl{args.pred_len}_{ts}.csv")

    sigmoid_rows = []
    for s in args.sigmoid_alphas:
        exp_name = f"{args.dataset_tag}_pl{args.pred_len}_sigmoid{s}"
        log_path = os.path.join(args.log_dir, f"{exp_name}.log")
        flags = [
            "--use_ed_sra",
            "--ed_sra_sigmoid_alpha", str(s),
        ]
        res = run_one(args, exp_name, flags, log_path)
        sigmoid_rows.append(
            {
                "sigmoid_alpha": str(s),
                "mse": f"{res['mse']:.6f}" if res["mse"] == res["mse"] else "",
                "mae": f"{res['mae']:.6f}" if res["mae"] == res["mae"] else "",
                "rmse": f"{res['rmse']:.6f}" if res["rmse"] == res["rmse"] else "",
                "returncode": str(int(res["returncode"])),
            }
        )

    threshold_rows = []
    for g in args.threshold_scales:
        exp_name = f"{args.dataset_tag}_pl{args.pred_len}_gamma{g}"
        log_path = os.path.join(args.log_dir, f"{exp_name}.log")
        flags = [
            "--use_ed_sra",
            "--ed_sra_threshold_scale", str(g),
        ]
        res = run_one(args, exp_name, flags, log_path)
        threshold_rows.append(
            {
                "threshold_scale_gamma": str(g),
                "mse": f"{res['mse']:.6f}" if res["mse"] == res["mse"] else "",
                "mae": f"{res['mae']:.6f}" if res["mae"] == res["mae"] else "",
                "rmse": f"{res['rmse']:.6f}" if res["rmse"] == res["rmse"] else "",
                "returncode": str(int(res["returncode"])),
            }
        )

    write_csv(csv_sigmoid, sigmoid_rows, ["sigmoid_alpha", "mse", "mae", "rmse", "returncode"])
    write_csv(csv_gamma, threshold_rows, ["threshold_scale_gamma", "mse", "mae", "rmse", "returncode"])

    t1 = markdown_table(
        ["sigmoid(alpha)", "MSE", "MAE", "RMSE"],
        [[r["sigmoid_alpha"], format_metric(float(r["mse"]) if r["mse"] else float("nan")),
          format_metric(float(r["mae"]) if r["mae"] else float("nan")),
          format_metric(float(r["rmse"]) if r["rmse"] else float("nan"))] for r in sigmoid_rows],
    )
    t2 = markdown_table(
        ["gamma(threshold scale)", "MSE", "MAE", "RMSE"],
        [[r["threshold_scale_gamma"], format_metric(float(r["mse"]) if r["mse"] else float("nan")),
          format_metric(float(r["mae"]) if r["mae"] else float("nan")),
          format_metric(float(r["rmse"]) if r["rmse"] else float("nan"))] for r in threshold_rows],
    )

    md = []
    md.append(f"# ED-SRA Sensitivity ({args.dataset_tag}, pred_len={args.pred_len})")
    md.append("")
    md.append("## Table A: Gating sharpness sensitivity")
    md.append("")
    md.append(t1)
    md.append("")
    md.append("## Table B: Dual-threshold scaling factor sensitivity")
    md.append("")
    md.append(t2)
    md.append("")
    md.append(f"- Raw CSV (sigmoid alpha): `{csv_sigmoid}`")
    md.append(f"- Raw CSV (threshold scale): `{csv_gamma}`")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("\n===== Sensitivity summary saved =====")
    print(f"Markdown: {summary_md}")
    print(f"CSV A:    {csv_sigmoid}")
    print(f"CSV B:    {csv_gamma}")


if __name__ == "__main__":
    main()
