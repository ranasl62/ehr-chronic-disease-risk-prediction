#!/usr/bin/env python3
"""
Export paper verification figures + sync Table 2 markdown from reports/paper/.

  PYTHONPATH=. python scripts/export_paper_figures.py \\
    --reports-dir reports/paper --figures-dir research-paper/figures/export
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from training.train import run_training


def _load_results(reports: Path) -> pd.DataFrame:
    path = reports / "results_table.csv"
    df = pd.read_csv(path)
    df = df[df["error"].isna()] if "error" in df.columns else df
    return df


def fig4_rocauc(df: pd.DataFrame, out: Path) -> None:
    order = df.sort_values("roc_auc").reset_index(drop=True)
    colors = ["#1f4e79" if c else "#8faadc" for c in order["calibrated"].astype(bool)]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    y = np.arange(len(order))
    ax.barh(y, order["roc_auc"], color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(order["tag"], fontsize=8)
    ax.set_xlabel("Hold-out ROC-AUC (illustrative, synthetic)")
    ax.set_xlim(0.5, max(0.85, float(order["roc_auc"].max()) + 0.05))
    ax.axvline(0.5, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_title("Figure 4. Verification matrix ROC-AUC")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig5_ece(df: pd.DataFrame, out: Path) -> None:
    multi = df[df["windows"].astype(str).str.contains("7")].copy()
    pairs = []
    for model in ["logreg", "random_forest", "xgboost"]:
        raw = multi[(multi["model"] == model) & (~multi["calibrated"].astype(bool))]
        cal = multi[(multi["model"] == model) & (multi["calibrated"].astype(bool))]
        if len(raw) and len(cal):
            pairs.append(
                {
                    "model": model,
                    "ece_raw": float(raw.iloc[0]["ece"]),
                    "ece_cal": float(cal.iloc[0]["ece"]),
                }
            )
    if not pairs:
        return
    p = pd.DataFrame(pairs)
    x = np.arange(len(p))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(x - w / 2, p["ece_raw"], w, label="Raw", color="#8faadc")
    ax.bar(x + w / 2, p["ece_cal"], w, label="Isotonic", color="#1f4e79")
    ax.set_xticks(x)
    ax.set_xticklabels(p["model"])
    ax.set_ylabel("ECE (illustrative)")
    ax.set_title("Figure 5. ECE before vs after calibration (multi-window)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig6_auc_brier(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for _, row in df.iterrows():
        marker = "o" if row["calibrated"] else "s"
        ax.scatter(
            row["roc_auc"],
            row["brier"],
            marker=marker,
            s=55,
            c="#1f4e79" if row["calibrated"] else "#8faadc",
            edgecolors="white",
            linewidths=0.4,
        )
        ax.annotate(row["tag"].replace("_", "\n"), (row["roc_auc"], row["brier"]), fontsize=5.5, alpha=0.75)
    ax.set_xlabel("ROC-AUC (↑ better)")
    ax.set_ylabel("Brier score (↓ better)")
    ax.set_title("Figure 6. Discrimination vs Brier (synthetic hold-out)")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig7_reliability(data: Path, reports: Path, export: Path) -> None:
    for calibrate, tag, fname in [
        (False, "xgboost_raw_w7-30-180", "fig7a_reliability_xgb_raw.png"),
        (True, "xgboost_cal_w7-30-180", "fig7b_reliability_xgb_cal.png"),
    ]:
        model_path = reports / f"model_{tag}.pkl"
        _, X_test, y_test, artifact = run_training(
            data_path=data,
            model_path=model_path if model_path.exists() else reports / f"_tmp_{tag}.pkl",
            model_kind="xgboost",
            data_format="longitudinal",
            windows_days=(7, 30, 180),
            window_days=180,
            calibrate=calibrate,
            skip_calibration_plot=True,
            split_by_patient=True,
            horizon_days=365,
            index_strategy="column",
            index_time_col="index_time",
            random_state=42,
            bootstrap_samples=None,
        )
        # Prefer existing artifact if present to avoid retrain drift; else use just-trained.
        model = artifact["model"]
        if model_path.exists():
            import joblib

            try:
                model = joblib.load(model_path)
            except Exception:
                pass
        prob = model.predict_proba(X_test)[:, 1]
        y = np.asarray(y_test).astype(int)
        fig, ax = plt.subplots(figsize=(5, 5))
        if len(np.unique(y)) < 2:
            ax.text(0.1, 0.5, "Single class")
            ax.set_axis_off()
        else:
            pt, pp = calibration_curve(y, prob, n_bins=8, strategy="uniform")
            ax.plot(pp, pt, marker="o", color="#1f4e79", label="Model")
            ax.plot([0, 1], [0, 1], "--", color="#888888", label="Ideal")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.legend(loc="lower right")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ax.set_title(
            "Reliability — XGBoost multi-window "
            + ("isotonic" if calibrate else "raw")
            + " (synthetic)"
        )
        fig.tight_layout()
        fig.savefig(export / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)


def fig9_leakage(e: dict, out: Path) -> None:
    labels = ["Integrity-aware", "Injected post-index\nglucose"]
    vals = [
        e["integrity_aware_logreg_roc_auc"],
        e["with_injected_post_index_glucose_feature_roc_auc"],
    ]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bars = ax.bar(labels, vals, color=["#1f4e79", "#c55a11"], width=0.55)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Hold-out ROC-AUC (logistic regression)")
    ax.set_title("Figure 9. Controlled leakage injection (synthetic)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=10)
    audit_txt = (
        f"Audit: truncated pass ({e['audit_truncated_feature_events_after_index']} post-index) · "
        f"raw fail ({e['audit_raw_as_features_events_after_index']} post-index)"
    )
    ax.text(0.5, -0.18, audit_txt, transform=ax.transAxes, ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_f_naive(f: dict, out: Path) -> None:
    seeds = [r["seed"] for r in f["per_seed"]]
    i_auc = [r["integrity_roc_auc"] for r in f["per_seed"]]
    n_auc = [r["naive_no_truncate_roc_auc"] for r in f["per_seed"]]
    x = np.arange(len(seeds))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(x - w / 2, i_auc, w, label="Integrity (truncate)", color="#1f4e79")
    ax.bar(x + w / 2, n_auc, w, label="Naive (no truncate)", color="#c55a11")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Hold-out seed")
    ax.set_ylabel("ROC-AUC (logistic regression)")
    ax.set_ylim(0.5, 1.05)
    ax.legend()
    ax.set_title(
        f"Experiment F: naive vs integrity (mean Δ={f['delta_auc_mean']:.3f}, "
        f"audit fail events={f['audit_naive_feature_events_after_index']})"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def sync_table2(df: pd.DataFrame, tables: Path, n_patients: int, top_ci: dict | None) -> None:
    n_test = int(df["n_test"].iloc[0]) if len(df) else 0
    lines = [
        "# Table 2. Illustrative hold-out metrics on the public synthetic cohort",
        "",
        (
            f"**Source.** Regenerated `reports/paper/results_table.csv` (paper experiment entry point) "
            f"on the public synthetic cohort (\(N={n_patients}\) patients; patient-disjoint 80%/20% split; "
            f"horizon \(H=365\) days; explicit `index_time`; generator seed 42)."
        ),
        "",
        (
            f"**Label.** **ILLUSTRATIVE — software verification only.** These numbers are **not** "
            f"clinical performance estimates. Hold-out size \(n_{{\\text{{test}}}}={n_test}\)."
        ),
        "",
        (
            "| Tag | Model | Calibrated | Windows (days) | "
            "\(n_{\\text{test}}\) | ROC-AUC | PR-AUC | Brier | ECE |"
        ),
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    best_tag = df.loc[df["roc_auc"].idxmax(), "tag"] if len(df) else ""
    for _, r in df.iterrows():
        tag = r["tag"]
        bold = tag == best_tag
        cells = [
            tag,
            r["model"],
            "Yes" if r["calibrated"] in (True, "True", 1) else "No",
            str(r["windows"]).replace(",", ", "),
            str(int(r["n_test"])),
            f"{float(r['roc_auc']):.3f}",
            f"{float(r['pr_auc']):.3f}",
            f"{float(r['brier']):.3f}",
            f"{float(r['ece']):.3f}",
        ]
        if bold:
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    roc_min = float(df["roc_auc"].min())
    roc_max = float(df["roc_auc"].max())
    top = df.loc[df["roc_auc"].idxmax()]
    ci_txt = ""
    if top_ci:
        ci_txt = (
            f" Bootstrap 95% CI for top cell ROC-AUC: "
            f"[{top_ci.get('roc_auc_ci_low'):.3f}, {top_ci.get('roc_auc_ci_high'):.3f}] "
            f"({int(top_ci.get('n_bootstrap', 0))} resamples)."
        )
    lines += [
        "",
        (
            f"**Summary.** ROC-AUC range across {len(df)} runs ≈ "
            f"**{roc_min:.2f}–{roc_max:.2f}**. Best illustrative cell: `{top['tag']}` "
            f"(ROC-AUC {float(top['roc_auc']):.3f}; PR-AUC {float(top['pr_auc']):.3f}; "
            f"Brier {float(top['brier']):.3f}; ECE {float(top['ece']):.3f})."
            f"{ci_txt}"
        ),
        "",
    ]
    (tables / "table2_illustrative_metrics.md").write_text("\n".join(lines) + "\n")


def sync_table4(e: dict, tables: Path) -> None:
    lines = [
        "# Table 4. Controlled leakage injection (Experiment E, synthetic)",
        "",
        "**Source.** `tables/experiment_e_leakage_contrast.json` (regenerable).",
        "",
        "**Disclaimer.** Illustrative software verification only. Not clinical performance.",
        "",
        "| Setting | Post-index feature events (audit) | Audit passed | Hold-out ROC-AUC (logistic regression) |",
        "|---|---:|:---:|---:|",
        f"| Integrity-aware multi-window matrix | {e['audit_truncated_feature_events_after_index']} | Yes | {e['integrity_aware_logreg_roc_auc']:.3f} |",
        f"| Same matrix + injected post-index mean glucose | — (engineered leak feature) | — | {e['with_injected_post_index_glucose_feature_roc_auc']:.3f} |",
        f"| Audit using truncated feature table | {e['audit_truncated_feature_events_after_index']} | Yes | — |",
        f"| Audit using raw events as feature table | {e['audit_raw_as_features_events_after_index']} | No | — |",
        "",
        (
            f"**Reading.** Injected post-index glucose moved illustrative ROC-AUC by "
            f"≈{e['delta_auc']:.2f} (N={e.get('n_patients')}, n_test={e.get('n_test')})."
        ),
        "",
    ]
    (tables / "table4_leakage_injection.md").write_text("\n".join(lines) + "\n")


def sync_table5(f: dict, tables: Path) -> None:
    lines = [
        "# Table 5. Naive (no truncate) vs integrity path (Experiment F, synthetic)",
        "",
        "**Source.** `tables/experiment_f_naive_vs_integrity.json`.",
        "",
        "**Disclaimer.** Illustrative multi-seed software verification. Not clinical performance.",
        "",
        "| Path | Audit passed | Post-index feature events | Mean ROC-AUC ± SD (logreg, 5 seeds) |",
        "|---|:---:|---:|---:|",
        (
            f"| Integrity (truncate to index) | {'Yes' if f['audit_integrity_passed'] else 'No'} | "
            f"{f['audit_integrity_feature_events_after_index']} | "
            f"{f['integrity_roc_auc_mean']:.3f} ± {f['integrity_roc_auc_std']:.3f} |"
        ),
        (
            f"| Naive (no truncate) | {'Yes' if f['audit_naive_passed'] else 'No'} | "
            f"{f['audit_naive_feature_events_after_index']} | "
            f"{f['naive_roc_auc_mean']:.3f} ± {f['naive_roc_auc_std']:.3f} |"
        ),
        "",
        (
            f"**Mean Δ (naive − integrity)** = {f['delta_auc_mean']:.3f} ± {f['delta_auc_std']:.3f}. "
            f"N={f.get('n_patients')}; prevalence={f.get('label_prevalence')}."
        ),
        "",
        "| Seed | Integrity ROC-AUC | Naive ROC-AUC | Δ |",
        "|---:|---:|---:|---:|",
    ]
    for r in f["per_seed"]:
        lines.append(
            f"| {r['seed']} | {r['integrity_roc_auc']:.3f} | "
            f"{r['naive_no_truncate_roc_auc']:.3f} | "
            f"{r['delta_auc_naive_minus_integrity']:.3f} |"
        )
    lines.append("")
    (tables / "table5_naive_vs_integrity.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", type=Path, default=Path("reports/paper"))
    ap.add_argument("--figures-dir", type=Path, default=Path("research-paper/figures/export"))
    ap.add_argument("--tables-dir", type=Path, default=Path("research-paper/tables"))
    ap.add_argument("--data", type=Path, default=Path("data/raw/paper_synthetic_cohort.csv"))
    args = ap.parse_args()
    reports = args.reports_dir
    export = args.figures_dir
    tables = args.tables_dir
    export.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    df = _load_results(reports)
    n_patients = int(pd.read_csv(args.data)["patient_id"].nunique())

    fig4_rocauc(df, export / "fig4_rocauc_matrix.png")
    fig5_ece(df, export / "fig5_ece_calibration.png")
    fig6_auc_brier(df, export / "fig6_auc_vs_brier.png")
    fig7_reliability(args.data, reports, export)

    e_path = reports / "experiment_e_leakage_contrast.json"
    if not e_path.exists():
        e_path = tables / "experiment_e_leakage_contrast.json"
    e = json.loads(e_path.read_text())
    fig9_leakage(e, export / "fig9_leakage_injection.png")
    sync_table4(e, tables)

    f_path = reports / "experiment_f_naive_vs_integrity.json"
    if f_path.exists():
        f = json.loads(f_path.read_text())
        fig_f_naive(f, export / "fig10_naive_vs_integrity.png")
        sync_table5(f, tables)

    # Copy SHAP if present
    shap_src = reports / "shap_summary.png"
    if shap_src.exists():
        import shutil

        shutil.copy2(shap_src, export / "fig8_shap_summary.png")

    # Top-cell bootstrap CI from summary
    top_ci = None
    summary_path = reports / "experiment_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        best = None
        for run in summary.get("runs", []):
            if "error" in run:
                continue
            auc = (run.get("metrics") or {}).get("roc_auc")
            if auc is None:
                continue
            if best is None or auc > best[0]:
                best = (auc, run.get("bootstrap_roc_auc"))
        if best:
            top_ci = best[1]

    sync_table2(df, tables, n_patients, top_ci)
    print(f"Exported figures to {export}")
    print(f"Synced tables under {tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
