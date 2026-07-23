"""CLI: ehr-ai start | train | evaluate | explain | report | init"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from utils.config import PROJECT_ROOT, REPORTS_DIR


def _cmd_doctor(_: argparse.Namespace) -> int:
    from utils.config import MODEL_PATH

    ok = True
    print(f"Project root: {PROJECT_ROOT}")
    demo = PROJECT_ROOT / "data" / "raw" / "ehr_data.csv"
    synth = PROJECT_ROOT / "data" / "raw" / "paper_synthetic_cohort.csv"
    print(f"  demo CSV: {'OK' if demo.is_file() else 'MISSING'} ({demo})")
    print(f"  synthetic CSV: {'OK' if synth.is_file() else 'MISSING'}")
    print(f"  tasks/: {'OK' if (PROJECT_ROOT / 'tasks').is_dir() else 'MISSING'}")
    print(f"  LIMITATIONS.md: {'OK' if (PROJECT_ROOT / 'LIMITATIONS.md').is_file() else 'MISSING'}")
    print(f"  model.pkl: {'present' if Path(MODEL_PATH).is_file() else 'not trained yet'}")
    docker = shutil.which("docker")
    print(f"  docker: {'OK' if docker else 'not found (native uvicorn path OK)'}")
    if not demo.is_file() and not synth.is_file():
        ok = False
    print("")
    print("Next: ehr-ai start   # or: docker compose up --build")
    print("UI → http://127.0.0.1:8080")
    return 0 if ok else 1


def _cmd_init(_: argparse.Namespace) -> int:
    uploads = PROJECT_ROOT / "data" / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Workspace ready under {PROJECT_ROOT}")
    print("Next: ehr-ai start   # or: docker compose up --build")
    return 0


def _cmd_start(args: argparse.Namespace) -> int:
    compose = shutil.which("docker")
    if compose:
        cmd = ["docker", "compose", "up", "--build"]
        if args.detach:
            cmd.append("-d")
        print("Starting stack via docker compose…")
        print("UI → http://127.0.0.1:8080  API → http://127.0.0.1:8000/docs")
        return subprocess.call(cmd, cwd=str(PROJECT_ROOT))
    print(
        "Docker not found. Start locally:\n"
        "  PYTHONPATH=. uvicorn api.main:app --reload --port 8000\n"
        "  cd web && npm start\n"
        "Or install Docker Desktop and re-run: ehr-ai start",
        file=sys.stderr,
    )
    return 1


def _cmd_train(args: argparse.Namespace) -> int:
    from openhealth.api import train
    from openhealth.task_spec import load_task

    kwargs = {}
    data = args.data
    if args.task:
        spec = load_task(args.task)
        params = spec.to_train_params(data)
        data = params.pop("data_path")
        kwargs.update({k: v for k, v in params.items() if k != "task_id"})
        if kwargs.get("windows_days"):
            kwargs["windows_days"] = tuple(kwargs["windows_days"])
    if args.model:
        kwargs["model_kind"] = args.model
    if args.format:
        kwargs["data_format"] = args.format
    train(data_path=data, out=args.out, **kwargs)
    print("Training complete.")
    return 0


def _cmd_evaluate(_: argparse.Namespace) -> int:
    from openhealth.api import evaluate
    import json

    ev = evaluate()
    print(json.dumps(ev.get("metrics") or ev, indent=2, default=str))
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    from openhealth.api import explain

    path = explain(out=args.out)
    print(f"SHAP written → {path}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from api.data_io import build_results_zip

    out = Path(args.out or (REPORTS_DIR / "ehr_risk_results_pack.zip"))
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(build_results_zip())
    print(f"Results pack → {out}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from openhealth.compare import compare_models
    from openhealth.task_spec import load_task

    kwargs: dict = {"data_path": args.data, "calibrate": args.calibrate}
    if args.task:
        spec = load_task(args.task)
        params = spec.to_train_params(args.data)
        kwargs["data_path"] = params["data_path"]
        kwargs.update(
            {
                "data_format": params["data_format"],
                "windows_days": params.get("windows_days"),
                "window_days": params.get("window_days", 180),
                "horizon_days": params.get("horizon_days"),
                "index_strategy": params.get("index_strategy", "last_event"),
                "index_time_col": params.get("index_time_col"),
                "feature_inclusive": params.get("feature_inclusive", True),
                "label_col": params.get("label_col"),
                "split_by_patient": params.get("split_by_patient", True),
                "temporal_split": params.get("temporal_split", False),
                "calibrate": params.get("calibrate", args.calibrate),
            }
        )
    summary = compare_models(**kwargs)
    print(f"Selected: {summary.get('selected_model')}")
    for row in summary.get("comparison") or []:
        mark = "*" if row.get("selected") else " "
        print(f" {mark} {row['model']}: ROC-AUC={row.get('roc_auc')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ehr-ai",
        description="OpenHealth EHR research CLI — train, compare, explain, start workbench",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="Create uploads/reports dirs")
    s.set_defaults(func=_cmd_init)

    s = sub.add_parser("doctor", help="Check demo data, Docker, LIMITATIONS")
    s.set_defaults(func=_cmd_doctor)

    s = sub.add_parser("start", help="Start Docker workbench (API + Angular)")
    s.add_argument("-d", "--detach", action="store_true")
    s.set_defaults(func=_cmd_start)

    s = sub.add_parser("train", help="Train a model (optional --task YAML)")
    s.add_argument("--task", help="Task id or path (e.g. diabetes or tasks/diabetes.yaml)")
    s.add_argument("--data", default=None, help="CSV path (overrides task suggested_path)")
    s.add_argument("--model", default=None)
    s.add_argument("--format", default=None, choices=["longitudinal", "tabular"])
    s.add_argument("--out", default=None)
    s.set_defaults(func=_cmd_train)

    s = sub.add_parser("evaluate", help="Print latest evaluation metrics")
    s.set_defaults(func=_cmd_evaluate)

    s = sub.add_parser("explain", help="Generate SHAP summary")
    s.add_argument("--out", default=None)
    s.set_defaults(func=_cmd_explain)

    s = sub.add_parser("report", help="Write downloadable results ZIP")
    s.add_argument("--out", default=None)
    s.set_defaults(func=_cmd_report)

    s = sub.add_parser("compare", help="Train classical models and rank by ROC-AUC")
    s.add_argument("--task", default=None)
    s.add_argument("--data", default="data/raw/ehr_data.csv")
    s.add_argument("--calibrate", action="store_true")
    s.set_defaults(func=_cmd_compare)

    args = p.parse_args(argv)
    if args.command == "train" and not args.data and not args.task:
        p.error("train requires --data and/or --task")
    if args.command == "train" and args.task and not args.data:
        from openhealth.task_spec import load_task

        args.data = load_task(args.task).suggested_path
        if not args.data:
            p.error("task has no suggested_path; pass --data")
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
