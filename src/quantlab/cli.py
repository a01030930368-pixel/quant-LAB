"""Command line front end (tests, automation, development). Uses the same services as the GUI."""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path


def _lab(args):
    from quantlab.services.lab import Lab

    return Lab(args.workspace)


def _print(obj) -> None:
    print(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="quantlab", description="QUANT LAB research app (command line)")
    ap.add_argument("--workspace", "-w", help="work folder (default: Documents/QuantLab)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest", help="run the full flow on synthetic data in a temporary folder")
    k = sub.add_parser("set-key", help="store the Tiingo API key in the OS credential manager")
    k.add_argument("--provider", default="tiingo")
    f = sub.add_parser("fetch", help="download daily prices from Tiingo")
    f.add_argument("symbols", nargs="+")
    f.add_argument("--start", required=True)
    f.add_argument("--end")
    c = sub.add_parser("import-csv", help="import one symbol from a CSV file")
    c.add_argument("symbol")
    c.add_argument("path")
    d = sub.add_parser("dataset", help="build a dataset from the latest data of each symbol")
    d.add_argument("symbols", nargs="+")
    sub.add_parser("datasets", help="list datasets")
    i = sub.add_parser("import-strategy", help="import a .qls.yaml strategy spec")
    i.add_argument("path")
    sub.add_parser("strategies", help="list strategies")
    b = sub.add_parser("backtest", help="exploratory backtest on TRAIN+VALIDATION data")
    b.add_argument("strategy_hash")
    b.add_argument("dataset")
    b.add_argument("--param", action="append", default=[], help="name=value")
    s = sub.add_parser("search", help="parameter search on TRAIN+VALIDATION data")
    s.add_argument("strategy_hash")
    s.add_argument("dataset")
    s.add_argument("--workers", type=int, default=0)
    s.add_argument("--resume", help="batch id to resume")
    cf = sub.add_parser("confirm", help="fix parameters and policy for the final evaluation")
    cf.add_argument("strategy_hash")
    cf.add_argument("--param", action="append", default=[])
    cf.add_argument("--note", default="")
    fe = sub.add_parser("final", help="final evaluation (reads the holdout once)")
    fe.add_argument("strategy_hash")
    fe.add_argument("dataset")
    fe.add_argument("--yes", action="store_true", help="confirm you understand the holdout is read once")
    r = sub.add_parser("report", help="write the research report for an experiment")
    r.add_argument("experiment_id")
    e = sub.add_parser("export", help="export a result packet (ZIP) for outside analysis")
    e.add_argument("experiment_id")
    rp = sub.add_parser("reproduce", help="re-run an exploratory experiment and compare")
    rp.add_argument("experiment_id")
    sub.add_parser("history", help="experiment history")

    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        from quantlab.services.selftest import run_selftest

        res = run_selftest()
        _print(res)
        return 0 if res.get("ok") else 1

    lab = _lab(args)
    try:
        if args.cmd == "set-key":
            lab.set_api_key(getpass.getpass("API key (input hidden): ").strip(), args.provider)
            print("stored in the OS credential manager")
        elif args.cmd == "fetch":
            for rec in lab.fetch(args.symbols, args.start, args.end):
                print(f"{rec.symbol}: sha256 {rec.raw_sha256[:12]}, fatal {rec.qc.fatal}, warnings {rec.qc.warnings}")
        elif args.cmd == "import-csv":
            rec = lab.import_csv(args.symbol, args.path)
            print(f"{rec.symbol}: fatal {rec.qc.fatal}, warnings {rec.qc.warnings}")
        elif args.cmd == "dataset":
            v, man = lab.build_dataset(args.symbols)
            _print({"dataset_version": v, "qc_fatal": man["qc_fatal"], "qc_warnings": man["qc_warnings"]})
        elif args.cmd == "datasets":
            _print([{k: m[k] for k in ("dataset_version", "symbols", "qc_fatal")} for m in lab.data.datasets()])
        elif args.cmd == "import-strategy":
            res = lab.import_strategy(args.path)
            _print({"strategy": f"{res.strategy_id}@{res.version}", "hash": res.strategy_hash, "state": res.state,
                    "new": res.new, "problems": res.problems})
        elif args.cmd == "strategies":
            _print(lab.registry.strategies())
        elif args.cmd == "backtest":
            out = lab.backtest(args.strategy_hash, args.dataset, _params(args.param))
            _print({"experiment_id": out.experiment_id, **out.metrics})
        elif args.cmd == "search":
            _print(lab.search(args.strategy_hash, args.dataset, workers=args.workers, resume=args.resume))
        elif args.cmd == "confirm":
            _print(lab.confirm_candidate(args.strategy_hash, _params(args.param), args.note))
        elif args.cmd == "final":
            if not args.yes:
                print("The holdout of this strategy family is evaluated once. A strategy changed after seeing the "
                      "result cannot use this period as independent evidence. Re-run with --yes to proceed.")
                return 2
            out = lab.final_evaluation(args.strategy_hash, args.dataset)
            _print({"experiment_id": out.experiment_id, "verdict": out.overall,
                     "gates": [(g.gate, g.verdict, g.reasons) for g in out.gates]})
        elif args.cmd == "report":
            print(lab.write_report(args.experiment_id))
        elif args.cmd == "export":
            print(lab.export_packet(args.experiment_id))
        elif args.cmd == "reproduce":
            _print(lab.reproduce(args.experiment_id))
        elif args.cmd == "history":
            _print(lab.registry.history())
    finally:
        lab.close()
    return 0


def _params(items: list[str]) -> dict:
    out = {}
    for it in items:
        name, _, val = it.partition("=")
        try:
            out[name] = json.loads(val)
        except json.JSONDecodeError:
            out[name] = val
    return out


if __name__ == "__main__":
    sys.exit(main())
