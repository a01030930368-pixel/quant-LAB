"""Research report (protocol 제28조, 14 items) and the result packet for outside analysis
(03 v2.0.0 §4.4). Reports are written in Korean for the user; numbers come straight
from the registry record, never recomputed with different settings.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import markdown as md
import pandas as pd
import yaml

from quantlab.analysis import charts
from quantlab.core import app_info
from quantlab.core.hashing import canonical_json

SECTIONS = ["1. 전략 ID와 버전", "2. 연구 가설과 경제적 논리", "3. 거래 규칙과 파라미터", "4. 데이터 출처와 품질",
            "5. 백테스트 및 검증 구간", "6. 수수료·슬리피지·체결 가정", "7. 기간별 수익률과 주요 위험지표",
            "8. 표본 외 성과", "9. 통계 검정과 다중검정 고려사항", "10. 강건성 및 비용 민감도",
            "11. 벤치마크 비교", "12. 주요 실패 가능성과 한계", "13. 최종 판정과 그 근거", "14. 실험 및 코드 재현 정보"]

STANDING_LIMITS = [
    "과거 성과는 미래 수익을 보장하지 않는다(헌장 제22조).",
    "공개 시장 데이터의 과거 구간은 연구자(사람·AI)가 이미 알고 있을 수 있어, 최종 평가 구간도 완전한 미지 데이터가 아니다. "
    "등록 이후 새로 쌓이는 데이터(전진 검증)가 가장 강한 증거다.",
    "세전·USD 기준이다. 한국 거주자 해외주식 양도소득세와 환전 비용은 반영하지 않았다.",
    "배당은 배당락일에 현금으로 입금되는 것으로 단순화했다(실제 지급일과 다름).",
]


def _pct(x) -> str:
    return "—" if x is None or x != x else f"{x * 100:.2f}%"


def _num(x, nd=3) -> str:
    return "—" if x is None or x != x else f"{x:.{nd}f}"


def _metrics_table(rows: dict[str, dict]) -> str:
    keys = [("cagr", "CAGR", _pct), ("total_return", "총수익", _pct), ("ann_vol", "연변동성", _pct),
            ("sharpe", "샤프", _num), ("sortino", "소르티노", _num), ("max_drawdown", "MDD", _pct),
            ("max_drawdown_sessions", "최장 낙폭 기간(거래일)", lambda v: str(v)), ("calmar", "칼마", _num),
            ("n_trades", "거래 수", lambda v: str(v)), ("turnover", "연 회전율(배)", _num),
            ("top5pct_sessions_share", "상위 5% 거래일 수익 기여", _pct), ("psr_vs_0", "PSR(샤프>0 확률)", _num)]
    head = "| 지표 | " + " | ".join(rows) + " |\n|---|" + "---|" * len(rows) + "\n"
    body = "".join(f"| {label} | " + " | ".join(fmt(r.get(k)) for r in rows.values()) + " |\n" for k, label, fmt in keys)
    return head + body


def _trade_lines(label: str, ts: dict | None) -> list[str]:
    """Trade-level checks (descriptive, not gate criteria)."""
    if not ts:
        return []
    if not ts.get("n"):
        return [f"- 거래 단위 점검({label}): 거래 없음"]
    return [f"- 거래 단위 점검({label}, 판정 기준 아님): {ts['n']}건 · 비용 전 평균 {_pct(ts['mean_gross'])}, "
            f"중앙값 {_pct(ts['median_gross'])}, 상위 10건 뺀 평균 {_pct(ts.get('mean_gross_ex_top10'))}, "
            f"비용 후 평균 {_pct(ts['mean_net'])}, 비용 후 승률 {_pct(ts['win_rate_net'])}, "
            f"같은 날 진입을 묶은 t값 {_num(ts.get('t_day_clustered'), 2)} ({ts['entry_days']}일), "
            f"평균 보유 {_num(ts['mean_sessions'], 1)}거래일"]


def build_report(lab, experiment_id: str) -> tuple[str, dict]:
    reg = lab.registry
    e = reg.experiment(experiment_id)
    m = e["metrics"]
    strat = reg.strategy(e["strategy_hash"]) if e.get("strategy_hash") else None
    spec = (strat or {}).get("body", {}).get("spec", {}) if strat else {}
    policy = reg.get_policy(e["validation_policy_version"]) if e.get("validation_policy_version") else {}
    dataset = next((d for d in lab.data.datasets() if d["dataset_version"] == e["dataset_version"]), None)
    verdicts = reg.verdicts(experiment_id)
    formal = e["kind"] == "formal"
    lines = [f"# 연구 보고서 — {e['strategy_id']}@{e['strategy_version']}",
             f"실험 {experiment_id} · {'정식 검증' if formal else '탐색 실험(TRAIN+VALIDATION만 사용)'} · "
             f"작성 {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC", ""]
    S = {}
    S[0] = [f"- 전략: `{e['strategy_id']}` 버전 `{e['strategy_version']}`",
            f"- 전략 해시: `{e.get('strategy_hash') or '—'}` · 출처: {e.get('strategy_origin') or '—'}",
            f"- 계보: 부모 {strat.get('parent') if strat else '—'}, 보류 결과를 본 뒤 파생: "
            f"{'예' if strat and strat['derived_after_holdout_view'] else '아니오'}"]
    hy = spec.get("hypothesis", {})
    S[1] = [f"- 분류: {hy.get('class', '—')}", f"- 경제적 논리: {str(hy.get('economic_rationale', '—')).strip()}",
            "- 예상 위험: " + ", ".join(map(str, hy.get("expected_risks", []) or ["—"])),
            f"- 반증 조건: {str(hy.get('falsification', '—')).strip()}"]
    S[2] = ["```yaml", yaml.safe_dump({"rules": spec.get("rules"), "execution": spec.get("execution"),
                                       "parameters_used": e["config"].get("params") or e["config"].get("strategy", {}).get("params")},
                                      allow_unicode=True, sort_keys=False).strip(), "```"]
    if dataset:
        S[3] = [f"- 데이터셋 `{dataset['dataset_version'][:16]}` (처리 버전 {dataset['processing_version']})"] + [
            f"- {s}: {v['source'].get('provider')} · {v['first']}~{v['last']} · {v['rows']}행 · 원본 sha256 `{v['raw_sha256'][:12]}`"
            for s, v in dataset["symbols"].items()] + [
            f"- 품질 검사 치명 항목: {len(dataset['qc_fatal'])}건" + "".join(f"\n  - {x}" for x in dataset["qc_fatal"]),
            f"- 품질 검사 경고: {len(dataset['qc_warnings'])}건" + "".join(f"\n  - {x}" for x in dataset["qc_warnings"][:10])]
    else:
        S[3] = ["- 데이터셋 정보를 작업 폴더에서 찾을 수 없음"]
    sp = policy.get("splits", {})
    S[4] = [f"- TRAIN {sp.get('train')}, VALIDATION {sp.get('validation')}, FINAL HOLDOUT {sp.get('holdout')}",
            f"- 이 실험이 사용한 구간: {'TRAIN~HOLDOUT (보류 구간 1회 열람)' if formal and 'holdout' in m else m.get('window', 'TRAIN+VALIDATION')}"]
    c = policy.get("costs", {})
    S[5] = [f"- 편도 수수료 {c.get('commission_bps')}bp, 슬리피지 {c.get('slippage_bps')}bp, 고정 수수료 {c.get('fixed_fee')}"
            + (f", 매도 거래세 {c['sell_tax_bps']}bp" if c.get("sell_tax_bps") else ""),
            "- 신호: 장 마감 가격으로 판단 → 다음 거래일 시가 체결(같은 종가 체결 없음), 매도 먼저 후 매수, 정수 주식, 현금 한도 내",
            f"- 판정 설정: `{policy.get('policy_id')}` v{policy.get('version')} (해시 `{(e.get('validation_policy_version') or '')[:12]}`)"]
    perf_src = m.get("holdout") or m.get("dev") or m
    S[6] = [_metrics_table({"전략": perf_src.get("strategy", {}), "벤치마크": perf_src.get("benchmark", {})})]
    if formal and "dev_train" in m:
        S[6].append("TRAIN / VALIDATION 구간별 샤프 차이: "
                    f"{_num(m['dev_train']['sharpe_diff'])} / {_num(m['dev_validation']['sharpe_diff'])}")
    if formal and "holdout" in m:
        ho = m["holdout"]
        S[7] = [f"- 보류 구간 샤프 차이 {_num(ho['sharpe_diff'])}, MDD 비율 {_num(ho['mdd_ratio'])}, "
                f"포지션 변경 {ho['position_changes']}회, 이전 열람 {ho['prior_holdout_accesses']}회"]
    else:
        S[7] = ["- 이 실험은 보류 구간을 열지 않았다(탐색 실험이거나 Gate 2에서 중단)."]
    ho = m.get("holdout", {})
    S[8] = [f"- 블록 부트스트랩(블록 {ho['bootstrap']['block']}일, {ho['bootstrap']['n_boot']}회, 시드 {ho['bootstrap']['seed']}): "
            f"관측 샤프 차이 {_num(ho['bootstrap']['observed'])}, 단측 p = {_num(ho['bootstrap']['p_value'])}, "
            f"95% 구간 {[round(x, 3) for x in ho['bootstrap']['ci95']]}" if ho.get("bootstrap") else "- 통계 검정 없음(보류 구간 미열람)",
            f"- 이 전략 계열의 누적 시행 수: {lab.registry.count_trials_for_family(e['strategy_id'])} (DSR 보정에 사용)",
            f"- DSR/PSR: {_num(ho.get('dsr'))}" if ho else ""]
    g4 = []
    if "gate4_cost_stress" in m:
        g4.append(f"- 비용 {m['gate4_cost_stress']['cost_multiplier']}배: 샤프 차이 {_num(m['gate4_cost_stress']['sharpe_diff'])}")
    for k, v in (m.get("gate4_neighbors") or {}).items():
        g4.append(f"- 인접 파라미터 {k}: 샤프 차이 {_num(v)}")
    S[9] = g4 or ["- Gate 4(일부)는 Gate 3을 통과한 경우에만 실행된다. 이번 실험에서는 실행되지 않았다."]
    S[9] += _trade_lines("보류 구간", (m.get("holdout") or {}).get("trades"))
    S[9] += _trade_lines("TRAIN+VALIDATION", (m.get("dev") or {}).get("trades") or m.get("trades"))
    wb = (m.get("dev") or {}).get("without_best_year") or m.get("without_best_year")
    if wb:
        S[9].append(f"- 가장 좋은 해({wb['best_year']}, {_pct(wb['best_year_return'])})를 뺀 CAGR: "
                    f"{_pct(wb['cagr_ex_best_year'])} (전체 {_pct(wb['cagr_all'])}, TRAIN+VALIDATION)")
    S[10] = [f"- 벤치마크: {policy.get('benchmark_symbol')} 매수 후 보유(같은 엔진·같은 비용)",
             f"- 샤프 차이 {_num(perf_src.get('sharpe_diff'))}, 초과 CAGR {_pct(perf_src.get('excess_cagr'))}"]
    S[11] = [f"- {x}" for x in STANDING_LIMITS] + [f"- 전략 고유 위험: {', '.join(map(str, hy.get('expected_risks', [])))}"]
    if verdicts:
        S[12] = [f"- **{g['gate']}: {g['verdict']}** — " + "; ".join(g["reasons"]) for g in verdicts]
    else:
        S[12] = ["- 탐색 실험에는 판정이 없다. 판정은 후보 확정 후 최종 평가에서만 내려진다."]
    S[13] = [f"- 앱 버전 {e['app_version']}, 코드 해시 `{e['app_code_hash'][:16]}`",
             f"- 데이터셋 `{e['dataset_version'][:16]}`, 설정 버전 `{e['configuration_version'][:16]}`, 난수 시드 {e['random_seed']}",
             f"- 결과 해시 `{e['result_hash'][:16]}`; 탐색 실험은 `quantlab reproduce {experiment_id}`로 재실행해 비교한다."]
    for i, title in enumerate(SECTIONS):
        lines += [f"## {title}", ""] + [x for x in S[i] if x] + [""]
    return "\n".join(lines), {"experiment": e, "verdicts": verdicts, "policy": policy, "dataset": dataset, "spec": spec}


def write_report(lab, experiment_id: str) -> Path:
    text, ctx = build_report(lab, experiment_id)
    out = lab.ws.path("reports", experiment_id)
    out.mkdir(parents=True, exist_ok=True)
    imgs = []
    try:
        eq, _ = lab.artifacts(experiment_id)
        hs = ctx["policy"].get("splits", {}).get("holdout", [None])[0] if ctx["experiment"]["kind"] == "formal" else None
        for name, fig in (("equity.png", charts.equity(eq, hs)), ("drawdown.png", charts.drawdown(eq)),
                          ("monthly.png", charts.monthly_returns(eq["strategy"]))):
            charts.save(fig, out / name)
            imgs.append(name)
    except FileNotFoundError:
        pass
    if imgs:
        text += "\n## 차트\n\n" + "\n".join(f"![{n}]({n})" for n in imgs) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    html = md.markdown(text, extensions=["tables", "fenced_code"])
    (out / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>QUANT LAB report</title>"
        "<style>body{font-family:system-ui,'Malgun Gothic',sans-serif;max-width:900px;margin:2em auto;padding:0 1em;line-height:1.5}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px;text-align:right}"
        "td:first-child,th:first-child{text-align:left}img{max-width:100%}pre{background:#f5f5f5;padding:8px}</style>"
        + html, encoding="utf-8")
    return out / "report.md"


def export_packet(lab, experiment_id: str) -> Path:
    """ZIP for outside analysis (e.g. Claude). Excludes API keys and raw market data."""
    reg = lab.registry
    e = reg.experiment(experiment_id)
    report_path = write_report(lab, experiment_id)
    strat = reg.strategy(e["strategy_hash"]) if e.get("strategy_hash") else None
    includes_holdout = e["kind"] == "formal" and "holdout" in e["metrics"]
    buf = io.BytesIO()
    contents = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        def add(name, data):
            z.writestr(name, data)
            contents.append(name)
        add("report.md", report_path.read_text(encoding="utf-8"))
        add("metrics.json", json.dumps(json.loads(canonical_json(e["metrics"])), indent=1, ensure_ascii=False))  # NaN as "NaN": valid JSON
        add("experiment.json", json.dumps({k: v for k, v in e.items() if k != "metrics"}, indent=1, ensure_ascii=False, default=str))
        add("verdicts.json", json.dumps(reg.verdicts(experiment_id), indent=1, ensure_ascii=False))
        if e.get("validation_policy_version"):
            add("policy.yaml", yaml.safe_dump(reg.get_policy(e["validation_policy_version"]), allow_unicode=True, sort_keys=False))
        if strat:
            add("strategy.qls.yaml", yaml.safe_dump(strat["body"]["spec"], allow_unicode=True, sort_keys=False))
            if strat["body"].get("code"):
                add("strategy.py", strat["body"]["code"])
            fam = [x for x in reg.experiments(strat["strategy_id"])]
            add("trials_summary.json", json.dumps({"family": strat["strategy_id"], "experiments": len(fam),
                                                   "family_trials": reg.count_trials_for_family(strat["strategy_id"])}))
        try:
            eq, trades = lab.artifacts(experiment_id)
            add("daily_equity.csv", eq.to_csv())
            add("trades.csv", trades.to_csv(index=False))
        except FileNotFoundError:
            pass
        for img in sorted(report_path.parent.glob("*.png")):
            z.write(img, f"charts/{img.name}")
            contents.append(f"charts/{img.name}")
        add("app.json", json.dumps({"app_version": app_info.app_version(), "app_code_hash": app_info.app_code_hash(),
                                    "includes_holdout_results": includes_holdout,
                                    "notice": "Strategies derived from this packet must declare "
                                              "derived_after_holdout_view: true" if includes_holdout else ""}))
    data = buf.getvalue()
    from quantlab.data.providers import get_api_key

    key = get_api_key("tiingo")
    if key and key.encode() in data:
        raise RuntimeError("refusing to export: an API key string was found in the packet")
    out = lab.ws.path("exports", f"{experiment_id}.zip")
    out.write_bytes(data)
    reg.record_export(e["strategy_id"], includes_holdout, str(out), contents)
    return out
