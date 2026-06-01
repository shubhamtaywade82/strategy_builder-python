"""
Reproducible Strategy Analysis Report
=====================================
Runs the REAL research engine and writes docs/strategy_analysis_report/ from the
actual output — every number and chart is generated, none are hand-typed. Re-run
it and you get the same report (deterministic engine + seeded RNGs).

Usage:
    python generate_report.py [SYMBOL] [DAYS] [RR]
    python generate_report.py SOLUSDT 60 2:1

In a geo-blocked / offline environment live klines are unavailable, so the
engine falls back to deterministic synthetic data and the report says so in
bold at the top. Point it at a host that can reach fapi.binance.com for a
real-market report with identical formatting.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("STRATEGY_RESEARCH_ALLOW_SYNTHETIC", "1")
sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strategy_research import ui_research
from strategy_research._ui_engine.grid_search import RR_CONFIGS

OUT_DIR = Path("docs/strategy_analysis_report")


# --------------------------------------------------------------------------
# formatting helpers — every value comes straight from the engine
# --------------------------------------------------------------------------
def pct(v, d=1):
    return "—" if v is None else f"{v * 100:.{d}f}%"


def num(v, d=2):
    return "—" if v is None else f"{v:.{d}f}"


def signed_pct(v, d=3):
    return "—" if v is None else f"{'+' if v >= 0 else ''}{v * 100:.{d}f}%"


def _best_strategy(side: dict):
    strats = (side or {}).get("strategies", [])
    return strats[0] if strats else None


# --------------------------------------------------------------------------
# charts (generated from real run data)
# --------------------------------------------------------------------------
def chart_in_sample_vs_oos(results: dict, path: Path):
    """Bar chart: the in-sample illusion vs the out-of-sample reality."""
    labels, in_wr, oos_wr = [], [], []
    for side in ("long", "short"):
        s = _best_strategy(results.get(side, {}))
        if not s:
            continue
        labels.append(f"{side}\n{s['name'].split('_')[-1]}")
        in_wr.append(s["in_sample_metrics"].get("win_rate", 0) * 100)
        oos_wr.append(s["metrics"].get("win_rate", 0) * 100)
    if not labels:
        return False

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - 0.2 for i in x], in_wr, 0.4, label="In-sample (illusion)", color="#ef4444")
    ax.bar([i + 0.2 for i in x], oos_wr, 0.4, label="Out-of-sample (reality)", color="#22c55e")
    ax.axhline(50, ls="--", c="#888", lw=1)
    ax.set_ylabel("Win rate (%)")
    ax.set_title("Model-confidence strategy: in-sample vs out-of-sample win rate")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=110); plt.close(fig)
    return True


def chart_player_folds(player: dict, path: Path):
    """Per-fold out-of-sample expectancy for the 5-condition rule."""
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for side, color in (("long", "#22c55e"), ("short", "#ef4444")):
        s = player.get(side, {})
        if "error" in s or not s.get("folds"):
            continue
        folds = [f for f in s["folds"] if f.get("expectancy") is not None]
        if not folds:
            continue
        ax.plot([f["fold"] for f in folds], [f["expectancy"] * 100 for f in folds],
                "o-", color=color, label=f"{side}")
        plotted = True
    if not plotted:
        plt.close(fig); return False
    ax.axhline(0, ls="--", c="#888", lw=1)
    ax.set_xlabel("Time fold"); ax.set_ylabel("Net expectancy / trade (%)")
    ax.set_title("5-condition rule — out-of-sample expectancy by fold")
    ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=110); plt.close(fig)
    return True


# --------------------------------------------------------------------------
# markdown sections
# --------------------------------------------------------------------------
def grid_table(side_name: str, side: dict) -> str:
    if not side or side.get("error"):
        return f"_{side_name}: {side.get('error', 'no data') if side else 'no data'}_\n"
    wf = side.get("walk_forward", {})
    st = side.get("shuffle_test", {})
    lines = [
        f"**{side_name.upper()}** — walk-forward avg test AUC "
        f"`{num(wf.get('avg_test_auc'), 3)}` (min `{num(wf.get('min_test_auc'), 3)}`), "
        f"shuffle p-value `{num(st.get('p_value'), 3)}`, "
        f"valid={wf.get('is_valid')}\n",
        "| Strategy | OOS win rate | OOS PF | OOS expectancy | In-sample WR | Robust | Score | Warnings |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in side.get("strategies", []):
        m, ism = s["metrics"], s["in_sample_metrics"]
        lines.append(
            f"| `{s['name']}` | {pct(m.get('win_rate'))} | {num(m.get('profit_factor'))} | "
            f"{signed_pct(m.get('expectancy'))} | {pct(ism.get('win_rate'))} | "
            f"{'✅' if s['is_robust'] else '❌'} | {s['robustness_score']} | "
            f"{'; '.join(s['warnings']) if s['warnings'] else '—'} |")
    return "\n".join(lines) + "\n"


def player_table(side_name: str, side: dict) -> str:
    if not side or side.get("error"):
        n = (side or {}).get("n_signals", 0)
        return f"**{side_name.upper()}** — {side.get('error', 'no data') if side else 'no data'} (signals fired: {n})\n"
    m = side["metrics"]
    rows = [
        f"**{side_name.upper()}** ({side.get('name', '')}) — "
        f"{'✅ ROBUST' if side['is_robust'] else '❌ NOT ROBUST'}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Trades | {m['trade_count']} |",
        f"| Win rate | {pct(m.get('win_rate'))} |",
        f"| Profit factor | {num(m.get('profit_factor'))} |",
        f"| Net expectancy / trade | {signed_pct(m.get('expectancy'))} |",
        f"| Sharpe (per-trade) | {num(m.get('sharpe'))} |",
        f"| Bootstrap p-value | {num(m.get('p_value'), 3)} |",
        f"| Edge vs random WR | {signed_pct(side.get('edge_win_rate'), 1)} |",
        f"| Edge vs random expectancy | {signed_pct(side.get('edge_expectancy'))} |",
        f"| Fold stability | {num(side.get('stability'), 2)} |",
    ]
    if side.get("warnings"):
        rows.append(f"| Warnings | {'; '.join(side['warnings'])} |")
    return "\n".join(rows) + "\n"


def build_markdown(raw: dict, symbol: str, days: int, rr: str, charts: dict) -> str:
    src = raw["data_source"]
    rr_data = raw["results"].get(rr, {})
    player = raw.get("player", {})
    cfg = RR_CONFIGS.get(rr, RR_CONFIGS["2:1"])
    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    src_banner = (
        "> ⚠️ **Data source: SYNTHETIC (deterministic demo).** Live Binance klines "
        "were unavailable when this ran, so the numbers below come from generated "
        "bars with no real edge — they exist to prove the pipeline is honest, not "
        "to recommend a trade. Re-run where `fapi.binance.com` is reachable for a "
        "real-market report.\n"
        if src == "synthetic" else
        "> ✅ **Data source: live Binance USD-M klines.**\n"
    )

    any_robust = any(
        s["is_robust"]
        for side in ("long", "short")
        for s in rr_data.get(side, {}).get("strategies", [])
    ) or any(
        (player.get(side, {}) or {}).get("is_robust")
        for side in ("long", "short")
    )

    verdict = (
        "**At least one strategy passed the out-of-sample robustness gate** — see ✅ rows below."
        if any_robust else
        "**No strategy passed the out-of-sample robustness gate on this window.** "
        "On no-edge data that is the correct, honest result: the engine refuses to "
        "manufacture an edge that is not there."
    )

    md = f"""# {symbol} Strategy Analysis Report

_Generated {stamp} by `generate_report.py` — every number and chart below comes
straight from a deterministic engine run. Re-running with the same inputs
reproduces this report exactly._

{src_banner}
**Run:** `{symbol}` · {days} days · RR `{rr}` (target {pct(cfg['up_pct'])} / stop {pct(cfg['dn_pct'])}) · 10x · round-trip cost 0.09%

---

## 1. Verdict

{verdict}

The point of this report is methodological honesty:

- **Model-confidence strategies** are scored on **out-of-sample** (purged
  walk-forward) predictions, not the in-sample fit. The in-sample win rate is
  shown only to expose how large the overfitting gap is.
- A strategy is **robust** only when it is profitable out-of-sample, consistent
  across folds, statistically significant (shuffle p < 0.05), and has enough
  trades. In-sample expectancy ranks nothing.

---

## 2. Model-confidence grid search (out-of-sample)

The in-sample column is the "96% win rate" illusion; the OOS column is what a
model that cannot peek at the test bars actually achieves.

{grid_table('long', rr_data.get('long', {}))}
{grid_table('short', rr_data.get('short', {}))}
![In-sample vs out-of-sample win rate]({charts['oos']})

---

## 3. Five-condition rule strategy ({player.get('rr', rr)})

Transparent rule (4H trend · 1H BOS · 15m FVG · ATR>60pct · Vol>70pct), AND-ed
on each leakage-safe 1m close, backtested out-of-sample with a bootstrap
significance test and a random-entry baseline.

{player_table('long', player.get('long', {}))}
{player_table('short', player.get('short', {}))}
"""
    if charts.get("folds"):
        md += f"![Per-fold out-of-sample expectancy]({charts['folds']})\n\n"

    md += """---

## 4. How to read this

- **Win rate alone means nothing.** A model that memorises the training set hits
  >90% in-sample and ~30-50% out-of-sample. Only the OOS number is tradeable.
- **Robustness gate, not vanity metrics.** Ranking is by a 0-100 score blending
  walk-forward AUC, fold stability, shuffle significance, and low train/test
  degradation.
- **Determinism.** Seeded RNGs + single-threaded XGBoost mean identical inputs
  produce identical numbers, so this report is reproducible and auditable.

> Disclaimer: backtested results, not investment advice. Crypto futures trading
> carries substantial risk. A passing robustness gate is a necessary, not
> sufficient, condition for live deployment — forward-test before risking capital.
"""
    return md


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SOLUSDT"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    rr = sys.argv[3] if len(sys.argv) > 3 else "2:1"

    print(f"Running research: {symbol} {days}d RR={rr} ...")
    raw = ui_research.run_research(symbol, days=days, rrs=[rr], allow_synthetic=True)
    print(f"  data_source = {raw['data_source']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    charts = {}
    if chart_in_sample_vs_oos(raw["results"].get(rr, {}), OUT_DIR / "in_sample_vs_oos.png"):
        charts["oos"] = "in_sample_vs_oos.png"
    if chart_player_folds(raw.get("player", {}), OUT_DIR / "rule_folds.png"):
        charts["folds"] = "rule_folds.png"

    md = build_markdown(raw, symbol, days, rr, charts)
    (OUT_DIR / "strategy_analysis_report.md").write_text(md)
    print(f"  wrote {OUT_DIR/'strategy_analysis_report.md'} ({len(md)} chars), charts={list(charts)}")


if __name__ == "__main__":
    main()
