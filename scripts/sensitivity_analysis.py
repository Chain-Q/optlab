"""
thetalab.scripts.sensitivity_analysis — 回测红线的量化体检（用户认知清单回应）

    1. 参数平原：delta_target × stop_multiple × exit_dte = 36 组网格回测
       （红线：参数≤3、参数平原越宽实盘存活率越高）
    2. 成本压力：手续费 5→15 元/张、滑点 ×1→×1.5（红线：成本被低估则实盘由盈转亏）
    3. 未来函数审计：每笔成交日必须晚于决策日
    4. 持仓 Greeks 化验单：平均 Delta 等值 / Vega / Theta / 保证金占用率
运行：python -m thetalab.scripts.sensitivity_analysis
"""
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

import thetalab.engine.broker as broker_mod
from thetalab.core.models import FeeRule
from thetalab.data.provider import ParquetStore, SseOptionProvider
from thetalab.engine.runner import BacktestRunner, SellStrangleStrategy

UNDERLYING = "510300"


def load():
    store = ParquetStore("thetalab_data/store")
    provider = SseOptionProvider(min_interval=0.3)
    risk = store.read("risk_indicators")
    risk = risk[risk["underlying"] == UNDERLYING].copy()
    daily = pd.read_parquet(store.root / "contract_daily" / "all.parquet")
    days = sorted(risk["trade_date"].unique())
    udl = provider.underlying_daily(UNDERLYING, days[0], days[-1])
    close = pd.Series(udl["close"].astype(float).values, index=udl["date"].values)
    return store, risk, daily, close, days[0], days[-1]


def audit_no_lookahead(res):
    """成交日必须晚于决策日（未来函数审计）"""
    bad = 0
    for o in res.orders_log:
        d_day = pd.Timestamp(o["decision_day"]).date()
        for t in res.trades:
            if t.instrument.symbol == o["symbol"] and t.ts.date() == d_day \
                    and t.offset.value == "OPEN":
                bad += 1
    return bad


def greeks_report(res):
    """持仓 Greeks 化验单：有持仓日的平均暴露"""
    held = [s for s in res.states if s.delta != 0.0 or s.vega != 0.0]
    if not held:
        return {}
    return {
        "持仓天数": len(held),
        "平均Delta等值(股)": round(sum(s.delta for s in held) / len(held)),
        "平均Vega(元/vol点)": round(sum(s.vega for s in held) / len(held), 1),
        "平均Theta(元/日)": round(sum(s.theta for s in held) / len(held), 1),
        "平均保证金占用率": f"{sum(s.margin_used / s.equity for s in held) / len(held):.1%}",
    }


def main():
    store, risk, daily, close, start, end = load()
    print(f"回测窗口: {start} ~ {end}\n")

    # ---------- 1. 参数平原 ----------
    grid = []
    for dt in (0.15, 0.20, 0.25, 0.30):
        for sm in (1.5, 2.0, 3.0):
            for xd in (5, 7, 10):
                strategy = SellStrangleStrategy(delta_target=dt, stop_multiple=sm,
                                                exit_dte=xd, lots_per_side=10)
                res = BacktestRunner(risk, daily, close).run(
                    strategy, start=start, end=end, cash=1_000_000.0)
                m = res.metrics
                grid.append({"delta": dt, "stop": sm, "exit_dte": xd,
                             "总收益%": round(m.get("total_return", 0) * 100, 2),
                             "夏普": m.get("sharpe"),
                             "回撤%": round(m.get("max_drawdown", 0) * 100, 2),
                             "胜率%": round((m.get("win_rate") or 0) * 100, 0)})
    g = pd.DataFrame(grid)
    print("=== 1. 参数平原（36 组：delta × stop × exit_dte）===")
    print(g.pivot_table(index=["delta", "stop"], columns="exit_dte",
                        values="总收益%").to_string())
    profitable = (g["总收益%"] > 0).mean()
    print(f"\n盈利组合占比: {profitable:.0%} | 收益范围 [{g['总收益%'].min():.2f}%, "
          f"{g['总收益%'].max():.2f}%] | 均值 {g['总收益%'].mean():.2f}% | "
          f"标准差 {g['总收益%'].std():.2f}")
    verdict = "平原宽（稳健）" if profitable >= 0.7 and g["总收益%"].min() > -5 \
        else "平原窄（对参数敏感，实盘存活率低）"
    print(f"参数平原判定: {verdict}")

    # ---------- 2. 成本压力 ----------
    print("\n=== 2. 成本压力测试（delta=0.20, stop=2.0, exit=7 基准参数）===")
    base = SellStrangleStrategy(delta_target=0.20, stop_multiple=2.0, exit_dte=7, lots_per_side=10)
    orig_spread = broker_mod.estimate_spread_pct
    rows = []
    for fee, slip_mult, label in (
            (5.0, 1.0, "基准: 5元/张 + 实测价差"),
            (15.0, 1.0, "手续费 15元/张"),
            (5.0, 1.5, "滑点 ×1.5"),
            (15.0, 1.5, "手续费15 + 滑点×1.5（最坏）")):
        broker_mod.estimate_spread_pct = (
            lambda p, m, d, t=0.0001, _k=slip_mult: min(1.0, _k * orig_spread(p, m, d, t)))
        res = BacktestRunner(risk, daily, close).run(
            base, start=start, end=end, cash=1_000_000.0,
            fee_rule=FeeRule(per_contract=fee))
        rows.append({"情形": label,
                     "总收益%": round(res.metrics.get("total_return", 0) * 100, 2),
                     "费用占比%": round((res.metrics.get("fee_share") or 0) * 100, 1)})
    broker_mod.estimate_spread_pct = orig_spread
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------- 3. 未来函数审计（基准参数回测）----------
    res0 = BacktestRunner(risk, daily, close).run(
        SellStrangleStrategy(delta_target=0.20, stop_multiple=2.0, exit_dte=7, lots_per_side=10),
        start=start, end=end, cash=1_000_000.0)
    bad = audit_no_lookahead(res0)
    print(f"\n=== 3. 未来函数审计 ===\n决策日当天即成交的笔数: {bad}（必须为 0）")

    # ---------- 4. 持仓 Greeks 化验单 ----------
    print("\n=== 4. 持仓 Greeks 化验单（红线：不要只看贵不贵）===")
    for k, v in greeks_report(res0).items():
        print(f"  {k}: {v}")

    g.to_csv(store.root.parent / "sensitivity_grid.csv", index=False)
    print(f"\n网格明细已存: thetalab_data/sensitivity_grid.csv")


if __name__ == "__main__":
    main()
