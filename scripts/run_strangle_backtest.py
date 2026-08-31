"""
thetalab.scripts.run_strangle_backtest — P2 验收：510300 卖出宽跨式回测（约1年）

产出：绩效指标、希腊字母损益归因、成交与拒单统计、净值曲线 CSV
运行：python -m thetalab.scripts.run_strangle_backtest
"""
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from thetalab.data.provider import ParquetStore, SseOptionProvider
from thetalab.engine.metrics import fee_share, performance_metrics, trade_stats
from thetalab.engine.runner import BacktestRunner, SellStrangleStrategy

UNDERLYING = "510300"


def load(store: ParquetStore, provider: SseOptionProvider):
    risk = store.read("risk_indicators")
    risk = risk[risk["underlying"] == UNDERLYING].copy()
    daily = pd.read_parquet(store.root / "contract_daily" / "all.parquet")
    days = sorted(risk["trade_date"].unique())
    udl = provider.underlying_daily(UNDERLYING, days[0], days[-1])
    close = pd.Series(udl["close"].astype(float).values, index=udl["date"].values)
    return risk, daily, close


def main():
    store = ParquetStore("thetalab_data/store")
    provider = SseOptionProvider(min_interval=0.3)
    risk, daily, close = load(store, provider)
    days = sorted(risk["trade_date"].unique())
    print(f"数据: {len(risk):,} 行风险指标, {len(daily):,} 行合约日线, "
          f"{len(days)} 个交易日 ({days[0]} ~ {days[-1]})")

    runner = BacktestRunner(risk, daily, close)
    strategy = SellStrangleStrategy(delta_target=0.20, entry_dte_min=25,
                                    exit_dte=7, stop_multiple=2.0, lots_per_side=10)
    res = runner.run(strategy, start=days[0], end=days[-1], cash=1_000_000.0)

    print("\n=== 绩效指标（卖出宽跨式 Δ20/20, 每边10张, 100万本金）===")
    for k, v in res.metrics.items():
        print(f"  {k}: {v}")

    if not res.attribution.empty:
        cum = res.attribution[["delta", "gamma", "vega", "theta", "residual", "total"]].sum()
        print("\n=== 希腊字母损益归因（全期累计, 元）===")
        for k, v in cum.items():
            print(f"  {k}: {v:,.0f}")

    print(f"\n成交笔数: {len(res.trades)}, 拒单: {len(res.rejects)}")
    from collections import Counter
    reasons = Counter(r.split(":")[0].split("，")[0] for _, r in res.rejects)
    for r, n in reasons.most_common(5):
        print(f"  拒单[{r}]: {n}")
    print("\n决策日志（前 8 条）:")
    for o in res.orders_log[:8]:
        print(f"  {o['decision_day']} {o['symbol']} {o['action']} x{o['qty']} — {o['reason']}")

    eq = pd.DataFrame(res.equity_curve, columns=["date", "equity"])
    out = store.root.parent / "strangle_equity_curve.csv"
    eq.to_csv(out, index=False)
    res.attribution.to_csv(store.root.parent / "strangle_attribution.csv", index=False)
    print(f"\n净值曲线: {out}")
    print(f"期初 1,000,000 → 期末 {eq['equity'].iloc[-1]:,.0f}")


if __name__ == "__main__":
    main()
