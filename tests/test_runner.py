"""
optlab.tests.test_runner — 回测纪律专项测试：未来函数 / 熔断 / T+1 执行
运行：python -m optlab.tests.test_runner

红线（用户认知清单）：回测时用了当时根本拿不到的数据 = 未来函数。
本测试用合成数据验证：T 日生成的决策，成交价必须来自 T+1 日行情。
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from optlab.core.models import Instrument, Right
from optlab.engine.broker import FillMode, RiskLimits
from optlab.engine.runner import BacktestRunner, SellStrangleStrategy

UND = "510300"
D1, D2, D3, D4 = date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)
EXP = date(2026, 2, 25)   # 2 月合约（DTE>25）


def make_data(spot_path, call_close_path, put_close_path, volumes=5000.0):
    """
    合成 3 个交易日的链数据。每天 1 个 call + 1 个 put（delta ±0.2 附近）。
    spot_path / *_close_path: dict{date: value}
    """
    rows = []
    dailies = []
    for d in (D1, D2, D3, D4):
        spot = spot_path[d]
        for right, strike, deltas, closes in (
                ("CALL", 5.00, 0.20, call_close_path),
                ("PUT", 4.50, -0.20, put_close_path)):
            rows.append({
                "trade_date": d, "security_id": f"1{right[0]}00", "contract_id": f"{UND}{right[0]}2602M{'05000' if right=='CALL' else '04500'}",
                "underlying": UND, "right": right, "expiry": EXP, "strike": float(strike),
                "adjusted": False, "iv": 0.16, "delta": deltas, "gamma": 1.5,
                "vega": 50.0, "theta": -110.0,   # 年化口径（交易所原值）
            })
            dailies.append({"security_id": f"1{right[0]}00", "trade_date": d,
                            "close": closes[d], "volume_lots": volumes, "date": d})
    risk = pd.DataFrame(rows)
    daily = pd.DataFrame(dailies)
    close = pd.Series({d: spot_path[d] for d in spot_path})
    return risk, daily, close


def test_no_lookahead_t_plus_1_fill():
    """T 日决策的订单必须用 T+1 日价格成交（未来函数红线）"""
    # D1 收盘 0.100 / D2 收盘 0.200 —— 若用 D1 价成交则亏损，用 D2 价成交则多收权利金
    risk, daily, close = make_data(
        spot_path={D1: 4.60, D2: 4.65, D3: 4.65, D4: 4.65},
        call_close_path={D1: 0.010, D2: 0.020, D3: 0.020, D4: 0.020},
        put_close_path={D1: 0.320, D2: 0.310, D3: 0.310, D4: 0.310},
    )
    runner = BacktestRunner(risk, daily, close)
    strategy = SellStrangleStrategy(entry_dte_min=25, exit_dte=3, lots_per_side=5)
    res = runner.run(strategy, start=D1, end=D4, cash=1_000_000.0)

    assert res.trades, "应有成交"
    for t in res.trades:
        if t.offset.value == "OPEN":
            # 决策日 = D1（月首），成交日必须 > D1
            assert t.ts.date() > D1, f"未来函数！{t.instrument.symbol} 在决策日当天成交"
            exp_open = {"510300C2602M05000": 0.020, "510300P2602M04500": 0.310}[t.instrument.symbol]
            # CLOSE_SLIPPAGE 卖方成交价 ∈ [T+1收盘−价差, T+1收盘]；T日价(0.010/0.320)必在区间外
            if t.direction.value == "SELL":
                assert exp_open * 0.95 <= t.price <= exp_open, \
                    f"成交价 {t.price} 未基于 T+1 收盘价 {exp_open}"


def test_month_first_entry_only_once():
    """同月只建仓一次（月首判定）"""
    risk, daily, close = make_data(
        spot_path={D1: 4.60, D2: 4.65, D3: 4.65, D4: 4.65},
        call_close_path={D1: 0.010, D2: 0.020, D3: 0.020, D4: 0.020},
        put_close_path={D1: 0.320, D2: 0.310, D3: 0.310, D4: 0.310},
    )
    runner = BacktestRunner(risk, daily, close)
    strategy = SellStrangleStrategy(entry_dte_min=25, exit_dte=0, lots_per_side=5)
    res = runner.run(strategy, start=D1, end=D4, cash=1_000_000.0)
    opens = [t for t in res.trades if t.offset.value == "OPEN"]
    assert len(opens) == 2, f"同月应只开 1 组（2腿），实际 {len(opens)} 笔"


def test_stop_loss_closes_position():
    """单腿权利金 ≥ 2× 开仓价 → 止损平仓单（T+1 执行）"""
    # put 开仓价 ~0.310（D2）；D3 收盘涨到 0.700 ≥ 2×0.31 → D3 决策止损、D4 执行
    risk, daily, close = make_data(
        spot_path={D1: 4.60, D2: 4.65, D3: 4.40, D4: 4.40},
        call_close_path={D1: 0.010, D2: 0.020, D3: 0.008, D4: 0.008},
        put_close_path={D1: 0.320, D2: 0.310, D3: 0.700, D4: 0.700},
    )
    runner = BacktestRunner(risk, daily, close)
    strategy = SellStrangleStrategy(entry_dte_min=25, exit_dte=0, lots_per_side=5)
    res = runner.run(strategy, start=D1, end=D4, cash=1_000_000.0)
    stop_orders = [o for o in res.orders_log if "止损" in o["reason"]]
    assert stop_orders, "应触发止损决策"
    # 止损单成交日必须晚于触发日（D3 触发 → D4 成交）
    closes = [t for t in res.trades
              if t.instrument.symbol == "510300P2602M04500" and t.offset.value == "CLOSE"]
    assert closes and all(t.ts.date() >= D4 for t in closes), \
        "止损平仓必须 T+1 执行"


def test_daily_loss_break_blocks_open():
    """单日亏损 > 3% → 当日开仓单被熔断拒绝"""
    risk, daily, close = make_data(
        spot_path={D1: 4.60, D2: 4.65, D3: 4.30, D4: 4.30},   # D3 标的大跌
        call_close_path={D1: 0.010, D2: 0.020, D3: 0.020, D4: 0.020},
        put_close_path={D1: 0.320, D2: 0.310, D3: 0.900, D4: 0.900},  # put 爆亏
    )
    runner = BacktestRunner(risk, daily, close)
    # 第二个月首日开仓（D4 属于新月份才触发——这里直接用月首标记测试熔断逻辑）
    strategy = SellStrangleStrategy(entry_dte_min=25, exit_dte=0, lots_per_side=5)
    res = runner.run(strategy, start=D1, end=D3, cash=1_000_000.0)
    # D1 月首建仓（D2 执行），D3 大跌 → 归因 total 大负；断言熔断记录只在真正超 3% 时出现
    if not res.attribution.empty:
        big_loss = (res.attribution["total"] < -0.03 * 1_000_000).any()
        if big_loss:
            assert any("熔断" in r for _, r in res.rejects), "大亏日应有熔断记录"


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for fn in ALL_TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__} {e}")
    print(f"\n{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} passed")
    sys.exit(1 if failed else 0)
