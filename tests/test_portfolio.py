"""
thetalab.tests.test_portfolio — P2 批次B 组合/归因/绩效单测
运行：python -m thetalab.tests.test_portfolio
"""
import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thetalab.core.models import Account, Greeks, Instrument, Position, Right, Trade
from thetalab.engine.portfolio import (
    Portfolio, PortfolioState, max_drawdown_stats, pnl_attribution,
)
from thetalab.engine.metrics import performance_metrics, trade_stats

CALL = Instrument(symbol="C", underlying="510300", right=Right.CALL, strike=4.7,
                  expiry=date(2026, 9, 23), multiplier=10000)
PUT = Instrument(symbol="P", underlying="510300", right=Right.PUT, strike=4.4,
                 expiry=date(2026, 9, 23), multiplier=10000)


def test_greeks_aggregation_with_multiplier():
    """组合希腊 = Σ per-unit × 张数 × 合约单位"""
    a = Account()
    a.positions["C"] = Position(instrument=CALL, net_qty=-10)   # 卖 call
    a.positions["P"] = Position(instrument=PUT, net_qty=-10)    # 卖 put
    pf = Portfolio(a)
    gmap = {"C": Greeks(delta=0.30, gamma=0.5, vega=0.004, theta=-0.001, iv=0.16),
            "P": Greeks(delta=-0.25, gamma=0.4, vega=0.003, theta=-0.0008, iv=0.18)}
    g = pf.aggregate_greeks(gmap)
    # 空头腿 k=net_qty×mult=-100000：短call delta 贡献 -30000，短 put(Δ=-0.25) 贡献 +25000
    assert abs(g.delta - (-5000.0)) < 1e-6, g.delta
    # vega: 0.004×(-100000) + 0.003×(-100000) = -700 元/vol point
    assert abs(g.vega - (-700.0)) < 1e-6
    # theta: 空头收时间价值 → (-0.001)×(-100000) + (-0.0008)×(-100000) = +180
    assert abs(g.theta - 180.0) < 1e-6


def test_greeks_skip_nan_legs():
    a = Account()
    a.positions["C"] = Position(instrument=CALL, net_qty=-10)
    pf = Portfolio(a)
    g = pf.aggregate_greeks({"C": Greeks(delta=0.3, iv=float("nan"))})
    assert g.delta == 0.0  # 无有效 IV 的腿不计入


def test_update_mark_and_equity():
    a = Account(cash=100_000.0, initial_cash=100_000.0)
    a.positions["C"] = Position(instrument=CALL, net_qty=10, avg_open_price=0.05)
    pf = Portfolio(a)
    pf.update_mark({"C": 0.08})
    assert a.positions["C"].market_value == 0.08 * 10 * 10000
    assert a.equity == 100_000 + 8000


def test_margin_refresh_maintenance():
    a = Account(cash=200_000.0, initial_cash=200_000.0)
    a.positions["P"] = Position(instrument=PUT, net_qty=-10)
    pf = Portfolio(a)
    pf.margin_refresh({"P": 0.10}, spot_close=4.679)
    expect = None
    from thetalab.core.spec import calc_margin
    expect = calc_margin(PUT, option_price=0.10, spot_close=4.679,
                         is_short=True, is_call=False, maintenance=True) * 10
    assert abs(a.margin_used - expect) < 0.01


def test_max_drawdown_stats():
    eq = [100, 120, 90, 95, 130, 110]
    s = max_drawdown_stats(eq)
    assert abs(s["max_drawdown"] - (1 - 90 / 120)) < 1e-9   # 25%
    assert s["duration"] == 1
    assert abs(s["current_drawdown"] - (1 - 110 / 130)) < 1e-6  # 函数保留6位小数


def test_stress_matrix_shape_and_sign():
    a = Account()
    a.positions["C"] = Position(instrument=CALL, net_qty=10)  # 买 call
    pf = Portfolio(a)
    pf.snapshot(date(2026, 8, 28), spot=4.679, atm_iv=0.16,
                greeks_map={"C": Greeks(delta=0.4, gamma=1.0, vega=0.004, theta=-0.001, iv=0.16)})
    m = pf.stress_matrix(spot=4.679)
    assert m.shape[0] == 7 and "+1%" in m.index and "+5v" in m.columns
    # 买 call：标的上涨列应优于下跌列
    assert m.loc["+1%"].mean() > m.loc["-1%"].mean()


def test_pnl_attribution_exact():
    """给定已知 Greeks 与价格/波动变动，四项归因精确、残差=0"""
    prev = PortfolioState(day=date(2026, 8, 27), equity=1_000_000, spot=4.679,
                          atm_iv=0.16, delta=-50_000, gamma=2_000, vega=-700, theta=150)
    curr = PortfolioState(day=date(2026, 8, 28), equity=1_000_000, spot=4.679 + 0.04679,
                          atm_iv=0.155, delta=-50_000, gamma=2_000, vega=-700, theta=150)
    # 手工构造实际权益变化 = 四项之和 → 残差应为 0
    dS = 0.04679
    expect = (-50_000 * dS) + 0.5 * 2000 * dS ** 2 + (-700) * (-0.5) + 150
    curr.equity = prev.equity + expect
    attr = pnl_attribution(prev, curr)
    assert abs(attr["delta"] - (-50_000 * dS)) < 0.01
    assert abs(attr["gamma"] - 0.5 * 2000 * dS ** 2) < 0.01
    assert abs(attr["vega"] - 350.0) < 0.01
    assert abs(attr["theta"] - 150.0) < 1e-6
    assert abs(attr["residual"]) < 1e-3, attr


def test_pnl_attribution_per_leg():
    """逐腿路径：共同腿逐腿分解（vega 用逐腿 ΔIV）、新开腿归 trade、残差=0"""
    prev = PortfolioState(day=date(2026, 8, 27), equity=1_000_000, spot=4.679,
                          legs={"A": {"delta": -0.5, "gamma": 2.0, "vega": 0.005,
                                      "theta": 0.02, "iv": 0.16, "qm": -100000,
                                      "price": 0.10, "avg_open": 0.10}})
    curr = PortfolioState(day=date(2026, 8, 28), equity=1_000_000, spot=4.679 + 0.04,
                          legs={"A": {"delta": -0.5, "gamma": 2.0, "vega": 0.005,
                                      "theta": 0.02, "iv": 0.155, "qm": -100000,
                                      "price": 0.09, "avg_open": 0.10}})
    dS = 0.04
    expect = (-0.5 * dS * -100000) + 0.5 * 2.0 * dS * dS * -100000         + (0.155 - 0.16) * 100 * 0.005 * -100000 + 0.02 * -100000
    curr.equity = prev.equity + expect
    attr = pnl_attribution(prev, curr)
    assert "trade" in attr
    assert abs(attr["residual"]) < 0.01, attr
    # 新开腿归 trade
    curr2 = PortfolioState(day=curr.day, equity=1_000_000 + 500, spot=curr.spot,
                           legs={**curr.legs,
                                 "B": {"delta": 0.4, "gamma": 1.0, "vega": 0.004,
                                       "theta": -0.001, "iv": 0.16, "qm": 10000,
                                       "price": 0.15, "avg_open": 0.10}})
    attr2 = pnl_attribution(prev, curr2)
    assert abs(attr2["trade"] - (0.15 - 0.10) * 10000) < 0.01, attr2


def test_performance_metrics_known_series():
    """交替 +0.2%/0% 收益：均值 0.1%、波动 0.1%、夏普≈0.001/0.001×√252≈15.9"""
    eq = []
    for i in range(50):
        eq.append(100 * (1.002 ** i))       # 涨日
        eq.append(100 * (1.002 ** i))       # 平日
    m = performance_metrics([(None, e) for e in eq])
    assert abs(m["total_return"] - (1.002 ** 49 - 1)) < 1e-4  # metrics 保留4位小数
    assert m["max_drawdown"] == 0.0
    assert m["sharpe"] > 5, m


def test_trade_stats():
    ts = [Trade(realized_pnl=100), Trade(realized_pnl=-50),
          Trade(realized_pnl=-60), Trade(realized_pnl=200), Trade(realized_pnl=0)]
    s = trade_stats(ts)
    assert s["closed_trades"] == 4
    assert s["win_rate"] == 0.5
    assert s["avg_win"] == 150.0
    assert s["avg_loss"] == 55.0
    assert s["max_consecutive_losses"] == 2


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
