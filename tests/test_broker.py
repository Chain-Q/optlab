"""
thetalab.tests.test_broker — P2 批次A 撮合内核单测
运行：python -m thetalab.tests.test_broker
"""
import math
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thetalab.core.models import (
    Account, Direction, Instrument, Offset, Order, Right,
)
from thetalab.engine.broker import Broker, FillMode, MarketRow, RiskLimits, estimate_spread_pct

DAY = date(2026, 8, 28)
CALL = Instrument(symbol="510300C2609M04700", underlying="510300", right=Right.CALL,
                  strike=4.7, expiry=date(2026, 9, 23), multiplier=10000)
PUT = Instrument(symbol="510300P2609M04500", underlying="510300", right=Right.PUT,
                 strike=4.5, expiry=date(2026, 9, 23), multiplier=10000)


def mk_row(inst=CALL, close=0.065, volume=47117.0, **kw):
    d = dict(instrument=inst, trade_date=DAY, close=close, volume=volume,
             spot_close=4.679, spot_prev_close=4.691, pre_close=close,
             pre_settle=close, is_trading_day=True)
    d.update(kw)
    return MarketRow(**d)


def mk_account(cash=1_000_000.0):
    a = Account(initial_cash=cash, cash=cash)
    return a


def mk_order(inst=CALL, direction=Direction.SELL, offset=Offset.OPEN, qty=10, price=0.065):
    return Order(instrument=inst, direction=direction, offset=offset, qty=qty,
                 price=price, strategy_id="test")


def test_estimate_spread():
    """价差模型：ATM 活跃 ~0.3-0.5%，低价深度虚值 tick 主导，实测校准"""
    atm = estimate_spread_pct(0.2, 0.0, 25)
    assert 0.003 <= atm <= 0.01, atm
    # tick 主导：price=0.001 → 10%
    assert abs(estimate_spread_pct(0.001, 0.2, 25) - 0.10) < 1e-9
    # 虚值度增长
    assert estimate_spread_pct(0.05, 0.10, 25) > estimate_spread_pct(0.05, 0.02, 25)
    # 期限增长
    assert estimate_spread_pct(0.05, 0.05, 90) > estimate_spread_pct(0.05, 0.05, 25)
    # 上限
    assert estimate_spread_pct(0.0001, 0.3, 5) <= 1.0


def test_buy_open_cash_and_equity_identity():
    """买入开仓：cash 减少 premium+fee；equity 只减少手续费"""
    b = Broker(fill_mode=FillMode.CLOSE)
    a = mk_account()
    eq0 = a.equity
    trades, reject = b.match(mk_order(direction=Direction.BUY), mk_row(), a)
    assert reject is None and len(trades) == 1
    t = trades[0]
    assert t.amount == 0.065 * 10 * 10000
    assert a.cash == eq0 - t.amount - t.fee
    assert a.positions["510300C2609M04700"].net_qty == 10
    assert abs(a.equity - (eq0 - t.fee)) < 1e-6, "开仓后权益只应减少手续费"


def test_sell_open_margin_freeze():
    """卖出开仓：权利金进 cash、保证金冻结、可用资金双降"""
    b = Broker(fill_mode=FillMode.CLOSE)
    a = mk_account()
    cash0 = a.cash
    trades, reject = b.match(mk_order(), mk_row(), a)
    assert reject is None
    t = trades[0]
    assert a.cash == cash0 + t.amount - t.fee
    assert a.margin_used > 0
    assert a.margin_available() == a.cash - a.margin_used
    pos = a.positions["510300C2609M04700"]
    assert pos.net_qty == -10
    # 保证金应等于交易所公式：前结算0.065, S=4.691, K=4.7, 虚值=0
    from thetalab.core.spec import calc_margin
    m_ref = calc_margin(CALL, option_price=0.065, spot_close=4.691,
                        is_short=True, is_call=True) * 10
    assert abs(a.margin_used - m_ref) < 0.01, (a.margin_used, m_ref)


def test_reject_low_liquidity():
    b = Broker()
    a = mk_account()
    _, reject = b.match(mk_order(), mk_row(volume=50.0), a)
    assert reject and "流动性" in reject


def test_reject_margin_shortage():
    b = Broker()
    a = mk_account(cash=1000.0)  # 保证金约 2000+ 需要更多
    _, reject = b.match(mk_order(qty=100), mk_row(), a)
    assert reject and "保证金" in reject


def test_reject_expired_contract():
    b = Broker()
    a = mk_account()
    row = mk_row()
    row.trade_date = date(2026, 9, 23)  # 到期日当天
    _, reject = b.match(mk_order(), row, a)
    assert reject and "到期" in reject


def test_volume_cap():
    """单笔 ≤ 当日成交量 × 2%"""
    b = Broker(limits=RiskLimits(fill_ratio=0.02))
    a = mk_account()
    trades, _ = b.match(mk_order(qty=100), mk_row(volume=1000.0), a)
    assert trades[0].qty == 20


def test_close_slippage_directional():
    """CLOSE_SLIPPAGE：买入价 > 收盘、卖出价 < 收盘（保守）"""
    b = Broker(fill_mode=FillMode.CLOSE_SLIPPAGE)
    row = mk_row()
    o_buy = mk_order(direction=Direction.BUY)
    p_buy = b._fill_price(o_buy, row)
    o_sell = mk_order(direction=Direction.SELL)
    p_sell = b._fill_price(o_sell, row)
    assert p_buy > row.close > p_sell, (p_buy, row.close, p_sell)


def test_close_short_realized_pnl():
    """买平义务仓：pnl = (开仓均价 − 平仓价)×mult×qty − fee"""
    b = Broker(fill_mode=FillMode.CLOSE)
    a = mk_account()
    b.match(mk_order(), mk_row(close=0.065), a)                    # 卖开 @0.065
    trades, _ = b.match(mk_order(direction=Direction.BUY, offset=Offset.CLOSE, qty=10),
                        mk_row(close=0.130), a)                    # 买平 @0.130
    t = trades[0]
    expect = (0.065 - 0.130) * 10 * 10000 - t.fee
    assert abs(t.realized_pnl - expect) < 1e-6, (t.realized_pnl, expect)
    assert "510300C2609M04700" not in a.positions
    assert a.margin_used == 0.0


def test_settle_expiry_itm_call_long():
    """到期实值认购权利仓：现金流入 (S−K)×mult×qty − 行权费"""
    b = Broker(fill_mode=FillMode.CLOSE)
    a = mk_account()
    b.match(mk_order(direction=Direction.BUY, qty=5), mk_row(close=0.20), a)
    cash0 = a.cash
    trades = b.settle_expiry(a, spot=4.9, on=date(2026, 9, 23))
    assert len(trades) == 1 and "510300C2609M04700" not in a.positions
    inflow = (4.9 - 4.7) * 5 * 10000
    assert a.cash == cash0 + inflow - 5 * b.fee_rule.exercise_fee


def test_settle_expiry_assigned_put_short():
    """到期被指派认沽义务仓（实值）：现金流出内在价值、保证金释放"""
    b = Broker(fill_mode=FillMode.CLOSE)
    a = mk_account()
    b.match(mk_order(inst=PUT), mk_row(inst=PUT, close=0.15), a)
    cash0, margin0 = a.cash, a.margin_used
    assert margin0 > 0
    trades = b.settle_expiry(a, spot=4.3, on=date(2026, 9, 23))
    outflow = (4.5 - 4.3) * 10 * 10000
    assert abs(a.cash - (cash0 - outflow)) < 1e-6
    assert a.margin_used == 0.0
    assert trades[0].realized_pnl < (0.15 * 10 * 10000)  # 收了权利金仍亏损


def test_settle_expiry_otm_void():
    """虚值到期作废：权利仓价值归零、义务仓释放保证金"""
    b = Broker(fill_mode=FillMode.CLOSE)
    a = mk_account()
    b.match(mk_order(), mk_row(), a)
    margin0 = a.margin_used
    trades = b.settle_expiry(a, spot=4.6, on=date(2026, 9, 23))  # K=4.7 虚值
    assert trades == []
    assert a.margin_used == 0.0 and margin0 > 0


def test_full_cycle_equity_conservation():
    """完整往返：卖开→买平，equity 变化 == −2×fee（无价格变动）"""
    b = Broker(fill_mode=FillMode.CLOSE)
    a = mk_account()
    eq0 = a.equity
    b.match(mk_order(), mk_row(close=0.065), a)
    b.match(mk_order(direction=Direction.BUY, offset=Offset.CLOSE, qty=10),
            mk_row(close=0.065), a)
    fee_side = b.fee_rule.calc(10, 0.065, 10000)
    assert abs(a.equity - (eq0 - 2 * fee_side)) < 1e-6, a.equity - eq0


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
