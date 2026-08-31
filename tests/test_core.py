"""
thetalab.tests.test_core — P0 内核验收单测

验收标准（设计方案 §8 P0）：
    - BS 价格与标准值误差 < 1e-6
    - IV 反解往返误差 < 1e-6
    - 保证金公式与交易所规则手工对账一致
    - 涨跌停公式与上交所官方公式一致
    - 到期日 = 到期月第四个周三（顺延）

运行：python -m thetalab.tests.test_core
"""
import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thetalab.core.pricing import (
    bs_price, bs_greeks, black76_price, crr_price, implied_vol,
)
from thetalab.core.spec import (
    calc_margin, calc_limit_prices, expiry_of_month, strike_step,
    build_option_symbol, TradingCalendar,
)
from thetalab.core.models import Right, Instrument, MarginRule, OptionQuote, Tick


def test_bs_price_reference():
    """BS vs 标准 textbook 值（Hull 表）"""
    c = bs_price(100, 100, 1.0, 0.05, 0.20, Right.CALL)
    p = bs_price(100, 100, 1.0, 0.05, 0.20, Right.PUT)
    assert abs(c - 10.450584) < 1e-6, f"call {c}"
    assert abs(p - 5.573526) < 1e-6, f"put {p}"
    # put-call parity
    assert abs((c - p) - (100 - 100 * math.exp(-0.05))) < 1e-9
    # 退化边界：到期、零波动
    assert bs_price(105, 100, 0.0, 0.05, 0.2, Right.CALL) == 5.0
    assert bs_price(95, 100, 0.0, 0.05, 0.2, Right.PUT) == 5.0


def test_iv_roundtrip_european():
    """IV 反解往返 < 1e-6（欧式 BS 路径 —— 回归测试：此前三元表达式 bug 导致此路径崩溃）"""
    for S, K, T, r, right in [
        (100, 110, 0.5, 0.03, Right.PUT),
        (4.0, 4.2, 30 / 365, 0.02, Right.CALL),
        (2.8, 2.8, 12 / 365, 0.02, Right.PUT),
    ]:
        for target in (0.12, 0.25, 0.45, 0.80):
            px = bs_price(S, K, T, r, target, right)
            iv = implied_vol(px, S, K, T, r, right)
            assert iv == iv, f"IV nan for target={target}"  # not nan
            assert abs(iv - target) < 1e-6, f"iv={iv} target={target}"


def test_iv_roundtrip_american_crr():
    """美式 CRR 路径 IV 往返（较宽容差：树离散化）"""
    px = crr_price(100, 100, 0.5, 0.03, 0.30, Right.PUT, steps=100, american=True)
    iv = implied_vol(px, 100, 100, 0.5, 0.03, Right.PUT, american=True, steps=100)
    assert abs(iv - 0.30) < 2e-3, f"iv={iv}"


def test_iv_unsolvable_returns_nan():
    """低于内在价值 / 超理论上限 → NaN（禁止 0 兜底）"""
    assert math.isnan(implied_vol(0.3, 4.0, 3.5, 0.05, 0.02, Right.CALL))
    assert math.isnan(implied_vol(99.0, 4.0, 4.0, 0.05, 0.02, Right.CALL))
    assert math.isnan(implied_vol(1.0, 4.0, 4.0, 0.0, 0.02, Right.CALL))  # T=0


def test_greeks_conventions():
    """Greeks 口径：vega=每1vol点、theta=每自然日(负)、与有限差分对拍"""
    S, K, T, r, sig = 4.0, 4.0, 30 / 365, 0.02, 0.18
    g = bs_greeks(S, K, T, r, sig, Right.CALL)
    assert g.theta < 0, "多头 theta 应为负（时间损耗）"
    assert 0.3 < g.delta < 0.7, f"ATM delta={g.delta}"
    # vega 有限差分对拍（σ+0.0001 → 价格差 / 1 vol point）
    p0 = bs_price(S, K, T, r, sig, Right.CALL)
    p1 = bs_price(S, K, T, r, sig + 0.01, Right.CALL)
    assert abs(g.vega - (p1 - p0)) < 1e-6, f"vega={g.vega} fd={p1-p0}"
    # theta 对拍：T-1天 的价格差（1天步长的二阶误差 <5e-5）
    p_t = bs_price(S, K, T - 1 / 365, r, sig, Right.CALL)
    assert abs(g.theta - (p_t - p0)) < 5e-5, f"theta={g.theta} fd={p_t-p0}"


def test_margin_open_call():
    """开仓保证金手工对账：510300, S_prev=4.0, K=4.2 call, 前结算=0.05, 单位10000
    虚值=0.2；max(0.48-0.2, 0.28)=0.28 → (0.05+0.28)×10000=3300"""
    inst = Instrument(symbol='510300C2609M04200', underlying='510300',
                      right=Right.CALL, strike=4.2, multiplier=10000)
    m = calc_margin(inst, option_price=0.05, spot_close=4.0, is_short=True, is_call=True)
    assert abs(m - 3300.0) < 0.01, m


def test_margin_open_put():
    """认沽开仓：S_prev=4.0, K=3.8, 前结算=0.05 → 虚值0.2; max(0.48-0.2, 0.07×3.8=0.266)=0.28 → 3300"""
    inst = Instrument(symbol='510300P2609M03800', underlying='510300',
                      right=Right.PUT, strike=3.8, multiplier=10000)
    m = calc_margin(inst, option_price=0.05, spot_close=4.0, is_short=True, is_call=False)
    assert abs(m - 3300.0) < 0.01, m


def test_margin_put_cap():
    """认沽保证金 min(..., 行权价) 封顶：深实值 put K=5.0, price=1.0 → (1.0+0.48)×10000=14800 < 50000"""
    inst = Instrument(symbol='510300P2609M05000', underlying='510300',
                      right=Right.PUT, strike=5.0, multiplier=10000)
    m = calc_margin(inst, option_price=1.0, spot_close=4.0, is_short=True, is_call=False)
    assert abs(m - 14800.0) < 0.01, m


def test_margin_maintenance_same_coefficients():
    """维持保证金与开仓同系数 12%/7%（区别仅取价时点）—— 回归测试：此前误用 10%/5%"""
    inst = Instrument(symbol='510300C2609M04200', underlying='510300',
                      right=Right.CALL, strike=4.2, multiplier=10000)
    # 当日结算价 0.06、标的当日收盘 4.1：虚值=0.1; max(0.12×4.1-0.1, 0.07×4.1)=max(0.392,0.287)=0.392
    m = calc_margin(inst, option_price=0.06, spot_close=4.1, is_short=True,
                    is_call=True, maintenance=True)
    assert abs(m - (0.06 + 0.392) * 10000) < 0.01, m
    rule = MarginRule()
    assert rule.maint_a == 0.12 and rule.maint_b == 0.07


def test_limit_prices_official():
    """涨跌停 vs 上交所官方公式 —— 回归测试：此前跌幅公式错用涨幅对称项"""
    S = 4.0
    # OTM call K=4.2, P=0.05: 涨幅=max{0.02, min(3.8,4)×10%=0.38}=0.38 → up=0.43; 跌幅=0.4 → dn=0.0001
    up, dn = calc_limit_prices(0.05, S, 4.2, True)
    assert abs(up - 0.43) < 1e-9 and abs(dn - 0.0001) < 1e-9, (up, dn)
    # 深实值 call K=3.0, P=1.02: up=1.02+0.4=1.42; dn=max(1.02-0.4,0.0001)=0.62
    up, dn = calc_limit_prices(1.02, S, 3.0, True)
    assert abs(up - 1.42) < 1e-9 and abs(dn - 0.62) < 1e-9, (up, dn)
    # 实值 put K=4.4, P=0.5: 涨幅=max{K×0.5%=0.022, min(4.8,4)×10%=0.4}=0.4 → up=0.9; dn=max(0.5-0.4,..)=0.1
    up, dn = calc_limit_prices(0.5, S, 4.4, False)
    assert abs(up - 0.9) < 1e-9 and abs(dn - 0.1) < 1e-9, (up, dn)


def test_expiry_fourth_wednesday():
    cases = [(2026, 9, "2026-09-23"), (2026, 8, "2026-08-26"),
             (2025, 2, "2025-02-26"), (2025, 10, "2025-10-22")]
    for y, m, expect in cases:
        e = expiry_of_month(y, m)
        assert str(e) == expect and e.weekday() == 2, (y, m, e)


def test_strike_step_table():
    assert strike_step(2.5) == 0.05   # 50ETF
    assert strike_step(4.0) == 0.10   # 300ETF
    assert strike_step(6.5) == 0.25   # 500ETF
    assert strike_step(1.05) == 0.05  # 科创50ETF


def test_otm_pct_sign():
    """虚值百分比符号：虚值>0 实值<0 —— 回归测试：此前认购方向反了"""
    q = OptionQuote(
        instrument=Instrument(symbol='X', right=Right.CALL, strike=4.4, multiplier=10000),
        tick=Tick(instrument=None, ts=None), spot=4.0)
    assert q.otm_pct > 0, "虚值认购应为正"
    q2 = OptionQuote(
        instrument=Instrument(symbol='X', right=Right.CALL, strike=3.6, multiplier=10000),
        tick=Tick(instrument=None, ts=None), spot=4.0)
    assert q2.otm_pct < 0, "实值认购应为负"
    q3 = OptionQuote(
        instrument=Instrument(symbol='X', right=Right.PUT, strike=3.6, multiplier=10000),
        tick=Tick(instrument=None, ts=None), spot=4.0)
    assert q3.otm_pct > 0, "虚值认沽应为正"


def test_crr_vs_bs_european():
    crr = crr_price(100, 110, 0.5, 0.03, 0.25, Right.PUT, steps=800, american=False)
    bs = bs_price(100, 110, 0.5, 0.03, 0.25, Right.PUT)
    assert abs(crr - bs) < 3e-3, (crr, bs)  # CRR 800步 O(1/N) 收敛残差
    am = crr_price(100, 110, 0.5, 0.03, 0.25, Right.PUT, steps=800, american=True)
    assert am >= crr


def test_black76_consistency():
    """Black76(F,K) = BS(S=F, q=r)"""
    F, K, T, r, sig = 4000.0, 4100.0, 0.25, 0.03, 0.18
    assert abs(black76_price(F, K, T, r, sig, Right.PUT)
               - bs_price(F, K, T, r, sig, Right.PUT, q=r)) < 1e-12


def test_trading_calendar_weekend():
    assert not TradingCalendar.is_trading_day(date(2026, 8, 29))  # 周六
    assert TradingCalendar.is_trading_day(date(2026, 8, 28))      # 周五
    nxt = TradingCalendar.next_trading_day(date(2026, 8, 28))
    assert nxt == date(2026, 8, 31)  # 周一（非节假日）


def test_symbol_builder():
    assert build_option_symbol("510300", Right.CALL, date(2026, 9, 23), 4.0) \
        == "510300C2609M04000"


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
