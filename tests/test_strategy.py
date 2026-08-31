"""
thetalab.tests.test_strategy — P3-D 策略 DSL / 模板库 / 盈亏结构 单测
运行：python -m thetalab.tests.test_strategy
"""
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from thetalab.core.models import Instrument, Right
from thetalab.strategy.spec import (
    ExpirySelector, LegSpec, Sizing, StrikeSelector, StrategySpec, resolve_legs,
)
from thetalab.strategy.templates import TEMPLATES, get_template
from thetalab.strategy.payoff import PayoffLeg, build_curves

TODAY = date(2026, 8, 28)
SPOT = 4.679
EXP1 = date(2026, 9, 23)


def synth_chain():
    """ATM±4档 × C/P，delta 由行权价近似给定"""
    rows = []
    for k, dc, dp in [(4.4, 0.72, -0.10), (4.5, 0.62, -0.18), (4.6, 0.50, -0.30),
                      (4.7, 0.38, -0.48), (4.8, 0.28, -0.68), (4.9, 0.18, -0.82),
                      (5.0, 0.10, -0.92)]:
        for right, delta, close in (("CALL", dc, max(4.679 - k, 0.05) + 0.06),
                                    ("PUT", dp, max(k - 4.679, 0.05) + 0.06)):
            inst = Instrument(symbol=f"510300{right[0]}2609M{round(k*1000):05d}",
                              underlying="510300", right=Right[right],
                              strike=k, expiry=EXP1, multiplier=10000)
            rows.append({"contract_id": inst.symbol, "security_id": inst.symbol,
                         "strike": k, "right": right, "expiry": EXP1, "delta": delta,
                         "close": round(close, 4), "volume": 8000.0,
                         "open_interest": 15000.0, "_instrument": inst})
    return pd.DataFrame(rows)


def test_templates_count_and_validate():
    """P3 验收：≥6 个模板，全部通过 DSL 校验"""
    assert len(TEMPLATES) >= 6
    for tid, spec in TEMPLATES.items():
        errs = spec.validate()
        assert not errs, f"{tid}: {errs}"
        # JSON 往返
        spec2 = StrategySpec.from_json(spec.to_json())
        assert spec2.name == spec.name and len(spec2.legs) == len(spec.legs)


def test_json_roundtrip_fields():
    spec = get_template("short_strangle")
    spec.validate()   # 校验时自动给卖方腿追加权限提示
    d = json.loads(spec.to_json())
    assert d["legs"][0]["strike_selector"]["type"] == "DELTA"
    assert "三级权限" in d["notes"]  # 卖方模板自动附权限提示


def test_strike_selector_delta():
    chain = synth_chain()
    sel = StrikeSelector(type="DELTA", target=0.20, tol=0.10)
    row = sel.pick(chain, SPOT)
    assert row is not None
    # 最接近 |delta|=0.20 的档位：call 4.9(0.18) 或 put 4.5(-0.18)
    assert abs(abs(row["delta"]) - 0.20) <= 0.10


def test_strike_selector_fixed_missing():
    chain = synth_chain()
    sel = StrikeSelector(type="FIXED", strike=9.99)
    assert sel.pick(chain, SPOT) is None


def test_expiry_selector():
    exps = [date(2026, 9, 23), date(2026, 10, 28), date(2026, 12, 23)]
    s1 = ExpirySelector(type="DTE_RANGE", min=25, max=45).pick(exps, TODAY)
    assert s1 == date(2026, 9, 23)        # DTE=26
    s2 = ExpirySelector(type="NEAREST_EXPIRY", n=2).pick(exps, TODAY)
    assert s2 == date(2026, 10, 28)


def test_resolve_legs_short_strangle():
    """宽跨式：两条腿各选 Δ≈0.20，张数受权益预算约束"""
    spec = get_template("short_strangle")
    spec.sizing = Sizing(value=0.10, max_contracts=50)
    chain = synth_chain()
    orders, skips = resolve_legs(spec, chain, TODAY, SPOT, equity=1_000_000.0,
                                 margin_of=lambda row, s, e, t: 3000.0)
    assert not skips, skips
    assert len(orders) == 2
    for o in orders:
        assert o.direction.value == "SELL"
        # 保证金 3000/张, 预算 10万 → 33 张
        assert 0 < o.qty <= 33, o.qty


def test_resolve_legs_skip_on_low_oi():
    spec = get_template("short_strangle")
    spec.filters["min_open_interest"] = 99999
    orders, skips = resolve_legs(spec, synth_chain(), TODAY, SPOT, 1_000_000.0)
    assert orders == [] and skips


def test_resolve_legs_iron_condor_four_legs():
    spec = get_template("iron_condor")
    orders, skips = resolve_legs(spec, synth_chain(), TODAY, SPOT, 1_000_000.0,
                                 margin_of=lambda row, s, e, t: 3000.0)
    assert len(orders) == 4, (len(orders), skips)
    sells = [o for o in orders if o.direction.value == "SELL"]
    buys = [o for o in orders if o.direction.value == "BUY"]
    assert len(sells) == 2 and len(buys) == 2


def test_payoff_short_strangle_breakevens():
    """宽跨式到期：盈亏平衡点 = K_put − credit 与 K_call + credit（每单位口径）"""
    k_c, k_p = 5.0, 4.4
    c, p = 0.10, 0.10
    legs = [PayoffLeg(right=Right.CALL, strike=k_c, expiry=EXP1, qty=-10,
                      entry_price=c, iv=0.16),
            PayoffLeg(right=Right.PUT, strike=k_p, expiry=EXP1, qty=-10,
                      entry_price=p, iv=0.16)]
    cur = build_curves(legs, spot=4.679, asof=TODAY)
    # 每单位 credit = 0.20 → 平衡点 ≈ 4.2 与 5.2
    assert len(cur.breakevens) == 2, cur.breakevens
    lo, hi = cur.breakevens
    assert abs(lo - (k_p - 0.20)) < 0.02 and abs(hi - (k_c + 0.20)) < 0.02
    # 最大收益 = credit（区间内），曲线元口径 = 0.20×10×10000
    assert abs(cur.max_profit - 0.20 * 10 * 10000) < 2000
    assert cur.max_loss < 0     # 两端无上限亏损（边界内为有限显示值）
    assert cur.net_credit > 0


def test_payoff_covered_call_capped():
    """备兑：上行收益封顶 = (K−S)×mult×qty + credit"""
    k = 5.0
    legs = [PayoffLeg(right=None, strike=4.679, expiry=EXP1, qty=10, multiplier=1.0),
            PayoffLeg(right=Right.CALL, strike=k, expiry=EXP1, qty=-10,
                      entry_price=0.10, iv=0.16, multiplier=1.0)]
    cur = build_curves(legs, spot=4.679, asof=TODAY)
    # 上端 4.679×1.3≈6.08 > K → 封顶
    cap = (k - 4.679) * 10 * 1.0 + 0.10 * 10 * 1.0
    assert abs(cur.max_profit - cap) < 1.0
    assert len(cur.breakevens) == 1 and abs(cur.breakevens[0] - (4.679 - 0.10)) < 0.02


def test_payoff_t0_below_expiry_for_short():
    """卖方 T+0 理论曲线在平值附近低于到期线（时间价值对卖方不利未实现部分）"""
    legs = [PayoffLeg(right=Right.CALL, strike=4.7, expiry=EXP1, qty=-10,
                      entry_price=0.065, iv=0.16)]
    cur = build_curves(legs, spot=4.679, asof=TODAY)
    i = np.argmin(np.abs(cur.spots - 4.679))
    assert cur.t0[i] < cur.at_expiry[i] + 1e-6  # 含未实现浮亏口径

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
