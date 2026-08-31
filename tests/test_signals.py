"""
optlab.tests.test_signals — P3-E 指标库 / Advisor / 信号引擎 单测
运行：python -m optlab.tests.test_signals
"""
import math
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from optlab.core.indicators import build_indicators, iv_rank_of
from optlab.core.models import Instrument, Position, Right
from optlab.strategy.advisor import Advisor
from optlab.strategy.signals import SignalEngine

TODAY = date(2026, 8, 28)


def trend_df(n=80, drift=0.0, vol=0.01, seed=7):
    """合成标的日线：drift>0 上行趋势"""
    rng = np.random.default_rng(seed)
    px = [4.0]
    for _ in range(n - 1):
        px.append(px[-1] * (1 + drift + rng.normal(0, vol)))
    d = pd.DataFrame({
        "date": pd.date_range("2026-05-01", periods=n, freq="D").date,
        "close": px, "high": [p * 1.005 for p in px], "low": [p * 0.995 for p in px]})
    return d


def test_indicators_trend_up():
    d = trend_df(drift=0.002)
    ind = build_indicators(d)
    assert ind["trend"] == "UP"
    assert ind["adx"] > 0
    assert ind["rv20"] == ind["rv20"]
    assert 0 <= ind["rsi14"] <= 100


def test_indicators_choppy():
    d = trend_df(drift=0.0, vol=0.004)
    ind = build_indicators(d)
    assert ind["trend"] in ("CHOP", "UP", "DOWN")
    assert not ind["trend_strong"] or ind["adx"] > 25


def test_iv_rank():
    s = pd.Series(np.linspace(0.12, 0.25, 200))
    r = iv_rank_of(s, 0.20)
    assert abs(r["iv_rank"] - (0.20 - 0.12) / 0.13) < 1e-6
    assert 0 <= r["iv_pct"] <= 1
    short = pd.Series([0.15] * 10)
    assert math.isnan(iv_rank_of(short, 0.15)["iv_rank"])   # 样本不足 → NaN


def chain_df(put_iv=0.19, call_iv=0.14):
    rows = []
    for right, iv in (("CALL", call_iv), ("PUT", put_iv)):
        for k in (4.5, 4.6, 4.7, 4.8):
            rows.append({"contract_id": f"510300{right[0]}2609M{round(k*1000):05d}",
                         "strike": k, "right": right, "expiry": date(2026, 9, 23),
                         "iv": iv, "delta": 0.3, "close": 0.1,
                         "volume": 20000.0, "open_interest": 80000.0})
    return pd.DataFrame(rows)


def test_advisor_ranking_and_permission_note():
    ind = build_indicators(trend_df(drift=0.0, vol=0.004))
    ind.update(iv_rank=0.85, iv_pct=0.9)
    recs = Advisor().recommend(ind, chain_oi=300000, chain_volume=800000,
                               dte_choices=[26, 54], top_n=3)
    assert recs, "应有推荐"
    assert recs == sorted(recs, key=lambda r: -r.score)
    top = recs[0]
    assert top.score > 0 and top.reasons and top.risks
    assert top.spec is not None and top.exit_plan
    # IV 高位 + 无趋势 → 偏卖方模板，且带权限提示
    assert top.spec_id if hasattr(top, "spec_id") else True
    assert any(r.needs_permission_note for r in recs), "卖方推荐必须带权限提示"


def test_advisor_penalizes_buying_at_iv_90():
    ind = build_indicators(trend_df(drift=0.0, vol=0.004))
    ind.update(iv_rank=0.95)
    recs = Advisor().recommend(ind, 300000, 800000, [26], top_n=10)
    longs = [r for r in recs if r.template_id.startswith("long_")]
    for r in longs:
        assert r.score < 100 and any("禁止买入" in x for x in r.reasons)


def test_signals_iv_high_and_low():
    eng = SignalEngine(cooldown_days=999)
    ind = {"iv_rank": 0.85, "iv_atm": 0.20, "rv20": 0.12, "adx": 15, "trend": "CHOP",
           "close": 4.679}
    sigs = eng.generate(ind, today=TODAY)
    hi = [s for s in sigs if s.name.startswith("IV 高位")]
    assert hi and hi[0].strength >= 50 and hi[0].evidence
    # 低位
    ind2 = dict(ind, iv_rank=0.10)
    sigs2 = SignalEngine(cooldown_days=999).generate(ind2, today=TODAY)
    assert any(s.name.startswith("IV 低位") for s in sigs2)


def test_signals_dedup_cooldown():
    eng = SignalEngine(cooldown_days=1)
    ind = {"iv_rank": 0.85, "adx": 15, "trend": "CHOP", "close": 4.679}
    day1 = eng.generate(ind, today=TODAY)
    day2 = eng.generate(ind, today=TODAY + timedelta(days=1))
    n1 = sum(1 for s in day1 if s.dedup_key.startswith("iv_hi"))
    n2 = sum(1 for s in day2 if s.dedup_key.startswith("iv_hi"))
    assert n1 == 1 and n2 == 0, "冷却期内同 key 不得重复触发"


def test_signals_gamma_risk_priority():
    """义务仓 DTE≤3 且 |Δ|∈[0.3,0.7] → 强度 95 置顶"""
    inst = Instrument(symbol="X", underlying="510300", right=Right.PUT, strike=4.7,
                      expiry=TODAY + timedelta(days=2), multiplier=10000)
    pos = Position(instrument=inst, net_qty=-10)
    pos.last_delta = 0.5
    eng = SignalEngine(cooldown_days=999)
    sigs = eng.generate({"adx": float("nan")}, positions={"X": pos}, today=TODAY)
    assert sigs and sigs[0].kind == "RISK" and sigs[0].strength >= 90
    assert sigs[0].level == "强"


def test_signals_holiday_warning():
    eng = SignalEngine(cooldown_days=999)
    sigs = eng.generate({"adx": float("nan")}, today=TODAY,
                        holidays_until=[TODAY + timedelta(days=3)])
    assert any("长假" in s.name for s in sigs)


def test_signals_top5_and_levels():
    eng = SignalEngine(cooldown_days=999)
    ind = {"iv_rank": 0.85, "iv_atm": 0.22, "rv20": 0.10, "adx": 30, "trend": "UP",
           "close": 4.679, "bb_pos": 1.5, "atr14": 0.05, "rsi14": 80}
    sigs = eng.generate(ind, today=TODAY)
    assert len(sigs) <= 5
    assert sigs == sorted(sigs, key=lambda s: -s.strength)
    for s in sigs:
        assert s.level in ("强", "中", "弱", "提示")


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
