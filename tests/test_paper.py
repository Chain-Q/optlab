"""
thetalab.tests.test_paper — P4-F 模拟盘端到端单测（合成数据，两天闭环）
运行：python -m thetalab.tests.test_paper

验证：月首挂单 → 人工确认 → T+1 撮合成交 → 状态持久化/恢复 → 报告 JSON。
"""
import json
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from thetalab.data.persist import StateStore
from thetalab.engine.paper import PaperTradingRunner
from thetalab.engine.runner import BacktestRunner, SellStrangleStrategy
from thetalab.strategy.templates import get_template

UND = "510300"
D1, D2, D3, D4 = date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)
EXP = date(2026, 2, 25)


def make_data():
    rows, dailies = [], []
    spot = {D1: 4.60, D2: 4.65, D3: 4.65, D4: 4.65}
    for d in (D1, D2, D3, D4):
        for right, strike, delta, closes in (
                ("CALL", 5.00, 0.20, {D1: 0.010, D2: 0.020, D3: 0.020, D4: 0.020}),
                ("PUT", 4.50, -0.20, {D1: 0.320, D2: 0.310, D3: 0.310, D4: 0.310})):
            cid = f"{UND}{right[0]}2602M{round(strike*1000):05d}"
            rows.append({"trade_date": d, "security_id": f"1{right[0]}00",
                         "contract_id": cid, "underlying": UND, "right": right,
                         "expiry": EXP, "strike": float(strike), "adjusted": False,
                         "iv": 0.16, "delta": delta, "gamma": 1.5, "vega": 50.0,
                         "theta": -110.0})
            dailies.append({"security_id": f"1{right[0]}00", "trade_date": d,
                            "close": closes[d], "volume_lots": 5000.0, "date": d})
    risk = pd.DataFrame(rows)
    daily = pd.DataFrame(dailies)
    close = pd.Series(spot)
    return risk, daily, close


def fresh_runner(tmp, dry_run=False, cash=1_000_000.0):
    risk, daily, close = make_data()
    feed = BacktestRunner(risk, daily, close)
    store = StateStore(Path(tmp) / "paper.db")
    strategy = get_template("short_strangle")
    runner = PaperTradingRunner(
        feed, store, data_dir=tmp, strategy=SellStrangleStrategy(entry_dte_min=25, exit_dte=0,
                                                                 lots_per_side=5),
        dry_run=dry_run)
    # 保证金折算函数注入（resolver 需要）
    return runner, store


def test_paper_two_day_cycle(tmp_path=None):
    tmp = tempfile.mkdtemp()
    runner, store = fresh_runner(tmp)

    # Day1（月首）：生成策略建议（PENDING），不成交
    r1 = runner.daily_update(D1)
    assert r1.equity > 0
    assert len(r1.strategy_orders) == 2, r1.strategy_orders
    assert len(r1.fills) == 0, "确认前不得成交"
    pending = store.pending()
    assert len(pending) == 2 and all(p["status"] == "PENDING" for p in pending)

    # 报告 JSON 落盘且含关键字段
    rep = json.loads((Path(tmp) / "report.json").read_text(encoding="utf-8"))
    for k in ("day", "equity", "signals", "recommendations", "positions", "chain"):
        assert k in rep, k

    # 人工确认 → Day2 撮合成交（T+1 纪律）
    runner.confirm_orders()
    r2 = runner.daily_update(D2)
    fills = [f for f in r2.fills if f["type"] == "成交"]
    assert len(fills) == 2, r2.fills
    acct = store.load_account()
    assert acct and len(acct.positions) == 2
    assert acct.margin_used > 0

    # 状态恢复：新实例读同一库，持仓一致
    store2 = StateStore(Path(tmp) / "paper.db")
    acct2 = store2.load_account()
    assert set(acct2.positions) == set(acct.positions)
    assert abs(acct2.margin_used - acct.margin_used) < 0.01

    # 净值曲线已记录
    eq = store.equity_curve()
    assert len(eq) >= 2 and eq[-1][0] == str(D2)


def test_paper_dry_run_blocks_orders():
    tmp = tempfile.mkdtemp()
    runner, store = fresh_runner(tmp, dry_run=True)
    r1 = runner.daily_update(D1)
    assert r1.strategy_orders == [], "干运行不得挂单"
    assert store.pending() == []


def test_paper_month_guard():
    """同月第二次 daily_update 不再重复挂单（持仓也在，双重保护）"""
    tmp = tempfile.mkdtemp()
    runner, store = fresh_runner(tmp)
    runner.daily_update(D1)
    r2 = runner.daily_update(D2)   # 非月首 + 已有持仓确认中
    assert r2.strategy_orders == []


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    results = []
    for fn in ALL_TESTS:
        try:
            if fn.__name__ == "test_paper_two_day_cycle":
                fn()
            else:
                fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print(f"ERROR {fn.__name__}: {type(e).__name__} {e}")
    print(f"\n{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} passed")
    sys.exit(1 if failed else 0)
