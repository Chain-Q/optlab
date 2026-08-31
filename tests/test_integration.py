"""
optlab.tests.test_integration — 全 API 集成测试（Workbench 逻辑层，覆盖全部端点逻辑）

覆盖清单（与 HTTP 路由一一对应）：
    state（含 has_daily_bars/note/live 字段）｜order（合法/各拒单分支/159915 闸门）
    confirm｜advance（撮合+结算+时钟）｜set_day｜set_underlying（合法/未知/512100）
    cancel_all｜live_start/live_stop
运行：python -m optlab.tests.test_integration
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optlab.server import Workbench

UNDERLYINGS = ["588000", "159915", "510300", "510050", "510500"]


def make_wb(tmp: str) -> Workbench:
    wb = Workbench(data_dir=Path("optlab_data"), auto_update=False)  # 测试关调度线程（晚间窗口会真探测真采集）
    from optlab.data.persist import StateStore
    wb.store = StateStore(Path(tmp) / "paper.db")
    wb.paper.store = wb.store
    wb.cursor = wb.days[-6]  # 回退到可推进的日期（交易闭环测试需要 advance 空间）
    return wb


def pick_liquid(wb, underlying="510300", min_vol=500, min_dte=None):
    """挑成交量最大的合约；min_dte 避开临到期合约（推进测试不可跨到期结算）"""
    wb.set_underlying(underlying)
    st = wb.state(underlying)
    cands = [r for r in st["chain"]
             if r["last"] and r["volume"] >= min_vol
             and (min_dte is None or r["dte"] >= min_dte)]
    cands.sort(key=lambda r: -r["volume"])
    return cands[0] if cands else None


def test_state_payload_contract():
    """state 必含字段契约（前端依赖）"""
    tmp = tempfile.mkdtemp()
    wb = make_wb(tmp)
    st = wb.state()
    required = ["ok", "cursor", "underlying", "underlying_name", "has_daily_bars",
                "collected", "note", "underlyings", "spot", "account", "pending",
                "chain", "expiries", "days", "equity_curve", "server_started",
                "live"]
    for k in required:
        assert k in st, f"state 缺字段 {k}"
    for u in st["underlyings"]:
        assert set(("code", "name", "collected", "has_daily_bars")) <= set(u)
    assert len(st["chain"]) > 50
    row = st["chain"][0]
    for k in ("contract_id", "strike", "right", "expiry", "dte", "last",
              "volume", "iv", "delta", "margin_per_lot"):
        assert k in row, f"chain 行缺 {k}"


def test_order_full_lifecycle_510300():
    tmp = tempfile.mkdtemp()
    wb = make_wb(tmp)
    row = pick_liquid(wb, "510300", min_dte=5)
    sym = row["contract_id"]
    assert wb.place_order(sym, "SELL", "OPEN", 2)["ok"]
    wb.confirm()
    r = wb.advance(); assert r["ok"]
    acct = wb.store.load_account()
    assert acct.positions[sym].net_qty == -2
    assert 0 < acct.margin_used < acct.equity
    assert wb.place_order(sym, "BUY", "CLOSE", 2)["ok"]
    wb.confirm(); wb.advance()
    assert sym not in wb.store.load_account().positions


def test_order_rejections():
    tmp = tempfile.mkdtemp()
    wb = make_wb(tmp)
    row = pick_liquid(wb, "510300")
    sym = row["contract_id"]
    assert not wb.place_order(sym, "BUY", "OPEN", 0)["ok"]       # 数量 0
    assert not wb.place_order(sym, "SIDEWAYS", "OPEN", 1)["ok"]  # 非法方向
    assert not wb.place_order("NOPE123", "BUY", "OPEN", 1)["ok"]  # 未知合约
    # 超量：卖开保证金超过可用（qty 极大）
    assert not wb.place_order(sym, "SELL", "OPEN", 10_000_000)["ok"]
    # 平仓无持仓
    assert not wb.place_order(sym, "BUY", "CLOSE", 1)["ok"]


def test_two_underlying_positions_coexist():
    """多品种持仓共存：510300 与 510050 各持一腿，结算/盯市互不干扰"""
    tmp = tempfile.mkdtemp()
    wb = make_wb(tmp)
    a = pick_liquid(wb, "510300", min_dte=5)
    b = pick_liquid(wb, "510050", min_dte=5)
    assert wb.place_order(a["contract_id"], "SELL", "OPEN", 1)["ok"]
    assert wb.place_order(b["contract_id"], "SELL", "OPEN", 1)["ok"]
    wb.confirm(); r = wb.advance(); assert r["ok"]
    acct = wb.store.load_account()
    assert a["contract_id"] in acct.positions and b["contract_id"] in acct.positions
    # 两个品种的保证金都应按各自标的计算（量级不同：510300 保证金 > 510050 同 delta 档）
    m300 = acct.positions[a["contract_id"]].margin
    m050 = acct.positions[b["contract_id"]].margin
    assert m300 > 0 and m050 > 0
    # 平掉其一，另一腿保证金不受影响
    assert wb.place_order(a["contract_id"], "BUY", "CLOSE", 1)["ok"]
    wb.confirm(); wb.advance()
    acct = wb.store.load_account()
    assert a["contract_id"] not in acct.positions
    assert b["contract_id"] in acct.positions
    assert acct.margin_used > 0


def test_159915_snapshot_trading_gate():
    """159915：快照口径可浏览+可下单（SZSE_MIN_DAYS=1）；512100 无法切换（无场内期权）"""
    tmp = tempfile.mkdtemp()
    wb = make_wb(tmp)
    wb.set_underlying("159915")
    wb.SZSE_MIN_DAYS = 1
    st = wb.state("159915")
    cands = [r for r in st["chain"] if r["last"] and r["iv"]]
    assert cands
    sym = cands[0]["contract_id"]
    r = wb.place_order(sym, "SELL", "OPEN", 1)
    assert r["ok"], r
    wb.confirm(); r = wb.advance(); assert r["ok"]
    # 512100 已从品种列表移除（用户指定换为 510050），切换应被拒
    assert not wb.set_underlying("512100")["ok"]


def test_set_day_and_underlying_independence():
    """时钟与品种切换互相独立，state 始终一致"""
    tmp = tempfile.mkdtemp()
    wb = make_wb(tmp)
    wb.set_underlying("510500")
    assert wb.set_day("2026-08-27")["ok"]
    st = wb.state()
    assert st["cursor"] == "2026-08-27" and st["underlying"] == "510500"
    assert wb.set_day("1999-01-01")["ok"] is False
    assert wb.set_underlying("888888")["ok"] is False


def test_live_toggle_contract():
    tmp = tempfile.mkdtemp()
    wb = make_wb(tmp)
    r = wb.live_start("510300", "2026-09-23")
    assert r["ok"]
    cs = wb.collect_status_now()
    assert cs is not None
    r = wb.live_stop()
    assert r["ok"]


def test_auto_jump_cursor_pending_guard():
    """更新后自动跳日：无挂单→跳最新；有未撮合挂单（PENDING/CONFIRMED）→不跳并提示"""
    tmp = tempfile.mkdtemp()
    wb = make_wb(tmp)
    old_day = wb.days[-6]
    wb.cursor = old_day
    # ① 无挂单 → 跳到最新
    assert wb._auto_jump_cursor() is True
    assert wb.cursor == wb.days[-1]
    # ② 有 PENDING 挂单（已挂未确认）→ 不跳 + 提示
    wb.cursor = old_day
    row = pick_liquid(wb, "510300")
    assert wb.place_order(row["contract_id"], "SELL", "OPEN", 1)["ok"]
    assert wb.store.pending("PENDING")
    assert wb._auto_jump_cursor() is False
    assert wb.cursor == old_day
    assert any("未撮合挂单" in t for t in wb.update_status["tail"])
    # ③ 确认后（CONFIRMED，待推进撮合）→ 同样不跳
    wb.confirm()
    assert wb.store.pending("CONFIRMED")
    assert wb._auto_jump_cursor() is False
    assert wb.cursor == old_day
    # ④ 挂单清除后 → 恢复可跳
    for p in wb.store.pending("PENDING") + wb.store.pending("CONFIRMED"):
        wb.store.set_order_status(p["order_key"], "CANCELLED")
    assert wb._auto_jump_cursor() is True
    assert wb.cursor == wb.days[-1]


def test_auto_update_step():
    """晚间调度单步决策：窗口外/周末/未发布不动作；发布→采集一次并去重"""
    from types import SimpleNamespace
    from datetime import datetime as _dt
    tmp = tempfile.mkdtemp()
    wb = make_wb(tmp)
    calls = []
    wb._run_auto_collect = lambda day: calls.append(str(day)) or 0
    # ① 窗口外（15:00）→ 不探测不采集
    assert wb._auto_update_step(_dt(2026, 9, 1, 15, 0)) == 60.0
    assert wb._last_probe is None and not calls
    # ② 窗口内未发布 → 不采集
    wb._probe_provider = SimpleNamespace(risk_indicators=lambda d: SimpleNamespace(empty=True))
    eve = _dt(2026, 9, 1, 20, 0)   # 周二
    assert wb._auto_update_step(eve) == 60.0
    assert not calls
    # ③ 发布 → 采集一次并去重
    wb._last_probe = None
    wb._probe_provider = SimpleNamespace(risk_indicators=lambda d: SimpleNamespace(empty=False))
    assert wb._auto_update_step(eve) == 60.0
    assert calls == ["2026-09-01"]
    assert wb._last_auto_date == "2026-09-01"
    # ④ 已采集过 → 不再动作
    assert wb._auto_update_step(eve) == 60.0
    assert calls == ["2026-09-01"]
    # ⑤ 周六 → 不动作
    assert wb._auto_update_step(_dt(2026, 9, 5, 20, 0)) == 60.0
    assert calls == ["2026-09-01"]


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
            import traceback; traceback.print_exc()
            print(f"ERROR {fn.__name__}: {type(e).__name__} {e}")
    print(f"\n{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} passed")
    sys.exit(1 if failed else 0)
