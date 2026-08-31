"""
optlab.data.persist — 模拟盘状态持久化（SQLite）

事务型状态：账户/持仓/订单/成交/信号/净值曲线。
模拟盘的全部长期价值在于可复盘（§7.3）——每笔都留 reason/evidence。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..core.models import (
    Account, Direction, Greeks, Instrument, Offset, Order, Position, Right, Trade,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    updated_at TEXT, cash REAL, margin_used REAL, initial_cash REAL, total_fee REAL,
    positions_json TEXT);
CREATE TABLE IF NOT EXISTS trades (
    trade_id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, symbol TEXT,
    direction TEXT, offset TEXT, qty REAL, price REAL, amount REAL, fee REAL,
    realized_pnl REAL, strategy_id TEXT, order_id TEXT);
CREATE TABLE IF NOT EXISTS signals (
    ts TEXT, kind TEXT, name TEXT, strength REAL, level TEXT, reason TEXT,
    evidence_json TEXT, action TEXT, dedup_key TEXT);
CREATE TABLE IF NOT EXISTS equity_curve (
    day TEXT PRIMARY KEY, equity REAL, margin_used REAL,
    delta REAL, vega REAL, theta REAL);
CREATE TABLE IF NOT EXISTS pending_orders (
    order_key TEXT PRIMARY KEY, decision_day TEXT, symbol TEXT, direction TEXT,
    offset TEXT, qty REAL, reason TEXT, status TEXT, instrument_json TEXT);
"""


def _locked(fn):
    """跨线程访问保护：ThreadingHTTPServer 每请求一线程，SQLite 共享连接需串行化"""
    import functools
    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        with self._lock:
            return fn(self, *a, **kw)
    return wrapper


class StateStore:
    _lock = None  # 占位，实际锁在 __init__
    def __init__(self, path: str | Path = "optlab_data/paper.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 服务器模式（ThreadingHTTPServer）跨线程访问：关闭同线程检查 + 写锁
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------ 账户
    @_locked
    def save_account(self, account: Account, updated_at: Optional[datetime] = None):
        positions = {sym: self._pos_dict(p) for sym, p in account.positions.items()}
        self._conn.execute(
            "INSERT OR REPLACE INTO accounts (id, updated_at, cash, margin_used, "
            "initial_cash, total_fee, positions_json) VALUES (1,?,?,?,?,?,?)",
            ((updated_at or datetime.now()).isoformat(),
             account.cash, account.margin_used, account.initial_cash,
             account.total_fee, json.dumps(positions, ensure_ascii=False)))
        self._conn.commit()

    @_locked
    def load_account(self) -> Optional[Account]:
        row = self._conn.execute(
            "SELECT cash, margin_used, initial_cash, total_fee, positions_json "
            "FROM accounts WHERE id=1").fetchone()
        if row is None:
            return None
        a = Account(initial_cash=row[2], cash=row[0])
        a.margin_used, a.total_fee = row[1], row[3]
        for sym, pd_ in json.loads(row[4]).items():
            a.positions[sym] = self._pos_from(pd_)
        return a

    @staticmethod
    def _pos_dict(p: Position) -> dict:
        inst = p.instrument
        return {
            "instrument": {"symbol": inst.symbol, "underlying": inst.underlying,
                           "right": inst.right.value if inst.right else None,
                           "strike": inst.strike,
                           "expiry": inst.expiry.isoformat() if inst.expiry else None,
                           "multiplier": inst.multiplier},
            "net_qty": p.net_qty, "avg_open_price": p.avg_open_price,
            "total_fee": p.total_fee, "realized_pnl": p.realized_pnl,
            "margin": p.margin, "last_price": p.last_price,
            "strategy_id": p.strategy_id}

    @staticmethod
    def _pos_from(d: dict) -> Position:
        i = d["instrument"]
        inst = Instrument(symbol=i["symbol"], underlying=i["underlying"],
                          right=Right[i["right"]] if i["right"] else None,
                          strike=i["strike"],
                          expiry=date.fromisoformat(i["expiry"]) if i["expiry"] else None,
                          multiplier=i["multiplier"])
        return Position(instrument=inst, net_qty=d["net_qty"],
                        avg_open_price=d["avg_open_price"], total_fee=d["total_fee"],
                        realized_pnl=d["realized_pnl"], margin=d["margin"],
                        last_price=d["last_price"], strategy_id=d["strategy_id"])

    # ------------------------------------------------------------ 成交/信号/净值
    @_locked
    def append_trades(self, trades: List[Trade]):
        for t in trades:
            self._conn.execute(
                "INSERT INTO trades (ts, symbol, direction, offset, qty, price, "
                "amount, fee, realized_pnl, strategy_id, order_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (t.ts.isoformat() if t.ts else "",
                 t.instrument.symbol if t.instrument else "",
                 t.direction.value, t.offset.value, t.qty, t.price, t.amount,
                 t.fee, t.realized_pnl, t.strategy_id, t.order_id))
        self._conn.commit()

    @_locked
    def trades(self) -> List[dict]:
        cur = self._conn.execute(
            "SELECT ts, symbol, direction, offset, qty, price, amount, fee, "
            "realized_pnl, strategy_id FROM trades ORDER BY id")
        cols = ["ts", "symbol", "direction", "offset", "qty", "price", "amount",
                "fee", "realized_pnl", "strategy_id"]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    @_locked
    def append_signals(self, signals, day):
        for s in signals:
            self._conn.execute(
                "INSERT INTO signals (ts, kind, name, strength, level, reason, "
                "evidence_json, action, dedup_key) VALUES (?,?,?,?,?,?,?,?,?)",
                (day.isoformat(), s.kind, s.name, s.strength, s.level, s.reason,
                 json.dumps(s.evidence, ensure_ascii=False, default=str),
                 s.action, s.dedup_key))
        self._conn.commit()

    @_locked
    def append_equity(self, day: str, state):
        self._conn.execute(
            "INSERT OR REPLACE INTO equity_curve (day, equity, margin_used, delta, vega, theta) "
            "VALUES (?,?,?,?,?,?)",
            (day, state.equity, state.margin_used, state.delta, state.vega, state.theta))
        self._conn.commit()

    @_locked
    def equity_curve(self) -> List[tuple]:
        return self._conn.execute(
            "SELECT day, equity FROM equity_curve ORDER BY day").fetchall()

    # ------------------------------------------------------------ 待确认订单
    @_locked
    def put_pending(self, order_key: str, decision_day: str, symbol: str,
                    direction: str, offset: str, qty: float, reason: str,
                    instrument: dict):
        self._conn.execute(
            "INSERT OR REPLACE INTO pending_orders VALUES (?,?,?,?,?,?,?,?,?)",
            (order_key, decision_day, symbol, direction, offset, qty, reason,
             "PENDING", json.dumps(instrument, ensure_ascii=False)))
        self._conn.commit()

    @_locked
    def pending(self, status: str = "PENDING") -> List[dict]:
        cur = self._conn.execute(
            "SELECT order_key, decision_day, symbol, direction, offset, qty, reason, "
            "status, instrument_json FROM pending_orders WHERE status=? ORDER BY decision_day",
            (status,))
        out = []
        for r in cur.fetchall():
            d = dict(zip(["order_key", "decision_day", "symbol", "direction", "offset",
                          "qty", "reason", "status", "instrument"], r))
            d["instrument"] = json.loads(d["instrument"])
            out.append(d)
        return out

    @_locked
    def set_order_status(self, order_key: str, status: str):
        self._conn.execute("UPDATE pending_orders SET status=? WHERE order_key=?",
                           (status, order_key))
        self._conn.commit()

    @_locked
    def close(self):
        self._conn.close()
