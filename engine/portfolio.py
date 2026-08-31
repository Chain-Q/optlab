"""
optlab.engine.portfolio — 组合内核：盯市 / 希腊聚合 / 压力矩阵 / 损益归因（§5.3、§7.2）

口径约定（与 pricing.py 一致，组合汇总须 ×张数×合约单位）：
    组合 delta = Σ delta_i × qty_i × mult  → 标的等值股数
    组合 vega  = Σ vega_i  × qty_i × mult  → 每 1 vol point 的金额
    组合 theta = Σ theta_i × qty_i × mult  → 每自然日金额
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..core.models import Account, Greeks, Position, Right
from ..core.spec import calc_margin

__all__ = ["PortfolioState", "Portfolio", "max_drawdown_stats", "pnl_attribution"]


@dataclass
class PortfolioState:
    """单日组合快照（归因输入）"""
    day: date
    equity: float
    spot: float
    atm_iv: float = float("nan")
    delta: float = 0.0     # 标的等值股数
    gamma: float = 0.0     # Δdelta/元
    vega: float = 0.0      # 元 / vol point
    theta: float = 0.0     # 元 / 自然日
    margin_used: float = 0.0
    # 逐腿明细 {sym: {delta,gamma,vega,theta,iv,qm,price,avg_open}}
    # qm = net_qty × 合约单位。提供时归因走逐腿路径（P0-3 修复）
    legs: Optional[Dict[str, Dict]] = None


class Portfolio:
    """账户的组合视图：盯市、希腊聚合、维持保证金盯市"""

    def __init__(self, account: Account):
        self.account = account
        self.states: List[PortfolioState] = []

    # ---- 盯市
    def update_mark(self, prices: Dict[str, float]) -> None:
        for sym, pos in self.account.positions.items():
            if sym in prices and prices[sym] == prices[sym]:
                pos.last_price = prices[sym]

    def margin_refresh(self, settle_map: Dict[str, float], spot_close) -> None:
        """
        义务仓按维持保证金盯市（结算价基准，缺失时用当前价近似）。
        多品种：spot_close 可传 callable(instrument) -> 该合约标的当日收盘。
        """
        spot_of = spot_close if callable(spot_close) else (lambda inst: spot_close)
        total = 0.0
        for sym, pos in self.account.positions.items():
            if pos.net_qty >= 0:
                pos.margin = 0.0
                continue
            inst = pos.instrument
            opt_price = settle_map.get(sym)
            if opt_price is None or opt_price <= 0:
                opt_price = pos.last_price if pos.last_price == pos.last_price else pos.avg_open_price
            m = calc_margin(inst, option_price=opt_price, spot_close=spot_of(inst),
                            is_short=True, is_call=inst.right is Right.CALL,
                            maintenance=True)
            pos.margin = m * abs(pos.net_qty)
            total += pos.margin
        self.account.margin_used = total

    # ---- 希腊聚合
    def aggregate_greeks(self, greeks_map: Dict[str, Greeks]) -> Greeks:
        agg = Greeks()
        for sym, pos in self.account.positions.items():
            g = greeks_map.get(sym)
            if g is None or g.iv != g.iv:  # 无有效 Greeks 的腿跳过（计数留痕）
                continue
            k = pos.net_qty * pos.instrument.multiplier
            agg.delta += g.delta * k
            agg.gamma += g.gamma * k
            agg.vega += g.vega * k
            agg.theta += g.theta * k
        return agg

    # ---- 快照与回撤
    def snapshot(self, day: date, spot: float, atm_iv: float = float("nan"),
                 greeks_map: Optional[Dict[str, Greeks]] = None,
                 legs: Optional[Dict[str, Dict]] = None) -> PortfolioState:
        g = self.aggregate_greeks(greeks_map) if greeks_map else Greeks()
        st = PortfolioState(day=day, equity=self.account.equity, spot=spot,
                            atm_iv=atm_iv, delta=g.delta, gamma=g.gamma,
                            vega=g.vega, theta=g.theta,
                            margin_used=self.account.margin_used, legs=legs)
        self.states.append(st)
        return st

    def margin_ratio(self) -> float:
        return self.account.margin_ratio

    def equity_curve(self) -> List[Tuple[date, float]]:
        return [(s.day, s.equity) for s in self.states]

    # ---- 压力矩阵（Greeks 一阶+二阶近似，§5.3）
    def stress_matrix(self, spot: float, spot_pcts=(-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03),
                      vol_shifts=(-2.0, -1.0, 0.0, 1.0, 2.0, 5.0)) -> pd.DataFrame:
        st = self.states[-1] if self.states else None
        if st is None:
            return pd.DataFrame()
        rows = {}
        for pct in spot_pcts:
            dS = spot * pct
            row = {}
            for dv in vol_shifts:
                pnl = (st.delta * dS + 0.5 * st.gamma * dS * dS
                       + st.vega * dv + st.theta)   # theta 含 1 日衰减
                row[f"{dv:+.0f}v"] = round(pnl, 0)
            rows[f"{pct:+.0%}"] = row
        return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------- 归因

def pnl_attribution(prev: PortfolioState, curr: PortfolioState) -> Dict[str, float]:
    """
    每日损益分解（§7.2）。两条路径：

    逐腿路径（prev/curr 均提供 legs 时，P0-3 修复后的主路径）：
        对两日共同持有的腿逐腿分解——vega 用该腿自身 IV 日变化（而非 ATM 序列，
        规避 ATM 定位跳档噪声）；新开腿的价格变化单独归 trade 项。
    组合路径（legs 缺失时，向后兼容）：
        ΔP = delta×ΔS + ½gamma×ΔS² + vega×Δσ_atm + theta×1日 + 残差
    """
    dS = curr.spot - prev.spot
    actual = curr.equity - prev.equity

    if prev.legs is not None and curr.legs is not None:
        common = prev.legs.keys() & curr.legs.keys()
        delta_pnl = gamma_pnl = vega_pnl = theta_pnl = 0.0
        for sym in common:
            a, b = prev.legs[sym], curr.legs[sym]
            qm = a["qm"]
            delta_pnl += a["delta"] * dS * qm
            gamma_pnl += 0.5 * a["gamma"] * dS * dS * qm
            if a["iv"] == a["iv"] and b["iv"] == b["iv"]:
                vega_pnl += (b["iv"] - a["iv"]) * 100.0 * a["vega"] * qm
            theta_pnl += a["theta"] * qm
        # 新开腿：开仓后到当日收盘的价格变化（属交易效应而非风险因子）
        trade_pnl = 0.0
        for sym in curr.legs.keys() - prev.legs.keys():
            b = curr.legs[sym]
            if b["price"] == b["price"] and b["avg_open"]:
                trade_pnl += (b["price"] - b["avg_open"]) * b["qm"]
        approx = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + trade_pnl
        return {"delta": round(delta_pnl, 2), "gamma": round(gamma_pnl, 2),
                "vega": round(vega_pnl, 2), "theta": round(theta_pnl, 2),
                "trade": round(trade_pnl, 2),
                "residual": round(actual - approx, 2),
                "total": round(actual, 2)}

    dvol_points = (curr.atm_iv - prev.atm_iv) * 100.0 \
        if (curr.atm_iv == curr.atm_iv and prev.atm_iv == prev.atm_iv) else 0.0
    delta_pnl = prev.delta * dS
    gamma_pnl = 0.5 * prev.gamma * dS * dS
    vega_pnl = prev.vega * dvol_points
    theta_pnl = prev.theta
    approx = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
    return {
        "delta": round(delta_pnl, 2),
        "gamma": round(gamma_pnl, 2),
        "vega": round(vega_pnl, 2),
        "theta": round(theta_pnl, 2),
        "residual": round(actual - approx, 2),
        "total": round(actual, 2),
    }


def max_drawdown_stats(equity: List[float]) -> Dict[str, float]:
    """最大回撤 / 回撤持续期（峰谷最长时间）"""
    if len(equity) < 2:
        return {"max_drawdown": 0.0, "current_drawdown": 0.0, "duration": 0}
    peak = equity[0]
    mdd, peak_i, trough_i, cur_peak_i = 0.0, 0, 0, 0
    for i, e in enumerate(equity):
        if e > peak:
            peak, cur_peak_i = e, i
        dd = 1.0 - e / peak
        if dd > mdd:
            mdd, peak_i, trough_i = dd, cur_peak_i, i
    return {"max_drawdown": round(mdd, 6),
            "current_drawdown": round(1.0 - equity[-1] / max(equity), 6),
            "duration": trough_i - peak_i}
