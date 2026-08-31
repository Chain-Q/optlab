"""
optlab.strategy.payoff — 盈亏结构计算（设计方案 §5.4）

输出四条曲线：到期损益 / T+0 理论 / D-7 / D-14，并求盈亏平衡点。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.models import Right
from ..core.pricing import bs_price

__all__ = ["PayoffLeg", "PayoffCurves", "build_curves"]


@dataclass
class PayoffLeg:
    right: Optional[Right]        # None = 标的
    strike: float
    expiry: date
    qty: float                    # 正=权利仓，负=义务仓（张）
    multiplier: float = 10000.0
    entry_price: float = 0.0      # 开仓权利金（每单位）
    iv: float = 0.16


@dataclass
class PayoffCurves:
    spots: np.ndarray = None
    at_expiry: np.ndarray = None       # 到期损益（元）
    t0: np.ndarray = None              # 当前理论（元）
    d7: np.ndarray = None              # 时间推进 7 天
    d14: np.ndarray = None
    breakevens: List[float] = field(default_factory=list)
    max_profit: float = float("nan")
    max_loss: float = float("nan")
    net_credit: float = 0.0            # 权利金净收支（正=收）


def _leg_pnl_at(leg: PayoffLeg, S: np.ndarray, T_years: float, r: float) -> np.ndarray:
    """单腿在价格序列上的每单位损益 × qty × mult（含开仓成本）"""
    if leg.right is None:                      # 标的
        return (S - leg.strike) * leg.qty * leg.multiplier
    if T_years <= 0:                           # 到期：内在价值
        px = np.maximum(0.0, (S - leg.strike) if leg.right is Right.CALL
                        else (leg.strike - S))
    else:                                      # 理论价
        px = np.array([bs_price(float(s), leg.strike, T_years, r, leg.iv, leg.right)
                       for s in S])
    return (px - leg.entry_price) * leg.qty * leg.multiplier


def build_curves(legs: List[PayoffLeg], spot: float, asof: Optional[date] = None,
                 n: int = 200, width: float = 0.30, r: float = 0.03) -> PayoffCurves:
    asof = asof or date.today()
    S = np.linspace(spot * (1 - width), spot * (1 + width), n)

    def total(T_years: float) -> np.ndarray:
        out = np.zeros(n)
        for leg in legs:
            out += _leg_pnl_at(leg, S, T_years, r)
        return out

    dte = max((max(l.expiry for l in legs if l.expiry) - asof).days, 0) \
        if any(l.expiry for l in legs) else 0
    at_exp = total(0.0)
    t0 = total(dte / 365.0)
    d7 = total(max(dte - 7, 0) / 365.0)
    d14 = total(max(dte - 14, 0) / 365.0)

    net_credit = sum(-l.entry_price * l.qty * l.multiplier for l in legs)
    breakevens = _find_breakevens(S, at_exp)
    max_profit = float(at_exp.max())
    max_loss = float(at_exp.min())
    return PayoffCurves(spots=S, at_expiry=at_exp, t0=t0, d7=d7, d14=d14,
                        breakevens=breakevens, max_profit=max_profit,
                        max_loss=max_loss, net_credit=net_credit)


def _find_breakevens(S: np.ndarray, pnl: np.ndarray) -> List[float]:
    """到期损益曲线的零点（数值求根）"""
    roots = []
    for i in range(1, len(S)):
        if pnl[i - 1] == 0:
            roots.append(float(S[i - 1]))
        if pnl[i - 1] * pnl[i] < 0:
            # 线性插值
            x0, x1, y0, y1 = S[i - 1], S[i], pnl[i - 1], pnl[i]
            roots.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
    return [round(x, 4) for x in roots]
