"""
thetalab.engine.metrics — 绩效评估（设计方案 §7.2）
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from ..core.models import Trade

__all__ = ["performance_metrics", "trade_stats", "fee_share"]


def _returns(equity: Sequence[float]) -> List[float]:
    return [(equity[i] / equity[i - 1] - 1.0) for i in range(1, len(equity)) if equity[i - 1] > 0]


def performance_metrics(equity_curve: List[Tuple], risk_free: float = 0.0,
                        periods: int = 252) -> Dict[str, float]:
    """
    equity_curve: [(date, equity), ...]
    返回总收益/年化/波动/夏普/Sortino/Calmar/最大回撤等。
    口径说明（复检 2026-08-29）：rf 默认 0。曾用 rf=2%，对年化波动仅 ~2% 的
    日频卖方策略，rf 会把正收益组的夏普系统性打成负值（实测复现 -0.133），
    导致按夏普选参数选到最差组——故默认归零，需要 rf 时显式传入。
    """
    eq = [e for _, e in equity_curve] if equity_curve and isinstance(equity_curve[0], (tuple, list)) \
        else list(equity_curve)
    if len(eq) < 2:
        return {}
    total_return = eq[-1] / eq[0] - 1.0
    n = len(eq) - 1
    ann_return = (1.0 + total_return) ** (periods / n) - 1.0
    rets = _returns(eq)
    mean_r = sum(rets) / len(rets)
    var = sum((r - mean_r) ** 2 for r in rets) / max(len(rets) - 1, 1)
    vol = math.sqrt(var)
    ann_vol = vol * math.sqrt(periods)
    rf_daily = risk_free / periods
    sharpe = ((mean_r - rf_daily) / vol * math.sqrt(periods)) if vol > 1e-12 else float("nan")
    downside = [r - rf_daily for r in rets if r < rf_daily]
    dvar = sum(d * d for d in downside) / max(len(downside), 1)
    dvol = math.sqrt(dvar)
    sortino = ((mean_r - rf_daily) / dvol * math.sqrt(periods)) if dvol > 1e-12 else float("nan")
    from .portfolio import max_drawdown_stats
    mdd = max_drawdown_stats(eq)["max_drawdown"]
    calmar = ann_return / mdd if mdd > 1e-9 else float("nan")
    return {
        "total_return": round(total_return, 4),
        "annual_return": round(ann_return, 4),
        "annual_vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown": round(mdd, 4),
        "calmar": round(calmar, 3),
        "trading_days": n,
    }


def trade_stats(trades: List[Trade]) -> Dict[str, float]:
    """胜率结构（仅统计主动平仓成交，行权结算计入但单独标注）"""
    closed = [t for t in trades if t.realized_pnl != 0.0]
    wins = [t for t in closed if t.realized_pnl > 0]
    losses = [t for t in closed if t.realized_pnl <= 0]
    avg_win = sum(t.realized_pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(t.realized_pnl for t in losses) / len(losses)) if losses else 0.0
    return {
        "closed_trades": len(closed),
        "win_rate": round(len(wins) / len(closed), 4) if closed else float("nan"),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_ratio": round(avg_win / avg_loss, 3) if avg_loss > 0 else float("nan"),
        "max_consecutive_losses": _max_consec_losses(closed),
    }


def _max_consec_losses(trades: List[Trade]) -> int:
    best = cur = 0
    for t in trades:
        if t.realized_pnl <= 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def fee_share(trades: List[Trade], total_pnl: float) -> float:
    total_fee = sum(t.fee for t in trades)
    return round(total_fee / abs(total_pnl), 4) if total_pnl else float("nan")
