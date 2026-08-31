"""
optlab.core.indicators — 标的技术指标与 UnderlyingSnapshot 构建（§4.1 / §5.5 输入）

全部输入为标的日线 DataFrame（date/open/high/low/close），输出快照数据类。
IV 相关字段（iv_atm/iv_rank）由数据层（atm_iv 序列）注入。
"""
from __future__ import annotations

import math
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = ["UnderlyingIndicators", "build_indicators", "iv_rank_of"]


class UnderlyingIndicators(dict):
    """指标字典：ma5/ma20/ma60/atr14/adx/rv20/rv60/trend/rsi14/bb_pos/high20/low20"""


def build_indicators(df: pd.DataFrame) -> UnderlyingIndicators:
    """
    df: 标的日线（date 升序, close/high/low 必需），取截至 asof 的窗口。
    返回指标字典（最新一根的值）。
    """
    d = df.sort_values("date").reset_index(drop=True)
    c = d["close"].astype(float)
    out = UnderlyingIndicators()

    out["close"] = float(c.iloc[-1])
    for n in (5, 20, 60):
        out[f"ma{n}"] = float(c.rolling(n).mean().iloc[-1]) if len(c) >= n else float("nan")
    # ATR14（Wilder 平滑近似）
    if len(d) >= 15:
        tr = pd.concat([d["high"] - d["low"],
                        (d["high"] - c.shift()).abs(),
                        (d["low"] - c.shift()).abs()], axis=1).max(axis=1)
        out["atr14"] = float(tr.rolling(14).mean().iloc[-1])
    else:
        out["atr14"] = float("nan")
    # ADX14（简化 Wilder）
    out["adx"] = _adx(d, 14) if len(d) >= 28 else float("nan")
    # 已实现波动率（对数收益年化）
    ret = np.log(c / c.shift())
    for n in (20, 60):
        out[f"rv{n}"] = float(ret.rolling(n).std().iloc[-1] * math.sqrt(252)) \
            if len(c) > n else float("nan")
    # RSI14
    out["rsi14"] = _rsi(c, 14) if len(c) >= 15 else float("nan")
    # ATR% 及其历史低分位（蓄势信号阈值）
    if len(d) >= 40:
        tr_s = pd.concat([d["high"] - d["low"],
                          (d["high"] - c.shift()).abs(),
                          (d["low"] - c.shift()).abs()], axis=1).max(axis=1)
        atr_pct_s = tr_s.rolling(14).mean() / c
        out["atr_pct"] = float(atr_pct_s.iloc[-1])
        out["atr_pct_low"] = float(atr_pct_s.rolling(120, min_periods=40)
                                   .quantile(0.20).iloc[-1])
    # 布林位置（20日）
    if len(c) >= 20:
        ma = c.rolling(20).mean().iloc[-1]
        sd = c.rolling(20).std().iloc[-1]
        out["bb_pos"] = (out["close"] - ma) / (2 * sd) if sd > 0 else 0.0   # >1 突破上轨
        out["high20"] = float(c.rolling(20).max().iloc[-2])
        out["low20"] = float(c.rolling(20).min().iloc[-2])
    else:
        out["bb_pos"] = 0.0
        out["high20"] = out["low20"] = float("nan")
    # 趋势状态
    ma5, ma20, ma60 = out.get("ma5"), out.get("ma20"), out.get("ma60")
    adx = out.get("adx", float("nan"))
    if ma5 > ma20 > ma60:
        out["trend"] = "UP"
    elif ma5 < ma20 < ma60:
        out["trend"] = "DOWN"
    else:
        out["trend"] = "CHOP"
    out["trend_strong"] = (adx == adx and adx > 25)
    out["choppy"] = (adx == adx and adx < 20)
    return out


def _adx(d: pd.DataFrame, n: int) -> float:
    up = d["high"].diff()
    dn = -d["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - d["close"].shift()).abs(),
                    (d["low"] - d["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    pdi = 100 * pd.Series(plus_dm, index=d.index).rolling(n).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=d.index).rolling(n).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return float(dx.rolling(n).mean().iloc[-1])


def _rsi(c: pd.Series, n: int) -> float:
    diff = c.diff()
    gain = diff.clip(lower=0).rolling(n).mean().iloc[-1]
    loss = (-diff.clip(upper=0)).rolling(n).mean().iloc[-1]
    if loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + gain / loss)


def iv_rank_of(atm_iv_series: pd.Series, current_iv: float) -> Dict[str, float]:
    """IV rank（区间位置）与百分位；序列需 ≥60 点否则返回 NaN"""
    s = atm_iv_series.dropna()
    if len(s) < 60 or current_iv != current_iv:
        return {"iv_rank": float("nan"), "iv_pct": float("nan"), "n": len(s)}
    lo, hi = float(s.min()), float(s.max())
    rank = (current_iv - lo) / (hi - lo) if hi > lo else float("nan")
    pct = float((s < current_iv).mean())
    return {"iv_rank": rank, "iv_pct": pct, "n": len(s),
            "iv_low": lo, "iv_high": hi}
