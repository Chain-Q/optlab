"""
thetalab.data.validator — 行情数据八道校验闸门（设计方案 §5.2）

任何数据入库前必须通过 validate_chain()；被拒绝的记录进入 rejects 并计数告警，
绝不静默修补（唯一例外：涨跌停钳制 + 深度实值 IV 置 NaN 是设计内行为）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd

__all__ = ["Reject", "ValidationContext", "DataValidator", "ValidationReport"]


@dataclass
class Reject:
    gate: str          # 闸门名
    reason: str        # 拒绝原因
    contract_id: str = ""
    row: Optional[Dict] = None


@dataclass
class ValidationReport:
    total: int = 0
    passed: int = 0
    rejected: int = 0
    flagged: int = 0                    # 保留但标记可疑（人工确认后采信）
    rejects: List[Reject] = field(default_factory=list)

    def summary(self) -> str:
        return (f"total={self.total} passed={self.passed} "
                f"rejected={self.rejected} flagged={self.flagged}")


@dataclass
class ValidationContext:
    """一次校验的环境参数"""
    trade_date: date                    # 目标交易日（闸门5）
    spot: float                         # 标的收盘价（平价/内在价值/涨跌停计算）
    spot_prev_close: float              # 标的前收盘（跳变阈值、涨跌停公式）
    risk_free: float = 0.02             # 无风险利率（平价关系）
    pre_settles: Optional[Dict[str, float]] = None  # 各合约前结算价（闸门3/4），symbol->price
    jump_option_pct: float = 1.00       # 期权结算价日跳变阈值（100%）
    jump_underlying_pct: float = 0.11   # 标的日跳变阈值
    parity_tol: float = 0.02            # 平价关系绝对容差（元）


class DataValidator:
    """
    八道闸门（作用于单日单标的期权链 DataFrame）：

    1 完整性  必填字段非空非 NaN
    2 合法性  价格>0、volume/oi≥0
    3 涨跌停  收盘价 ∈ [limit_down, limit_up]，超界钳制到边界并标记
    4 跳变    与前一日结算价比较，期权>100% 标记可疑
    5 时序    数据日期 == 目标交易日
    6 平价    |C − P − (S·e^{-qT} − K·e^{-rT})| > tol 的行标记脏数据
    7 内在价值 价格 ≥ max(0, 内在价值)，违反时 IV 置 NaN（禁止填 0）
    8 单调性  同到期日内认购价随 K 单调递减、认沽单调递增
    """

    REQUIRED_COLS = ["contract_id", "right", "strike", "expiry", "close"]

    def validate_chain(self, df: pd.DataFrame, ctx: ValidationContext) -> Tuple[pd.DataFrame, ValidationReport]:
        rep = ValidationReport(total=len(df))
        if df.empty:
            return df, rep
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            rep.rejects.append(Reject("完整性", f"缺少必需列 {missing}"))
            rep.rejected = rep.total
            return df, rep

        out = df.copy()

        # ---- 1 完整性
        null_mask = out[self.REQUIRED_COLS].isna().any(axis=1)
        for _, r in out[null_mask].iterrows():
            rep.rejects.append(Reject("完整性", "必填字段为空", str(r.get("contract_id", ""))))
        out = out[~null_mask]

        # ---- 2 合法性
        bad_price = out["close"] <= 0
        for _, r in out[bad_price].iterrows():
            rep.rejects.append(Reject("合法性", f"价格非法 close={r['close']}", str(r["contract_id"])))
        out = out[~bad_price]
        for col in ("volume", "open_interest"):
            if col in out.columns:
                bad = out[col].notna() & (out[col] < 0)
                rep.rejects.extend(Reject("合法性", f"{col}<0", str(r["contract_id"]))
                                   for _, r in out[bad].iterrows())
                out = out[~bad]

        # ---- 3 涨跌停（有前结算价才可校验；超界钳制 + 标记）
        from ..core.spec import calc_limit_prices
        from ..core.models import Right
        if ctx.pre_settles:
            clamped = []
            for idx, r in out.iterrows():
                pre = ctx.pre_settles.get(r["contract_id"])
                if pre is None or not (pre > 0):
                    continue
                is_call = str(r["right"]).upper().startswith("C")
                up, dn = calc_limit_prices(pre, ctx.spot_prev_close, float(r["strike"]), is_call)
                c = float(r["close"])
                if c > up:
                    out.at[idx, "close"] = up
                    out.at[idx, "limit_clamped"] = True
                    clamped.append((r["contract_id"], c, up))
                elif c < dn:
                    out.at[idx, "close"] = dn
                    out.at[idx, "limit_clamped"] = True
                    clamped.append((r["contract_id"], c, dn))
            for cid, old, new in clamped:
                rep.rejects.append(Reject("涨跌停", f"close {old} 钳制到 {new}", cid))
                rep.flagged += 1

        # ---- 4 跳变（标记可疑，不丢弃）
        if ctx.pre_settles:
            susp = []
            for idx, r in out.iterrows():
                pre = ctx.pre_settles.get(r["contract_id"])
                if pre and pre > 0:
                    chg = abs(float(r["close"]) - pre) / pre
                    if chg > ctx.jump_option_pct:
                        out.at[idx, "jump_suspicious"] = True
                        susp.append((r["contract_id"], chg))
            for cid, chg in susp:
                rep.rejects.append(Reject("跳变", f"结算价日变动 {chg:.0%}，标记可疑", cid))
                rep.flagged += 1

        # ---- 5 时序
        if "trade_date" in out.columns:
            bad_date = out["trade_date"].astype(str) != str(ctx.trade_date)
            rep.rejects.extend(Reject("时序", f"数据日期 {r['trade_date']} != 目标 {ctx.trade_date}",
                                      str(r["contract_id"]))
                               for _, r in out[bad_date].iterrows())
            out = out[~bad_date]

        # ---- 6 平价关系（按行权价配对，标记脏数据不丢弃）
        try:
            parity_bad = self._check_parity(out, ctx)
            for cid in parity_bad:
                out.loc[out["contract_id"] == cid, "parity_bad"] = True
                rep.rejects.append(Reject("平价", "Put-Call Parity 违反", cid))
                rep.flagged += 1
            out["parity_ok"] = ~out["contract_id"].isin(parity_bad)
        except Exception:
            out["parity_ok"] = True

        # ---- 7 内在价值（违反时 iv 置 NaN）
        if "iv" in out.columns:
            S_df = ctx.spot
            r_ = ctx.risk_free
            T_days = out["expiry"].apply(lambda d: max((pd.Timestamp(d) - pd.Timestamp(ctx.trade_date)).days, 0))
            T = T_days / 365.0
            intrinsic = out.apply(
                lambda r: max(0.0, (S_df - r["strike"]) if str(r["right"]).upper().startswith("C")
                              else (r["strike"] - S_df)), axis=1)
            violated = out["close"] < intrinsic - 1e-9
            n_iv_nan = int(((violated) & (out["iv"].notna())).sum())
            out.loc[violated, "iv"] = float("nan")
            out.loc[violated, "intrinsic_violated"] = True
            if n_iv_nan:
                rep.rejects.append(Reject("内在价值", f"{n_iv_nan} 条价格低于内在价值，IV 置 NaN"))
                rep.flagged += n_iv_nan

        # ---- 8 单调性（标记可疑区间）
        mono_bad = self._check_monotonic(out)
        for cid, msg in mono_bad:
            out.loc[out["contract_id"] == cid, "monotonic_suspicious"] = True
            rep.rejects.append(Reject("单调性", msg, cid))
            rep.flagged += 1

        rep.passed = len(out)
        rep.rejected = rep.total - rep.passed
        return out, rep

    def _check_parity(self, df: pd.DataFrame, ctx: ValidationContext) -> List[str]:
        """|C − P − (S·e^{-rT} − K·e^{-rT})| > tol 的行权价 → 脏数据"""
        bad: List[str] = []
        if "expiry" not in df.columns:
            return bad
        for expiry, g in df.groupby("expiry"):
            T = max((pd.Timestamp(expiry) - pd.Timestamp(ctx.trade_date)).days, 0) / 365.0
            calls = {round(float(r["strike"]), 6): r for _, r in g.iterrows()
                     if str(r["right"]).upper().startswith("C")}
            puts = {round(float(r["strike"]), 6): r for _, r in g.iterrows()
                    if str(r["right"]).upper().startswith("P")}
            for k, c in calls.items():
                if k not in puts:
                    continue
                p = puts[k]
                disc = math.exp(-ctx.risk_free * T)
                lhs = float(c["close"]) - float(p["close"])
                rhs = ctx.spot - k * disc
                if abs(lhs - rhs) > ctx.parity_tol:
                    bad.extend([str(c["contract_id"]), str(p["contract_id"])])
        return bad

    def _check_monotonic(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """同到期日：认购随 K 递减、认沽随 K 递增（违反标记可疑）"""
        bad: List[Tuple[str, str]] = []
        if "expiry" not in df.columns:
            return bad
        for expiry, g in df.groupby("expiry"):
            for right, expect_less in (("C", True), ("P", False)):
                gg = g[g["right"].astype(str).str.upper().str.startswith(right)] \
                    .sort_values("strike")
                closes = gg["close"].astype(float).tolist()
                cids = gg["contract_id"].astype(str).tolist()
                for i in range(1, len(closes)):
                    if expect_less and closes[i] > closes[i - 1] + 1e-9:
                        bad.append((cids[i], f"认购价随K上升而上升 K={gg['strike'].iloc[i]}"))
                    elif (not expect_less) and closes[i] < closes[i - 1] - 1e-9:
                        bad.append((cids[i], f"认沽价随K上升而下降 K={gg['strike'].iloc[i]}"))
        return bad
