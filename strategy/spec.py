"""
optlab.strategy.spec — 声明式策略 DSL（设计方案 §5.4）

策略不写死合约，而写「选择规则」（Selector）：
    strike_selector: DELTA(target,tol) / MONEYNESS(pct) / ATM / FIXED(strike)
    expiry_selector: DTE_RANGE(min,max) / NEAREST_EXPIRY(n)
JSON 可序列化（保存/分享/回测），Resolver 把 spec + 当日链 → 订单意图。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..core.models import Direction, Instrument, Offset, Order, Right

__all__ = ["StrikeSelector", "ExpirySelector", "LegSpec", "ExitRules", "Sizing",
           "StrategySpec", "resolve_legs"]


# ---------------------------------------------------------------- 选择器


@dataclass
class StrikeSelector:
    type: str = "DELTA"          # DELTA / MONEYNESS / ATM / FIXED
    target: float = 0.20         # DELTA 目标（绝对值）/ MONEYNESS 虚值比例
    tol: float = 0.10            # DELTA 容差
    strike: float = 0.0          # FIXED 用

    def pick(self, chain: pd.DataFrame, spot: float) -> Optional[pd.Series]:
        # P1-2（2026-08-29 检验）：IV 缺失系统性偏向实值档，任何选合约路径都必须显式排除
        if "iv" in chain.columns:
            chain = chain[chain["iv"].notna()]
        g = chain
        if self.type == "DELTA":
            g = g[g["delta"].notna()]
            if g.empty:
                return None
            g = g.assign(_err=(g["delta"].abs() - self.target).abs())
            row = g.loc[g["_err"].idxmin()]
            if abs(row["delta"]) > self.target + self.tol:
                return None
            return row
        if self.type == "MONEYNESS":
            # 虚值 side 的行权价：call 上方 / put 下方
            g = g.assign(_err=(g["strike"] / spot - 1.0).abs())
            row = g.loc[g["_err"].idxmin()]
            if abs(row["strike"] / spot - 1.0) > self.target * 2:
                return None
            return row
        if self.type == "ATM":
            return g.loc[(g["strike"] - spot).abs().idxmin()]
        if self.type == "FIXED":
            g2 = g[g["strike"] == self.strike]
            return g2.iloc[0] if not g2.empty else None
        raise ValueError(f"未知 strike_selector.type: {self.type}")


@dataclass
class ExpirySelector:
    type: str = "DTE_RANGE"      # DTE_RANGE / NEAREST_EXPIRY
    min: int = 25
    max: int = 45
    n: int = 1                   # NEAREST_EXPIRY: 第 n 个到期月

    def pick(self, expiries: List[date], today: date) -> Optional[date]:
        exps = sorted(set(expiries))
        if self.type == "DTE_RANGE":
            ok = [e for e in exps if self.min <= (e - today).days <= self.max]
            return ok[0] if ok else None
        if self.type == "NEAREST_EXPIRY":
            future = [e for e in exps if e >= today]
            return future[self.n - 1] if len(future) >= self.n else None
        raise ValueError(f"未知 expiry_selector.type: {self.type}")


# ---------------------------------------------------------------- 腿与规则


@dataclass
class LegSpec:
    kind: str = "OPTION"                       # OPTION / UNDERLYING
    right: Optional[str] = None                # CALL / PUT（UNDERLYING 时 None）
    direction: str = "SELL"                    # BUY / SELL
    ratio: int = 1                             # 张数倍数（蝶式中间腿 = 2）
    strike_selector: StrikeSelector = field(default_factory=StrikeSelector)
    expiry_selector: ExpirySelector = field(default_factory=ExpirySelector)

    def validate(self) -> Optional[str]:
        if self.kind not in ("OPTION", "UNDERLYING"):
            return f"非法 kind {self.kind}"
        if self.direction not in ("BUY", "SELL"):
            return f"非法 direction {self.direction}"
        if self.kind == "OPTION" and self.right not in ("CALL", "PUT"):
            return "OPTION 腿必须指定 CALL/PUT"
        return None


@dataclass
class ExitRules:
    """退出计划（§5.4：止盈/止损/时间/移仓）"""
    take_profit_pct: float = 0.60      # 赚取权利金 60% 即平（卖方）
    stop_multiple: float = 2.0         # 亏损达权利金 2 倍止损（卖方）
    time_stop_dte: int = 7             # 剩余 7 天强制平仓/移仓
    roll_enable: bool = True


@dataclass
class Sizing:
    type: str = "PCT_OF_EQUITY"        # PCT_OF_EQUITY / FIXED_LOTS
    value: float = 0.10                # 每腿权利金名义/保证金 ≤ 权益×value
    max_contracts: int = 50


@dataclass
class StrategySpec:
    id: str = ""
    name: str = ""
    underlying: str = "510300"
    category: str = "volatility"       # direction / spread / volatility / hedge
    legs: List[LegSpec] = field(default_factory=list)
    filters: Dict = field(default_factory=lambda: {
        "min_open_interest": 1000, "max_spread_pct": 0.10, "min_dte": 20})
    entry: Dict = field(default_factory=lambda: {"conditions": []})   # 如 ["iv_rank > 0.5"]
    exit: ExitRules = field(default_factory=ExitRules)
    sizing: Sizing = field(default_factory=Sizing)
    notes: str = ""

    # ---- 序列化
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "StrategySpec":
        d = json.loads(s)
        legs = [LegSpec(
            kind=l["kind"], right=l.get("right"), direction=l["direction"],
            ratio=l.get("ratio", 1),
            strike_selector=StrikeSelector(**l["strike_selector"]),
            expiry_selector=ExpirySelector(**l["expiry_selector"])) for l in d["legs"]]
        return cls(id=d["id"], name=d["name"], underlying=d["underlying"],
                   category=d.get("category", "volatility"), legs=legs,
                   filters=d.get("filters", {}), entry=d.get("entry", {}),
                   exit=ExitRules(**d["exit"]), sizing=Sizing(**d["sizing"]),
                   notes=d.get("notes", ""))

    def validate(self) -> List[str]:
        errs = []
        if not self.legs:
            errs.append("至少 1 条腿")
        for i, l in enumerate(self.legs):
            e = l.validate()
            if e:
                errs.append(f"腿{i}: {e}")
        # 卖方腿存在时必须标注实盘权限（§5.5 固定提示）
        if any(l.direction == "SELL" for l in self.legs):
            self.notes = (self.notes + " | ⚠️实盘需三级权限+50万验资").strip(" |")
        return errs


# ---------------------------------------------------------------- Resolver


def resolve_legs(spec: StrategySpec, chain: pd.DataFrame, today: date,
                 spot: float, equity: float, fee_per_lot: float = 5.0,
                 margin_of: Optional[object] = None) -> Tuple[List[Order], List[str]]:
    """
    spec + 当日链 → 订单列表。返回 (orders, skip_reasons)。
    chain 需含: contract_id/strike/right/expiry/delta/close/volume/_instrument/oi。
    仓位：卖方腿按保证金上限（PCT_OF_EQUITY）、买方腿按权利金上限折算张数。
    """
    orders: List[Order] = []
    skips: List[str] = []
    min_dte = spec.filters.get("min_dte", 20)
    min_oi = spec.filters.get("min_open_interest", 1000)
    min_vol = spec.filters.get("min_volume", 100)          # 复检 P1：成交量闸门
    max_mny = spec.filters.get("max_moneyness_dev", 0.15)  # 复检 P1：|K/S-1| 上限
                                                          # （深度实值流动性折价，收盘价可低于内在价值）

    # 信用类（含卖方）：额度 = 权益 × sizing.value，按最贵腿保证金折算
    is_credit = any(l.direction == "SELL" for l in spec.legs)
    budget = equity * spec.sizing.value

    for i, leg in enumerate(spec.legs):
        err = leg.validate()
        if err:
            skips.append(f"腿{i}: {err}")
            continue
        if leg.kind == "UNDERLYING":
            # 标的腿（备兑/领口）：按合约单位匹配期权张数，回测中用 ETF 替代
            orders.append(Order(instrument=Instrument(symbol=spec.underlying, name=spec.underlying,
                                                      underlying=spec.underlying, multiplier=1.0),
                                direction=Direction[leg.direction], offset=Offset.OPEN,
                                qty=0.0,   # 张数由期权腿数量决定（resolve 后补）
                                strategy_id=spec.id, reason=f"{spec.name} 标的腿"))
            continue
        g = chain[chain["right"] == leg.right]
        if g.empty:
            skips.append(f"腿{i}: 链上无 {leg.right}")
            continue
        exp = leg.expiry_selector.pick(sorted(chain["expiry"].unique()), today)
        if exp is None:
            skips.append(f"腿{i}: 无满足 {leg.expiry_selector.type} 的到期月")
            continue
        g2 = g[g["expiry"] == exp]
        row = leg.strike_selector.pick(g2, spot)
        if row is None:
            skips.append(f"腿{i}: {leg.strike_selector.type} 无满足档位")
            continue
        if (row["expiry"] - today).days < min_dte:
            skips.append(f"腿{i}: DTE {(row['expiry'] - today).days} < {min_dte}")
            continue
        # OI 仅在数据真实存在时比较（risk_indicators 源无 OI 列 → 不拒，交给成交量闸门）
        oi = row.get("open_interest", row.get("oi", float("nan")))
        if pd.notna(oi) and oi > 0 and oi < min_oi:
            skips.append(f"腿{i}: OI {oi} < {min_oi}")
            continue
        vol = row.get("volume", float("nan"))
        if pd.notna(vol) and vol < min_vol:
            skips.append(f"腿{i}: 成交量 {vol:.0f} < {min_vol}")
            continue
        if spec.filters.get("exclude_adjusted", True) and                 str(row.get("contract_id", "")).find("A") >= 0:
            skips.append(f"腿{i}: 调整合约(A)流动性折价，默认排除")
            continue
        if spot > 0 and abs(float(row["strike"]) / spot - 1.0) > max_mny:
            skips.append(f"腿{i}: 在值度偏离 {float(row['strike']) / spot - 1.0:.1%} 超限")
            continue
        inst = row["_instrument"]
        if inst is None:
            skips.append(f"腿{i}: 当日无行情")
            continue

        qty = _size_leg(leg, row, budget, spec.sizing.max_contracts,
                        equity, fee_per_lot, margin_of, spot, exp, today)
        if qty <= 0:
            skips.append(f"腿{i}: 仓位折算为 0")
            continue
        orders.append(Order(instrument=inst, direction=Direction[leg.direction],
                            offset=Offset.OPEN, qty=qty, strategy_id=spec.id,
                            reason=f"{spec.name} 腿{i}: {leg.direction} {leg.right} "
                                   f"K={row['strike']} Δ={row.get('delta', float('nan')):.2f}"))
    return orders, skips


def _size_leg(leg: LegSpec, row: pd.Series, budget: float, max_contracts: int,
              equity: float, fee_per_lot: float, margin_of, spot: float,
              exp: date, today: date) -> int:
    """按腿类型折算张数：卖方按保证金、买方按权利金；ratio 为倍数"""
    per_lot_cost = float(row["close"]) * 10000.0   # ETF 期权合约单位 10000
    if leg.direction == "SELL" and margin_of is not None:
        try:
            per_lot = margin_of(row, spot, exp, today)
        except Exception:
            per_lot = per_lot_cost * 2.0
    else:
        per_lot = per_lot_cost
    if per_lot <= 0:
        return 0
    qty = int(min(budget / per_lot, max_contracts) // max(leg.ratio, 1)) * max(leg.ratio, 1)
    return max(0, qty)
