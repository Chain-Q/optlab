"""
optlab.engine.broker — 撮合内核（回测与模拟盘共用，设计方案 §5.3）

设计铁律（§3）：回测与模拟盘唯一差异是数据来源与时钟，撮合/费用/保证金代码一字不差。
撮合哲学：日频数据无盘口，**宁可低估收益，不可高估**。

价差模型（2026-08-29 用 510300 近月 30 合约真实盘口实测校准）：
    主导项是「最小变动价位 / 权利金」——深度虚值低价合约 spread 可达 10%~35%；
    比例项 ≈ 0.3% + 0.4×|K/S−1| + 0.15×max(DTE−30,0)/365（ATM 实测 0.3%~0.5%）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

from ..core.models import (
    Account, Direction, FeeRule, Greeks, Instrument, Offset, Order,
    OrderStatus, Position, Right, Trade,
)
from ..core.spec import calc_margin

__all__ = ["FillMode", "RiskLimits", "estimate_spread_pct", "Broker"]


class FillMode(str, Enum):
    SETTLE = "SETTLE"                # 当日结算价（最接近真实成交均价）
    CLOSE = "CLOSE"                  # 收盘价
    CLOSE_SLIPPAGE = "CLOSE_SLIPPAGE"  # 收盘价 ± 估计价差/2（默认，无盘口时）
    MID = "MID"                      # (bid+ask)/2，需延时快照
    CROSS = "CROSS"                  # 买用 ask 卖用 bid（最保守）


def estimate_spread_pct(price: float, moneyness_dev: float,
                        dte: int, tick: float = 0.0001) -> float:
    """
    估计相对价差（实测校准版）。price=权利金，moneyness_dev=|K/S−1|。
    - tick 项：0.0001/price —— 深度虚值低价合约主导项（实测 10%~35%）
    - 比例项：ATM 实测 0.3%~0.5%，随虚值度与期限增长
    上限 100% 防爆表；price<=0 返回 1（视为不可成交量级）。
    """
    if price <= 0:
        return 1.0
    tick_term = tick / price
    ratio_term = 0.003 + 0.4 * moneyness_dev + 0.15 * max(dte - 30, 0) / 365.0
    return min(1.0, max(tick_term, ratio_term))


@dataclass
class RiskLimits:
    """风控限额（撮合前置校验 6/7）"""
    max_margin_ratio: float = 0.50   # 保证金占用/权益 上限
    max_position_per_contract: float = 200.0   # 单合约最大持仓（张）
    max_position_per_underlying: float = 1000.0  # 单标的最大持仓（张）
    min_volume_lots: float = 100.0   # 流动性闸门：日成交量下限（张）
    fill_ratio: float = 0.02         # 单笔不超过当日成交量的比例


@dataclass
class MarketRow:
    """撮合所需的单合约当日市场数据（由数据层拼装，回测=历史行，模拟盘=当日行）"""
    instrument: Instrument
    trade_date: date
    close: float                     # 收盘价（或结算价，fill basis）
    volume: float = 0.0              # 当日成交量（张）
    open_interest: float = float("nan")
    bid: float = float("nan")
    ask: float = float("nan")
    pre_close: float = float("nan")  # 前收盘（涨跌停校验基准）
    pre_settle: float = float("nan") # 前结算价（保证金基准；缺失用 pre_close 近似）
    spot_close: float = float("nan") # 标的当日收盘（维持保证金）
    spot_prev_close: float = float("nan")  # 标的前收盘
    is_trading_day: bool = True


class Broker:
    """
    撮合器。账户会计约定：
      - 权利金进出走 cash；义务仓保证金冻结在 margin_used（不减 cash）
      - equity = cash + Σ position.market_value（义务仓市值为负负债）
      - 开仓瞬间 equity 仅减少手续费（premium 进 cash、market_value 为负，两者相抵）
    """

    def __init__(self, fee_rule: Optional[FeeRule] = None,
                 fill_mode: FillMode = FillMode.CLOSE_SLIPPAGE,
                 limits: Optional[RiskLimits] = None):
        self.fee_rule = fee_rule or FeeRule()
        self.fill_mode = fill_mode
        self.limits = limits or RiskLimits()
        self.reject_log: List[Tuple[str, str]] = []

    # ------------------------------------------------------------ 撮合
    def match(self, order: Order, row: MarketRow,
              account: Account) -> Tuple[List[Trade], Optional[str]]:
        """前置校验（任一失败拒单）→ 按模式定价 → 成交量约束 → 落账。"""
        reject = self._preflight(order, row, account)
        if reject:
            order.status = OrderStatus.REJECTED
            order.reject_msg = reject
            self.reject_log.append((order.instrument.symbol if order.instrument else "?", reject))
            return [], reject

        price = self._fill_price(order, row)
        if price is None or price <= 0:
            return [], "无有效成交价"
        # 成交量约束：单笔 ≤ 当日成交量 × fill_ratio
        if row.volume > 0:
            cap = max(1.0, math.floor(row.volume * self.limits.fill_ratio))
            qty = min(order.remaining, cap)
        else:
            qty = order.remaining  # 无量数据时不额外限制（闸门4已挡）
        if qty < 1:
            self.reject_log.append((row.instrument.symbol, "成交量约束: 0 张可成交"))
            return [], "成交量约束"
        qty = math.floor(qty)

        trade = self._apply_fill(order, row, price, qty, account)
        order.filled_qty += qty
        order.status = OrderStatus.FILLED if order.filled_qty >= order.qty else OrderStatus.PARTIAL
        return [trade], None

    def _preflight(self, order: Order, row: MarketRow, account: Account) -> Optional[str]:
        """撮合前置校验 1~7（§5.3）"""
        inst = order.instrument
        # 1 交易日
        if not row.is_trading_day:
            return "非交易日"
        # 2 合约有效
        if inst.expiry and row.trade_date >= inst.expiry:
            return "合约已到期"
        # 3 价格合法（涨跌停区间内）
        if not (row.close > 0):
            return "收盘价非法"
        if row.pre_close == row.pre_close and row.pre_close > 0:
            # 数据层已提供涨跌停时校验；这里只做粗检（±15% 防脏数据）
            if not (0.05 * row.pre_close <= row.close <= 5.0 * row.pre_close):
                return f"收盘价异常跳变 close={row.close} pre={row.pre_close}"
        # 4 流动性达标
        if row.volume == row.volume and row.volume < self.limits.min_volume_lots:
            return f"流动性不足 volume={row.volume:.0f}张 < {self.limits.min_volume_lots:.0f}"
        # 5 资金/保证金 + 6/7 限额
        is_open = order.offset is Offset.OPEN
        is_short = is_open and order.direction is Direction.SELL
        if is_open and order.direction is Direction.BUY:
            cost = row.close * order.remaining * inst.multiplier
            fee = self.fee_rule.calc(order.remaining, row.close, inst.multiplier)
            if account.cash < cost + fee:
                return f"资金不足 需要{cost+fee:.0f} 现金{account.cash:.0f}"
        elif is_short:
            margin_per = self._margin_per_lot(inst, row, maintenance=False)
            need = margin_per * order.remaining
            fee = self.fee_rule.calc(order.remaining, row.close, inst.multiplier)
            avail = account.cash - account.margin_used
            if need + fee > avail:
                return f"保证金不足 需要{need+fee:.0f} 可用{avail:.0f}"
            if account.margin_ratio > self.limits.max_margin_ratio:
                return f"保证金占用率 {account.margin_ratio:.0%} 超限"
        # 6 持仓限额
        pos = account.positions.get(inst.symbol)
        cur = pos.net_qty if pos else 0.0
        signed = order.remaining if (order.direction is Direction.BUY) == (order.offset is Offset.OPEN) \
            else -order.remaining
        new_net = cur + (signed if is_open else
                         (order.remaining if order.direction is Direction.SELL else -order.remaining))
        if abs(new_net) > self.limits.max_position_per_contract:
            return f"单合约持仓超限 {abs(new_net):.0f}"
        return None

    def _fill_price(self, order: Order, row: MarketRow) -> Optional[float]:
        """按模式确定成交价；CLOSE_SLIPPAGE 按方向加价差/2"""
        base = row.close
        if self.fill_mode is FillMode.MID and row.bid == row.bid and row.ask == row.ask:
            base = (row.bid + row.ask) / 2
        elif self.fill_mode is FillMode.CROSS and row.bid == row.bid and row.ask == row.ask:
            base = row.ask if order.direction is Direction.BUY else row.bid
        elif self.fill_mode is FillMode.CLOSE_SLIPPAGE:
            spot = row.spot_close if row.spot_close == row.spot_close else row.spot_prev_close
            m_dev = abs(row.instrument.strike / spot - 1.0) if spot and spot > 0 else 0.05
            dte = max((row.instrument.expiry - row.trade_date).days, 0) \
                if row.instrument.expiry else 30
            sp = estimate_spread_pct(base, m_dev, dte, row.instrument.multiplier and 0.0001)
            half = base * sp / 2.0
            # 买入付价差上半段、卖出收价差下半段（保守）
            base = base + half if order.direction is Direction.BUY else max(base - half, 0.0001)
        return round(base, 4)

    def _margin_per_lot(self, inst: Instrument, row: MarketRow,
                        maintenance: bool = False) -> float:
        """每张保证金。基准价优先级：结算价 > 收盘价（近似，已在报告中标注）"""
        opt_price = row.pre_settle if row.pre_settle == row.pre_settle and row.pre_settle > 0 \
            else row.close
        spot = row.spot_prev_close if row.spot_prev_close == row.spot_prev_close \
            else row.spot_close
        return calc_margin(inst, option_price=opt_price, spot_close=spot,
                           is_short=True, is_call=inst.right is Right.CALL,
                           maintenance=maintenance)

    # ------------------------------------------------------------ 落账
    def _apply_fill(self, order: Order, row: MarketRow, price: float,
                    qty: float, account: Account) -> Trade:
        inst = order.instrument
        fee = self.fee_rule.calc(qty, price, inst.multiplier)
        amount = price * qty * inst.multiplier
        trade = Trade(ts=datetime.combine(row.trade_date, datetime.min.time()),
                      instrument=inst, direction=order.direction, offset=order.offset,
                      qty=qty, price=price, amount=amount, fee=fee,
                      strategy_id=order.strategy_id)
        sym = inst.symbol
        pos = account.positions.get(sym)

        if order.offset is Offset.OPEN:
            if order.direction is Direction.BUY:      # 买入开仓（权利仓）
                account.cash -= amount + fee
                margin_delta = 0.0
            else:                                     # 卖出开仓（义务仓）
                account.cash += amount - fee
                margin_delta = self._margin_per_lot(inst, row) * qty
                account.margin_used += margin_delta
            trade.margin_delta = margin_delta
            if pos is None:
                pos = Position(instrument=inst, strategy_id=order.strategy_id)
                account.positions[sym] = pos
            # avg_open_price 口径=每单位权利金；加权平均含方向（义务仓为负持仓）
            total = pos.net_qty * pos.avg_open_price + qty * price
            pos.net_qty += qty if order.direction is Direction.BUY else -qty
            pos.avg_open_price = abs(total) / max(abs(pos.net_qty), 1e-9)
            pos.total_fee += fee
            if margin_delta > 0:
                pos.margin += margin_delta
        else:  # CLOSE
            closing_short = order.direction is Direction.BUY  # 买平=平义务仓
            open_avg = pos.avg_open_price if pos else price
            if closing_short:
                pnl = (open_avg - price) * qty * inst.multiplier
                margin_delta = -(pos.margin / max(abs(pos.net_qty), 1e-9)) * qty if pos else 0.0
                account.margin_used = max(0.0, account.margin_used + margin_delta)
                account.cash -= amount + fee
            else:  # 卖平=平权利仓
                pnl = (price - open_avg) * qty * inst.multiplier
                margin_delta = 0.0
                account.cash += amount - fee
            trade.realized_pnl = pnl - fee
            trade.margin_delta = margin_delta
            if pos:
                pos.net_qty += qty if closing_short else -qty
                pos.realized_pnl += trade.realized_pnl
                pos.total_fee += fee
                if abs(pos.net_qty) < 1e-9:
                    del account.positions[sym]
        account.total_fee += fee
        account.orders.append(order)
        account.trades.append(trade)
        return trade

    # ------------------------------------------------------------ 到期结算
    def settle_expiry(self, account: Account, spot, on: date) -> List[Trade]:
        """
        T 日到期合约：实值自动行权（现金结算简化，见 §5.3 模拟与实盘差异）、虚值作废。
        多品种：spot 可传 callable(instrument) -> 该合约标的现价；传 float 时按单标的兼容。
        """
        spot_of = spot if callable(spot) else (lambda inst: spot)
        trades: List[Trade] = []
        for sym in list(account.positions.keys()):
            pos = account.positions[sym]
            inst = pos.instrument
            if not inst.is_option or inst.expiry != on:
                continue
            spot = spot_of(inst)
            intrinsic = max(0.0, (spot - inst.strike) if inst.right is Right.CALL
                            else (inst.strike - spot))
            qty = abs(pos.net_qty)
            if intrinsic > 0:
                settle_amount = intrinsic * qty * inst.multiplier
                fee = self.fee_rule.exercise_fee * qty
                if pos.net_qty > 0:      # 权利仓行权：流入内在价值
                    account.cash += settle_amount - fee
                    pnl = settle_amount - pos.avg_open_price * qty * inst.multiplier - fee
                else:                    # 义务仓被指派：流出内在价值、释放保证金
                    account.cash -= settle_amount
                    account.margin_used = max(0.0, account.margin_used - pos.margin)
                    pnl = (pos.avg_open_price - intrinsic) * qty * inst.multiplier
                t = Trade(ts=datetime.combine(on, datetime.min.time()), instrument=inst,
                          direction=Direction.SELL if pos.net_qty > 0 else Direction.BUY,
                          offset=Offset.CLOSE, qty=qty, price=intrinsic,
                          amount=settle_amount, fee=fee, realized_pnl=pnl,
                          strategy_id=pos.strategy_id)
                pos.realized_pnl += pnl
                trades.append(t)
            else:
                if pos.net_qty < 0:  # 义务仓释放保证金
                    account.margin_used = max(0.0, account.margin_used - pos.margin)
            account.total_fee += self.fee_rule.exercise_fee * qty if intrinsic > 0 else 0.0
            del account.positions[sym]
        return trades
