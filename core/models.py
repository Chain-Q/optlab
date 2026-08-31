"""
thetalab.core.models — 核心数据模型与枚举定义

模型分六类，与方案文档 §4 一一对应：
    Instrument 合约(标的/期权)   Quote/Bar/Tick 行情
    Order/Trade 订单             Position/Account 持仓与账户
    StrategySpec 策略            Signal 信号

设计原则：
    1. 全部使用 dataclass，字段类型显式标注，便于序列化与落库；
    2. 时间统一用 UTC+8 无时区 datetime，落库时转 ISO 字符串；
    3. 金额/价格统一 float，数量统一 float（张），避免整数除法陷阱；
    4. 合约对象 frozen + 缓存，可作为 dict key 与跨模块共享。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

__all__ = [
    "Right", "AssetClass", "ExerciseStyle", "DeliveryType",
    "Direction", "Offset", "OrderType", "OrderStatus", "Interval",
    "Instrument", "Bar", "Tick", "OptionQuote", "OptionChain",
    "Greeks", "Order", "Trade", "Position", "Account", "MarginRule", "FeeRule",
    "UnderlyingSnapshot",
]

# ---------------------------------------------------------------- 基础枚举


class Right(str, Enum):
    """期权方向"""
    CALL = "CALL"
    PUT = "PUT"

    @property
    def cn(self) -> str:
        return "认购" if self is Right.CALL else "认沽"

    @property
    def sign(self) -> int:
        """内在价值计算符号：CALL=+1, PUT=-1"""
        return 1 if self is Right.CALL else -1


class AssetClass(str, Enum):
    """标的大类，决定合约单位、保证金模型、交易日历"""
    ETF_OPTION = "ETF_OPTION"        # 境内 ETF 期权（510050/510300/159915 ...）
    INDEX_OPTION = "INDEX_OPTION"    # 股指期权（IO/HO/MO），现金交割，点×100
    COMMODITY_OPTION = "COMMODITY_OPTION"  # 商品期权，多为美式
    EQUITY_OPTION = "EQUITY_OPTION"  # 境外个股期权，美式、单位 100 股


class ExerciseStyle(str, Enum):
    EUROPEAN = "EUROPEAN"   # 欧式，到期日行权
    AMERICAN = "AMERICAN"   # 美式，可提前行权（需二叉树定价）


class DeliveryType(str, Enum):
    PHYSICAL = "PHYSICAL"   # 实物交割（ETF 份额）
    CASH = "CASH"           # 现金交割


class Direction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Offset(str, Enum):
    """开平标志 —— A 股期权采用「开/平」语义而非多空语义"""
    OPEN = "OPEN"      # 开仓：买入开仓=权利仓，卖出开仓=义务仓
    CLOSE = "CLOSE"    # 平仓：卖出平仓=平权利仓，买入平仓=平义务仓

    @classmethod
    def from_cn(cls, s: str) -> "Offset":
        return cls.OPEN if "开" in s else cls.CLOSE


class OrderType(str, Enum):
    LIMIT = "LIMIT"    # 限价
    MARKET = "MARKET"  # 市价（转 FAK）
    FAK = "FAK"        # 立即成交剩余撤销
    FOK = "FOK"        # 全部成交否则撤销


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Interval(str, Enum):
    TICK = "tick"
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "60m"
    D1 = "1d"

    @property
    def seconds(self) -> int:
        return {
            "tick": 0, "1m": 60, "5m": 300, "15m": 900,
            "30m": 1800, "60m": 3600, "1d": 86400,
        }[self.value]


# ---------------------------------------------------------------- 合约模型


@dataclass(frozen=True)
class Instrument:
    """
    统一合约描述。标的价格用同一结构表达（right=None 即为标的）。

    合约代码规范（境内 ETF 期权，上交所 8 位）：
        '510300C2609M04000'  → 标的 510300 + C/P + 年月 + M/A/B/C(到期月) + 行权价×1000
    回测/模拟环境允许自定义代码，只要保证唯一。
    """
    symbol: str
    name: str = ""
    underlying: str = ""
    asset_class: AssetClass = AssetClass.ETF_OPTION
    right: Optional[Right] = None          # None => 标的本身
    strike: float = 0.0
    expiry: Optional[date] = None
    multiplier: float = 10000.0             # 合约单位（份/点）
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    delivery: DeliveryType = DeliveryType.PHYSICAL
    exchange: str = "SSE"

    # ---------------- 派生属性
    @property
    def is_option(self) -> bool:
        return self.right is not None

    @property
    def is_underlying(self) -> bool:
        return self.right is None

    @property
    def kind(self) -> str:
        if self.is_underlying:
            return "UNDERLYING"
        return f"{self.right.value}"

    def dte(self, today: Optional[date] = None) -> int:
        """剩余自然日"""
        if self.expiry is None:
            return 0
        today = today or date.today()
        return max(0, (self.expiry - today).days)

    def t_years(self, now: Optional[datetime] = None, day_count: int = 365) -> float:
        """剩余年化时间。now 为空时用日期粒度。"""
        if self.expiry is None:
            return 0.0
        if now is None:
            return max(1e-6, self.dte() / day_count)
        # 到期日 15:00 为境内期权到期时点
        end = datetime.combine(self.expiry, datetime.min.time()).replace(hour=15)
        return max(1e-6, (end - now).total_seconds() / (day_count * 86400.0))

    def intrinsic(self, spot: float) -> float:
        """每单位内在价值"""
        if not self.is_option:
            return spot
        return max(0.0, self.right.sign * (spot - self.strike))

    def moneyness(self, spot: float) -> float:
        """价值状态：spot/strike。>1 认购实值 / 认沽虚值"""
        if not self.is_option or self.strike <= 0:
            return 1.0
        return spot / self.strike

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["asset_class"] = self.asset_class.value
        d["right"] = self.right.value if self.right else None
        d["exercise_style"] = self.exercise_style.value
        d["delivery"] = self.delivery.value
        d["expiry"] = self.expiry.isoformat() if self.expiry else None
        return d


# ---------------------------------------------------------------- 行情模型


@dataclass
class Greeks:
    """希腊字母（与 pricing.py 口径一致：每单位标的）。

    delta：0~1 / -1~0；gamma：每元标的价格变动的 delta 变化；
    vega：波动率 +1 vol point 的价格变化；theta：每自然日价格变化（负=损耗）；
    rho：利率 +1 vol point 的价格变化。
    组合汇总须再乘 张数×合约单位（Position 层负责）。
    """
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    iv: float = float("nan")  # 隐含波动率（年化小数）

    def scale(self, qty: float) -> "Greeks":
        return Greeks(
            delta=self.delta * qty, gamma=self.gamma * qty, vega=self.vega * qty,
            theta=self.theta * qty, rho=self.rho * qty, iv=self.iv,
        )

    def __add__(self, other: "Greeks") -> "Greeks":
        return Greeks(
            self.delta + other.delta, self.gamma + other.gamma,
            self.vega + other.vega, self.theta + other.theta,
            self.rho + other.rho, float("nan"),
        )

    def to_dict(self) -> Dict[str, float]:
        return {"delta": self.delta, "gamma": self.gamma, "vega": self.vega,
                "theta": self.theta, "rho": self.rho, "iv": self.iv}


@dataclass
class Tick:
    """逐笔/快照行情（期权链一行即一个 Tick）"""
    instrument: Instrument
    ts: datetime
    last: float = float("nan")
    open: float = float("nan")
    high: float = float("nan")
    low: float = float("nan")
    pre_close: float = float("nan")
    pre_settle: float = float("nan")
    bid: float = float("nan")
    ask: float = float("nan")
    bid_vol: float = 0.0
    ask_vol: float = 0.0
    volume: float = 0.0
    turnover: float = 0.0
    open_interest: float = 0.0
    limit_up: float = float("nan")
    limit_down: float = float("nan")
    greeks: Optional[Greeks] = None

    # ---- 派生
    @property
    def mid(self) -> float:
        if not (math.isnan(self.bid) or math.isnan(self.ask)) and self.ask > 0 and self.bid > 0:
            return (self.bid + self.ask) / 2.0
        return self.last

    @property
    def spread(self) -> float:
        if math.isnan(self.bid) or math.isnan(self.ask):
            return float("nan")
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        m = self.mid
        if math.isnan(m) or m <= 0:
            return float("nan")
        s = self.spread
        return s / m if not math.isnan(s) else float("nan")

    def fair_price(self, mode: Literal["mid", "last", "bid", "ask", "conservative"] = "mid") -> float:
        """
        用于撮合/估值的参考价。
        conservative：权利仓买入用 ask、卖出用 bid；义务仓反过来 —— 由 broker 侧决定方向后调用。
        """
        if mode == "mid":
            return self.mid
        if mode == "last":
            return self.last
        if mode == "bid":
            return self.bid if not math.isnan(self.bid) else self.last
        if mode == "ask":
            return self.ask if not math.isnan(self.ask) else self.last
        return self.mid


@dataclass
class Bar:
    """K 线"""
    instrument: Instrument
    ts: datetime
    interval: Interval = Interval.D1
    open: float = float("nan")
    high: float = float("nan")
    low: float = float("nan")
    close: float = float("nan")
    volume: float = 0.0
    turnover: float = 0.0
    open_interest: float = 0.0
    iv: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.instrument.symbol, "ts": self.ts.isoformat(),
            "interval": self.interval.value, "open": self.open, "high": self.high,
            "low": self.low, "close": self.close, "volume": self.volume,
            "turnover": self.turnover, "oi": self.open_interest, "iv": self.iv,
        }


@dataclass
class OptionQuote:
    """期权链展示用聚合行（Tick + 静态属性 + 希腊字母）"""
    instrument: Instrument
    tick: Tick
    spot: float = float("nan")

    @property
    def iv(self) -> float:
        return self.tick.greeks.iv if self.tick.greeks else float("nan")

    @property
    def delta(self) -> float:
        return self.tick.greeks.delta if self.tick.greeks else float("nan")

    @property
    def otm_pct(self) -> float:
        """虚值百分比（%）：虚值>0，实值<0，平值≈0"""
        if not self.instrument.is_option or math.isnan(self.spot):
            return float("nan")
        return (self.instrument.strike / self.spot - 1.0) * 100.0 * self.right_sign()

    def right_sign(self) -> int:
        """虚值方向符号：CALL 平值以上为虚值(+1)，PUT 平值以下为虚值(-1)"""
        return 1 if self.instrument.right is Right.CALL else -1

    def to_row(self) -> Dict[str, Any]:
        g = self.tick.greeks or Greeks()
        return {
            "symbol": self.instrument.symbol,
            "right": self.instrument.right.value if self.instrument.right else "",
            "strike": self.instrument.strike,
            "expiry": self.instrument.expiry.isoformat() if self.instrument.expiry else "",
            "dte": self.instrument.dte(),
            "last": self.tick.last, "bid": self.tick.bid, "ask": self.tick.ask,
            "bid_vol": self.tick.bid_vol, "ask_vol": self.tick.ask_vol,
            "volume": self.tick.volume, "oi": self.tick.open_interest,
            "spread_pct": self.tick.spread_pct,
            "iv": g.iv, "delta": g.delta, "gamma": g.gamma,
            "vega": g.vega, "theta": g.theta,
            "otm_pct": self.otm_pct,
        }


@dataclass
class OptionChain:
    """某标的某到期日的完整期权链"""
    underlying: str
    spot: float
    expiry: date
    ts: datetime
    quotes: List[OptionQuote] = field(default_factory=list)
    forward: float = float("nan")   # 远期价（用于 Black76 / 合成期货）
    risk_free: float = 0.03

    def atm_strike(self) -> float:
        if math.isnan(self.forward):
            f = self.spot
        else:
            f = self.forward
        return min(self.quotes, key=lambda q: abs(q.instrument.strike - f)).instrument.strike

    def by_right(self, right: Right) -> List[OptionQuote]:
        return sorted(
            [q for q in self.quotes if q.instrument.right is right],
            key=lambda q: q.instrument.strike,
        )

    def total_oi(self) -> float:
        return sum(q.tick.open_interest or 0 for q in self.quotes)

    def pcr_oi(self) -> float:
        """认沽/认购 持仓量比，>1 偏谨慎"""
        p = sum(q.tick.open_interest or 0 for q in self.quotes if q.instrument.right is Right.PUT)
        c = sum(q.tick.open_interest or 0 for q in self.quotes if q.instrument.right is Right.CALL)
        return p / c if c > 0 else float("nan")


@dataclass
class UnderlyingSnapshot:
    """标的快照（含已实现波动率、历史分位等派生指标，供策略推荐/信号使用）"""
    instrument: Instrument
    ts: datetime
    last: float
    change_pct: float = float("nan")
    rv_20d: float = float("nan")      # 20 日已实现波动率（年化）
    rv_60d: float = float("nan")
    iv_atm: float = float("nan")      # 平值隐含波动率
    iv_rank: float = float("nan")     # IV 历史分位 0~1
    iv_pct: float = float("nan")      # IV 百分位 0~100
    ma5: float = float("nan")
    ma20: float = float("nan")
    ma60: float = float("nan")
    atr14: float = float("nan")
    trend: str = "UNKNOWN"            # UP / DOWN / CHOP
    adx: float = float("nan")


# ---------------------------------------------------------------- 交易模型


@dataclass
class Order:
    """订单"""
    order_id: str = ""
    ts: Optional[datetime] = None
    instrument: Optional[Instrument] = None
    direction: Direction = Direction.BUY
    offset: Offset = Offset.OPEN
    qty: float = 1.0
    price: float = float("nan")        # LIMIT 时有效
    order_type: OrderType = OrderType.LIMIT
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    filled_amount: float = 0.0         # 成交金额（不含费）
    fee: float = 0.0
    margin: float = 0.0                # 开仓占用/释放
    strategy_id: str = ""
    reason: str = ""                   # 下单理由（信号触发写这里）
    reject_msg: str = ""

    @property
    def is_open(self) -> bool:
        return self.offset is Offset.OPEN

    @property
    def remaining(self) -> float:
        return max(0.0, self.qty - self.filled_qty)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "ts": self.ts.isoformat() if self.ts else "",
            "symbol": self.instrument.symbol if self.instrument else "",
            "direction": self.direction.value, "offset": self.offset.value,
            "qty": self.qty, "price": self.price, "type": self.order_type.value,
            "status": self.status.value, "filled_qty": self.filled_qty,
            "filled_amount": self.filled_amount, "fee": self.fee, "margin": self.margin,
            "strategy_id": self.strategy_id, "reason": self.reason,
        }


@dataclass
class Trade:
    """成交明细"""
    trade_id: str = ""
    ts: Optional[datetime] = None
    order_id: str = ""
    instrument: Optional[Instrument] = None
    direction: Direction = Direction.BUY
    offset: Offset = Offset.OPEN
    qty: float = 0.0
    price: float = 0.0
    amount: float = 0.0
    fee: float = 0.0
    margin_delta: float = 0.0
    strategy_id: str = ""
    realized_pnl: float = 0.0          # 平仓成交产生的已实现盈亏


@dataclass
class Position:
    """
    持仓。约定：
        net_qty > 0 → 权利仓（long options）
        net_qty < 0 → 义务仓（short options / 卖方）
        net_qty 为「张数」，金额口径需 × multiplier
    """
    instrument: Instrument = None  # type: ignore
    net_qty: float = 0.0
    avg_open_price: float = 0.0     # 开仓均价（期权金/单位）
    total_fee: float = 0.0
    realized_pnl: float = 0.0
    margin: float = 0.0             # 当前占用保证金（仅义务仓）
    last_price: float = float("nan")
    opened_at: Optional[datetime] = None
    strategy_id: str = ""

    @property
    def is_long(self) -> bool:
        return self.net_qty > 0

    @property
    def is_short(self) -> bool:
        return self.net_qty < 0

    @property
    def market_value(self) -> float:
        """按最新价计市值（权利仓为正资产，义务仓为负负债）"""
        p = self.last_price if not math.isnan(self.last_price) else self.avg_open_price
        return p * self.net_qty * self.instrument.multiplier

    @property
    def cost_value(self) -> float:
        return self.avg_open_price * self.net_qty * self.instrument.multiplier

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_value - self.total_fee

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    def exposure_notional(self, spot: float) -> float:
        """名义敞口（标的等值）"""
        return spot * abs(self.net_qty) * self.instrument.multiplier


@dataclass
class MarginRule:
    """保证金规则参数（上交所/深交所 ETF 期权口径）

    开仓保证金：[合约前结算价 + max(12%×标的前收盘 − 虚值, 7%×标的前收盘或行权价)] × 单位
    维持保证金：[合约当日结算价 + max(12%×标的当日收盘 − 虚值, 7%×标的当日收盘或行权价)] × 单位
    —— 两者系数同为 12%/7%，差异只在取价时点（前收盘 vs 当日收盘）。
    """
    short_call_init_a: float = 0.12   # 开仓：12% × 标的前收盘
    short_call_init_b: float = 0.07   # 开仓：7% × 标的前收盘（认沽为 7% × 行权价）
    short_put_init_a: float = 0.12
    short_put_init_b: float = 0.07
    maint_a: float = 0.12             # 维持：12% × 标的当日收盘（与开仓同系数）
    maint_b: float = 0.07             # 维持：7% × 标的当日收盘（认沽为 7% × 行权价）
    min_per_unit: float = 0.0         # 每张最低保证金


@dataclass
class FeeRule:
    """手续费规则：按张固定 + 按成交金额比例，取两者之和"""
    per_contract: float = 5.0        # 元/张（券商佣金+交易所经手费，双边收取）
    pct_of_amount: float = 0.0       # 按权利金金额比例
    min_per_order: float = 0.0
    exercise_fee: float = 0.6        # 行权手续费 元/张

    def calc(self, qty: float, price: float, multiplier: float) -> float:
        amount = abs(price * qty * multiplier)
        f = self.per_contract * abs(qty) + self.pct_of_amount * amount
        return max(self.min_per_order, round(f, 2))


@dataclass
class Account:
    """模拟账户"""
    account_id: str = "SIM-001"
    initial_cash: float = 1_000_000.0
    cash: float = 1_000_000.0
    margin_used: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    orders: List[Order] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[tuple] = field(default_factory=list)  # [(ts, equity)]
    total_fee: float = 0.0

    # ---------------- 派生指标
    @property
    def position_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def equity(self) -> float:
        """权益 = 现金 + 持仓市值（义务仓市值为负）"""
        return self.cash + self.position_value

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values())

    @property
    def margin_ratio(self) -> float:
        """保证金占用率"""
        return self.margin_used / self.equity if self.equity > 1e-9 else float("inf")

    @property
    def total_return(self) -> float:
        return self.equity / self.initial_cash - 1.0

    def margin_available(self) -> float:
        return max(0.0, self.cash - self.margin_used)


# ---------------------------------------------------------------- 工具


def d(x: Any) -> str:
    """统一时间格式化"""
    return x.isoformat(sep=" ") if isinstance(x, (datetime, date)) else str(x)
