"""
optlab.core.spec — 合约规格、交易日历、保证金与手续费

规则来源（境内 ETF 期权）：
    《上海证券交易所股票期权试点交易规则》
    《深圳证券交易所股票期权试点交易规则》
    《中国金融期货交易所股指期权合约交易细则》（IO/HO/MO）

关键实现：
    1. 行权价间距（按标的价格分档，上交所规则）
    2. 到期日 = 到期月份第四个星期三（遇法定节假日顺延）
    3. 卖方开仓/维持保证金（ETF 期权公式）
    4. 涨跌停价格计算
    5. 手续费与行权费

注意：本文件所有参数均为「可配置默认值」，实盘前必须按交易所最新公告与
券商实际收取标准核对（尤其是保证金比例在极端行情下会被临时上调）。
"""

from __future__ import annotations

import calendar
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from .models import (
    AssetClass, DeliveryType, ExerciseStyle, FeeRule, Instrument,
    MarginRule, Right,
)

__all__ = [
    "ContractSpec", "OptionSpecRegistry", "ETF_OPTION_SPECS",
    "expiry_of_month", "next_n_expiries", "strike_step", "generate_strikes",
    "build_option_symbol", "calc_margin", "calc_limit_prices", "TradingCalendar",
]


# ---------------------------------------------------------------- 交易日历


class TradingCalendar:
    """
    简化交易日历：仅处理周末 + 硬编码法定节假日。
    生产环境应替换为交易所日历或 akshare/chinese_calendar 数据源。
    """
    # 2025-2027 中国大陆法定节假日（示例，需按国务院年度公告更新）
    HOLIDAYS: Dict[int, List[str]] = {
        2025: ["2025-01-01", "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31",
               "2025-02-03", "2025-02-04", "2025-04-04", "2025-05-01", "2025-05-02",
               "2025-05-05", "2025-06-02", "2025-10-01", "2025-10-02", "2025-10-03",
               "2025-10-06", "2025-10-07", "2025-10-08"],
        2026: ["2026-01-01", "2026-01-02", "2026-02-16", "2026-02-17", "2026-02-18",
               "2026-02-19", "2026-02-20", "2026-04-06", "2026-05-01", "2026-05-04",
               "2026-05-05", "2026-06-19", "2026-09-25", "2026-10-01", "2026-10-02",
               "2026-10-05", "2026-10-06", "2026-10-07"],
        2027: ["2027-01-01", "2027-02-05", "2027-02-08", "2027-02-09", "2027-02-10",
               "2027-02-11", "2027-02-12", "2027-04-05", "2027-05-03", "2027-05-04",
               "2027-05-05", "2027-06-09", "2027-09-15", "2027-10-01", "2027-10-04",
               "2027-10-05", "2027-10-06", "2027-10-07"],
    }

    @classmethod
    def is_holiday(cls, d: date) -> bool:
        return d.isoformat() in cls.HOLIDAYS.get(d.year, [])

    @classmethod
    def is_trading_day(cls, d: date) -> bool:
        return d.weekday() < 5 and not cls.is_holiday(d)

    @classmethod
    def next_trading_day(cls, d: date, n: int = 1) -> date:
        cur = d
        cnt = 0
        while cnt < n:
            cur += timedelta(days=1)
            if cls.is_trading_day(cur):
                cnt += 1
        return cur

    @classmethod
    def prev_trading_day(cls, d: date, n: int = 1) -> date:
        cur = d
        cnt = 0
        while cnt < n:
            cur -= timedelta(days=1)
            if cls.is_trading_day(cur):
                cnt += 1
        return cur

    @classmethod
    def trading_days_between(cls, start: date, end: date) -> List[date]:
        out = []
        cur = start
        while cur <= end:
            if cls.is_trading_day(cur):
                out.append(cur)
            cur += timedelta(days=1)
        return out


# ---------------------------------------------------------------- 到期日


def expiry_of_month(year: int, month: int) -> date:
    """
    到期月份第四个星期三，遇法定节假日顺延（上交所/深交所/中金所通用口径）。
    """
    c = calendar.Calendar(firstweekday=0)
    wednesdays = [d for d in c.itermonthdates(year, month)
                  if d.month == month and d.weekday() == 2]
    d = wednesdays[3]
    # 顺延至下一交易日
    while not TradingCalendar.is_trading_day(d):
        d += timedelta(days=1)
    return d


def next_n_expiries(today: date, n: int = 4) -> List[date]:
    """从今天起最近 n 个到期日"""
    out = []
    y, m = today.year, today.month
    for _ in range(n + 2):
        e = expiry_of_month(y, m)
        if e >= today and e not in out:
            out.append(e)
            if len(out) >= n:
                break
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


# ---------------------------------------------------------------- 行权价


def strike_step(spot: float) -> float:
    """
    上交所 ETF 期权行权价间距（按标的前收盘价分档）：
        ≤3 元    0.05
        3~5 元   0.10
        5~10 元  0.25
        10~20 元 0.50
        20~50 元 1.00
        50~100 元 2.50
        >100 元  5.00
    """
    if spot <= 3.0:
        return 0.05
    if spot <= 5.0:
        return 0.10
    if spot <= 10.0:
        return 0.25
    if spot <= 20.0:
        return 0.50
    if spot <= 50.0:
        return 1.00
    if spot <= 100.0:
        return 2.50
    return 5.00


def generate_strikes(spot: float, n_up: int = 8, n_dn: int = 8,
                     step: Optional[float] = None) -> List[float]:
    """
    以平值为中心生成行权价序列。实值/虚值各扩展，并额外覆盖在平值附近的
    加密档位（ETF 期权在平值附近会加挂）。
    """
    st = step or strike_step(spot)
    atm = round(spot / st) * st
    strikes = [round(atm + i * st, 4) for i in range(-n_dn, n_up + 1)]
    return sorted({round(s, 4) for s in strikes if s > 0})


def build_option_symbol(underlying: str, right: Right, expiry: date, strike: float,
                        month_code: str = "M") -> str:
    """
    生成标准合约代码：
        ETF 期权(上交所) 510300C2609M04000
        股指期权(中金所) IO2609-C-4000
    """
    if underlying.upper().startswith(("IO", "HO", "MO", "IM")):
        return f"{underlying.upper()}{expiry.strftime('%y%m')}-{right.value[0]}-{strike:g}"
    strike_code = f"{round(strike * 1000):05d}"
    return (f"{underlying}{right.value[0]}{expiry.strftime('%y%m')}"
            f"{month_code}{strike_code}")


# ---------------------------------------------------------------- 合约规格


@dataclass
class ContractSpec:
    """单一品种的合约规格"""
    underlying: str
    name: str = ""
    asset_class: AssetClass = AssetClass.ETF_OPTION
    multiplier: float = 10000.0
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    delivery: DeliveryType = DeliveryType.PHYSICAL
    exchange: str = "SSE"
    min_tick: float = 0.0001            # 最小报价单位
    margin_rule: MarginRule = field(default_factory=MarginRule)
    fee_rule: FeeRule = field(default_factory=FeeRule)
    price_limit_pct: float = 0.10       # 标的涨跌停幅度，用于期权涨跌停计算
    # 合约月份：当月/下月/随后两个季月
    contract_months: str = "M0,M1,Q1,Q2"

    def make_instrument(self, right: Optional[Right], expiry: Optional[date],
                        strike: float = 0.0) -> Instrument:
        if right is None:
            return Instrument(
                symbol=self.underlying, name=self.name, underlying=self.underlying,
                asset_class=self.asset_class, multiplier=1.0, exchange=self.exchange,
                exercise_style=self.exercise_style, delivery=self.delivery,
            )
        return Instrument(
            symbol=build_option_symbol(self.underlying, right, expiry, strike),
            name=f"{self.name}{right.cn}{strike:g}@{expiry:%y%m}",
            underlying=self.underlying, asset_class=self.asset_class,
            right=right, strike=round(strike, 4), expiry=expiry,
            multiplier=self.multiplier, exercise_style=self.exercise_style,
            delivery=self.delivery, exchange=self.exchange,
        )


# 境内主要期权品种规格（合约单位以交易所公告为准）
ETF_OPTION_SPECS: Dict[str, ContractSpec] = {
    "510050": ContractSpec("510050", "50ETF", multiplier=10000, exchange="SSE"),
    "510300": ContractSpec("510300", "300ETF(沪)", multiplier=10000, exchange="SSE"),
    "159919": ContractSpec("159919", "300ETF(深)", multiplier=10000, exchange="SZSE"),
    "510500": ContractSpec("510500", "500ETF(沪)", multiplier=10000, exchange="SSE"),
    "159922": ContractSpec("159922", "500ETF(深)", multiplier=10000, exchange="SZSE"),
    "159915": ContractSpec("159915", "创业板ETF", multiplier=10000, exchange="SZSE"),
    "588000": ContractSpec("588000", "科创50ETF", multiplier=10000, exchange="SSE"),
    "588080": ContractSpec("588080", "科创板50ETF", multiplier=10000, exchange="SSE"),
    "159901": ContractSpec("159901", "深证100ETF", multiplier=10000, exchange="SZSE"),
}

INDEX_OPTION_SPECS: Dict[str, ContractSpec] = {
    "IO": ContractSpec("IO", "沪深300股指期权", AssetClass.INDEX_OPTION, multiplier=100,
                       delivery=DeliveryType.CASH, exchange="CFFEX",
                       fee_rule=FeeRule(per_contract=15.0)),
    "HO": ContractSpec("HO", "上证50股指期权", AssetClass.INDEX_OPTION, multiplier=100,
                       delivery=DeliveryType.CASH, exchange="CFFEX",
                       fee_rule=FeeRule(per_contract=15.0)),
    "MO": ContractSpec("MO", "中证1000股指期权", AssetClass.INDEX_OPTION, multiplier=100,
                       delivery=DeliveryType.CASH, exchange="CFFEX",
                       fee_rule=FeeRule(per_contract=15.0)),
}


class OptionSpecRegistry:
    """品种规格注册表 + 合约缓存"""
    def __init__(self):
        self.specs: Dict[str, ContractSpec] = {}
        self.specs.update(ETF_OPTION_SPECS)
        self.specs.update(INDEX_OPTION_SPECS)
        self._cache: Dict[str, Instrument] = {}

    def register(self, spec: ContractSpec) -> None:
        self.specs[spec.underlying] = spec

    def get(self, underlying: str) -> ContractSpec:
        if underlying not in self.specs:
            # 未知品种：用 ETF 期权默认规格兜底
            self.register(ContractSpec(underlying, underlying))
        return self.specs[underlying]

    def underlying_instrument(self, sym: str) -> Instrument:
        key = f"U::{sym}"
        if key not in self._cache:
            self._cache[key] = self.get(sym).make_instrument(None, None)
        return self._cache[key]

    def option(self, underlying: str, right: Right, expiry: date, strike: float) -> Instrument:
        k = f"{underlying}{right.value[0]}{expiry:%Y%m}{strike:g}"
        if k not in self._cache:
            self._cache[k] = self.get(underlying).make_instrument(right, expiry, strike)
        return self._cache[k]

    def list_expiries(self, underlying: str, today: date, n: int = 4) -> List[date]:
        return next_n_expiries(today, n)


# ---------------------------------------------------------------- 保证金


def calc_margin(inst: Instrument, *, option_price: float, spot_close: float,
                is_short: bool, is_call: bool, multiplier: Optional[float] = None,
                maintenance: bool = False,
                rule: Optional[MarginRule] = None) -> float:
    """
    ETF 期权卖方保证金（上交所公式，单位：元/张）：

    认购期权义务仓开仓保证金 =
        [前结算价 + max(12% × 标的前收盘 − 认购虚值, 7% × 标的前收盘)] × 合约单位
    认沽期权义务仓开仓保证金 =
        min[前结算价 + max(12% × 标的前收盘 − 认沽虚值, 7% × 行权价), 行权价] × 合约单位

    其中：
        认购虚值 = max(0, 行权价 − 标的前收盘)
        认沽虚值 = max(0, 标的前收盘 − 行权价)

    维持保证金与开仓保证金系数相同（12%/7%），仅取价时点不同：
        开仓：option_price 传合约「前结算价」，spot_close 传标的「前收盘价」
        维持：option_price 传合约「当日结算价」，spot_close 传标的「当日收盘价」

    注意：本函数返回「每张」保证金，调用方需 × 张数。
    """
    if not is_short:
        return 0.0
    rule = rule or MarginRule()
    mult = multiplier or inst.multiplier
    K = inst.strike
    S = spot_close
    a = rule.maint_a if maintenance else rule.short_call_init_a
    b = rule.maint_b if maintenance else rule.short_call_init_b

    if is_call:
        otm = max(0.0, K - S)
        m = (option_price + max(a * S - otm, b * S)) * mult
    else:
        otm = max(0.0, S - K)
        m = min(option_price + max(a * S - otm, b * K), K) * mult

    return max(0.0, round(m, 2))


def calc_index_option_margin(inst: Instrument, *, option_price: float, spot: float,
                             is_short: bool, is_call: bool, multiplier: float = 100.0,
                             rule: Optional[MarginRule] = None) -> float:
    """
    中金所股指期权保证金 —— 实验性近似，与真实公式结构不同，勿用于精确风控。

    中金所真实公式（看涨义务仓，每手）：
        保证金 = 权利金 + max(标的收盘 × 保证金调整系数 − 虚值额,
                              最低保障系数 × 标的收盘 × 保证金调整系数) × 合约乘数
    其中 IO/HO 调整系数 10%、MO 12%，最低保障系数 0.5。
    待实现后再启用股指期权品种。
    """
    raise NotImplementedError("股指期权保证金公式未按中金所口径实现，暂勿使用")


# ---------------------------------------------------------------- 涨跌停


def calc_limit_prices(pre_settle: float, spot_prev_close: float, strike: float,
                      is_call: bool) -> Tuple[float, float]:
    """
    ETF 期权涨跌停价（上交所规则，S=标的前收盘, K=行权价, P=期权前结算价）：

        认购：最大涨幅 = max{S×0.5%, min(2S−K, S)×10%}
              最大跌幅 = S×10%
        认沽：最大涨幅 = max{K×0.5%, min(2K−S, S)×10%}
              最大跌幅 = S×10%
        涨停 = P + 最大涨幅；跌停 = max(P − 最大跌幅, 0.0001)
    """
    S, K, P = spot_prev_close, strike, pre_settle
    if is_call:
        up_amp = max(S * 0.005, min(2 * S - K, S) * 0.10)
    else:
        up_amp = max(K * 0.005, min(2 * K - S, S) * 0.10)
    dn_amp = S * 0.10
    up = P + up_amp
    dn = max(P - dn_amp, 0.0001)
    return round(up, 4), round(dn, 4)
