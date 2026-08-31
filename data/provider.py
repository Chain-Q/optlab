"""
optlab.data.provider — 数据源抽象 + 上交所期权 Provider（akshare 实现）

接口实测基准（akshare 1.18.94，2026-08-29）：
    ak.option_risk_indicator_sse(date)     逐日全市场 IV/Greeks（交易所官方口径，
                                           深度实虚值合约 IV=0 占位，须过滤为 NaN）
    ak.option_current_day_sse()            当日全市场快照（含逐合约成交量/持仓量，
                                           用于自建历史 OI 库）
    ak.option_sse_daily_sina(security_id)  逐合约历史日线（成交量单位=份额，÷10000=张）
    ak.fund_etf_hist_sina(symbol)          标的 ETF 日线
    ak.tool_trade_date_hist_sina()         交易日历（替代硬编码节假日表）

数据源可插拔：业务代码只依赖 MarketDataProvider 协议（设计方案 §3 铁律3）。
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..core.spec import expiry_of_month

__all__ = ["MarketDataProvider", "SseOptionProvider", "ParquetStore"]


class MarketDataProvider(ABC):
    """数据源协议——业务层只认这个接口，不认 akshare"""

    @abstractmethod
    def risk_indicators(self, day: date) -> pd.DataFrame:
        """单日全市场期权风险指标（contract_id/right/strike/expiry/iv/greeks）"""

    @abstractmethod
    def underlying_daily(self, underlying: str, start: date, end: date) -> pd.DataFrame:
        """标的日线（date/close/...）"""


# ---------------------------------------------------------------- 合约代码解析

_RE_CONTRACT = re.compile(
    r"^(?P<und>\d{6}|[A-Z]{2})(?P<right>[CP])(?P<ym>\d{4})(?P<adj>[MABCD])(?P<strike>\d{5})$")


def parse_contract_id(contract_id: str) -> Optional[Dict]:
    """
    解析上交所交易代码 510300C2609M04500：
        und=510300 right=C ym=2609 adj=M strike=04500(÷1000=4.5)
    返回 None 表示非标格式（如股指期权 IO2609-C-4000）。
    """
    m = _RE_CONTRACT.match(str(contract_id).strip().upper())
    if not m:
        return None
    ym = m.group("ym")
    y, mo = 2000 + int(ym[:2]), int(ym[2:])
    try:
        exp = expiry_of_month(y, mo)
    except Exception:
        return None
    return {
        "underlying": m.group("und"),
        "right": "CALL" if m.group("right") == "C" else "PUT",
        "expiry": exp,
        "strike": int(m.group("strike")) / 1000.0,
        "adjusted": m.group("adj") != "M",
    }


class SseOptionProvider(MarketDataProvider):
    """上交所 ETF 期权数据源（akshare 后端，带进程内缓存与限速）"""

    def __init__(self, min_interval: float = 0.35):
        self._risk_cache: Dict[str, pd.DataFrame] = {}
        self._daily_cache: Dict[str, pd.DataFrame] = {}
        self._last_call: float = 0.0
        self.min_interval = min_interval   # 简单限速，避免触发上游频控

    def _throttle(self):
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    # ---- 风险指标（交易所官方，全市场一次一调用）
    def risk_indicators(self, day: date) -> pd.DataFrame:
        key = day.strftime("%Y%m%d")
        if key in self._risk_cache:
            return self._risk_cache[key]
        import akshare as ak
        self._throttle()
        raw = ak.option_risk_indicator_sse(date=key)
        df = raw.rename(columns={
            "CONTRACT_ID": "contract_id", "SECURITY_ID": "security_id",
            "TRADE_DATE": "trade_date", "IMPLC_VOLATLTY": "iv",
            "DELTA_VALUE": "delta", "GAMMA_VALUE": "gamma",
            "VEGA_VALUE": "vega", "THETA_VALUE": "theta", "RHO_VALUE": "rho",
        })
        parsed = df["contract_id"].map(parse_contract_id)
        ok = parsed.notna()
        df = df[ok].copy()
        for col in ("underlying", "right", "expiry", "strike", "adjusted"):
            df[col] = [p[col] for p in parsed[ok]]
        # 交易所对深度实虚值合约给 IV=0/Greeks=0 占位 → 一律置 NaN（禁止 0 兜底）
        for col in ("iv", "delta", "gamma", "vega", "theta"):
            df.loc[df[col] == 0, col] = float("nan")
        # A(分红调整)合约的 strike 字段是名义行权价——从合约简称解析真实值覆盖
        # （复检报告 2026-08-29：26372/26372 行 100% 名义值，均值偏差 0.073 元）
        if "CONTRACT_SYMBOL" in df.columns:
            import re as _re

            def _real_strike(sym_name):
                m = _re.search(r"(\d{3,5})A?$", str(sym_name))
                return int(m.group(1)) / 1000.0 if m else None

            adj = df["adjusted"] == True
            real = df.loc[adj, "CONTRACT_SYMBOL"].map(_real_strike)
            df.loc[adj, "strike"] = real.fillna(df.loc[adj, "strike"])
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        self._risk_cache[key] = df
        return df

    # ---- 当日合约静态表（合约单位/到期日/交收日/上市日 → instruments 主表）
    def contract_static(self) -> pd.DataFrame:
        """注意：接口名虽叫 option_current_day_sse，实际返回当日合约静态表，
        不含价格与量/仓。量/仓请走 contract_spot()。"""
        import akshare as ak
        self._throttle()
        df = ak.option_current_day_sse()
        rename = {}
        for c in df.columns:
            if "合约交易代码" in c:
                rename[c] = "contract_id"
            elif c == "合约编码":
                rename[c] = "security_id"
            elif c == "行权价":
                rename[c] = "strike"
            elif c == "合约单位":
                rename[c] = "multiplier"
            elif c == "到期日":
                rename[c] = "expiry"
            elif c == "期权行权日":
                rename[c] = "exercise_date"
            elif c == "行权交收日":
                rename[c] = "delivery_date"
            elif c == "开始日期":
                rename[c] = "list_date"
            elif c == "类型":
                rename[c] = "right_cn"
        df = df.rename(columns=rename)
        if "contract_id" in df.columns:
            parsed = df["contract_id"].map(parse_contract_id)
            ok = parsed.notna()
            df = df[ok].copy()
            for col in ("underlying", "right", "expiry"):
                df[col] = [p[col] for p in parsed[ok]]
            if "expiry" in df.columns:
                df["expiry"] = pd.to_datetime(df["expiry"], format="%Y%m%d").dt.date
        return df

    # ---- 逐合约实时/收盘快照（sina，五档盘口 + 量/仓；单位=张，已实证）
    def contract_spot(self, security_id: str) -> Dict:
        """返回单合约快照 dict。单位已实证：成交量/持仓量均为「张」
        （成交量×均价×合约单位≈成交额，与交易所标的级统计吻合）。
        涨停价/跌停价为交易所计算口径，可用于校验 calc_limit_prices。"""
        import akshare as ak
        self._throttle()
        raw = ak.option_sse_spot_price_sina(symbol=str(security_id))
        kv = dict(zip(raw["字段"].astype(str), raw["值"]))
        num = lambda k: float(kv.get(k, "nan") or "nan")
        return {
            "security_id": str(security_id),
            "short_name": kv.get("期权合约简称"),
            "underlying": kv.get("标的股票"),
            "last": num("最新价"), "bid": num("买价"), "ask": num("卖价"),
            "bid_vol": num("买量"), "ask_vol": num("卖量"),
            "open": num("开盘价"), "high": num("最高价"), "low": num("最低价"),
            "pre_close": num("昨收价"),
            "limit_up": num("涨停价"), "limit_down": num("跌停价"),
            "volume": num("成交量"), "open_interest": num("持仓量"),
            "quote_time": kv.get("行情时间"),
        }

    # ---- 逐合约历史日线（sina，成交量单位=份额，统一换算为张）
    def contract_daily(self, security_id: str) -> pd.DataFrame:
        if security_id in self._daily_cache:
            return self._daily_cache[security_id]
        import akshare as ak
        self._throttle()
        df = ak.option_sse_daily_sina(symbol=str(security_id))
        df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high",
                                "最低": "low", "收盘": "close", "成交量": "volume"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        # sina 期权日线成交量单位为「份额」，÷合约单位(10000) = 张
        if "volume" in df.columns:
            df["volume_lots"] = df["volume"] / 10000.0
        df["volume_unit"] = "lots"
        self._daily_cache[security_id] = df
        return df

    # ---- 标的日线
    def underlying_daily(self, underlying: str, start: date, end: date) -> pd.DataFrame:
        import akshare as ak
        self._throttle()
        # 深市 ETF（159xxx）用 sz 前缀
        prefix = "sz" if underlying.startswith("1") else "sh"
        raw = ak.fund_etf_hist_sina(symbol=f"{prefix}{underlying}")
        df = raw.rename(columns={"日期": "date", "开盘": "open", "最高": "high",
                                 "最低": "low", "收盘": "close", "成交量": "volume"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        mask = (df["date"] >= start) & (df["date"] <= end)
        return df[mask].reset_index(drop=True)

    # ---- 交易日历（国务院/交易所口径，替代 spec.TradingCalendar 硬编码表）
    def trading_days(self, start: Optional[date] = None, end: Optional[date] = None) -> List[date]:
        import akshare as ak
        self._throttle()
        cal = ak.tool_trade_date_hist_sina()
        days = pd.to_datetime(cal["trade_date"]).dt.date
        if start:
            days = days[days >= start]
        if end:
            days = days[days <= end]
        return sorted(days)

    # ---- 链表组装（风险指标 → 单标的单到期日）
    def chain_table(self, day: date, underlying: str,
                    expiry: Optional[date] = None) -> pd.DataFrame:
        df = self.risk_indicators(day)
        g = df[df["underlying"] == underlying]
        if expiry is not None:
            g = g[g["expiry"] == expiry]
        return g.reset_index(drop=True)


# ---------------------------------------------------------------- 存储


class ParquetStore:
    """行情事实表存储：optlab_data/store/<name>/<year>-<month>.parquet"""

    def __init__(self, root: str | Path = "optlab_data/store"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, df: pd.DataFrame, partition_col: str = "trade_date") -> int:
        if df.empty:
            return 0
        part = df[partition_col].iloc[0]
        key = part.strftime("%Y-%m") if isinstance(part, (date, datetime)) else str(part)[:7]
        out_dir = self.root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{key}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            df = pd.concat([old, df], ignore_index=True) \
                   .drop_duplicates(subset=[c for c in ("contract_id", "trade_date")
                                            if c in df.columns], keep="last")
        df.to_parquet(path, index=False)
        return len(df)

    def read(self, name: str, start: Optional[date] = None,
             end: Optional[date] = None) -> pd.DataFrame:
        out_dir = self.root / name
        if not out_dir.exists():
            return pd.DataFrame()
        frames = [pd.read_parquet(p) for p in sorted(out_dir.glob("*.parquet"))]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        col = "trade_date" if "trade_date" in df.columns else "date"
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.date
            if start:
                df = df[df[col] >= start]
            if end:
                df = df[df[col] <= end]
        return df.reset_index(drop=True)
