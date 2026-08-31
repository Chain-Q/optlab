"""
optlab.strategy.signals — 择时信号引擎（设计方案 §5.6）

四类信号源：波动率偏离 / 技术形态 / 到期临近 / 事件日历。
强制要求：每条信号带证据链（evidence），dedup_key 冷却去重，界面只展示 Top 5，
矛盾信号并列展示由人决策——系统不替人做最终决定。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

__all__ = ["Signal", "SignalEngine"]


@dataclass
class Signal:
    kind: str                    # VOL / TECH / EXPIRY / EVENT / RISK
    name: str
    strength: float              # 0~100
    reason: str                  # 人话理由
    evidence: Dict = field(default_factory=dict)   # 触发依据原始数据（可审计）
    action: str = ""             # 建议（提示性，不自动下单）
    dedup_key: str = ""

    @property
    def level(self) -> str:
        if self.strength >= 80:
            return "强"
        if self.strength >= 60:
            return "中"
        if self.strength >= 40:
            return "弱"
        return "提示"


class SignalEngine:
    def __init__(self, cooldown_days: int = 1):
        self.cooldown = cooldown_days
        self._last_fired: Dict[str, date] = {}

    def generate(self, ind: dict, chain_today=None, chain_prev=None,
                 positions: Optional[Dict] = None, today: date = None,
                 next_expiry: Optional[date] = None,
                 holidays_until: Optional[List[date]] = None) -> List[Signal]:
        """
        ind: build_indicators 输出（含 iv_rank/rv20/bb_pos/adx/trend）
        chain_today/chain_prev: 当日/5日前链 DataFrame（IV 偏斜用，可选）
        positions: {symbol: Position}（到期预警需要）
        holidays_until: 未来 N 个交易日内的节假日（长假提示）
        """
        today = today or date.today()
        sigs: List[Signal] = []
        sigs += self._vol_signals(ind, chain_today, chain_prev, today)
        sigs += self._tech_signals(ind, today)
        sigs += self._expiry_signals(positions or {}, today, next_expiry)
        sigs += self._event_signals(today, next_expiry, holidays_until or [])
        # 冷却去重
        out = []
        for s in sigs:
            if not s.dedup_key:
                out.append(s)
                continue
            last = self._last_fired.get(s.dedup_key)
            if last and (today - last).days <= self.cooldown:
                continue
            self._last_fired[s.dedup_key] = today
            out.append(s)
        out.sort(key=lambda s: -s.strength)
        return out[:5]                     # 降噪：只出 Top 5

    # ------------------------------------------------------------ 波动率
    def _vol_signals(self, ind, chain_today, chain_prev, today) -> List[Signal]:
        sigs = []
        iv_rank = ind.get("iv_rank", float("nan"))
        iv_atm = ind.get("iv_atm", float("nan"))
        rv20 = ind.get("rv20", float("nan"))
        if iv_rank == iv_rank and iv_rank > 0.8:
            sigs.append(Signal(
                kind="VOL", name="IV 高位 → 偏卖方",
                strength=min(100.0, (iv_rank - 0.8) * 250 + 50),
                reason=f"IV 分位 {iv_rank:.0%} 处于历史高位，偏卖方策略",
                evidence={"iv_rank": iv_rank, "iv_atm": iv_atm},
                action="考虑卖出宽跨式/铁鹰", dedup_key="iv_hi"))
        elif iv_rank == iv_rank and iv_rank < 0.2:
            sigs.append(Signal(
                kind="VOL", name="IV 低位 → 偏买方",
                strength=min(100.0, (0.2 - iv_rank) * 250 + 50),
                reason=f"IV 分位 {iv_rank:.0%} 处于历史低位，权利金便宜",
                evidence={"iv_rank": iv_rank},
                action="考虑买入跨式/宽跨式", dedup_key="iv_lo"))
        # IV − RV 价差 z 分数（简化：用 (iv−rv)/rv 的相对水平）
        if iv_atm == iv_atm and rv20 == rv20 and rv20 > 0:
            spread = iv_atm - rv20
            z = spread / max(rv20 * 0.25, 0.02)     # 经验标准化
            if abs(z) > 1.5:
                direction = "做空波动率" if z > 0 else "做多波动率"
                sigs.append(Signal(
                    kind="VOL", name=f"IV−RV 价差异常（{direction}）",
                    strength=min(100.0, abs(z) * 30),
                    reason=f"IV {iv_atm:.1%} vs RV20 {rv20:.1%}，偏离 {z:.1f}σ",
                    evidence={"iv": iv_atm, "rv20": rv20, "z": z},
                    action=f"倾向{direction}", dedup_key=f"ivrv_{"pos" if z > 0 else "neg"}"))
        # 偏斜突变（put ATM IV − call ATM IV，5 日对比）
        if chain_today is not None and chain_prev is not None:
            sk_now = self._skew(chain_today)
            sk_prev = self._skew(chain_prev)
            if sk_now == sk_now and sk_prev == sk_prev and abs(sk_now - sk_prev) > 0.03:
                sigs.append(Signal(
                    kind="VOL", name="偏斜突变（恐慌升温）" if sk_now > sk_prev
                    else "偏斜回落",
                    strength=min(100.0, abs(sk_now - sk_prev) * 1000),
                    reason=f"Put-Call ATM IV 差 {sk_prev:+.1%} → {sk_now:+.1%}",
                    evidence={"skew_now": sk_now, "skew_prev": sk_prev},
                    action="下行保护需求变化", dedup_key=f"skew_{"up" if sk_now > sk_prev else "dn"}"))
        return sigs

    @staticmethod
    def _skew(chain) -> float:
        """ATM Put IV − ATM Call IV（正值=下行保护更贵）"""
        import pandas as pd
        if chain is None or chain.empty or "iv" not in chain.columns:
            return float("nan")
        g = chain[chain["iv"].notna()]
        if g.empty:
            return float("nan")
        mid_k = g["strike"].median()
        out = {}
        for right in ("CALL", "PUT"):
            gg = g[g["right"] == right]
            if gg.empty:
                return float("nan")
            k = gg["strike"].iloc[int((gg["strike"] - mid_k).abs().values.argmin())]
            out[right] = float(gg[gg["strike"] == k]["iv"].iloc[0])
        return out["PUT"] - out["CALL"]

    # ------------------------------------------------------------ 技术
    def _tech_signals(self, ind, today) -> List[Signal]:
        sigs = []
        trend, adx = ind.get("trend", "CHOP"), ind.get("adx", float("nan"))
        if trend == "UP" and adx == adx and adx > 25:
            sigs.append(Signal(
                kind="TECH", name="多头趋势确立",
                strength=min(100.0, 50 + adx), reason=f"MA5>MA20>MA60 且 ADX={adx:.0f}",
                evidence={"trend": trend, "adx": adx},
                action="偏多方向策略", dedup_key="trend_up"))
        elif trend == "DOWN" and adx == adx and adx > 25:
            sigs.append(Signal(
                kind="TECH", name="空头趋势确立",
                strength=min(100.0, 50 + adx), reason=f"MA5<MA20<MA60 且 ADX={adx:.0f}",
                evidence={"trend": trend, "adx": adx},
                action="偏空/保护策略", dedup_key="trend_dn"))
        bb = ind.get("bb_pos", 0.0)
        atr = ind.get("atr14", float("nan"))
        if bb == bb and abs(bb) > 1.0 and atr == atr:
            sigs.append(Signal(
                kind="TECH", name="布林上轨突破" if bb > 0 else "布林下轨突破",
                strength=min(100.0, 50 + abs(bb) * 20),
                reason=f"价格偏离 20 日均线 {bb:.1f} 个标准差（ATR={atr:.4f}）",
                evidence={"bb_pos": bb, "atr14": atr},
                action="突破跟随或回归博弈，结合 IV 判断", dedup_key=f"bb_{"up" if bb > 0 else "dn"}"))
        # 波动收敛（蓄势）：ATR% 处于自身历史 20 分位以下
        atr_pct = ind.get("atr_pct", float("nan"))
        low_thr = ind.get("atr_pct_low", float("nan"))
        if atr_pct == atr_pct and low_thr == low_thr and atr_pct <= low_thr:
            sigs.append(Signal(
                kind="TECH", name="波动收敛（突破前夜）",
                strength=55, reason=f"ATR/价格 {atr_pct:.2%} 处于历史低位（阈值 {low_thr:.2%}）",
                evidence={"atr_pct": atr_pct, "threshold": low_thr},
                action="买方蓄势信号", dedup_key="squeeze"))
        rsi = ind.get("rsi14", float("nan"))
        if rsi == rsi and (rsi > 75 or rsi < 25):
            sigs.append(Signal(
                kind="TECH", name="超买" if rsi > 75 else "超卖",
                strength=55, reason=f"RSI(14)={rsi:.0f}",
                evidence={"rsi": rsi}, action="回归风险提示",
                dedup_key=f"rsi_{"hi" if rsi > 75 else "lo"}"))
        return sigs

    # ------------------------------------------------------------ 到期
    def _expiry_signals(self, positions, today, next_expiry) -> List[Signal]:
        sigs = []
        for sym, pos in positions.items():
            inst = getattr(pos, "instrument", None)
            if inst is None or not inst.is_option:
                continue
            dte = (inst.expiry - today).days
            delta = abs(pos.last_delta) if getattr(pos, "last_delta", None) else 0.5
            if dte <= 3 and pos.net_qty < 0 and 0.3 <= delta <= 0.7:
                sigs.append(Signal(
                    kind="RISK", name="Gamma 风险预警（强）",
                    strength=95, reason=f"{sym} DTE={dte} 且 |Δ|={delta:.2f} 平值附近义务仓",
                    evidence={"dte": dte, "delta": delta},
                    action="立即平仓或移仓", dedup_key=f"gamma_{sym}"))
            elif dte <= 14 and pos.net_qty > 0:
                sigs.append(Signal(
                    kind="EXPIRY", name="Theta 加速预警（买方）",
                    strength=min(90.0, 40 + (14 - dte) * 4),
                    reason=f"{sym} DTE={dte}，买方时间损耗加速",
                    evidence={"dte": dte}, action="考虑平仓/移仓",
                    dedup_key=f"theta_{sym}"))
            elif dte <= 7:
                sigs.append(Signal(
                    kind="EXPIRY", name="移仓窗口",
                    strength=60, reason=f"{sym} DTE={dte}≤7",
                    evidence={"dte": dte}, action="评估移仓至下月",
                    dedup_key=f"roll_{sym}"))
        return sigs

    # ------------------------------------------------------------ 事件
    def _event_signals(self, today, next_expiry, holidays) -> List[Signal]:
        sigs = []
        if next_expiry:
            dte = (next_expiry - today).days
            if 0 <= dte <= 5:
                sigs.append(Signal(
                    kind="EVENT", name="期权到期日临近",
                    strength=50, reason=f"到期日 {next_expiry}（{dte} 天）",
                    evidence={"expiry": str(next_expiry), "dte": dte},
                    action="实值合约决定行权/平仓", dedup_key=f"expiry_{next_expiry}"))
        soon = [h for h in holidays if 0 <= (h - today).days <= 5]
        if soon:
            sigs.append(Signal(
                kind="EVENT", name="长假临近，谨慎卖权",
                strength=75, reason=f"{soon[0]} 起休市，期间无法调整持仓",
                evidence={"holiday": str(soon[0])},
                action="卖方降仓或对冲", dedup_key=f"holiday_{soon[0]}"))
        return sigs
