"""
thetalab.strategy.advisor — 策略自动推荐（设计方案 §5.5）

四维打分：流动性 0.30 + IV 吸引力 0.30 + 期限适配 0.20 + 结构匹配 0.20 − 惩罚项。
规则矩阵给出候选模板；每张卡片带 ≤3 条数据依据 + 针对性风险 + 退出计划。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from .templates import get_template
from .spec import StrategySpec

__all__ = ["Recommendation", "Advisor"]


@dataclass
class Recommendation:
    template_id: str
    name: str
    score: float                    # 0~100
    reasons: List[str]              # ≤3 条，每条带数据依据
    risks: List[str]                # 针对该策略的风险（非套话）
    exit_plan: Dict = field(default_factory=dict)
    spec: Optional[StrategySpec] = None
    needs_permission_note: bool = False   # 含卖方腿时 True（前端固定展示权限提示）


class Advisor:
    def __init__(self, templates=None):
        self._templates = templates or list(__import__(
            "thetalab.strategy.templates", fromlist=["TEMPLATES"]).TEMPLATES.keys())

    def recommend(self, ind: dict, chain_oi: float, chain_volume: float,
                  dte_choices: List[int], top_n: int = 3) -> List[Recommendation]:
        """
        ind: build_indicators 输出 + iv_rank/iv_pct/rv20
        chain_oi / chain_volume: 当日链合计持仓量/成交量（流动性分）
        dte_choices: 当前可选到期月的 DTE 列表
        """
        iv_rank = ind.get("iv_rank", float("nan"))
        iv_pct = ind.get("iv_pct", iv_rank)
        adx = ind.get("adx", float("nan"))
        trend = ind.get("trend", "CHOP")
        choppy = ind.get("choppy", adx == adx and adx < 20)
        strong_trend = ind.get("trend_strong", adx == adx and adx > 25)
        has_valid_iv = iv_rank == iv_rank

        # ---- 规则矩阵（§5.5）→ 候选与理由
        candidates: List[tuple] = []   # (template_id, [reasons], base_score)
        iv_hi = has_valid_iv and iv_rank > 0.70
        iv_lo = has_valid_iv and iv_rank < 0.30
        if choppy or not strong_trend:
            if iv_hi:
                candidates += [("short_strangle",
                                [f"IV 分位 {iv_rank:.0%}，处于高位，均值回复概率大",
                                 f"ADX={adx:.0f} 无趋势，区间震荡可同时赚 Theta 与 Vega 回落"], 90),
                               ("iron_condor",
                                ["同宽跨式逻辑，但加买远腿封顶尾部亏损"], 82)]
            elif iv_lo:
                candidates += [("long_straddle",
                                [f"IV 分位 {iv_rank:.0%}，处于低位，波动率便宜",
                                 "震荡蓄势后方向选择代价低"], 80)]
            else:
                candidates += [("long_call_butterfly",
                                ["IV 中性+无趋势，低成本赌到期落点"], 62),
                               ("calendar_spread",
                                ["中性市赚近月 Theta 加速"], 60)]
        elif strong_trend and trend == "UP":
            candidates += [("bull_call_spread", [f"趋势向上 (ADX={adx:.0f})，价差降成本"], 78),
                           ("covered_call", ["上行趋势中持股增强"], 70)]
            if iv_hi:
                candidates.append(("short_put",
                                   [f"高 IV (分位 {iv_rank:.0%}) 卖认沽权利金丰厚",
                                    "趋势向上被行权概率低"], 75))
        elif strong_trend and trend == "DOWN":
            candidates += [("protective_put", [f"趋势向下 (ADX={adx:.0f})，下行保护"], 74),
                           ("bear_put_spread", ["看跌价差降低保险成本"], 70)]
        # 长假/事件惩罚由 signals 层提示，此处不重复

        # ---- 打分
        out: List[Recommendation] = []
        for tid, reasons, base in candidates:
            spec = get_template(tid)
            liq = self._liquidity_score(chain_oi, chain_volume)
            iv_fit = self._iv_fit_score(tid, iv_rank, iv_pct)
            tenor = self._tenor_score(tid, dte_choices)
            struct = self._structure_score(tid, ind)
            score = 0.30 * liq + 0.30 * iv_fit + 0.20 * tenor + 0.20 * struct
            score = score * 0.6 + base * 0.4          # 规则矩阵先验 + 打分
            # 惩罚项
            penalties = []
            if has_valid_iv and iv_rank > 0.90 and tid.startswith("long_"):
                score -= 25
                penalties.append("IV 极高禁止买入（Vega/Theta 双杀）")
            if has_valid_iv and iv_rank < 0.15 and any(
                    l.direction == "SELL" for l in spec.legs if l.kind == "OPTION"):
                score -= 30
                penalties.append("IV 极低禁止裸卖（权利金不足以补偿尾部）")
            score = max(0.0, min(100.0, score))
            has_short = any(l.direction == "SELL" for l in spec.legs)
            out.append(Recommendation(
                template_id=tid, name=spec.name, score=round(score, 1),
                reasons=reasons[:3] + penalties,
                risks=self._template_risks(tid),
                exit_plan={"take_profit_pct": spec.exit.take_profit_pct,
                           "stop_multiple": spec.exit.stop_multiple,
                           "time_stop_dte": spec.exit.time_stop_dte},
                spec=spec, needs_permission_note=has_short))
        out.sort(key=lambda r: -r.score)
        return out[:top_n]

    # ------------------------------------------------------------ 打分子项
    @staticmethod
    def _liquidity_score(oi: float, vol: float) -> float:
        f_oi = min(1.0, max(0.0, (oi - 5e4) / (3e5 - 5e4))) if oi == oi else 0.5
        f_v = min(1.0, max(0.0, (vol - 1e4) / (5e5 - 1e4))) if vol == vol else 0.5
        return 100 * (0.5 * f_oi + 0.5 * f_v)

    @staticmethod
    def _iv_fit_score(tid: str, iv_rank: float, iv_pct: float) -> float:
        if iv_rank != iv_rank:
            return 50.0
        seller = tid.startswith(("short", "covered")) or tid == "calendar_spread"
        if seller:
            return 100 * min(1.0, max(0.0, iv_rank))
        return 100 * min(1.0, max(0.0, 1 - iv_rank))

    @staticmethod
    def _tenor_score(tid: str, dte_choices: List[int]) -> float:
        if not dte_choices:
            return 50.0
        target_lo, target_hi = (25, 45) if not tid.startswith("long_") else (30, 60)
        best = max((1.0 if target_lo <= d <= target_hi else
                    max(0.0, 1 - min(abs(d - target_lo), abs(d - target_hi)) / 60.0))
                   for d in dte_choices)
        return 100 * best

    @staticmethod
    def _structure_score(tid: str, ind: dict) -> float:
        trend, choppy = ind.get("trend", "CHOP"), ind.get("choppy", False)
        table = {"short_strangle": choppy, "iron_condor": choppy,
                 "long_straddle": choppy, "short_straddle": choppy,
                 "calendar_spread": choppy, "long_call_butterfly": choppy,
                 "bull_call_spread": trend == "UP", "covered_call": trend == "UP",
                 "short_put": trend == "UP",
                 "bear_put_spread": trend == "DOWN", "protective_put": trend == "DOWN",
                 "collar": trend == "DOWN"}
        return 100.0 if table.get(tid) else 45.0

    @staticmethod
    def _template_risks(tid: str) -> List[str]:
        risks = {
            "short_strangle": ["单边突破一侧亏损无上限，严格止损不可免"],
            "iron_condor": ["四腿手续费 ×2；突破近腿后 Delta 加速"],
            "long_straddle": ["双杀：Theta 持续损耗 + IV 不升反降"],
            "short_straddle": ["平值 Gamma 风险极大，DTE≤14 严禁持有"],
            "bull_call_spread": ["趋势反转；上行收益被封顶"],
            "covered_call": ["上行收益封顶；急跌时仍有标的全部回撤"],
            "short_put": ["急跌被指派，接货成本高于市价"],
            "protective_put": ["保险费持续磨损 Theta"],
            "bear_put_spread": ["下行收益有限；反弹则权利金双亏"],
            "long_call_butterfly": ["收益上限低；手续费占比高"],
            "calendar_spread": ["标的大幅跳动双侧受损；移仓成本"],
            "collar": ["反弹收益被封顶"],
        }
        return risks.get(tid, ["策略与当前市场状态不匹配"])
