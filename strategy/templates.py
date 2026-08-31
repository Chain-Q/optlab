"""
optlab.strategy.templates — 内置策略模板库（设计方案 §5.4，P3 验收：≥6 个）

每条腿独立选择行权价/到期月，适应行情变化；JSON 可序列化可分享。
"""
from __future__ import annotations

from .spec import ExpirySelector, LegSpec, Sizing, StrikeSelector, StrategySpec

__all__ = ["TEMPLATES", "get_template"]


def _opt(right, direction, s_type="DELTA", target=0.20, ratio=1,
         s_min=25, s_max=45) -> LegSpec:
    return LegSpec(kind="OPTION", right=right, direction=direction, ratio=ratio,
                   strike_selector=StrikeSelector(type=s_type, target=target),
                   expiry_selector=ExpirySelector(type="DTE_RANGE", min=s_min, max=s_max))


def _und(direction="BUY") -> LegSpec:
    return LegSpec(kind="UNDERLYING", direction=direction)


TEMPLATES: dict = {
    # ---------- 波动率类 ----------
    "short_strangle": StrategySpec(
        id="short_strangle", name="卖出宽跨式", category="volatility",
        legs=[_opt("CALL", "SELL", "DELTA", 0.20), _opt("PUT", "SELL", "DELTA", 0.20)],
        entry={"conditions": ["iv_rank > 0.5"]},
        notes="震荡+IV高位适用；单边突破一侧亏损无上限，严格止损"),
    "iron_condor": StrategySpec(
        id="iron_condor", name="铁鹰", category="volatility",
        legs=[_opt("CALL", "SELL", "DELTA", 0.25), _opt("CALL", "BUY", "DELTA", 0.08),
              _opt("PUT", "SELL", "DELTA", 0.25), _opt("PUT", "BUY", "DELTA", 0.08)],
        entry={"conditions": ["iv_rank > 0.5"]},
        notes="亏损有限的卖方组合； IV 回落+区间震荡双重盈利"),
    "long_straddle": StrategySpec(
        id="long_straddle", name="买入跨式", category="volatility",
        legs=[_opt("CALL", "BUY", "ATM"), _opt("PUT", "BUY", "ATM")],
        entry={"conditions": ["iv_rank < 0.3"]},
        notes="IV 低位赌波动放大；双杀风险（Theta+IV 回落）"),
    "short_straddle": StrategySpec(
        id="short_straddle", name="卖出跨式", category="volatility",
        legs=[_opt("CALL", "SELL", "ATM"), _opt("PUT", "SELL", "ATM")],
        entry={"conditions": ["iv_rank > 0.7"]},
        notes="高 IV 收租；Gamma 尾部风险极大，DTE≤14 严禁持有"),
    "short_put": StrategySpec(
        id="short_put", name="卖出认沽（现金担保）", category="direction",
        legs=[_opt("PUT", "SELL", "DELTA", 0.25)],
        entry={"conditions": ["iv_rank > 0.5", "trend == UP"]},
        notes="高IV+上行趋势收权利金；急跌被指派需备好接货资金"),
    # ---------- 方向/价差类 ----------
    "bull_call_spread": StrategySpec(
        id="bull_call_spread", name="牛市认购价差", category="spread",
        legs=[_opt("CALL", "BUY", "DELTA", 0.40), _opt("CALL", "SELL", "DELTA", 0.20)],
        notes="看涨降成本；收益上限=价差宽度"),
    "bear_put_spread": StrategySpec(
        id="bear_put_spread", name="熊市认沽价差", category="spread",
        legs=[_opt("PUT", "BUY", "DELTA", 0.40), _opt("PUT", "SELL", "DELTA", 0.20)],
        notes="看跌降成本；适合对冲而非重仓方向"),
    "long_call_butterfly": StrategySpec(
        id="long_call_butterfly", name="认购蝶式", category="spread",
        legs=[_opt("CALL", "BUY", "DELTA", 0.35), _opt("CALL", "SELL", "DELTA", 0.20, ratio=2),
              _opt("CALL", "BUY", "DELTA", 0.08)],
        notes="低成本赌到期价落中间；手续费占比高"),
    "calendar_spread": StrategySpec(
        id="calendar_spread", name="日历价差", category="spread",
        legs=[LegSpec(kind="OPTION", right="CALL", direction="SELL",
                      strike_selector=StrikeSelector(type="ATM"),
                      expiry_selector=ExpirySelector(type="NEAREST_EXPIRY", n=1)),
              LegSpec(kind="OPTION", right="CALL", direction="BUY",
                      strike_selector=StrikeSelector(type="ATM"),
                      expiry_selector=ExpirySelector(type="NEAREST_EXPIRY", n=2))],
        notes="赚近月 Theta 加速；标的大幅跳动则受损"),
    # ---------- 对冲/增强类 ----------
    "covered_call": StrategySpec(
        id="covered_call", name="备兑开仓", category="hedge",
        legs=[_und("BUY"), _opt("CALL", "SELL", "DELTA", 0.30)],
        notes="持股增强收益；上行收益被封顶；保证金可全额抵扣"),
    "protective_put": StrategySpec(
        id="protective_put", name="保护性认沽", category="hedge",
        legs=[_und("BUY"), _opt("PUT", "BUY", "DELTA", 0.30)],
        notes="下行保险；持续磨损 Theta，适合事件/长假前"),
    "collar": StrategySpec(
        id="collar", name="领口", category="hedge",
        legs=[_und("BUY"), _opt("PUT", "BUY", "DELTA", 0.30), _opt("CALL", "SELL", "DELTA", 0.20)],
        notes="卖购补贴买沽；反弹收益被封顶"),
}


def get_template(tid: str) -> StrategySpec:
    if tid not in TEMPLATES:
        raise KeyError(f"未知模板 {tid}，可选: {list(TEMPLATES)}")
    import copy
    return copy.deepcopy(TEMPLATES[tid])
