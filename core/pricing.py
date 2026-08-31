"""
optlab.core.pricing — 期权定价、希腊字母与隐含波动率

支持三种模型：
    1. Black-Scholes-Merton（连续分红率 q）：适用于 ETF 期权（欧式）
    2. Black-76（期货期权，q = r）：适用于商品期权、股指期权的期货视角
    3. CRR 二叉树（N 步）：适用于美式期权 / 提前行权判断 / 障碍近似

数值约定（与交易台口径一致）：
    - 价格返回「每单位标的」价格，实际金额需 × multiplier
    - delta：每单位标的期权价格对标的价格的导数（0~1 / -1~0）
    - gamma：delta 对标的价格的导数
    - vega ：波动率上升 1 个百分点（1 vol point）时的价格变化
    - theta：每自然日时间损耗（价格单位），负数表示损耗
    - rho  ：无风险利率上升 1 个百分点时的价格变化
所有希腊字母均有 analytic 解，另提供 finite_difference 版本用于二叉树与美式期权。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from .models import ExerciseStyle, Greeks, Instrument, Right

__all__ = [
    "norm_cdf", "norm_pdf",
    "bs_price", "bs_greeks", "black76_price", "black76_greeks",
    "crr_price", "crr_greeks",
    "price", "greeks", "implied_vol",
    "PriceModel",
]

SQRT_2PI = math.sqrt(2.0 * math.pi)
_EPS = 1e-12


# ---------------------------------------------------------------- 正态函数


def norm_cdf(x: float) -> float:
    """标准正态累积分布，Abramowitz-Stegun 替代：使用 erfc 保证精度"""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


# ---------------------------------------------------------------- Black-Scholes-Merton


def _d1d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> Tuple[float, float]:
    if S <= 0 or K <= 0:
        # 内在价值极限：由调用方处理，这里给安全值
        S = max(S, _EPS)
        K = max(K, _EPS)
    vt = sigma * math.sqrt(max(T, _EPS))
    if vt < _EPS:
        # sigma→0 或 T→0 的退化情形
        fwd = S * math.exp((r - q) * T)
        if abs(fwd - K) < _EPS:
            return 0.0, 0.0
        return (math.inf if fwd > K else -math.inf,
                math.inf if fwd > K else -math.inf)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vt
    d2 = d1 - vt
    return d1, d2


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             right: Right, q: float = 0.0) -> float:
    """BSM 期权价格（每单位标的）"""
    if T <= 0:
        return max(0.0, (S - K) if right is Right.CALL else (K - S))
    if sigma <= 0:
        fwd = S * math.exp((r - q) * T)
        return max(0.0, math.exp(-r * T) * ((fwd - K) if right is Right.CALL else (K - fwd)))
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    if right is Right.CALL:
        return S * df_q * norm_cdf(d1) - K * df_r * norm_cdf(d2)
    return K * df_r * norm_cdf(-d2) - S * df_q * norm_cdf(-d1)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
              right: Right, q: float = 0.0) -> Greeks:
    """BSM 解析希腊字母。vega/theta/rho 已按「1 个百分点 / 1 天」换算。"""
    if T <= _EPS or sigma <= _EPS:
        # 到期/零波动：退化处理
        itm = (S > K) if right is Right.CALL else (S < K)
        delta = (1.0 if itm else 0.0) if right is Right.CALL else (-1.0 if itm else 0.0)
        return Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0, iv=sigma)

    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    pdf_d1 = norm_pdf(d1)
    df_q = math.exp(-q * T)
    df_r = math.exp(-r * T)
    sqrtT = math.sqrt(T)

    gamma = df_q * pdf_d1 / (S * sigma * sqrtT)
    vega_1pct = S * df_q * pdf_d1 * sqrtT * 0.01          # 每 +1 vol point

    if right is Right.CALL:
        delta = df_q * norm_cdf(d1)
        theta = (-(S * df_q * pdf_d1 * sigma) / (2.0 * sqrtT)
                 - r * K * df_r * norm_cdf(d2)
                 + q * S * df_q * norm_cdf(d1)) / 365.0
        rho_1pct = K * T * df_r * norm_cdf(d2) * 0.01
    else:
        delta = -df_q * norm_cdf(-d1)
        theta = (-(S * df_q * pdf_d1 * sigma) / (2.0 * sqrtT)
                 + r * K * df_r * norm_cdf(-d2)
                 - q * S * df_q * norm_cdf(-d1)) / 365.0
        rho_1pct = -K * T * df_r * norm_cdf(-d2) * 0.01

    return Greeks(delta=delta, gamma=gamma, vega=vega_1pct, theta=theta,
                  rho=rho_1pct, iv=sigma)


# ---------------------------------------------------------------- Black-76


def black76_price(F: float, K: float, T: float, r: float, sigma: float, right: Right) -> float:
    """期货期权：等价于 q = r 的 BSM"""
    return bs_price(F, K, T, r, sigma, right, q=r)


def black76_greeks(F: float, K: float, T: float, r: float, sigma: float, right: Right) -> Greeks:
    return bs_greeks(F, K, T, r, sigma, right, q=r)


# ---------------------------------------------------------------- CRR 二叉树（美式）


def crr_price(S: float, K: float, T: float, r: float, sigma: float,
              right: Right, q: float = 0.0, steps: int = 100,
              american: bool = True) -> float:
    """
    Cox-Ross-Rubinstein 二叉树。
    american=True 时每个节点取 max(持有价值, 立即行权价值)。
    """
    if T <= 0:
        return max(0.0, (S - K) if right is Right.CALL else (K - S))
    n = max(2, int(steps))
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)  # 数值保护

    sign = 1 if right is Right.CALL else -1
    # 到期价值
    vals = [max(0.0, sign * (S * (u ** (n - i)) * (d ** i) - K)) for i in range(n + 1)]
    # 回溯
    for j in range(n - 1, -1, -1):
        new_vals = []
        for i in range(j + 1):
            hold = disc * (p * vals[i] + (1 - p) * vals[i + 1])
            if american:
                exer = max(0.0, sign * (S * (u ** (j - i)) * (d ** i) - K))
                new_vals.append(max(hold, exer))
            else:
                new_vals.append(hold)
        vals = new_vals
    return vals[0]


def crr_greeks(S: float, K: float, T: float, r: float, sigma: float,
               right: Right, q: float = 0.0, steps: int = 100,
               american: bool = True) -> Greeks:
    """有限差分法求二叉树希腊字母（美式无解析解）"""
    h_s = max(S * 1e-4, 1e-6)
    h_v = 1e-4
    h_t = max(T / max(steps, 1), 1.0 / 365.0)

    p0 = crr_price(S, K, T, r, sigma, right, q, steps, american)
    p_up = crr_price(S + h_s, K, T, r, sigma, right, q, steps, american)
    p_dn = crr_price(S - h_s, K, T, r, sigma, right, q, steps, american)
    p_vol = crr_price(S, K, T, r, sigma + h_v, right, q, steps, american)
    p_t = crr_price(S, K, max(T - h_t, 1e-6), r, sigma, right, q, steps, american)

    delta = (p_up - p_dn) / (2 * h_s)
    gamma = (p_up - 2 * p0 + p_dn) / (h_s * h_s)
    vega = (p_vol - p0) * 0.01                    # 每 1 vol point
    theta = (p_t - p0) / (h_t * 365.0)            # 每日
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=float("nan"), iv=sigma)


# ---------------------------------------------------------------- 统一入口


def _model_inputs(inst: Instrument, spot: float, T: float, r: float,
                  sigma: float, q: float) -> Tuple[float, float]:
    """根据交割方式选择定价基准（现货 or 远期）"""
    if inst.delivery.value == "CASH" or inst.asset_class.value == "INDEX_OPTION":
        # 现金交割/股指：可直接用远期价做 Black76
        F = spot * math.exp((r - q) * T)
        return F, r
    return spot, q


def price(inst: Instrument, spot: float, T: float, r: float, sigma: float,
          q: float = 0.0, steps: int = 100) -> float:
    """统一价格入口：欧式走 BS，美式走 CRR"""
    if inst.exercise_style is ExerciseStyle.AMERICAN:
        return crr_price(spot, inst.strike, T, r, sigma, inst.right, q, steps, american=True)
    if inst.asset_class.value == "INDEX_OPTION":
        F = spot * math.exp((r - q) * T)
        return black76_price(F, inst.strike, T, r, sigma, inst.right)
    return bs_price(spot, inst.strike, T, r, sigma, inst.right, q)


def greeks(inst: Instrument, spot: float, T: float, r: float, sigma: float,
           q: float = 0.0, steps: int = 100) -> Greeks:
    """统一希腊字母入口"""
    if inst.exercise_style is ExerciseStyle.AMERICAN:
        g = crr_greeks(spot, inst.strike, T, r, sigma, inst.right, q, steps, american=True)
        g.iv = sigma
        return g
    if inst.asset_class.value == "INDEX_OPTION":
        F = spot * math.exp((r - q) * T)
        g = black76_greeks(F, inst.strike, T, r, sigma, inst.right)
        # Black76 的 delta 是相对期货价，转成相对现货：∂/∂S = ∂/∂F × e^{(r-q)T}
        adj = math.exp((r - q) * T)
        g.delta *= adj
        g.gamma *= adj * adj
        g.iv = sigma
        return g
    g = bs_greeks(spot, inst.strike, T, r, sigma, inst.right, q)
    g.iv = sigma
    return g


# ---------------------------------------------------------------- 隐含波动率


def implied_vol(market_price: float, S: float, K: float, T: float, r: float,
                right: Right, q: float = 0.0, *,
                american: bool = False, steps: int = 60,
                lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-6,
                max_iter: int = 80) -> float:
    """
    隐含波动率求解：Newton-Raphson 为主，失败回落二分法。
    返回 nan 表示无解（通常是价格低于内在价值或超出理论上限）。

    Newton 初值用 Brenner-Subrahmanyam 近似，收敛快且稳定。
    """
    if T <= _EPS:
        return float("nan")
    intrinsic = max(0.0,
                    (S * math.exp(-q * T) - K * math.exp(-r * T)) if right is Right.CALL
                    else (K * math.exp(-r * T) - S * math.exp(-q * T)))
    if market_price <= intrinsic + 1e-9:
        return float("nan")

    if american:
        f: Callable[[float], float] = (
            lambda v: crr_price(S, K, T, r, v, right, q, steps, american=True))
    else:
        f = lambda v: bs_price(S, K, T, r, v, right, q)

    # 上下界检查
    if market_price > f(hi):
        return float("nan")

    # 初值（Brenner-Subrahmanyam ATM 近似）
    fwd = S * math.exp((r - q) * T)
    v0 = math.sqrt(2 * math.pi / T) * market_price / (fwd + K)
    v = min(max(v0, lo * 2), hi * 0.9)

    for _ in range(max_iter):
        p = f(v)
        diff = p - market_price
        if abs(diff) < tol:
            return v
        # 数值 vega（中心差分）
        hv = max(v * 1e-4, 1e-6)
        vega = (f(v + hv) - f(max(v - hv, lo))) / (2 * hv)
        if abs(vega) < 1e-10:
            break
        step = diff / vega
        # 阻尼，避免跳飞
        step = max(min(step, 0.5), -0.5)
        v_new = v - step
        if not (lo < v_new < hi):
            break
        if abs(v_new - v) < 1e-10:
            v = v_new
            break
        v = v_new

    # 二分兜底
    a, b = lo, hi
    for _ in range(200):
        m = 0.5 * (a + b)
        if f(m) < market_price:
            a = m
        else:
            b = m
        if b - a < tol:
            return 0.5 * (a + b)
    return 0.5 * (a + b)


# ---------------------------------------------------------------- 定价上下文


@dataclass
class PriceModel:
    """定价上下文：捆绑利率/分红/模型参数，避免到处传参"""
    r: float = 0.03
    q: float = 0.0
    steps: int = 100

    def price(self, inst: Instrument, spot: float, T: float, sigma: float) -> float:
        return price(inst, spot, T, self.r, sigma, self.q, self.steps)

    def greeks(self, inst: Instrument, spot: float, T: float, sigma: float) -> Greeks:
        return greeks(inst, spot, T, self.r, sigma, self.q, self.steps)

    def iv(self, market_price: float, inst: Instrument, spot: float, T: float) -> float:
        am = inst.exercise_style is ExerciseStyle.AMERICAN
        return implied_vol(market_price, spot, inst.strike, T, self.r, inst.right,
                           self.q, american=am, steps=min(self.steps, 60))
