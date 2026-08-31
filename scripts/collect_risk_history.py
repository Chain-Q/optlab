"""
thetalab.scripts.collect_risk_history — 历史风险指标采集 + ATM IV 序列/IV rank 实测

用途（P1 数据层实测）：
    1. 循环拉取近 N 个交易日的 option_risk_indicator_sse，断点续跑落 Parquet
    2. 用 510300 验证 ATM IV 时间序列与 IV rank 的可计算性
运行：python -m thetalab.scripts.collect_risk_history [days=120]
"""
import sys
import time
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from thetalab.data.provider import SseOptionProvider, ParquetStore

UNDERLYING = "510300"


def main(days: int = 120):
    t0 = time.time()
    p = SseOptionProvider(min_interval=0.3)
    store = ParquetStore("thetalab_data/store")

    # 断点续跑：已采集日期集合
    done = set()
    if not store.read("risk_indicators").empty:
        done = set(store.read("risk_indicators")["trade_date"].unique())
    print(f"已采集 {len(done)} 天，目标新增 {days} 天")

    all_days = p.trading_days(end=date.today())
    todo = [d for d in all_days if d not in done][-days:]
    print(f"待采集 {len(todo)} 天: {todo[0]} ~ {todo[-1]}")

    ok = fail = 0
    for i, d in enumerate(todo):
        try:
            df = p.risk_indicators(d)
            store.write("risk_indicators", df)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  [{d}] FAIL {type(e).__name__}: {str(e)[:80]}")
        if (i + 1) % 20 == 0:
            print(f"  进度 {i+1}/{len(todo)} 成功{ok} 失败{fail} 耗时{time.time()-t0:.0f}s")

    print(f"\n采集完成: 成功 {ok} 失败 {fail}, 总耗时 {time.time()-t0:.0f}s")

    # ---- ATM IV 序列与 IV rank
    hist = store.read("risk_indicators")
    und = hist[hist["underlying"] == UNDERLYING].copy()
    if und.empty:
        print("无数据"); return
    # 标的收盘序列（定位每日 ATM 行权价）
    udl = p.underlying_daily(UNDERLYING, min(und["trade_date"]), max(und["trade_date"]))
    close_map = dict(zip(udl["date"], udl["close"].astype(float)))

    rows = []
    for d, g in und.groupby("trade_date"):
        if d not in close_map:
            continue
        S = close_map[d]
        exps = sorted(g["expiry"].unique())
        near = [e for e in exps if (e - d).days >= 7]
        if not near:
            continue
        gg = g[g["expiry"] == near[0]]
        gg = gg[gg["iv"].notna()]
        if gg.empty:
            continue
        atm_k = float(gg["strike"].iloc[int((gg["strike"] - S).abs().values.argmin())])
        row = {"date": d, "expiry": near[0], "atm_strike": atm_k}
        for right, key in (("CALL", "iv_call"), ("PUT", "iv_put")):
            r = gg[(gg["right"] == right) & (gg["strike"] == atm_k)]
            row[key] = float(r["iv"].iloc[0]) if len(r) else float("nan")
        rows.append(row)

    s = pd.DataFrame(rows).dropna(subset=["iv_call", "iv_put"], how="any")
    s["atm_iv"] = (s["iv_call"] + s["iv_put"]) / 2.0
    latest = s.iloc[-1]
    lo, hi = s["atm_iv"].min(), s["atm_iv"].max()
    rank = (latest["atm_iv"] - lo) / (hi - lo) if hi > lo else float("nan")
    pct = (s["atm_iv"] < latest["atm_iv"]).mean()

    print(f"\n=== {UNDERLYING} ATM IV 实测 ===")
    print(f"样本: {len(s)} 个交易日 ({s['date'].iloc[0]} ~ {latest['date']})")
    print(f"最新 ATM IV = {latest['atm_iv']:.4f} (认购 {latest['iv_call']:.4f} / 认沽 {latest['iv_put']:.4f})")
    print(f"区间 [{lo:.4f}, {hi:.4f}]")
    print(f"IV Rank = {rank:.2%} | IV 百分位 = {pct:.2%}")
    out = store.root.parent / "atm_iv_series.csv"
    s.to_csv(out, index=False)
    print(f"序列已保存: {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
