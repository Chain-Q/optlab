"""
optlab.scripts.collect_daily — 每日收盘后统一采集入口（设计文档 §5.2 日频批处理）

用途：server 晚间调度自动调用，或手动运行（即「每日更新.bat」）。做五件事：
    1. 当日 risk_indicators（IV/Greeks，全市场一次调用）
    2. 标的日线增量（510300）
    3. 逐合约 OI/量 快照（自建历史 OI 库的唯一来源——sina/交易所均无历史逐合约 OI）
    4. 重算 ATM IV 序列（追加）
    5. 重建 dashboard.html
运行：python -m optlab.scripts.collect_daily [date=今天]
"""
import sys
import time
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from optlab.data.provider import ParquetStore, SseOptionProvider
from optlab.scripts.collect_risk_history import UNDERLYING


def main(day: date = None):
    day = day or date.today()
    t0 = time.time()
    store = ParquetStore("optlab_data/store")
    p = SseOptionProvider(min_interval=0.3)
    log = []

    # 1) 风险指标（收盘后交易所发布有延迟：重试 3 次每次隔 60s；实测发布时点约 19:30~21:00+，更稳的是让 server 晚间调度盯发布）
    df = pd.DataFrame()
    for attempt in range(3):
        try:
            df = p.risk_indicators(day)
            if not df.empty:
                break
        except Exception:
            pass
        if attempt < 2:
            log.append(f"risk_indicators 第{attempt+1}次无数据，60s 后重试（交易所发布延迟约 19:30~21:00+）")
            time.sleep(60)
    try:
        if df.empty:
            log.append(f"risk_indicators: {day} 数据未发布（非交易日或发布延迟），跳过")
        else:
            n = store.write("risk_indicators", df)
            log.append(f"risk_indicators: +{n} 行")
    except Exception as e:
        log.append(f"risk_indicators FAIL: {type(e).__name__} {str(e)[:50]}")

    # 2) 标的日线（全品种增量）
    try:
        daily_all = pd.read_parquet(store.root / "contract_daily" / "all.parquet")
        d_min = daily_all["trade_date"].min()
        ud_path = store.root / "underlying_daily" / "all.parquet"
        ud_old = pd.read_parquet(ud_path) if ud_path.exists() else pd.DataFrame()
        frames = [ud_old] if len(ud_old) else []
        for und in ("510300", "510050", "510500", "588000", "588080", "159915"):
            try:
                u = p.underlying_daily(und, d_min, day)
                u["underlying"] = und
                frames.append(u)
            except Exception as e:
                log.append(f"标的日线 {und} FAIL: {type(e).__name__}")
        ud = pd.concat(frames, ignore_index=True)             .drop_duplicates(subset=["underlying", "date"], keep="last")
        ud.to_parquet(ud_path, index=False)
        log.append(f"underlying_daily: {len(ud)} 行 / {ud['underlying'].nunique()} 品种")
    except Exception as e:
        log.append(f"underlying_daily FAIL: {type(e).__name__} {str(e)[:60]}")

    # 2b) 深市静态快照（159915 等：前结算/OI/涨跌停/合约调整）
    try:
        import akshare as _ak
        df = _ak.option_current_day_szse()
        ren = {"合约代码": "contract_id", "行权价": "strike", "合约单位": "multiplier",
               "前结算价": "pre_settle", "合约总持仓": "oi", "涨停价格": "limit_up",
               "跌停价格": "limit_down", "到期日": "expiry", "合约类型": "right_cn",
               "合约调整": "adjusted", "标的证券简称(代码)": "und_name"}
        g = df.rename(columns=ren)
        g["trade_date"] = g["交易日期"]
        g["underlying"] = g["contract_id"].astype(str).str[:6]
        g["adjusted"] = g["adjusted"].astype(str) == "是"
        g["right"] = g["right_cn"].map(lambda x: "CALL" if "购" in str(x) else "PUT")
        g["security_id"] = g["合约编码"]
        out = store.root / "snapshots_szse"
        out.mkdir(parents=True, exist_ok=True)
        snap_day = str(g["trade_date"].iloc[0]).replace("-", "")
        f = out / f"{snap_day[:6]}.parquet"
        g.to_parquet(f, index=False)
        log.append(f"深市快照: {len(g)} 行（{snap_day}）")
    except Exception as e:
        log.append(f"深市快照 FAIL: {type(e).__name__} {str(e)[:60]}")

    # 3) 逐合约 OI/量 快照（当日全合约，自建 OI 库）
    try:
        try:
            risks = p.risk_indicators(day)
        except Exception:
            risks = pd.DataFrame()   # 非交易日：交易所接口返回空表头 → KeyError
        if risks.empty:
            log.append(f"OI 快照跳过：{day} 非交易日（无行情）")
        else:
            rows = []
            for sid in risks["security_id"].unique():
                s_ = p.contract_spot(sid)
                s_["trade_date"] = day
                rows.append(s_)
            oi_df = pd.DataFrame(rows)
            out = store.root / "oi_snapshots"
            out.mkdir(parents=True, exist_ok=True)
            f = out / f"{day:%Y%m}.parquet"
            if f.exists():
                old = pd.read_parquet(f)
                oi_df = pd.concat([old, oi_df], ignore_index=True) \
                    .drop_duplicates(subset=["security_id", "trade_date"], keep="last")
            oi_df.to_parquet(f, index=False)
            log.append(f"OI 快照: {len(oi_df)} 行（含持仓量，累计建库）")
    except Exception as e:
        log.append(f"OI 快照 FAIL: {type(e).__name__} {str(e)[:60]}")

    # 4) ATM IV 序列重建
    try:
        from optlab.scripts.collect_risk_history import main as _unused  # noqa
        hist = store.read("risk_indicators")
        und = hist[hist["underlying"] == UNDERLYING].copy()
        udl = pd.read_parquet(store.root / "underlying_daily" / "all.parquet")
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
            gg = g[(g["expiry"] == near[0]) & g["iv"].notna()]
            if gg.empty:
                continue
            k = gg["strike"].iloc[int((gg["strike"] - S).abs().values.argmin())]
            row = {"date": d, "expiry": near[0], "atm_strike": float(k)}
            for right, key in (("CALL", "iv_call"), ("PUT", "iv_put")):
                r2 = gg[(gg["right"] == right) & (gg["strike"] == k)]
                row[key] = float(r2["iv"].iloc[0]) if len(r2) else float("nan")
            rows.append(row)
        s2 = pd.DataFrame(rows).dropna(subset=["iv_call", "iv_put"], how="any")
        s2["atm_iv"] = (s2["iv_call"] + s2["iv_put"]) / 2.0
        s2.to_csv(store.root.parent / "atm_iv_series.csv", index=False)
        log.append(f"ATM IV 序列: {len(s2)} 天")
    except Exception as e:
        log.append(f"ATM IV FAIL: {type(e).__name__} {str(e)[:60]}")

    for x in log:
        print(f"  [{x}]")
    print(f"耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else None)
