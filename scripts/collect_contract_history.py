"""
thetalab.scripts.collect_contract_history — 逐合约历史日线批量采集（回测成交价来源）

数据源：ak.option_sse_daily_sina(security_id)，成交量单位=份额 →÷10000=张（已在 provider 换算）
存储：thetalab_data/store/contract_daily/all.parquet（多品种合并，含 contract_id 主键，断点续跑）
附带：采集完成后自动补该品种标的日线到 underlying_daily 表
运行：python -m thetalab.scripts.collect_contract_history [underlying=510300]
      underlying=all 采集全部五个沪市品种
"""
import sys
import time
from datetime import date
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from thetalab.data.provider import ParquetStore, SseOptionProvider

UNDERLYINGS = ["510300", "510050", "510500", "588000", "588080"]


def build_contract_id_map(risk_all: pd.DataFrame) -> pd.DataFrame:
    """(trade_date, security_id) → contract_id 日级映射 + security_id 全局众数兜底"""
    m_day = risk_all.drop_duplicates(["trade_date", "security_id"]).set_index(
        ["trade_date", "security_id"])["contract_id"]
    m_glob = risk_all.groupby("security_id")["contract_id"].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else None)
    return m_day, m_glob


def main(underlying: str = "510300"):
    t0 = time.time()
    p = SseOptionProvider(min_interval=0.3)
    store = ParquetStore("thetalab_data/store")
    targets = UNDERLYINGS if underlying == "all" else [underlying]

    risk_all = store.read("risk_indicators")
    m_day, m_glob = build_contract_id_map(risk_all)

    # 合约宇宙：risk_indicators 中该品种出现过的全部合约
    sids_all = sorted(risk_all[risk_all["underlying"].isin(targets)]
                      .groupby("security_id").head(1)
                      .loc[:, ["security_id", "underlying"]]
                      .itertuples(index=False, name=None))
    print(f"目标品种 {targets} 合约宇宙: {len(sids_all)} 个")

    out_path = store.root / "contract_daily" / "all.parquet"
    have, frames = set(), []
    if out_path.exists():
        old = pd.read_parquet(out_path)
        have = set(old.loc[old["security_id"].isin(
            [s for s, _ in sids_all]), "security_id"].unique())
        frames.append(old)
    todo = [(s, u) for s, u in sids_all if s not in have]
    print(f"已采集 {len(have)}，待采集 {len(todo)}")

    ok = empty = fail = 0
    new_frames = []
    for i, (sid, und) in enumerate(todo):
        try:
            df = p.contract_daily(sid)
            if df.empty:
                empty += 1
            else:
                df["security_id"] = sid
                df["underlying"] = und
                new_frames.append(df)
                ok += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  [{sid}] FAIL {type(e).__name__}: {str(e)[:80]}")
        if (i + 1) % 50 == 0:
            print(f"  进度 {i+1}/{len(todo)} ok={ok} empty={empty} fail={fail} "
                  f"耗时{time.time()-t0:.0f}s")

    if new_frames:
        new_df = pd.concat(new_frames, ignore_index=True)
        new_df["trade_date"] = new_df["date"]   # provider 列名=date，落库统一 trade_date
        key = list(zip(new_df["trade_date"], new_df["security_id"]))
        cid = pd.Series([m_day.get(k) for k in key]) \
            .fillna(new_df["security_id"].map(m_glob))
        new_df["contract_id"] = cid.fillna(new_df["security_id"])
        frames.append(new_df)
        print(f"新增 {len(new_df)} 行（contract_id 覆盖 "
              f"{(new_df['contract_id'] != new_df['security_id']).mean():.0%}）")

    if frames:
        all_df = pd.concat(frames, ignore_index=True) \
            .drop_duplicates(subset=["trade_date", "contract_id"], keep="last")
        all_df["trade_date"] = all_df["trade_date"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        all_df.to_parquet(out_path, index=False)
        print(f"落库 {len(all_df)} 行 → {out_path}")

    # ---- 各品种标的日线（合并写入）
    ud_path = store.root / "underlying_daily" / "all.parquet"
    ud_old = pd.read_parquet(ud_path) if ud_path.exists() else pd.DataFrame(
        columns=["date", "open", "high", "low", "close", "volume", "underlying"])
    ud_frames = [ud_old]
    for und in targets:
        d_min = min((df_["date"].min() for df_ in frames if len(frames)), default=None)
        try:
            u = p.underlying_daily(und, pd.Timestamp(d_min).date()
                                   if d_min else pd.Timestamp("2025-01-01").date(),
                                   date.today())
            u["underlying"] = und
            ud_frames.append(u)
            print(f"标的日线 {und}: {len(u)} 行")
        except Exception as e:
            print(f"标的日线 {und} FAIL: {type(e).__name__} {str(e)[:80]}")
    ud_all = pd.concat(ud_frames, ignore_index=True) \
        .drop_duplicates(subset=["underlying", "date"], keep="last")
    ud_path.parent.mkdir(parents=True, exist_ok=True)
    ud_all.to_parquet(ud_path, index=False)

    print(f"完成: ok={ok} empty={empty} fail={fail}, 耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "510300")
