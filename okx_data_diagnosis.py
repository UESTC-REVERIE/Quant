#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
okx_data_diagnosis.py
验证使用 OKX 公共行情 API 的数据获取是否可行，并判断是否满足训练最小样本量。

运行示例：
  python okx_data_diagnosis.py --inst-id BTC-USDT --start 2018-01-01 --bar 1D \
      --seq-len 60 --max-h 15 --min-train-windows 100

可选代理：
  python okx_data_diagnosis.py --proxy http://127.0.0.1:7897

依赖：
  pip install requests certifi pandas numpy
"""

import sys
import time, math, json
import argparse
from typing import Optional, Dict, Any, Tuple, List

import requests
import certifi
import numpy as np
import pandas as pd
from tqdm import tqdm
from okx import MarketData

BAR_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1H": 3_600_000, "2H": 7_200_000, "4H": 14_400_000, "6H": 21_600_000, "12H": 43_200_000,
    "1D": 86_400_000, "1W": 604_800_000,
}

def _rows_from_payload(arr: List[List[str]]) -> List[Tuple[int, float, float, float, float, float]]:
    # OKX返回: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    rows = []
    for r in arr:
        ts = int(r[0])
        o, h, l, c = map(float, r[1:5])
        vol = float(r[5])
        rows.append((ts, o, h, l, c, vol))
    return rows

def fetch_okx_candles(
    inst_id: str,
    start_date: str,
    bar: str = "1D",
    proxy_url: Optional[str] = None,  # SDK 不直接提供代理参数；多数场景下系统代理即可生效
    timeout: int = 20,                # SDK内部有默认超时，这里参数保留占位
    per_page: int = 100,
    max_loops: int = 5000,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    使用 python-okx (MarketData.MarketAPI) 抓取从 start_date 到当前的K线。
    - 先用 get_candlesticks 拿最近页，取其中“最旧一根”时间戳，作为 after 游标；
    - 再用 get_history_candlesticks + after 向更旧方向分页，直到到达/早于 start_date。
    - 进度条显示：本页近似字节、瞬时/平均网速（近似）、EMA每页时长、ETA。

    返回：DataFrame(index=Date)[Open, High, Low, Close, Volume]
    """
    if bar not in BAR_MS:
        raise ValueError(f"Unsupported bar '{bar}'. Supported: {sorted(BAR_MS.keys())}")
    bar_ms = BAR_MS[bar]

    # OKX SDK：0=实盘，1=模拟盘
    md = MarketData.MarketAPI(flag="0")

    start_ms = int(pd.to_datetime(start_date).tz_localize("UTC").timestamp() * 1000)
    rows: List[Tuple[int, float, float, float, float, float]] = []

    # 统计量（速度/ETA）
    total_pages = 0
    total_bytes = 0  # 近似：用json序列化后的长度
    t0 = time.time()
    ema_time = None  # 每页EMA耗时

    # ---------- 先取最近一页 ----------
    data = md.get_candlesticks(instId=inst_id, bar=bar, limit=str(per_page))
    arr = data.get("data", []) or []
    if not arr:
        return pd.DataFrame()
    rows.extend(_rows_from_payload(arr))
    size_bytes = len(json.dumps(data, ensure_ascii=False).encode("utf-8"))  # 近似“本页字节数”
    total_bytes += size_bytes
    total_pages += 1

    oldest_ts = int(arr[-1][0])  # 此页“最旧”的一根在最后
    # 估计剩余页数（用时间戳差 / (bar*每页数量)）
    remaining_bars = max(0, (oldest_ts - start_ms) // bar_ms)
    estimated_pages_left = math.ceil(remaining_bars / per_page)

    pbar = None
    if show_progress:
        pbar = tqdm(total=1 + estimated_pages_left, desc=f"Downloading {inst_id} ({bar})", unit="page")
        pbar.update(1)

    # ---------- 向更旧翻页 ----------
    for i in range(max_loops):
        if oldest_ts <= start_ms:
            if pbar: pbar.write("✅ 到达起始日期。")
            break

        t_start = time.time()
        # 注意：history 接口 + after=oldest_ts => 返回“更旧”的K线
        data = md.get_history_candlesticks(instId=inst_id, bar=bar, limit=str(per_page), after=str(oldest_ts))
        arr = data.get("data", []) or []
        elapsed = time.time() - t_start

        if not arr:
            if pbar: pbar.write("⚠️ 无更多数据，提前结束。")
            break

        # 吸收数据
        rows.extend(_rows_from_payload(arr))
        oldest_ts = int(arr[-1][0])

        if pbar:
            pbar.update(1)

        # 轻限速，避免 50011（Rate limit）
        # if elapsed < 0.12:
        #     time.sleep(0.12 - elapsed)

    if pbar:
        pbar.close()

    if not rows:
        return pd.DataFrame()

    # ---------- 转 DataFrame ----------
    df = (
        pd.DataFrame(rows, columns=["ts","Open","High","Low","Close","Volume"])
        .drop_duplicates("ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )
    df = df[df["ts"] >= start_ms]
    if df.empty:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.set_index("Date")
    # 汇总打印
    total_time = time.time() - t0
    print(
        f"\n✅ 完成！共抓取 {len(df)} 根K线。"
        f"\n⏱️ 总用时: {total_time:.1f}s, 近似下载: {total_bytes/1024:.1f}KB, 平均速度: {(total_bytes/1024)/max(total_time,1e-6):.1f}KB/s"
    )
    return df[["Open","High","Low","Close","Volume"]]
# -----------------------------
# 体检：原始 OHLCV 的基本合理性
# -----------------------------
def ohlc_sanity(df: pd.DataFrame) -> Dict[str, Any]:
    bad_open = (df["Open"] > df["High"]) | (df["Open"] < df["Low"])
    bad_close = (df["Close"] > df["High"]) | (df["Close"] < df["Low"])
    nonpos_price = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    neg_vol = (df["Volume"] < 0)

    idx_is_dt = isinstance(df.index, pd.DatetimeIndex)
    idx_dup = int(df.index.duplicated().sum()) if idx_is_dt else None
    idx_increasing = bool(df.index.is_monotonic_increasing) if idx_is_dt else None

    return {
        "rows": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "is_datetime_index": idx_is_dt,
        "duplicated_index": idx_dup,
        "is_monotonic_increasing": idx_increasing,
        "na_counts": df.isna().sum().to_dict(),
        "rows_bad_open": int(bad_open.sum()),
        "rows_bad_close": int(bad_close.sum()),
        "rows_nonpos_price": int(nonpos_price.sum()),
        "rows_neg_volume": int(neg_vol.sum()),
        "zero_volume_ratio": float((df["Volume"] == 0).mean()),
    }


# -----------------------------
# 特征工程（不依赖 ta）：MA/波动率/RSI/OBV
# -----------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy()
    feats["daily_return"] = feats["Close"].pct_change()
    feats["MA5"] = feats["Close"].rolling(5).mean()
    feats["MA10"] = feats["Close"].rolling(10).mean()
    feats["MA20"] = feats["Close"].rolling(20).mean()
    feats["volatility"] = feats["daily_return"].rolling(10).std()

    # RSI(14) - Wilder
    delta = feats["Close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=feats.index).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    roll_down = pd.Series(down, index=feats.index).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    feats["RSI"] = 100 - (100 / (1 + rs))

    # OBV
    sign = np.sign(feats["Close"].diff().fillna(0.0))
    feats["OBV"] = (sign * feats["Volume"]).fillna(0.0).cumsum()

    feats = feats.dropna().copy()
    return feats


# -----------------------------
# 训练可行性判定
# -----------------------------
def check_trainability(
    feats_len: int, seq_len: int, max_h: int, min_train_windows: int
) -> Tuple[bool, Dict[str, Any]]:
    """
    len(X) = feats_len - seq_len - max_h
    需要 len(X) >= min_train_windows
    """
    need_rows = min_train_windows + seq_len + max_h
    len_x = max(0, feats_len - seq_len - max_h)
    trainable = feats_len >= need_rows
    detail = {
        "feats_len": feats_len,
        "seq_len": seq_len,
        "max_h": max_h,
        "min_train_windows": min_train_windows,
        "need_rows_at_least": need_rows,
        "len_X_if_train": len_x,
        "trainable": trainable,
    }
    return trainable, detail


# -----------------------------
# 主流程
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Diagnose OKX data acquisition & trainability.")
    parser.add_argument("--inst-id", type=str, default="BTC-USDT", help="交易对，比如 BTC-USDT / ETH-USDT")
    parser.add_argument("--start", type=str, default="2018-01-01", help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--bar", type=str, default="1D", help="K线周期，例如 1D / 4H / 1H / 1m ...")
    parser.add_argument("--proxy", type=str, default=None, help="HTTP(S) 代理，如 http://127.0.0.1:7897")
    parser.add_argument("--seq-len", type=int, default=60, help="输入窗口长度")
    parser.add_argument("--max-h", type=int, default=15, help="预测步数")
    parser.add_argument("--min-train-windows", type=int, default=100, help="训练至少需要的样本窗口数")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP 超时秒")
    parser.add_argument("--max-loops", type=int, default=2000, help="向前翻页最大次数（保障能翻到较早历史）")
    args = parser.parse_args()

    print("=== 参数 ===")
    print(vars(args))

    # 1) 抓取
    try:
        raw = fetch_okx_candles(
            inst_id=args.inst_id,
            start_date=args.start,
            bar=args.bar
        )
    except Exception as e:
        print(f"\n[ERROR] 获取数据失败：{e}")
        print("建议：检查网络/代理；确保交易对存在；必要时降低 bar（改 1H/1m）或更改起始日期。")
        sys.exit(2)

    if raw is None or raw.empty:
        print("\n[ERROR] 获取到的 DataFrame 为空。可能原因：交易对不存在、历史过短、网络/代理限制。")
        sys.exit(2)

    # 2) 原始数据体检
    print("\n=== 原始数据体检 ===")
    s = ohlc_sanity(raw)
    for k, v in s.items():
        print(f"{k}: {v}")

    ok_raw = True
    reasons = []
    if not s["is_datetime_index"] or not s["is_monotonic_increasing"]:
        ok_raw = False
        reasons.append("索引不是递增 DatetimeIndex")
    if s["duplicated_index"] and s["duplicated_index"] > 0:
        ok_raw = False
        reasons.append(f"存在重复时间戳({s['duplicated_index']})")
    if s["rows"] < 50:
        ok_raw = False
        reasons.append("样本行数过少(<50)")
    if s["rows_bad_open"] > 0 or s["rows_bad_close"] > 0:
        ok_raw = False
        reasons.append("存在 Open/Close 超出 High/Low")
    if s["rows_nonpos_price"] > 0:
        ok_raw = False
        reasons.append("存在非正价格")
    if s["rows_neg_volume"] > 0:
        ok_raw = False
        reasons.append("存在负成交量")

    if not ok_raw:
        print("\n[ERROR] 原始数据存在异常：", "; ".join(reasons))
        sys.exit(3)
    else:
        print("\n[OK] 原始 OHLCV 数据通过基础体检。")

    # 3) 特征工程 + 判定可训练性
    feats = build_features(raw)
    print("\n=== 特征工程 ===")
    print("feats rows:", len(feats))
    na_counts = feats.isna().sum().to_dict()
    print("feats NA counts:", na_counts)

    if feats.empty or feats.isna().any().any():
        print("\n[ERROR] 特征数据为空或含缺失值（请检查滚动窗口/RSI等计算）。")
        sys.exit(3)

    trainable, detail = check_trainability(
        feats_len=len(feats),
        seq_len=args.seq_len,
        max_h=args.max_h,
        min_train_windows=args.min_train_windows,
    )
    print("\n=== 训练可行性 ===")
    for k, v in detail.items():
        print(f"{k}: {v}")

    if trainable:
        print("\n✅ 结论：数据完整、合理且数量充足，可以进入训练阶段。")
        sys.exit(0)
    else:
        print("\n⚠️ 结论：当前样本不足以训练。")
        print("建议：")
        print("  1) 减少 --seq-len 或 --max-h；")
        print("  2) 调低 --min-train-windows；")
        print("  3) 改用更细周期（如 --bar 1H / 1m）；")
        print("  4) 确保使用了 history-candles 并成功翻到足够早的历史；")
        print("  5) 尝试更早的 --start 或换交易对。")
        sys.exit(4)


if __name__ == "__main__":
    main()
