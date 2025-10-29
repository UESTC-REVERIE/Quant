#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import sys
import time
import math
import os
from datetime import datetime, timezone
from typing import List
import pandas as pd
from tqdm import tqdm

OkxMarketAPI = None
try:
    from okx.MarketData import MarketAPI as OkxMarketAPI
except Exception:
    try:
        import okx.MarketData as MarketData
        OkxMarketAPI = MarketData.MarketAPI
    except Exception:
        try:
            from okx import MarketAPI as OkxMarketAPI
        except Exception:
            print("pip install okx tqdm pandas", file=sys.stderr)
            sys.exit(1)

BAR_TO_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1H": 3600, "2H": 7200, "4H": 14400, "6H": 21600, "12H": 43200,
    "1D": 86400, "1W": 604800, "1M": 2592000,
    "6Hutc": 21600, "12Hutc": 43200, "1Dutc": 86400, "1Wutc": 604800, "1Mutc": 2592000,
    "3Mutc": 7776000, "6Mutc": 15552000, "1Yutc": 31536000,
}

CSV_HEADER = [
    "ts", "datetime_utc", "open", "high", "low", "close",
    "vol", "volCcy", "volCcyQuote", "confirm"
]

METHOD_TRY_ORDER = [
    "get_history_candlesticks",
    "get_candlesticks_history",
    "get_history_candles",
    "get_candlesticks",
    "getCandlesticksHistory",
    "getCandlesticks",
]

def parse_human_time(s: str) -> int:
    s = s.strip()
    fmts = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"]
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    dt = pd.to_datetime(s, utc=True)
    return int(dt.value // 1_000_000)

def ts_to_iso_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def estimate_total_rows(start_ms: int, end_ms: int, bar: str) -> int:
    sec = BAR_TO_SECONDS.get(bar)
    if not sec:
        return 0
    total_sec = max(0, (end_ms - start_ms) / 1000)
    return int(math.ceil(total_sec / sec))

def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(num_bytes)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}PB"

def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def read_last_ts_from_csv(path: str) -> int:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return -1
    try:
        with open(path, "rb") as f:
            f.seek(-min(4096, os.path.getsize(path)), os.SEEK_END)
            tail = f.read().decode("utf-8", errors="ignore").strip().splitlines()
            for line in reversed(tail):
                if line.startswith("ts,"):
                    continue
                parts = line.split(",")
                if parts and parts[0].isdigit():
                    return int(parts[0])
    except Exception:
        pass
    try:
        df = pd.read_csv(path, usecols=["ts"])
        if len(df) == 0:
            return -1
        return int(df["ts"].iloc[-1])
    except Exception:
        return -1

def write_rows_csv(path: str, rows: List[List[str]], bytes_counter: List[int]):
    ensure_parent_dir(path)
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(CSV_HEADER)
            bytes_counter[0] += len(",".join(CSV_HEADER)) + 1
        before = f.tell()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
        after = f.tell()
        bytes_counter[0] += (after - before)

def normalize_candle_row(raw: List[str]) -> List[str]:
    data = list(raw) + [""] * (9 - len(raw))
    ts = int(data[0])
    o, h, l, c = data[1], data[2], data[3], data[4]
    vol, volCcy, volCcyQuote, confirm = data[5], data[6], data[7], data[8]
    return [str(ts), ts_to_iso_utc(ts), o, h, l, c, vol, volCcy, volCcyQuote, confirm]

def call_okx(market, name, params):
    fn = getattr(market, name)
    resp = fn(**params)
    if isinstance(resp, dict):
        if str(resp.get("code", "0")) != "0":
            raise RuntimeError(f"OKX {name} {resp.get('code')}: {resp.get('msg')}")
        return resp.get("data", [])
    return resp

def fetch_after(market, inst_id: str, bar: str, cursor_ms: int, limit: int):
    params = {"instId": str(inst_id), "bar": str(bar), "after": str(cursor_ms), "limit": str(min(max(limit, 1), 100))}
    for name in METHOD_TRY_ORDER:
        if hasattr(market, name):
            try:
                data = call_okx(market, name, params)
                if data:
                    return data
            except Exception:
                continue
    return []

def fetch_before(market, inst_id: str, bar: str, cursor_ms: int, limit: int):
    params = {"instId": str(inst_id), "bar": str(bar), "before": str(cursor_ms), "limit": str(min(max(limit, 1), 100))}
    for name in METHOD_TRY_ORDER:
        if hasattr(market, name):
            try:
                data = call_okx(market, name, params)
                if data:
                    return data
            except Exception:
                continue
    return []

def robust_history_pull(market, inst_id: str, bar: str, start_ms: int, end_ms: int, limit: int = 100, sleep_sec: float = 0.25, max_retries: int = 5):
    mode = "after"
    cursor = end_ms
    last_edge = None
    switched = False
    while True:
        if mode == "after" and cursor <= start_ms:
            break
        if mode == "before" and cursor >= end_ms:
            break
        for attempt in range(max_retries):
            try:
                if mode == "after":
                    data = fetch_after(market, inst_id, bar, cursor, limit)
                else:
                    data = fetch_before(market, inst_id, bar, cursor, limit)
                if not data:
                    if not switched:
                        mode = "before"
                        cursor = start_ms
                        switched = True
                        time.sleep(sleep_sec)
                        break
                    return
                data_sorted = sorted(data, key=lambda x: int(x[0]))
                clipped = [normalize_candle_row(x) for x in data_sorted if start_ms <= int(x[0]) <= end_ms]
                if clipped:
                    yield clipped
                if mode == "after":
                    oldest_ts = int(data_sorted[0][0])
                    if last_edge is not None and oldest_ts >= last_edge:
                        return
                    last_edge = oldest_ts
                    cursor = oldest_ts
                    if cursor <= start_ms:
                        return
                else:
                    newest_ts = int(data_sorted[-1][0])
                    if last_edge is not None and newest_ts <= last_edge:
                        return
                    last_edge = newest_ts
                    cursor = newest_ts
                    if cursor >= end_ms:
                        return
                time.sleep(sleep_sec)
                break
            except Exception:
                if attempt + 1 >= max_retries:
                    raise
                time.sleep(1.0 * (attempt + 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst-id", type=str, default="BTC-USDT-SWAP")
    ap.add_argument("--bar", type=str, default="1m")
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--outfile", type=str, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--flag", type=int, default=0, choices=[0, 1])
    args = ap.parse_args()

    if args.bar not in BAR_TO_SECONDS:
        print("Unsupported bar", file=sys.stderr)
        sys.exit(2)

    start_ms = parse_human_time(args.start)
    end_ms = parse_human_time(args.end)
    if end_ms <= start_ms:
        print("Invalid time range", file=sys.stderr)
        sys.exit(2)

    ensure_parent_dir(args.outfile)
    if not os.path.exists(args.outfile) or os.path.getsize(args.outfile) == 0:
        with open(args.outfile, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)

    if args.resume and os.path.exists(args.outfile):
        last_ts = read_last_ts_from_csv(args.outfile)
        if last_ts > 0:
            start_ms = max(start_ms, last_ts + 1)

    total_rows_est = estimate_total_rows(start_ms, end_ms, args.bar)
    total_for_pb = total_rows_est if total_rows_est > 0 else None

    market = OkxMarketAPI(flag=str(args.flag))

    bytes_written = [os.path.getsize(args.outfile)]
    rows_downloaded = 0

    with tqdm(total=total_for_pb, unit="rows", desc=f"{args.inst_id} {args.bar}", dynamic_ncols=True) as pbar:
        try:
            for batch in robust_history_pull(
                market, args.inst_id, args.bar, start_ms, end_ms,
                limit=args.limit, sleep_sec=args.sleep
            ):
                write_rows_csv(args.outfile, batch, bytes_written)
                cnt = len(batch)
                rows_downloaded += cnt
                pbar.update(cnt)
                pbar.set_postfix(file=human_size(bytes_written[0]))
                pbar.refresh()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"\nError during fetch: {e}\nPartial data saved at {args.outfile}")
        finally:
            if os.path.exists(args.outfile) and os.path.getsize(args.outfile) > 0:
                try:
                    df = pd.read_csv(args.outfile)
                    if len(df) > 0:
                        df = df.sort_values("ts", kind="mergesort").drop_duplicates(subset=["ts"], keep="last")
                        df.to_csv(args.outfile, index=False)
                except Exception as e:
                    print(f"Post-process warning: {e}")

    if os.path.exists(args.outfile) and os.path.getsize(args.outfile) > 0 and rows_downloaded > 0:
        print(f"Done. Rows: {rows_downloaded}. Wrote: {args.outfile} ({human_size(os.path.getsize(args.outfile))})")
    else:
        print("No data written")

if __name__ == "__main__":
    main()
