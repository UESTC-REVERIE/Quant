# app.py
# -*- coding: utf-8 -*-
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import ta
import requests
import certifi
from datetime import datetime, timedelta, timezone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------
# 全局设置
# ----------------------------
st.set_page_config(page_title="加密货币多模型预测 (OKX+LSTM/GRU/Transformer/TCN)", layout="wide")
plt.rcParams.update({"figure.autolayout": True})

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 侧边栏参数
# ----------------------------
st.sidebar.header("配置参数")
st.sidebar.caption("数据源：OKX v5 公共行情 API（免 Key）")
inst_id = st.sidebar.text_input("交易对（例如：BTC-USDT / ETH-USDT / SOL-USDT）", value="BTC-USDT")
start_date = st.sidebar.date_input("起始日期", value=pd.to_datetime("2018-01-01"))
bar = st.sidebar.selectbox("K线周期", options=["1D", "4H", "1H"], index=0)  # OKX 支持的常用周期
use_proxy = st.sidebar.checkbox("使用 HTTP 代理（可选）", value=False)
proxy_url = st.sidebar.text_input("代理地址（示例：http://127.0.0.1:7897）", value="http://127.0.0.1:7897") if use_proxy else None

seq_len = st.sidebar.number_input("输入窗口长度（bar 数）", min_value=20, max_value=240, value=60, step=5)
max_h = st.sidebar.selectbox("预测步数（输出长度）", options=[5, 10, 15], index=2)
epochs = st.sidebar.slider("训练轮次", min_value=10, max_value=200, value=60, step=10)
batch_size = st.sidebar.selectbox("Batch Size", options=[32, 64, 128], index=1)
hidden_size = st.sidebar.selectbox("隐藏层维度（RNN/TCN/Transformer）", options=[64, 128, 192], index=0)
nhead = st.sidebar.selectbox("Transformer 多头数", options=[2, 4, 8], index=1)
dropout = st.sidebar.slider("Dropout", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
train_ratio = st.sidebar.slider("训练集比例", min_value=0.5, max_value=0.95, value=0.8, step=0.05)
horizons_eval = st.sidebar.multiselect("评估周期（可多选）", [1, 3, 5, 10, 15], default=[1, 3, 5, 10, 15])

models_to_run = st.sidebar.multiselect(
    "选择模型（至少一个）",
    ["LSTM", "GRU", "Transformer", "TCN"],
    default=["LSTM", "GRU", "Transformer", "TCN"],
)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 开始预测")

# ----------------------------
# 工具 & 数据抓取（OKX API）
# ----------------------------
def _safe_series(x):
    return pd.Series(np.array(x).squeeze(), index=x.index if hasattr(x, "index") else None)

@st.cache_data(show_spinner=True)
def fetch_okx_candles(inst_id: str, start_date: pd.Timestamp, bar: str = "1D", proxy_url: str | None = None) -> pd.DataFrame:
    import requests, certifi, time
    import pandas as pd
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    session.verify = certifi.where()
    if proxy_url:
        session.proxies.update({"http": proxy_url, "https": proxy_url})

    base = "https://www.okx.com"
    path_hist = "/api/v5/market/history-candles"  # ✅ 历史端点
    path_curr = "/api/v5/market/candles"          # 近期端点（可选）
    per_page = 100
    max_loops = 2000  # 足够大
    start_ms = int(pd.to_datetime(start_date).tz_localize("UTC").timestamp() * 1000)

    def fetch_page(path, params):
        r = session.get(base + path, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        if js.get("code") != "0":
            raise RuntimeError(js)
        return js.get("data", [])

    candles = []
    before = None

    # 1) 先用 history-candles 不断向“更早”翻页
    for _ in range(max_loops):
        params = {"instId": inst_id, "bar": bar, "limit": str(per_page)}
        if before is not None:
            params["before"] = str(before)
        arr = fetch_page(path_hist, params)
        if not arr:
            break
        for row in arr:
            ts = int(row[0])
            o, h, l, c = map(float, row[1:5])
            vol = float(row[5])
            candles.append((ts, o, h, l, c, vol))
        earliest_ts = int(arr[-1][0])
        if earliest_ts <= start_ms:  # 到达或早于起始时间就够了
            break
        before = earliest_ts - 1
        time.sleep(0.1)

    # （可选）2) 再用 /candles 补最近一段，防止 history 有些交易对最近页延迟
    params = {"instId": inst_id, "bar": bar, "limit": str(per_page)}
    arr = fetch_page(path_curr, params)
    for row in arr:
        ts = int(row[0])
        o, h, l, c = map(float, row[1:5])
        vol = float(row[5])
        candles.append((ts, o, h, l, c, vol))

    if not candles:
        return pd.DataFrame()

    # 去重 & 正序 & 过滤
    df = pd.DataFrame(candles, columns=["ts", "Open", "High", "Low", "Close", "Volume"]).drop_duplicates("ts")
    df = df.sort_values("ts").reset_index(drop=True)
    df = df[df["ts"] >= start_ms]
    df["Date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.set_index("Date")
    return df[["Open", "High", "Low", "Close", "Volume"]]


@st.cache_data(show_spinner=True)
def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["daily_return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["volatility"] = df["daily_return"].rolling(window=10).std()

    close = _safe_series(df["Close"])
    volume = _safe_series(df["Volume"])
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df = df.dropna().copy()
    return df

@st.cache_data(show_spinner=True)
def scale_data(df: pd.DataFrame):
    features = df.drop(columns=["Close"])
    target = df[["Close"]]
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target)
    scaled_data = np.concatenate([scaled_features, scaled_target], axis=1)
    return scaled_data, feature_scaler, target_scaler

def create_sequences_multiday(scaled_data: np.ndarray, seq_len: int, max_h: int):
    xs, ys, bases = [], [], []
    for i in range(len(scaled_data) - seq_len - max_h):
        x = scaled_data[i:i + seq_len]
        base = scaled_data[i + seq_len - 1, -1]
        future = [scaled_data[i + seq_len + h, -1] for h in range(1, max_h + 1)]
        xs.append(x); ys.append(future); bases.append(base)
    return np.array(xs), np.array(ys), np.array(bases)

class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# ----------------------------
# 模型定义
# ----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=15, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers; self.hidden_size = hidden_size
    def forward(self, x):
        B = x.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=15, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers; self.hidden_size = hidden_size
    def forward(self, x):
        B = x.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, num_layers=2, nhead=4, hidden_dim=128, output_size=15, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        self.input_proj = nn.Linear(input_size, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_size)
    def _positional_encoding(self, L, D, device):
        pos = torch.arange(0, L, device=device).unsqueeze(1)
        i = torch.arange(0, D, 2, device=device)
        angle = 1.0 / torch.pow(10000, (i.float() / D))
        pe = torch.zeros(L, D, device=device)
        pe[:, 0::2] = torch.sin(pos * angle); pe[:, 1::2] = torch.cos(pos * angle)
        return pe.unsqueeze(0)
    def forward(self, x):
        B, T, _ = x.shape
        z = self.input_proj(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=x.device))
        z = z + self._positional_encoding(T, self.d_model, x.device)
        out = self.encoder(z)[:, -1, :]
        return self.fc(out)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, ksz, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, ksz, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding); self.relu1 = nn.ReLU(); self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, ksz, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding); self.relu2 = nn.ReLU(); self.drop2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.drop1,
                                 self.conv2, self.chomp2, self.relu2, self.drop2)
        self.down = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        for m in self.net:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    def forward(self, x):
        out = self.net(x); res = x if self.down is None else self.down(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size=15, num_channels=[64, 64, 128], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i - 1]
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_ch, ch, kernel_size, 1, dilation, padding, dropout))
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
    def forward(self, x):
        z = x.transpose(1, 2)
        y = self.net(z)
        y = y[:, :, -1]
        return self.fc(y)

# ----------------------------
# 训练与评估
# ----------------------------
def inv_transform_each_day(arr_2d, scaler: MinMaxScaler):
    out = np.zeros_like(arr_2d, dtype=float)
    for k in range(arr_2d.shape[1]):
        out[:, k] = scaler.inverse_transform(arr_2d[:, [k]]).ravel()
    return out

def trend_accuracy_horizon(true_seq, pred_seq, base_price, horizon):
    true_dir = np.sign(true_seq[:, :horizon] - base_price[:, None])
    pred_dir = np.sign(pred_seq[:, :horizon] - base_price[:, None])
    return (true_dir == pred_dir).mean()

@st.cache_resource(show_spinner=False)
def build_model(name, input_size, output_size, hidden_size, nhead, dropout):
    if name == "LSTM":
        return LSTMModel(input_size, hidden_size=hidden_size, num_layers=2, output_size=output_size, dropout=dropout).to(device)
    if name == "GRU":
        return GRUModel(input_size, hidden_size=hidden_size, num_layers=2, output_size=output_size, dropout=dropout).to(device)
    if name == "Transformer":
        return TimeSeriesTransformer(input_size, num_layers=2, nhead=nhead, hidden_dim=hidden_size, output_size=output_size, dropout=dropout).to(device)
    if name == "TCN":
        return TCN(input_size, output_size=output_size, num_channels=[hidden_size, hidden_size, hidden_size * 2], kernel_size=3, dropout=dropout).to(device)
    raise ValueError("unknown model")

def train_and_evaluate(model_name, model, X_train, Y_train, X_test, Y_test, base_test, target_scaler, epochs, batch_size):
    start = time.time()
    loader = DataLoader(SeqDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train(); losses = []
    for ep in range(epochs):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        losses.append(total / max(1, len(loader)))
    train_time = time.time() - start

    model.eval()
    with torch.no_grad():
        Xtt = torch.tensor(X_test, dtype=torch.float32, device=device)
        preds_scaled = model(Xtt).cpu().numpy()

    preds = inv_transform_each_day(preds_scaled, target_scaler)
    true = inv_transform_each_day(Y_test, target_scaler)
    base_true = target_scaler.inverse_transform(base_test.reshape(-1, 1)).ravel()

    rows = []
    for H in horizons_eval:
        t = true[:, :H].ravel()
        p = preds[:, :H].ravel()
        rmse = math.sqrt(mean_squared_error(t, p))
        mae = mean_absolute_error(t, p)
        r2 = r2_score(t, p) if len(np.unique(t)) > 1 else np.nan
        acc = trend_accuracy_horizon(true, preds, base_true, H)
        rows.append([model_name, H, rmse, mae, r2, acc, train_time])
    return rows, preds, true, losses

# ----------------------------
# 主体流程
# ----------------------------
st.title("🪙 加密货币多模型预测（OKX / LSTM / GRU / Transformer / TCN）")
st.caption("输入交易对与参数，自动完成数据抓取、训练、预测、对比与建议。（仅用于研究与教学，不构成投资建议）")

if run_button:
    with st.spinner("正在从 OKX 获取数据并构建特征..."):
        raw = fetch_okx_candles(inst_id, start_date, bar, proxy_url if use_proxy else None)
        if raw.empty:
            st.error("获取数据失败，请检查交易对是否存在或网络/代理设置。示例：BTC-USDT、ETH-USDT、SOL-USDT")
            st.stop()
        feats = build_features(raw)
        scaled_data, feature_scaler, target_scaler = scale_data(feats)
        X, Y, base_close = create_sequences_multiday(scaled_data, seq_len, max_h)
        st.write(
            "raw rows:", len(raw),
            "feats rows:", len(feats),
            "scaled rows:", len(scaled_data),
            "need at least:", 100 + seq_len + max_h,
            "=> len(X):", len(X)
        )
        if len(X) < 100:
            st.warning("样本太少，无法训练。请缩短起始日期或减少 seq_len / max_h。")
            st.stop()

        split = int(len(X) * train_ratio)
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]
        base_train, base_test = base_close[:split], base_close[split:]

    st.success(f"数据准备完成：训练样本 {len(X_train)}，测试样本 {len(X_test)}。（数据源：OKX {bar} K线）")

    # 历史价格
    with st.expander("查看历史价格曲线", expanded=False):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(feats.index, feats["Close"])
        ax.set_title(f"{inst_id} 历史收盘价（{bar}）")
        ax.set_xlabel("Date"); ax.set_ylabel("Close")
        st.pyplot(fig)

    all_results = []; model_preds = {}
    cols = st.columns(len(models_to_run))
    for i, name in enumerate(models_to_run):
        with cols[i]:
            st.markdown(f"### {name}")
            model = build_model(
                name,
                input_size=scaled_data.shape[1],
                output_size=max_h,
                hidden_size=hidden_size,
                nhead=nhead,
                dropout=dropout,
            )
            rows, preds, true, losses = train_and_evaluate(
                name, model,
                X_train, Y_train, X_test, Y_test,
                base_test, target_scaler,
                epochs, batch_size
            )
            all_results += rows
            model_preds[name] = (preds, true, losses)

            fig_l, ax_l = plt.subplots(figsize=(4, 2.2))
            ax_l.plot(losses); ax_l.set_title("训练损失")
            ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("MSE")
            st.pyplot(fig_l)

    result_df = pd.DataFrame(
        all_results, columns=["Model", "Horizon", "RMSE", "MAE", "R2", "TrendAcc", "TrainTime(s)"]
    )
    st.subheader("📊 指标对比")
    st.dataframe(
        result_df.style.format({
            "RMSE": "{:.4f}", "MAE": "{:.4f}", "R2": "{:.4f}",
            "TrendAcc": "{:.4f}", "TrainTime(s)": "{:.2f}",
        }),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        for m in result_df["Model"].unique():
            dfm = result_df[result_df["Model"] == m]
            ax1.plot(dfm["Horizon"], dfm["TrendAcc"], marker="o", label=m)
        ax1.set_title("各模型趋势准确率对比"); ax1.set_xlabel("Horizon (bars)"); ax1.set_ylabel("TrendAcc")
        ax1.legend(); st.pyplot(fig1)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        for m in result_df["Model"].unique():
            dfm = result_df[result_df["Model"] == m]
            ax2.plot(dfm["Horizon"], dfm["RMSE"], marker="o", label=m)
        ax2.set_title("各模型 RMSE 对比"); ax2.set_xlabel("Horizon (bars)"); ax2.set_ylabel("RMSE")
        ax2.legend(); st.pyplot(fig2)

    st.subheader("🔭 测试集样本未来路径对比（随机示例）")
    if len(X_test) > 0:
        idx = np.random.randint(0, len(X_test))
        figp, axp = plt.subplots(figsize=(8, 3))
        for name, (preds, true, _) in model_preds.items():
            axp.plot(range(1, max_h + 1), true[idx], marker="o", label=f"{name}-True")
            axp.plot(range(1, max_h + 1), preds[idx], marker="o", linestyle="--", label=f"{name}-Pred")
        axp.set_title(f"样本 {idx} 的未来 {max_h} 个 {bar}（真实 vs 预测）")
        axp.set_xlabel("Steps Ahead"); axp.set_ylabel("Close")
        axp.legend(); st.pyplot(figp)

    st.subheader("🧭 未来建议（仅供参考，不构成投资建议）")
    suggestions = []
    for m in models_to_run:
        preds, true, _ = model_preds[m]
        path = preds[-1]
        slope = (path[-1] - path[0]) / max(1, (len(path) - 1))
        df_m = result_df[(result_df["Model"] == m) & (result_df["Horizon"] == max_h)]
        trend_acc = float(df_m["TrendAcc"].iloc[0]) if len(df_m) > 0 else np.nan
        if slope > 0 and trend_acc >= 0.55:
            view = "偏多（上行概率较高）"
        elif slope < 0 and trend_acc >= 0.55:
            view = "偏空（下行概率较高）"
        else:
            view = "中性（信号较弱）"
        suggestions.append({"Model": m, "ForecastSlope": slope, f"TrendAcc@{max_h}": trend_acc, "View": view})

    sug_df = pd.DataFrame(suggestions)
    st.dataframe(sug_df.style.format({"ForecastSlope": "{:.4f}", f"TrendAcc@{max_h}": "{:.3f}"}), use_container_width=True)

    votes = {"偏多": 0.0, "偏空": 0.0, "中性": 0.0}
    for r in suggestions:
        votes[r["View"].split("（")[0]] += max(0.0, (r.get(f"TrendAcc@{max_h}") or 0) - 0.5)
    final_view = max(votes, key=votes.get) if sum(votes.values()) > 0 else "中性"
    st.info(f"**综合参考观点：{final_view}**（基于各模型 TrendAcc>0.5 的加权投票）")

else:
    st.info("在左侧选择参数后，点击 **“🚀 开始预测”**。示例交易对：`BTC-USDT`，周期 `1D`，输入窗口 60，预测 15。")

with st.expander("方法说明与免责声明"):
    st.markdown("""
- **数据源**：OKX v5 公共行情 API（免 Key），K线周期：1D/4H/1H；
- **特征工程**：OHLCV、收益率、均线（MA5/10/20）、波动率、RSI、OBV；
- **归一化**：特征与目标分开 MinMaxScaler；
- **数据集**：固定输入窗口（默认 60 根），输出未来 `H` 步（默认 15）；
- **模型**：LSTM / GRU / Transformer / TCN；
- **评估**：RMSE、MAE、R²、趋势准确率；
- **建议**：基于预测路径斜率与趋势准确率（>0.55）的规则化投票，仅供参考，**不构成投资建议**。
""")
