# app.py
# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import ta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams["font.sans-serif"] = [
    "SimHei"
]  # 或 ['Microsoft YaHei'] / ['Arial Unicode MS']
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块
# st.set_option('deprecation.showPyplotGlobalUse', False)

# ----------------------------
# 全局设置
# ----------------------------
st.set_page_config(
    page_title="多模型股价预测平台 (LSTM/GRU/Transformer/TCN)", layout="wide"
)
plt.rcParams.update({"figure.autolayout": True})


# 复现性
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
ticker = st.sidebar.text_input(
    "股票代码（例如：603799.SS / AAPL / MSFT）", value="603799.SS"
)
start_date = st.sidebar.date_input("起始日期", value=pd.to_datetime("2018-01-01"))
use_proxy = st.sidebar.checkbox("使用 HTTP 代理（可选）", value=False)
proxy_url = (
    st.sidebar.text_input(
        "代理地址（示例：http://127.0.0.1:7897）", value="http://127.0.0.1:7897"
    )
    if use_proxy
    else None
)

seq_len = st.sidebar.number_input(
    "输入窗口长度（天）", min_value=20, max_value=240, value=60, step=5
)
max_h = st.sidebar.selectbox("预测天数（输出长度）", options=[5, 10, 15], index=2)
epochs = st.sidebar.slider("训练轮次", min_value=10, max_value=200, value=60, step=10)
batch_size = st.sidebar.selectbox("Batch Size", options=[32, 64, 128], index=1)
hidden_size = st.sidebar.selectbox(
    "隐藏层维度（RNN/TCN/Transformer内部）", options=[64, 128, 192], index=0
)
nhead = st.sidebar.selectbox("Transformer 多头注意力头数", options=[2, 4, 8], index=1)
dropout = st.sidebar.slider(
    "Dropout", min_value=0.0, max_value=0.5, value=0.2, step=0.05
)
train_ratio = st.sidebar.slider(
    "训练集比例", min_value=0.5, max_value=0.95, value=0.8, step=0.05
)
horizons_eval = st.sidebar.multiselect(
    "评估周期（可多选）", [1, 3, 5, 10, 15], default=[1, 3, 5, 10, 15]
)

models_to_run = st.sidebar.multiselect(
    "选择模型（至少一个）",
    ["LSTM", "GRU", "Transformer", "TCN"],
    default=["LSTM", "GRU", "Transformer", "TCN"],
)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 开始预测")


# ----------------------------
# 缓存：数据与特征工程
# ----------------------------
@st.cache_data(show_spinner=True)
def fetch_data(ticker, start_date, proxy_url=None):
    kwargs = dict(start=str(start_date))
    if proxy_url:
        kwargs["proxy"] = proxy_url
    df = yf.download(ticker, **kwargs)
    return df


def _safe_series(x):
    # 保证是一维 Series
    return pd.Series(
        np.array(x).squeeze(), index=x.index if hasattr(x, "index") else None
    )


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
        x = scaled_data[i : i + seq_len]
        base = scaled_data[i + seq_len - 1, -1]  # 当前价（scaled）
        future = [scaled_data[i + seq_len + h, -1] for h in range(1, max_h + 1)]
        xs.append(x)
        ys.append(future)
        bases.append(base)
    X = np.array(xs)
    Y = np.array(ys)
    base_close = np.array(bases)
    return X, Y, base_close


class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ----------------------------
# 模型定义：LSTM / GRU / Transformer / TCN
# ----------------------------
class LSTMModel(nn.Module):
    def __init__(
        self, input_size, hidden_size=64, num_layers=2, output_size=15, dropout=0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B = x.size(0)
        h0 = torch.zeros(2, B, self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros(2, B, self.lstm.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(
        self, input_size, hidden_size=64, num_layers=2, output_size=15, dropout=0.2
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B = x.size(0)
        h0 = torch.zeros(2, B, self.gru.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        num_layers=2,
        nhead=4,
        hidden_dim=128,
        output_size=15,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = hidden_dim
        self.input_proj = nn.Linear(input_size, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_size)

    def _positional_encoding(self, L, D, device):
        pos = torch.arange(0, L, device=device).unsqueeze(1)
        i = torch.arange(0, D, 2, device=device)
        angle = 1.0 / torch.pow(10000, (i.float() / D))
        pe = torch.zeros(L, D, device=device)
        pe[:, 0::2] = torch.sin(pos * angle)
        pe[:, 1::2] = torch.cos(pos * angle)
        return pe.unsqueeze(0)

    def forward(self, x):
        # x: (B, T, F)
        B, T, _ = x.shape
        z = self.input_proj(x) * torch.sqrt(
            torch.tensor(self.d_model, dtype=torch.float32, device=x.device)
        )
        z = z + self._positional_encoding(T, self.d_model, x.device)
        out = self.encoder(z)[:, -1, :]
        return self.fc(out)


# ---- TCN ----
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, ksz, stride, dilation, padding, dropout=0.2
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, ksz, stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, ksz, stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.drop1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.drop2,
        )
        self.down = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        for m in self.net:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        out = self.net(x)
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size=15,
        num_channels=[64, 64, 128],
        kernel_size=3,
        dropout=0.2,
    ):
        super().__init__()
        layers = []
        for i, ch in enumerate(num_channels):
            dilation = 2**i
            in_ch = input_size if i == 0 else num_channels[i - 1]
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(in_ch, ch, kernel_size, 1, dilation, padding, dropout)
            )
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        z = x.transpose(1, 2)
        y = self.net(z)  # (B, C, T)
        y = y[:, :, -1]  # 最后时刻
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
        return LSTMModel(
            input_size,
            hidden_size=hidden_size,
            num_layers=2,
            output_size=output_size,
            dropout=dropout,
        ).to(device)
    if name == "GRU":
        return GRUModel(
            input_size,
            hidden_size=hidden_size,
            num_layers=2,
            output_size=output_size,
            dropout=dropout,
        ).to(device)
    if name == "Transformer":
        return TimeSeriesTransformer(
            input_size,
            num_layers=2,
            nhead=nhead,
            hidden_dim=hidden_size,
            output_size=output_size,
            dropout=dropout,
        ).to(device)
    if name == "TCN":
        return TCN(
            input_size,
            output_size=output_size,
            num_channels=[hidden_size, hidden_size, hidden_size * 2],
            kernel_size=3,
            dropout=dropout,
        ).to(device)
    raise ValueError("unknown model")


def train_and_evaluate(
    model_name,
    model,
    X_train,
    Y_train,
    X_test,
    Y_test,
    base_test,
    target_scaler,
    epochs,
    batch_size,
):
    start = time.time()
    loader = DataLoader(
        SeqDataset(X_train, Y_train), batch_size=batch_size, shuffle=True
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    losses = []
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
        losses.append(total / len(loader))

    train_time = time.time() - start

    # 预测
    model.eval()
    with torch.no_grad():
        Xtt = torch.tensor(X_test, dtype=torch.float32, device=device)
        preds_scaled = model(Xtt).cpu().numpy()

    preds = inv_transform_each_day(preds_scaled, target_scaler)
    true = inv_transform_each_day(Y_test, target_scaler)
    base_true = target_scaler.inverse_transform(base_test.reshape(-1, 1)).ravel()

    # 各个 horizon 指标
    rows = []
    for H in horizons_eval:
        t = true[:, :H].ravel()
        p = preds[:, :H].ravel()
        rmse = np.sqrt(mean_squared_error(t, p))
        mae = mean_absolute_error(t, p)
        r2 = r2_score(t, p) if len(np.unique(t)) > 1 else np.nan
        acc = trend_accuracy_horizon(true, preds, base_true, H)
        rows.append([model_name, H, rmse, mae, r2, acc, train_time])

    return rows, preds, true, losses


# ----------------------------
# 主体流程
# ----------------------------
st.title("📈 多模型股价预测平台（LSTM / GRU / Transformer / TCN）")
st.caption(
    "输入股票代码与参数，自动完成数据抓取、训练、预测、对比与建议。（仅用于研究与教学，不构成投资建议）"
)

if run_button:
    with st.spinner("正在获取数据与构建特征..."):
        raw = fetch_data(ticker, start_date, proxy_url if use_proxy else None)
        if raw.empty:
            st.error("获取数据失败，请检查股票代码或代理设置。")
            st.stop()
        feats = build_features(raw)
        scaled_data, feature_scaler, target_scaler = scale_data(feats)
        X, Y, base_close = create_sequences_multiday(scaled_data, seq_len, max_h)
        if len(X) < 100:
            st.warning("样本太少，无法训练。请缩短起始日期或减少 seq_len / max_h。")
            st.stop()

        split = int(len(X) * train_ratio)
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]
        base_train, base_test = base_close[:split], base_close[split:]

    st.success(f"数据准备完成：训练样本 {len(X_train)}，测试样本 {len(X_test)}。")

    # 展示原始价格
    with st.expander("查看历史价格曲线", expanded=False):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(feats.index, feats["Close"])
        ax.set_title(f"{ticker} 历史收盘价")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close")
        st.pyplot(fig)

    all_results = []
    model_preds = {}

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
                name,
                model,
                X_train,
                Y_train,
                X_test,
                Y_test,
                base_test,
                target_scaler,
                epochs,
                batch_size,
            )
            all_results += rows
            model_preds[name] = (preds, true, losses)

            # 训练曲线
            fig_l, ax_l = plt.subplots(figsize=(4, 2.2))
            ax_l.plot(losses)
            ax_l.set_title("训练损失")
            ax_l.set_xlabel("Epoch")
            ax_l.set_ylabel("MSE")
            st.pyplot(fig_l)

    # 汇总指标表
    result_df = pd.DataFrame(
        all_results,
        columns=["Model", "Horizon", "RMSE", "MAE", "R2", "TrendAcc", "TrainTime(s)"],
    )
    st.subheader("📊 指标对比")
    st.dataframe(
        result_df.style.format(
            {
                "RMSE": "{:.4f}",
                "MAE": "{:.4f}",
                "R2": "{:.4f}",
                "TrendAcc": "{:.4f}",
                "TrainTime(s)": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

    # 关键指标可视化：TrendAcc / RMSE
    c1, c2 = st.columns(2)
    with c1:
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        for m in result_df["Model"].unique():
            dfm = result_df[result_df["Model"] == m]
            ax1.plot(dfm["Horizon"], dfm["TrendAcc"], marker="o", label=m)
        ax1.set_title("各模型趋势准确率对比")
        ax1.set_xlabel("Horizon (days)")
        ax1.set_ylabel("TrendAcc")
        ax1.legend()
        st.pyplot(fig1)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        for m in result_df["Model"].unique():
            dfm = result_df[result_df["Model"] == m]
            ax2.plot(dfm["Horizon"], dfm["RMSE"], marker="o", label=m)
        ax2.set_title("各模型 RMSE 对比")
        ax2.set_xlabel("Horizon (days)")
        ax2.set_ylabel("RMSE")
        ax2.legend()
        st.pyplot(fig2)

    # 展示每个模型的一个样本未来路径
    st.subheader("🔭 测试集样本未来路径对比（随机示例）")
    if len(X_test) > 0:
        idx = np.random.randint(0, len(X_test))
        figp, axp = plt.subplots(figsize=(8, 3))
        for name, (preds, true, _) in model_preds.items():
            axp.plot(range(1, max_h + 1), true[idx], marker="o", label=f"{name}-True")
            axp.plot(
                range(1, max_h + 1),
                preds[idx],
                marker="o",
                linestyle="--",
                label=f"{name}-Pred",
            )
        axp.set_title(f"样本 {idx} 的未来 {max_h} 天路径（真实 vs 预测）")
        axp.set_xlabel("Days Ahead")
        axp.set_ylabel("Close")
        axp.legend()
        st.pyplot(figp)

    # ----------------------------
    # 生成“未来建议”
    # ----------------------------
    st.subheader("🧭 未来建议（仅供参考，不构成投资建议）")

    # 以“在测试集末尾的最后一个样本”的预测均值斜率 & 趋势准确率来给出简单规则化建议
    suggestions = []
    for m in models_to_run:
        preds, true, _ = model_preds[m]
        # 用最后一个测试样本
        path = preds[-1]  # (max_h,)
        slope = (path[-1] - path[0]) / max(1, (len(path) - 1))
        # 从指标表中取该模型的 TrendAcc（选你挑选的 horizon，比如 max_h）
        df_m = result_df[(result_df["Model"] == m) & (result_df["Horizon"] == max_h)]
        trend_acc = float(df_m["TrendAcc"].iloc[0]) if len(df_m) > 0 else np.nan

        if slope > 0 and trend_acc >= 0.55:
            view = "偏多（上行概率较高）"
        elif slope < 0 and trend_acc >= 0.55:
            view = "偏空（下行概率较高）"
        else:
            view = "中性（信号较弱）"

        suggestions.append(
            {
                "Model": m,
                "ForecastSlope": slope,
                f"TrendAcc@{max_h}": trend_acc,
                "View": view,
            }
        )

    sug_df = pd.DataFrame(suggestions)
    st.dataframe(
        sug_df.style.format({"ForecastSlope": "{:.4f}", f"TrendAcc@{max_h}": "{:.3f}"}),
        use_container_width=True,
    )

    # 综合建议：投票 + 权重
    # 以 TrendAcc 为权重的“多模型投票”
    votes = {"偏多": 0.0, "偏空": 0.0, "中性": 0.0}
    for r in suggestions:
        votes[r["View"].split("（")[0]] += max(
            0.0, r[f"TrendAcc@{max_h}"] - 0.5
        )  # 大于0.5才给正权重
    final_view = max(votes, key=votes.get) if sum(votes.values()) > 0 else "中性"
    st.info(f"**综合参考观点：{final_view}**（基于各模型 TrendAcc>0.5 的加权投票）")

else:
    st.info(
        "在左侧选择参数后，点击 **“🚀 开始预测”**。默认示例：`603799.SS` 自 2018-01-01 至今，输入窗口 60 天，预测 15 天。"
    )

# ----------------------------
# 说明
# ----------------------------
with st.expander("方法说明与免责声明"):
    st.markdown(
        """
- **特征工程**：Open/High/Low/Close/Volume、收益率、均线（MA5/10/20）、波动率、RSI、OBV；
- **归一化**：特征与目标分开使用 MinMaxScaler；
- **数据集**：固定输入窗口（默认 60 天），输出未来 `H` 天路径（默认 15）；
- **模型**：LSTM / GRU / Transformer / TCN；
- **评估**：RMSE、MAE、R²、趋势准确率（相对当前价的方向一致率）；
- **建议**：基于预测路径斜率与趋势准确率（>0.55）进行规则化投票，仅供参考，**不构成投资建议**。
"""
    )
