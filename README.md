# Quant

使用 Transformer 来进行加密货币（BTC）价格波动的时间序列预测。

## TODO LIST

- [ ] 下载并处理比特币（BTC）的历史交易数据集 [Binance](https://data.binance.vision/?prefix=data/)、[Kaggle](https://www.kaggle.com/code/alibulut1/bitcoin-price-prediction-with-using-lstm/notebook?select=BTC-2017min.csv)、[Python-OKX](https://pypi.org/project/python-okx/)
- [ ] 确定预测指标，可暂定为闭盘价（Close）
- [ ] 使用 Streamlit 可视化
- [ ] 使用 Temporal Fusion Transformer 模型作为 Baseline，可以直接使用 [Pytorch Forecast](https://github.com/sktime/pytorch-forecasting?tab=readme-ov-file) 库中的实现

## Study

[OKX Docs](https://www.okx.com/docs-v5/zh/#overview)

## Installation

```bash
pip install -r requirements.txt
```

## Data

下载数据：

```bash
python okx/okx_fetch_data.py --inst-id BTC-USDT-SWAP --bar 15m  --start "2025-10-27" --end "2025-10-28" --outfile data/btc_swap_15m_20251027_20251028.csv
```

数据分析：
TODO: 改成读取 CSV 文件分析并可视化（写成 jupyter 的 .ipynb 文件）。

```bash
# python okx/okx_data_diagnosis.py --inst-id BTC-USDT --start 2018-01-01 --bar 1H
```

## Usage

```bash
streamlit run app.py
```
