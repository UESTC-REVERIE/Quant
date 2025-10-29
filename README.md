# Quant

使用 Transformer 来进行加密货币（BTC）价格波动的时间序列预测。

## TODO LIST

- [ ] 下载并处理比特币（BTC）的历史交易数据集 [Binance](https://data.binance.vision/?prefix=data/)、[Kaggle](https://www.kaggle.com/code/alibulut1/bitcoin-price-prediction-with-using-lstm/notebook?select=BTC-2017min.csv)、[Python-OKX](https://pypi.org/project/python-okx/)
- [ ] 确定预测指标，可暂定为闭盘价（Close）
- [ ] 使用 Streamlit 可视化
- [ ] 使用 Temporal Fusion Transformer 模型作为 Baseline，可以直接使用 [Pytorch Forecast](https://github.com/sktime/pytorch-forecasting?tab=readme-ov-file) 库中的实现

## Installation

```bash
pip install -r requirements.txt
```

## Data

下载数据：

```bash

```

数据分析：

```bash
python okx_data_diagnosis.py --inst-id BTC-USDT --start 2018-01-01 --bar 1H
```

## Usage

```bash
streamlit run app.py
```
