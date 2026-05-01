# QQQM 11-Panel Report (Text Version)

This repository includes a reproducible script and a text summary of the latest run, so no binary-only artifact is required.

## Files
- `qqqm_11panel_report.py` — generates the full 11-panel chart and summary CSV.
- `qqqm_performance_summary.csv` — machine-readable performance table.

## Latest Performance Summary (Overall)

| Strategy | Total Return | CAGR | Sharpe Ratio | Sortino Ratio | Max Drawdown | Win Rate (%) | Profit Factor | Total Trades |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Momentum Algo | 5.669412 | 0.141968 | 0.993956 | 1.148821 | -0.220919 | 62.672811 | 2.195621 | 217 |
| Mean Reversion Algo | 1.182512 | 0.056121 | 0.643235 | 0.423315 | -0.117480 | 69.603524 | 1.650406 | 227 |
| Volatility Algo (RSI) | 0.853965 | 0.044135 | 0.667264 | 0.324141 | -0.104917 | 68.604651 | 1.988946 | 86 |
| Reverse Arbitrage Algo (RSI) | 0.843092 | 0.043706 | 0.666913 | 0.349704 | -0.094814 | 70.886076 | 2.527697 | 79 |

## Reproduce
```bash
python3 qqqm_11panel_report.py
```
