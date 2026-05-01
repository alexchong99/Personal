import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# Config
# =============================
CFG = {
    "ticker_primary": "QQQM",
    "ticker_proxy": "QQQ",
    "use_proxy_preinception": True,
    "start": "2012-01-01",
    "end": None,
    "initial_cash": 100_000,
    "fee_per_side": 0.0005,  # 0.05% per side = 0.10% round-trip
    "rsi_period": 14,
    "meanrev_lookback": 20,
    "vol_lookback": 20,
    "target_daily_vol": 0.012,
    "rolling_sharpe_window": 252,
}

sns.set_theme(style="whitegrid", context="talk")


def _extract_close(df: pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        return df[("Close", ticker)].rename(ticker)
    return df["Close"].rename(ticker)


def download_price_series(cfg: dict) -> pd.Series:
    p = cfg["ticker_primary"]
    q = cfg["ticker_proxy"]
    start, end = cfg["start"], cfg["end"]

    primary = _extract_close(yf.download(p, start=start, end=end, auto_adjust=True, progress=False), p)
    if not cfg["use_proxy_preinception"]:
        return primary.dropna()

    proxy = _extract_close(yf.download(q, start=start, end=end, auto_adjust=True, progress=False), q)
    first_valid = primary.first_valid_index()
    if first_valid is None or first_valid not in proxy.index:
        return primary.dropna()

    scale = primary.loc[first_valid] / proxy.loc[first_valid]
    stitched = primary.combine_first(proxy * scale)
    return stitched.dropna()


def rsi(series: pd.Series, period: int) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def calc_metrics(returns: pd.Series, trades: pd.Series) -> dict:
    r = returns.dropna()
    eq = (1 + r).cumprod()
    total_return = eq.iloc[-1] - 1
    cagr = eq.iloc[-1] ** (252 / len(r)) - 1
    sharpe = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else np.nan
    downside = r[r < 0].std()
    sortino = (r.mean() / downside) * np.sqrt(252) if pd.notna(downside) and downside > 0 else np.nan
    max_dd = (eq / eq.cummax() - 1).min()
    wr = (trades > 0).mean() * 100 if len(trades) else np.nan
    gp, gl = trades[trades > 0].sum(), -trades[trades < 0].sum()
    pf = gp / gl if gl > 0 else np.nan
    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Win Rate (%)": wr,
        "Profit Factor": pf,
        "Total Trades": float(len(trades)),
    }


def extract_trade_returns(position: pd.Series, strategy_returns: pd.Series) -> pd.Series:
    pos = position.fillna(0)
    changes = pos.diff().fillna(pos)
    entries = (changes > 0)
    exits = (changes < 0)
    out, in_trade, cum = [], False, 1.0
    for i in range(len(pos)):
        if entries.iloc[i] and not in_trade and pos.iloc[i] > 0:
            in_trade, cum = True, 1.0
        if in_trade:
            cum *= (1 + strategy_returns.iloc[i])
        if exits.iloc[i] and in_trade:
            out.append(cum - 1)
            in_trade, cum = False, 1.0
    if in_trade:
        out.append(cum - 1)
    return pd.Series(out, dtype=float)


def build_strategies(price: pd.Series, cfg: dict) -> dict:
    ret = price.pct_change().fillna(0)
    vol = ret.rolling(cfg["vol_lookback"]).std().replace(0, np.nan)
    size = (cfg["target_daily_vol"] / vol).clip(0, 1).fillna(0)

    ma50, ma200 = price.rolling(50).mean(), price.rolling(200).mean()
    z = (price - price.rolling(cfg["meanrev_lookback"]).mean()) / price.rolling(cfg["meanrev_lookback"]).std()
    r = rsi(price, cfg["rsi_period"])

    momentum = (ma50 > ma200).astype(float) * size
    meanrev = ((z < -1.0).astype(float) - (z > 0).astype(float)).clip(lower=0) * size

    vol_rsi_raw = pd.Series(0.0, index=price.index)
    rev_arb_raw = pd.Series(0.0, index=price.index)
    trend = price > ma200
    for i in range(1, len(price)):
        vol_rsi_raw.iloc[i] = 1.0 if r.iloc[i] < 30 else (0.0 if r.iloc[i] > 55 else vol_rsi_raw.iloc[i - 1])
        rev_arb_raw.iloc[i] = 1.0 if (r.iloc[i] < 35 and trend.iloc[i]) else (0.0 if (r.iloc[i] > 65 or not trend.iloc[i]) else rev_arb_raw.iloc[i - 1])

    positions = {
        "Momentum Algo": momentum.shift(1).fillna(0),
        "Mean Reversion Algo": meanrev.shift(1).fillna(0),
        "Volatility Algo (RSI)": (vol_rsi_raw * size).shift(1).fillna(0),
        "Reverse Arbitrage Algo (RSI)": (rev_arb_raw * size).shift(1).fillna(0),
    }

    out = {}
    for name, pos in positions.items():
        costs = pos.diff().abs().fillna(0) * cfg["fee_per_side"]
        sret = pos * ret - costs
        eq = (1 + sret).cumprod() * cfg["initial_cash"]
        trades = extract_trade_returns(pos, sret)
        out[name] = {"position": pos, "returns": sret, "equity": eq, "trades": trades, "metrics": calc_metrics(sret, trades)}
    return out


def market_regimes(price: pd.Series) -> tuple[pd.Series, pd.Series]:
    r252 = price.pct_change(252)
    v63 = price.pct_change().rolling(63).std() * np.sqrt(252)
    rv = pd.Series(index=price.index, dtype=object)
    rv[r252 > 0.10] = "Bull"
    rv[r252 < -0.10] = "Bear"
    rv[(r252 >= -0.10) & (r252 <= 0.10)] = "Sideways"
    vv = pd.Series(np.where(v63 > v63.median(), "High Vol", "Low Vol"), index=price.index)
    return rv, vv


def monthly_returns(r: pd.Series) -> pd.Series:
    return r.resample("ME").apply(lambda x: (1 + x).prod() - 1)


def run_report(cfg: dict) -> pd.DataFrame:
    price = download_price_series(cfg)
    res = build_strategies(price, cfg)
    regime, vreg = market_regimes(price)

    perf = pd.DataFrame({k: v["metrics"] for k, v in res.items()}).T
    print("\nPerformance Summary (Overall)\n")
    print(perf)

    fig, axes = plt.subplots(6, 2, figsize=(24, 30))
    axes = axes.flatten()
    colors = dict(zip(res.keys(), sns.color_palette("tab10", 4)))

    axes[0].axis("off")
    t = axes[0].table(cellText=np.round(perf.values, 4), rowLabels=perf.index, colLabels=perf.columns, loc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1.0, 1.6)
    axes[0].set_title("1. Performance Summary (Overall)")

    for k, v in res.items():
        axes[1].plot(v["equity"], label=k, color=colors[k], linewidth=2)
    axes[1].set_yscale("log"); axes[1].legend(fontsize=9); axes[1].set_title("2. Equity Curve Comparison (Log Scale)")

    conds = ["Bull", "Bear", "Sideways", "High Vol", "Low Vol"]
    cond_perf = pd.DataFrame(index=conds)
    for k, v in res.items():
        rr = v["returns"]
        cond_perf[k] = [rr[regime=="Bull"].mean()*252, rr[regime=="Bear"].mean()*252, rr[regime=="Sideways"].mean()*252,
                        rr[vreg=="High Vol"].mean()*252, rr[vreg=="Low Vol"].mean()*252]
    cond_perf.plot(kind="bar", ax=axes[2]); axes[2].set_title("3. Performance by Market Condition")

    for k, v in res.items():
        dd = v["equity"] / v["equity"].cummax() - 1
        axes[3].plot(dd, label=k, color=colors[k])
    axes[3].legend(fontsize=9); axes[3].set_title("4. Drawdown Comparison")

    w = cfg["rolling_sharpe_window"]
    for k, v in res.items():
        rs = v["returns"].rolling(w).mean() / v["returns"].rolling(w).std() * np.sqrt(252)
        axes[4].plot(rs, label=k, color=colors[k])
    axes[4].axhline(0, color="black", lw=1); axes[4].legend(fontsize=9); axes[4].set_title(f"5. Rolling Sharpe Ratio ({w}-day)")

    hm = monthly_returns(res["Momentum Algo"]["returns"]).to_frame("ret")
    hm["Year"] = hm.index.year; hm["Month"] = hm.index.strftime("%b")
    piv = hm.pivot(index="Year", columns="Month", values="ret").reindex(columns=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    sns.heatmap(piv*100, cmap="RdYlGn", center=0, ax=axes[5], cbar=True)
    axes[5].set_title("6. Monthly Returns Heatmap (Momentum Algo, %)")

    wr = pd.DataFrame(index=conds)
    for k, v in res.items():
        rr=v["returns"]; wr[k]=[(rr[regime=="Bull"]>0).mean()*100,(rr[regime=="Bear"]>0).mean()*100,(rr[regime=="Sideways"]>0).mean()*100,(rr[vreg=="High Vol"]>0).mean()*100,(rr[vreg=="Low Vol"]>0).mean()*100]
    wr.plot(kind="bar", ax=axes[6]); axes[6].set_title("7. Win Rate by Market Condition (%)")

    for k, v in res.items():
        sns.kdeplot(monthly_returns(v["returns"]).dropna(), ax=axes[7], label=k, color=colors[k])
    axes[7].legend(fontsize=9); axes[7].set_title("8. Monthly Returns Distribution")

    for k, v in res.items():
        dd = v["equity"]/v["equity"].cummax()-1
        axes[8].plot(dd.cummin(), label=k, color=colors[k])
    axes[8].legend(fontsize=9); axes[8].set_title("9. Max Drawdown Over Time")

    trades_df = pd.DataFrame({k:v["trades"] for k,v in res.items()})
    sns.boxplot(data=trades_df, ax=axes[9]); axes[9].tick_params(axis="x", rotation=20); axes[9].set_title("10. Average Trade Return Distribution")

    for k,v in res.items():
        axes[10].plot((v["position"].diff().abs()>0).astype(int).cumsum(), label=k, color=colors[k])
    axes[10].legend(fontsize=9); axes[10].set_title("11. Trade Count Over Time (Cumulative)")

    axes[11].axis("off")
    axes[11].text(0.01, 0.8, f"Ticker: {cfg['ticker_primary']} (proxy {cfg['ticker_proxy']}: {cfg['use_proxy_preinception']})\nData: {price.index.min().date()} to {price.index.max().date()}\nCosts: {cfg['fee_per_side']*100:.02f}% per side\nNo look-ahead: entries shifted 1 bar", fontsize=12)

    plt.tight_layout(); plt.savefig("qqqm_11panel_report.png", dpi=180)
    perf.to_csv("qqqm_performance_summary.csv")
    return perf


if __name__ == "__main__":
    run_report(CFG)
