# -*- coding: utf-8 -*-
"""
============================================================
  FirstTrade — Nifty 50 Screener  |  Quant Research Edition
============================================================
  Signals:
    • Golden Cross (MA50 > MA200)
    • RSI Momentum (40–65 zone)
    • Volume confirmation
    • Volatility stability vs benchmark
    • MACD signal confirmation
    • Bollinger Band squeeze detector
    • Beta vs Nifty 50

  Quant Add-ons:
    • 21-day backtest with Sharpe + Max Drawdown
    • Factor contribution table (leave-one-out)
    • Rolling 1-year return vs benchmark (alpha proxy)
============================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries loaded!")


# ══════════════════════════════════════════════════════════
#  UNIVERSE & BENCHMARK
# ══════════════════════════════════════════════════════════

NIFTY50 = [
    'RELIANCE.NS','TCS.NS','HDFCBANK.NS','BHARTIARTL.NS','ICICIBANK.NS',
    'INFOSYS.NS','SBIN.NS','HINDUNILVR.NS','ITC.NS','LT.NS',
    'KOTAKBANK.NS','AXISBANK.NS','BAJFINANCE.NS','MARUTI.NS','HCLTECH.NS',
    'ASIANPAINT.NS','TITAN.NS','SUNPHARMA.NS','ULTRACEMCO.NS','WIPRO.NS',
    'NESTLEIND.NS','BAJAJFINSV.NS','TATAMOTORS.NS','POWERGRID.NS','NTPC.NS',
    'TECHM.NS','ONGC.NS','JSWSTEEL.NS','COALINDIA.NS','TATASTEEL.NS',
    'ADANIENT.NS','ADANIPORTS.NS','GRASIM.NS','CIPLA.NS','DIVISLAB.NS',
    'DRREDDY.NS','HEROMOTOCO.NS','HINDALCO.NS','INDUSINDBK.NS','M&M.NS',
    'APOLLOHOSP.NS','BAJAJ-AUTO.NS','BRITANNIA.NS','EICHERMOT.NS','TATACONSUM.NS',
    'BPCL.NS','SHRIRAMFIN.NS','SBILIFE.NS','HDFCLIFE.NS','BEL.NS'
]

BENCHMARK = '^NSEI'


# ══════════════════════════════════════════════════════════
#  HELPER: SAFE DOWNLOAD
# ══════════════════════════════════════════════════════════

def safe_download(ticker, period='1y', interval='1d'):
    """Download OHLCV; flatten MultiIndex; return empty DataFrame on failure."""
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True, threads=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════
#  TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════

def compute_rsi(series, period=14):
    """Wilder RSI."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    """Return (MACD line, Signal line, Histogram)."""
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(series, window=20, num_std=2):
    """Return (upper, mid, lower band, % bandwidth)."""
    mid   = series.rolling(window).mean()
    std   = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    bw    = (upper - lower) / mid * 100
    return upper, mid, lower, bw


def compute_beta(stock_returns, bench_returns):
    """OLS beta of stock vs benchmark over aligned returns."""
    df_ab = pd.concat([stock_returns, bench_returns], axis=1).dropna()
    if len(df_ab) < 30:
        return np.nan
    cov = np.cov(df_ab.iloc[:, 0], df_ab.iloc[:, 1])
    return round(cov[0, 1] / cov[1, 1], 2)


# ══════════════════════════════════════════════════════════
#  CORE SCREENER
# ══════════════════════════════════════════════════════════

def analyse_stock(ticker, benchmark_vol, bench_returns):
    """
    6 signals: Trend, Momentum, Volume, Stability, MACD, BB Squeeze.
    Returns scored dict or None.
    """
    try:
        df = safe_download(ticker)
        if df.empty or len(df) < 210:
            return None

        close  = df['Close'].squeeze()
        volume = df['Volume'].squeeze()

        # Original 4 signals
        ma50  = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        trend_pass = bool(ma50 > ma200)

        rsi_val       = float(compute_rsi(close).iloc[-1])
        momentum_pass = bool(40 <= rsi_val <= 65)

        volume_pass = bool(volume.iloc[-1] > volume.rolling(20).mean().iloc[-1])

        stock_vol      = float(close.pct_change().dropna().std() * np.sqrt(252))
        stability_pass = bool(stock_vol < benchmark_vol)

        # MACD confirmation
        macd_line, signal_line, _ = compute_macd(close)
        macd_pass = bool(macd_line.iloc[-1] > signal_line.iloc[-1])

        # Bollinger Band squeeze
        _, _, _, bw = compute_bollinger(close)
        bw_now     = float(bw.iloc[-1])
        bw_avg     = float(bw.rolling(50).mean().iloc[-1])
        bb_squeeze = bool(bw_now < bw_avg)

        # Beta
        stock_ret = close.pct_change().dropna()
        beta_val  = compute_beta(stock_ret, bench_returns)

        score = sum([trend_pass, momentum_pass, volume_pass, stability_pass])

        if score >= 3:
            verdict = '✅ Look closer'
        elif score >= 1:
            verdict = '⏳ Wait'
        else:
            verdict = '❌ Avoid'

        return {
            'Stock':      ticker.replace('.NS', ''),
            'Price (₹)':  round(float(close.iloc[-1]), 2),
            'Trend':      '✓' if trend_pass     else '✗',
            'Momentum':   '✓' if momentum_pass  else '✗',
            'Volume':     '✓' if volume_pass    else '✗',
            'Stability':  '✓' if stability_pass else '✗',
            'Score':      score,
            'RSI':        round(rsi_val, 1),
            'Volatility': round(stock_vol * 100, 1),
            'MACD ✓':     '✓' if macd_pass  else '✗',
            'BB Squeeze': '✓' if bb_squeeze else '✗',
            'Beta':       beta_val,
            'Verdict':    verdict,
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════
#  RUN SCREENER
# ══════════════════════════════════════════════════════════

print("\nFetching Nifty 50 benchmark data...")
nifty_df = safe_download(BENCHMARK)

if nifty_df.empty:
    raise Exception("❌ Could not fetch benchmark. Check internet connection.")

bench_returns = nifty_df['Close'].squeeze().pct_change().dropna()
benchmark_vol = float(bench_returns.std() * np.sqrt(252))
print(f"   Benchmark annualised vol: {benchmark_vol*100:.1f}%\n")

results = []
for i, ticker in enumerate(NIFTY50):
    print(f"  Analysing {ticker.replace('.NS',''):15s} ({i+1}/{len(NIFTY50)})", end='\r')
    r = analyse_stock(ticker, benchmark_vol, bench_returns)
    if r:
        results.append(r)

df_results = pd.DataFrame(results)

if df_results.empty:
    print("\n⚠️ No stock data fetched.")
else:
    df_results = df_results.sort_values('Score', ascending=False).reset_index(drop=True)
    print(f"\n✅ Done! Screened {len(df_results)} stocks.\n")
    print(df_results.to_string(index=False))


# ══════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════

if not df_results.empty:
    df = df_results

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('FirstTrade — Nifty 50 Quant Research Dashboard', fontsize=14, fontweight='bold')

    # Chart 1: Score Distribution
    score_counts = df['Score'].value_counts().sort_index()
    bar_colours  = [{0:'#dc3545',1:'#fd7e14',2:'#ffc107',3:'#28a745',4:'#1a7a3c'}[s]
                    for s in score_counts.index]
    axes[0,0].bar(score_counts.index.astype(str), score_counts.values, color=bar_colours)
    axes[0,0].set_title("Score Distribution")
    axes[0,0].set_xlabel("Score (0–4)")
    axes[0,0].set_ylabel("# Stocks")

    # Chart 2: Signal Pass Rate
    signals    = ['Trend','Momentum','Volume','Stability','MACD ✓','BB Squeeze']
    pass_rates = [((df[s]=='✓').sum() / len(df)) * 100 for s in signals]
    colours_h  = ['#4c72b0']*4 + ['#dd8452','#55a868']
    axes[0,1].barh(signals, pass_rates, color=colours_h)
    axes[0,1].set_title("Signal Pass Rate (%)")
    axes[0,1].set_xlim(0, 100)
    axes[0,1].axvline(50, color='grey', linestyle='--', linewidth=0.8)

    # Chart 3: RSI vs Volatility
    scatter_colours = df['Score'].map({0:'#dc3545',1:'#fd7e14',2:'#ffc107',
                                       3:'#28a745',4:'#1a7a3c'})
    axes[1,0].scatter(df['RSI'], df['Volatility'], c=scatter_colours, s=60, alpha=0.8)
    axes[1,0].set_title("RSI vs Annualised Volatility")
    axes[1,0].set_xlabel("RSI")
    axes[1,0].set_ylabel("Volatility (%)")
    axes[1,0].axvline(40, color='grey', linestyle='--', linewidth=0.7)
    axes[1,0].axvline(65, color='grey', linestyle='--', linewidth=0.7)
    for _, row in df.iterrows():
        axes[1,0].annotate(row['Stock'], (row['RSI'], row['Volatility']),
                           fontsize=5, alpha=0.6)

    # Chart 4: Beta Distribution
    beta_clean = df['Beta'].dropna()
    axes[1,1].hist(beta_clean, bins=15, color='#4c72b0', edgecolor='white')
    axes[1,1].axvline(1.0, color='red', linestyle='--', linewidth=1, label='Beta = 1')
    axes[1,1].axvline(beta_clean.mean(), color='green', linestyle='--',
                      linewidth=1, label=f'Mean = {beta_clean.mean():.2f}')
    axes[1,1].set_title("Beta Distribution vs Nifty 50")
    axes[1,1].set_xlabel("Beta")
    axes[1,1].set_ylabel("# Stocks")
    axes[1,1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('screener_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 Dashboard saved as screener_dashboard.png")


# ══════════════════════════════════════════════════════════
#  SIP SUITABILITY CHECK
# ══════════════════════════════════════════════════════════

top_stocks = df_results[df_results['Score'] >= 3]['Stock'].tolist() if not df_results.empty else []

if not top_stocks:
    print("\nNo stocks scored 3+ today.")
else:
    print(f"\n🏆 Stocks scoring 3+: {', '.join(top_stocks)}")

print("\n🔄 Checking SIP suitability...")
sip_results = []

for stock in top_stocks:
    ticker = stock + '.NS'
    monthly_scores = []

    for m in [1, 2, 3]:
        try:
            df_m = safe_download(ticker, period=f'{m+1}mo')
            if df_m.empty or len(df_m) < 50:
                continue
            c   = df_m['Close'].squeeze()
            v   = df_m['Volume'].squeeze()
            m50  = c.rolling(50).mean().iloc[-1]
            m200 = c.rolling(min(200, len(c))).mean().iloc[-1]
            rsi  = float(compute_rsi(c).iloc[-1])
            vol  = bool(v.iloc[-1] > v.rolling(20).mean().iloc[-1])
            std  = float(c.pct_change().std() * np.sqrt(252))
            score = sum([bool(m50 > m200), bool(40 <= rsi <= 65),
                         vol, bool(std < benchmark_vol)])
            monthly_scores.append(score)
        except Exception:
            continue

    if monthly_scores:
        sip_results.append({'Stock': stock,
                            '3-Month Avg Score': round(np.mean(monthly_scores), 1)})

if not sip_results:
    print("📭 No SIP candidates found.")
else:
    df_sip = pd.DataFrame(sip_results).sort_values('3-Month Avg Score', ascending=False)
    print(df_sip.to_string(index=False))


# ══════════════════════════════════════════════════════════
#  QUANT ADD-ON 1 — 21-Day Signal Backtest
# ══════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  📊 QUANT ADD-ON 1 — 21-Day Signal Backtest")
print("="*55)

trade_returns = []

for ticker in NIFTY50:
    df_s = safe_download(ticker)
    if df_s.empty or len(df_s) < 100:
        continue

    close  = df_s['Close'].squeeze()
    volume = df_s['Volume'].squeeze()
    rsi    = compute_rsi(close)
    ma50   = close.rolling(50).mean()
    ma200  = close.rolling(200).mean()

    stock_vol  = float(close.pct_change().std() * np.sqrt(252))
    stab_val   = int(stock_vol < benchmark_vol)

    trend_sig = (ma50 > ma200).astype(int)
    mom_sig   = ((rsi >= 40) & (rsi <= 65)).astype(int)
    vol_sig   = (volume > volume.rolling(20).mean()).astype(int)
    signal    = trend_sig + mom_sig + vol_sig + stab_val

    for i in range(len(close) - 21):
        if signal.iloc[i] >= 3:
            ret = float((close.iloc[i+21] - close.iloc[i]) / close.iloc[i])
            trade_returns.append(ret)

trade_returns = np.array(trade_returns)

if len(trade_returns) > 0:
    sharpe   = (np.mean(trade_returns) / np.std(trade_returns)) * np.sqrt(12)
    cum      = (1 + trade_returns).cumprod()
    peak     = np.maximum.accumulate(cum)
    dd       = (cum - peak) / peak
    win_rate = (trade_returns > 0).mean() * 100

    print(f"  Total signals  : {len(trade_returns)}")
    print(f"  Avg 21-day ret : {np.mean(trade_returns)*100:.2f}%")
    print(f"  Win rate       : {win_rate:.1f}%")
    print(f"  Sharpe ratio   : {sharpe:.2f}")
    print(f"  Max drawdown   : {dd.min()*100:.1f}%")

    plt.figure(figsize=(12, 4))
    plt.plot(cum, color='#1a7a3c', linewidth=1.5)
    plt.fill_between(range(len(cum)), cum, 1, where=(cum < 1),
                     color='#dc3545', alpha=0.3, label='Drawdown')
    plt.axhline(1, color='grey', linestyle='--', linewidth=0.8)
    plt.title("Equity Curve — Score ≥ 3 Signals (21-day exits)", fontweight='bold')
    plt.xlabel("Trade #")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig('backtest_equity_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("  ⚠️ Not enough data for backtest.")


# ══════════════════════════════════════════════════════════
#  QUANT ADD-ON 2 — Factor Contribution (Leave-One-Out)
# ══════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  🧪 QUANT ADD-ON 2 — Factor Contribution (Leave-One-Out)")
print("="*55)

all_factors    = ['Trend', 'Momentum', 'Volume', 'Stability']
factor_results = []

for factor in all_factors:
    factor_returns = []

    for ticker in NIFTY50:
        df_s = safe_download(ticker)
        if df_s.empty or len(df_s) < 100:
            continue

        close  = df_s['Close'].squeeze()
        volume = df_s['Volume'].squeeze()
        rsi    = compute_rsi(close)
        ma50   = close.rolling(50).mean()
        ma200  = close.rolling(200).mean()
        std    = float(close.pct_change().std() * np.sqrt(252))

        factor_sigs = {
            'Trend':     (ma50 > ma200).astype(int),
            'Momentum':  ((rsi >= 40) & (rsi <= 65)).astype(int),
            'Volume':    (volume > volume.rolling(20).mean()).astype(int),
            'Stability': pd.Series(int(std < benchmark_vol), index=close.index),
        }

        score = sum(v for k, v in factor_sigs.items() if k != factor)

        for i in range(len(close) - 21):
            if score.iloc[i] >= 2:
                ret = float((close.iloc[i+21] - close.iloc[i]) / close.iloc[i])
                factor_returns.append(ret)

    if factor_returns:
        avg = np.mean(factor_returns) * 100
        factor_results.append({'Removed Factor': factor,
                                'Avg Return (%)': round(avg, 3),
                                'N trades':       len(factor_returns)})

if factor_results:
    df_factors = pd.DataFrame(factor_results).sort_values('Avg Return (%)', ascending=False)
    print("  (Lower avg return = that factor was more important)\n")
    print(df_factors.to_string(index=False))

    plt.figure(figsize=(8, 4))
    plt.barh(df_factors['Removed Factor'], df_factors['Avg Return (%)'], color='#4c72b0')
    plt.axvline(0, color='grey', linewidth=0.8)
    plt.title("Avg 21-day Return When Each Factor is Removed", fontweight='bold')
    plt.xlabel("Avg Return (%)")
    plt.tight_layout()
    plt.savefig('factor_contribution.png', dpi=150, bbox_inches='tight')
    plt.show()


# ══════════════════════════════════════════════════════════
#  QUANT ADD-ON 3 — Alpha Proxy (Screen vs Nifty)
# ══════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  📈 QUANT ADD-ON 3 — Alpha Proxy (Screen vs Nifty)")
print("="*55)

if top_stocks:
    fig, ax = plt.subplots(figsize=(12, 5))
    nifty_close = nifty_df['Close'].squeeze()
    nifty_norm  = nifty_close / nifty_close.iloc[0]
    ax.plot(nifty_close.index, nifty_norm, color='grey',
            linewidth=2, linestyle='--', label='Nifty 50 (benchmark)', zorder=3)

    for stock in top_stocks[:5]:
        df_s = safe_download(stock + '.NS')
        if df_s.empty:
            continue
        close_s = df_s['Close'].squeeze()
        common  = close_s.index.intersection(nifty_close.index)
        if len(common) < 10:
            continue
        norm_s = close_s.loc[common] / close_s.loc[common].iloc[0]
        ax.plot(common, norm_s, linewidth=1.3, label=stock, alpha=0.85)

    ax.set_title("Top-Scoring Stocks vs Nifty 50 (1-Year Normalised Price)", fontweight='bold')
    ax.set_ylabel("Normalised Price (base = 1.0)")
    ax.legend(fontsize=8, loc='upper left')
    ax.axhline(1, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('alpha_proxy.png', dpi=150, bbox_inches='tight')
    plt.show()

    bench_ret_1y = float(nifty_close.iloc[-1] / nifty_close.iloc[0] - 1) * 100
    print(f"\n  {'Stock':<15} {'1Y Return':>10}  {'vs Nifty':>10}")
    print("  " + "-"*38)
    for stock in top_stocks:
        df_s = safe_download(stock + '.NS')
        if df_s.empty:
            continue
        c    = df_s['Close'].squeeze()
        ret  = float(c.iloc[-1] / c.iloc[0] - 1) * 100
        diff = ret - bench_ret_1y
        flag = '🟢' if diff > 0 else '🔴'
        print(f"  {stock:<15} {ret:>9.1f}%  {diff:>+9.1f}% {flag}")
    print(f"\n  {'Nifty 50':<15} {bench_ret_1y:>9.1f}%  {'(baseline)':>10}")

print("\n" + "="*55)
print("  ✅ FirstTrade Quant Research Screen complete.")
print("  ⚠️  Not financial advice. Past signals ≠ future returns.")
print("="*55)
