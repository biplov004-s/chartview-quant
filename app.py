import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Chartview Quant", page_icon=":chart_with_upwards_trend:", layout="wide")

st.title("Chartview Quant")
st.caption("Eagle-style Quant Research - Screener - Backtest - Paper Trading")
st.caption("Created by Biplov Soren")

DEFAULT_WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS",
    "HCLTECH.NS", "ASIANPAINT.NS", "WIPRO.NS", "ADANIENT.NS", "TATAMOTORS.NS",
    "ULTRACEMCO.NS", "NESTLEIND.NS", "BAJAJFINSV.NS", "POWERGRID.NS", "NTPC.NS",
    "M&M.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "ONGC.NS",
    "COALINDIA.NS", "GRASIM.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
    "EICHERMOT.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "BRITANNIA.NS", "TATACONSUM.NS",
    "APOLLOHOSP.NS", "ADANIPORTS.NS", "BPCL.NS", "IOC.NS", "SBILIFE.NS",
    "HDFCLIFE.NS", "INDUSINDBK.NS", "TECHM.NS", "SHREECEM.NS", "UPL.NS",
    "PIDILITIND.NS", "DABUR.NS", "GODREJCP.NS", "HAVELLS.NS", "SIEMENS.NS",
    "DLF.NS", "VEDL.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS",
    "AMBUJACEM.NS", "ACC.NS", "MOTHERSON.NS", "TVSMOTOR.NS", "BOSCHLTD.NS",
    "MARICO.NS", "COLPAL.NS", "BERGEPAINT.NS", "PAGEIND.NS", "TRENT.NS",
    "ZOMATO.NS", "NAUKRI.NS", "PIIND.NS", "TORNTPHARM.NS", "LUPIN.NS",
    "AUROPHARMA.NS", "BIOCON.NS", "GAIL.NS", "IGL.NS", "PETRONET.NS",
]

# -------------------- data --------------------
@st.cache_data(ttl=900, show_spinner=False)
def get_data(ticker, period="2y"):
    """Fetch OHLCV data. Returns empty DataFrame on any failure (bad ticker,
    network issue, etc.) instead of raising, so the UI can show a clean error."""
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()
    return df

# -------------------- indicators --------------------
def rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def adx(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    up = h.diff()
    down = -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atrv = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atrv
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atrv
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean(), plus_di, minus_di

def prepare(df):
    x = df.copy()
    x["EMA20"] = x["Close"].ewm(span=20, adjust=False).mean()
    x["EMA50"] = x["Close"].ewm(span=50, adjust=False).mean()
    x["EMA200"] = x["Close"].ewm(span=200, adjust=False).mean()
    x["RSI"] = rsi(x["Close"])
    x["ADX"], x["+DI"], x["-DI"] = adx(x)
    x["VolAvg20"] = x["Volume"].rolling(20).mean()
    x["VolRatio"] = x["Volume"] / x["VolAvg20"]
    x["ATR"] = atr(x)
    x["PrevHigh"] = x["High"].shift(1)
    x["SwingLow20"] = x["Low"].rolling(20).min()
    x["BullTrend"] = (x["Close"] > x["EMA20"]) & (x["EMA20"] > x["EMA50"]) & (x["EMA50"] > x["EMA200"])
    x["Momentum"] = x["RSI"].between(55, 70)
    x["TrendStrong"] = (x["ADX"] > 25) & (x["+DI"] > x["-DI"])
    x["VolumeStrong"] = x["VolRatio"] >= 1.5
    x["Breakout"] = x["Close"] > x["PrevHigh"]
    x["BUY"] = x["BullTrend"] & x["Momentum"] & x["TrendStrong"] & x["VolumeStrong"] & x["Breakout"]
    x["Score"] = (
        x["BullTrend"].astype(int) + x["Momentum"].astype(int) + x["TrendStrong"].astype(int)
        + x["VolumeStrong"].astype(int) + x["Breakout"].astype(int)
    )
    return x

# -------------------- realistic backtest --------------------
def backtest_realistic(d, hold_days=20, rr=2.0):
    """Simulate each BUY signal as an actual trade: hold until stop-loss or
    target is hit (using intraday High/Low), or exit at close after
    `hold_days` if neither is hit. This is far closer to real trading than a
    naive 'next day return' check."""
    rows = []
    signal_idx = np.where(d["BUY"].values)[0]
    n = len(d)
    for i in signal_idx:
        if i + 1 >= n:
            continue
        entry_date = d.index[i]
        entry = float(d["Close"].iloc[i])
        sl = float(d["SwingLow20"].iloc[i])
        if not np.isfinite(sl) or sl >= entry:
            sl = entry - 2 * float(d["ATR"].iloc[i])
        risk_per_share = max(entry - sl, 0.01)
        target = entry + rr * risk_per_share

        outcome, exit_price, exit_date = None, None, None
        window_end = min(i + 1 + hold_days, n)
        for j in range(i + 1, window_end):
            low_j = float(d["Low"].iloc[j])
            high_j = float(d["High"].iloc[j])
            if low_j <= sl:
                outcome, exit_price, exit_date = "SL", sl, d.index[j]
                break
            if high_j >= target:
                outcome, exit_price, exit_date = "TARGET", target, d.index[j]
                break
        if outcome is None:
            j = window_end - 1
            outcome = "TIME EXIT"
            exit_price = float(d["Close"].iloc[j])
            exit_date = d.index[j]

        ret_pct = (exit_price - entry) / entry * 100
        rows.append({
            "Entry date": entry_date, "Exit date": exit_date, "Outcome": outcome,
            "Entry": round(entry, 2), "Exit": round(exit_price, 2), "Return %": round(ret_pct, 2),
        })
    return pd.DataFrame(rows)

# -------------------- screener --------------------
@st.cache_data(ttl=900, show_spinner=False)
def get_batch_data(tickers_tuple, period="1y"):
    """Fetch many tickers in one batched, threaded call instead of looping
    one-by-one. Much faster and far less likely to be rate-limited than
    scanning 100-200 tickers sequentially."""
    tickers = list(tickers_tuple)
    try:
        raw = yf.download(
            tickers=tickers, period=period, interval="1d",
            auto_adjust=True, progress=False, group_by="ticker", threads=True,
        )
    except Exception:
        return {}
    out = {}
    if raw is None or raw.empty:
        return out
    for t in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[t].copy() if t in raw.columns.get_level_values(0) else pd.DataFrame()
            df = df.dropna()
            required = {"Open", "High", "Low", "Close", "Volume"}
            if df.empty or not required.issubset(set(df.columns)):
                continue
            out[t] = df
        except Exception:
            continue
    return out

def score_from_df(ticker, df):
    if df.empty or len(df) < 210:
        return None
    d = prepare(df)
    last = d.iloc[-1]
    return {
        "Ticker": ticker.replace(".NS", ""),
        "Price": round(float(last["Close"]), 2),
        "Score": int(last["Score"]),
        "Signal": "BUY" if bool(last["BUY"]) else "WAIT",
        "RSI": round(float(last["RSI"]), 1),
        "ADX": round(float(last["ADX"]), 1),
        "Vol x": round(float(last["VolRatio"]), 2),
    }

# -------------------- sidebar --------------------
st.sidebar.header("Controls")
mode = st.sidebar.radio("Mode", ["Screener", "Single Stock"])
period = st.sidebar.selectbox("History", ["1y", "2y", "5y"], index=1)

if mode == "Single Stock":
    ticker = st.sidebar.text_input("NSE ticker", "RELIANCE.NS")
    capital = st.sidebar.number_input("Paper capital (Rs)", min_value=1000.0, value=100000.0, step=5000.0)
    risk_pct = st.sidebar.slider("Risk per trade %", 0.5, 3.0, 1.0, 0.5)
    hold_days = st.sidebar.slider("Backtest max holding days", 5, 40, 20, 5)

    df = get_data(ticker, period)
    if df.empty:
        st.error("Data nahi mila. Sahi NSE ticker format use karein, jaise RELIANCE.NS. "
                  "Agar ticker sahi hai to ho sakta hai network/data-provider issue ho - thodi der baad try karein.")
        st.stop()
    if len(df) < 210:
        st.warning("Itna history nahi hai ki EMA200 jaise indicators reliable ho paayein. Lambi period select karein.")
        st.stop()

    d = prepare(df)
    last = d.iloc[-1]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price", f"Rs {last['Close']:,.2f}")
    c2.metric("RSI", f"{last['RSI']:.1f}")
    c3.metric("ADX", f"{last['ADX']:.1f}")
    c4.metric("Volume", f"{last['VolRatio']:.2f}x")
    c5.metric("Signal", "BUY" if bool(last["BUY"]) else "WAIT")

    st.subheader("All-In-One Market Dashboard")
    rows = [
        ("Direction", "BULLISH" if last["EMA20"] > last["EMA50"] else "BEARISH"),
        ("Trend", "UP" if last["+DI"] > last["-DI"] else "DOWN"),
        ("Trend Strength", "STRONG" if last["ADX"] >= 25 else "NEUTRAL/WEAK"),
        ("Momentum", "STRONG" if last["RSI"] >= 60 else ("WEAK" if last["RSI"] <= 40 else "NEUTRAL")),
        ("Volume", "STRONG" if last["VolRatio"] >= 1.5 else ("WEAK" if last["VolRatio"] < .8 else "NEUTRAL")),
        ("Breakout", "BREAKOUT" if last["Breakout"] else "NONE"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Factor", "Status"]), use_container_width=True, hide_index=True)

    st.subheader("Price & EMA")
    st.line_chart(d[["Close", "EMA20", "EMA50", "EMA200"]].tail(250), use_container_width=True)

    st.subheader("Paper Trade Plan")
    if bool(last["BUY"]):
        entry = float(last["Close"])
        sl = float(last["SwingLow20"])
        if not np.isfinite(sl) or sl >= entry:
            sl = entry - 2 * float(last["ATR"])
        risk_per_share = max(entry - sl, 0.01)
        risk_amount = capital * risk_pct / 100
        shares = int(risk_amount / risk_per_share)
        target = entry + 2 * risk_per_share
        st.success("A-GRADE BUY SETUP")
        a, b, c, e = st.columns(4)
        a.metric("Entry", f"Rs {entry:,.2f}")
        b.metric("Stop Loss", f"Rs {sl:,.2f}")
        c.metric("Target 1 (1:2)", f"Rs {target:,.2f}")
        e.metric("Shares (paper)", f"{max(shares, 0)}")
    else:
        st.info("WAIT - current candle does not satisfy all Eagle-style BUY filters.")

    st.subheader("Realistic Backtest (entry -> SL/Target/Time-exit)")
    trades = backtest_realistic(d, hold_days=hold_days, rr=2.0)
    if len(trades):
        win_rate = (trades["Return %"] > 0).mean() * 100
        avg_ret = trades["Return %"].mean()
        compounded = (1 + trades["Return %"] / 100).prod() - 1
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Trades", str(len(trades)))
        q2.metric("Win Rate", f"{win_rate:.1f}%")
        q3.metric("Avg Return / Trade", f"{avg_ret:.2f}%")
        q4.metric("Compounded Return", f"{compounded*100:.1f}%")
        st.dataframe(trades.sort_values("Entry date", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.warning("Selected history me qualifying BUY signals nahi mile.")

else:  # Screener
    st.sidebar.caption("Default watchlist ya apni comma-separated list daalein (Nifty 200/500 bhi chalega)")
    custom = st.sidebar.text_area("Tickers (NSE, comma-separated)", "")
    tickers = [t.strip().upper() for t in custom.split(",") if t.strip()] if custom.strip() else DEFAULT_WATCHLIST
    tickers = [t if t.endswith(".NS") else t + ".NS" for t in tickers]

    st.subheader(f"Screener - {len(tickers)} stocks")
    if len(tickers) > 250:
        st.warning("250 se zyada tickers ek saath scan karna free hosting pe unreliable ho sakta hai. "
                   "Behtar hoga list ko chhote batches (jaise 150-200) mein todo.")

    with st.spinner(f"Fetching data for {len(tickers)} tickers..."):
        batch = get_batch_data(tuple(tickers), period)

    results = []
    for t in tickers:
        df = batch.get(t, pd.DataFrame())
        r = score_from_df(t, df)
        if r is not None:
            results.append(r)

    missing = len(tickers) - len(results)
    if missing:
        st.caption(f"{missing} ticker(s) skip hue (invalid symbol ya kaafi history nahi mili).")

    if results:
        res_df = pd.DataFrame(results).sort_values(["Score", "RSI"], ascending=[False, False])
        buys = res_df[res_df["Signal"] == "BUY"]
        st.metric("BUY signals found", len(buys))
        st.dataframe(res_df, use_container_width=True, hide_index=True)
    else:
        st.error("Koi data nahi mila. Tickers check karein ya thodi der baad try karein.")

st.divider()
st.caption("Research/paper-trading tool. Signals are not guaranteed and are not investment advice.")
