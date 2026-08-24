import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Chartview Quant", page_icon="📈", layout="wide")

st.title("📈 Chartview Quant")
st.caption("Eagle-style Quant Research • Screener • Backtest • Paper Trading")

# -------------------- helpers --------------------
@st.cache_data(ttl=900)
def get_data(ticker, period="2y"):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

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
    return x

# -------------------- sidebar --------------------
st.sidebar.header("Controls")
ticker = st.sidebar.text_input("NSE ticker", "RELIANCE.NS")
period = st.sidebar.selectbox("History", ["1y", "2y", "5y"], index=1)
capital = st.sidebar.number_input("Paper capital (₹)", min_value=1000.0, value=100000.0, step=5000.0)
risk_pct = st.sidebar.slider("Risk per trade %", 0.5, 3.0, 1.0, 0.5)

df = get_data(ticker, period)

if df.empty:
    st.error("Data nahi mila. NSE ticker format use karein, example: RELIANCE.NS")
    st.stop()

d = prepare(df)
last = d.iloc[-1]

# -------------------- top metrics --------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", f"₹{last['Close']:,.2f}")
c2.metric("RSI", f"{last['RSI']:.1f}")
c3.metric("ADX", f"{last['ADX']:.1f}")
c4.metric("Volume", f"{last['VolRatio']:.2f}x")
c5.metric("Signal", "🟢 BUY" if bool(last["BUY"]) else "⚪ WAIT")

# -------------------- dashboard --------------------
st.subheader("All-In-One Market Dashboard")

rows = [
    ("Direction", "BULLISH" if last["EMA20"] > last["EMA50"] else "BEARISH"),
    ("Trend", "UP" if last["+DI"] > last["-DI"] else "DOWN"),
    ("Trend Strength", "STRONG" if last["ADX"] >= 25 else "NEUTRAL/WEAK"),
    ("Momentum", "STRONG" if last["RSI"] >= 60 else ("WEAK" if last["RSI"] <= 40 else "NEUTRAL")),
    ("Volume", "STRONG" if last["VolRatio"] >= 1.5 else ("WEAK" if last["VolRatio"] < .8 else "NEUTRAL")),
    ("Breakout", "BREAKOUT" if last["Breakout"] else "NONE"),
]
dash = pd.DataFrame(rows, columns=["Factor", "Status"])
st.dataframe(dash, use_container_width=True, hide_index=True)

# -------------------- chart --------------------
st.subheader("Price & EMA")
chart = d[["Close", "EMA20", "EMA50", "EMA200"]].tail(250)
st.line_chart(chart, use_container_width=True)

# -------------------- trade plan --------------------
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
    a,b,c,d4 = st.columns(4)
    a.metric("Entry", f"₹{entry:,.2f}")
    b.metric("Stop Loss", f"₹{sl:,.2f}")
    c.metric("Target 1 (1:2)", f"₹{target:,.2f}")
    d4.metric("Shares (paper)", f"{max(shares,0)}")
else:
    st.info("WAIT — current candle does not satisfy all Eagle-style BUY filters.")

# -------------------- backtest --------------------
st.subheader("Simple Historical Backtest")
bt = d.copy()
bt["Entry"] = bt["BUY"]
bt["Ret"] = bt["Close"].pct_change().shift(-1)
trades = bt.loc[bt["Entry"], "Ret"].dropna()
if len(trades):
    win = (trades > 0).mean() * 100
    total = (1 + trades).prod() - 1
    q1,q2,q3 = st.columns(3)
    q1.metric("Trades", str(len(trades)))
    q2.metric("Win Rate", f"{win:.1f}%")
    q3.metric("Next-day compounded return", f"{total*100:.1f}%")
    st.dataframe(pd.DataFrame({"Trade date": trades.index, "Next-day return %": trades.values*100}).tail(50),
                 use_container_width=True, hide_index=True)
else:
    st.warning("Selected history me qualifying BUY signals nahi mile.")

st.divider()
st.caption("Research/paper-trading tool. Signals are not guaranteed and are not investment advice.")
