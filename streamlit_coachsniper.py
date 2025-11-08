# streamlit_coachsniper.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import random

# ============================================================
#  Coach Sniper – Scanner S&P 500 (Heikin Ashi, Yahoo Finance)
#  Version: Ichimoku + RSI50 + Williams %R + Volume Oscillator
# ============================================================

st.set_page_config(page_title="Coach Swing – Heikin Ashi Scanner S&P 500", layout="wide")
st.title("🧭 Coach Swing – Scanner S&P 500 (Heikin Ashi)")

# ---------------------------------------------
# Heikin Ashi conversion
# ---------------------------------------------
def to_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit un DataFrame OHLC en Heikin Ashi.
    Attend des colonnes: Open, High, Low, Close.
    """
    df = df.copy()

    # HA-Close = (O+H+L+C)/4
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4

    # HA-Open = (prev_HA_Open + prev_HA_Close)/2 ; pour la première barre, (O+C)/2
    ha_open = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2

    # HA-High = max(High, HA-Open, HA-Close)
    ha_high = pd.concat([df["High"], ha_open, ha_close], axis=1).max(axis=1)

    # HA-Low = min(Low, HA-Open, HA-Close)
    ha_low = pd.concat([df["Low"], ha_open, ha_close], axis=1).min(axis=1)

    out = df.copy()
    out["Open"]  = ha_open
    out["High"]  = ha_high
    out["Low"]   = ha_low
    out["Close"] = ha_close
    return out

# ---------------------------------------------
# S&P 500 constituents (Wikipedia or fallback)
# ---------------------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_sp500_constituents():
    """
    Récupère la table du S&P 500 depuis Wikipedia avec un User-Agent.
    Fallback possible via st.secrets["SP500_CSV_URL"] (CSV avec colonnes 'Symbol' & 'Security').
    Retourne (df, tickers_yf).
    """
    # Fallback CSV
    csv_url = st.secrets.get("SP500_CSV_URL")
    if csv_url:
        try:
            df = pd.read_csv(csv_url)
            if "Symbol" not in df.columns or "Security" not in df.columns:
                raise ValueError("Le CSV doit contenir 'Symbol' et 'Security'")
            df["Symbol_yf"] = df["Symbol"].astype(str).replace(".", "-", regex=False)
            df = df.rename(
                columns={
                    "Security": "Company",
                    "GICS Sector": "Sector",
                    "GICS Sub-Industry": "SubIndustry",
                    "Headquarters Location": "HQ",
                    "Date first added": "DateAdded",
                }
            )
            return df, df["Symbol_yf"].tolist()
        except Exception as e:
            st.warning(f"CSV fallback échec ({e}). On tente Wikipedia…")

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0; +https://streamlit.io)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    last_err = None
    for _ in range(3):
        try:
            resp = requests.get(WIKI_URL, headers=headers, timeout=20)
            resp.raise_for_status()
            tables = pd.read_html(resp.text)
            df = tables[0].copy()
            df["Symbol_yf"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)
            df = df.rename(
                columns={
                    "Security": "Company",
                    "GICS Sector": "Sector",
                    "GICS Sub-Industry": "SubIndustry",
                    "Headquarters Location": "HQ",
                    "Date first added": "DateAdded",
                }
            )
            return df, df["Symbol_yf"].tolist()
        except Exception as e:
            last_err = e
            time.sleep(1.2 + random.random())

    raise RuntimeError(f"Échec de récupération du S&P 500 sur Wikipedia : {last_err}")

# ---------------------------------------------
# Utils & indicateurs
# ---------------------------------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """a croise au-dessus de b sur cette barre."""
    return (a > b) & (a.shift(1) <= b.shift(1))

def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    """a croise au-dessous de b sur cette barre."""
    return (a < b) & (a.shift(1) >= b.shift(1))

def cross_recent(cross: pd.Series, lookback: int = 3) -> pd.Series:
    """Vrai si le croisement a eu lieu dans les 0..lookback dernières barres."""
    out = cross.copy().astype(bool).fillna(False)
    for i in range(1, lookback + 1):
        out = out | cross.shift(i).fillna(False)
    return out

def ichimoku_components(high: pd.Series, low: pd.Series, len_tenkan=9, len_kijun=26, len_senkou_b=52):
    tenkan = (high.rolling(len_tenkan).max() + low.rolling(len_tenkan).min()) / 2.0
    kijun  = (high.rolling(len_kijun).max()  + low.rolling(len_kijun).min())  / 2.0
    spanA  = (tenkan + kijun) / 2.0
    spanB  = (high.rolling(len_senkou_b).max() + low.rolling(len_senkou_b).min()) / 2.0
    return tenkan, kijun, spanA, spanB

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    denom = (hh - ll).replace(0, np.nan)
    wr = -100 * (hh - close) / denom
    return wr.replace([np.inf, -np.inf], np.nan).fillna(method="bfill").fillna(method="ffill")

def volume_oscillator(volume: pd.Series, fast=5, slow=20) -> pd.Series:
    ema_fast = ema(volume, fast)
    ema_slow = ema(volume, slow)
    with np.errstate(divide="ignore", invalid="ignore"):
        vo = (ema_fast - ema_slow) / ema_slow * 100.0
    return vo.replace([np.inf, -np.inf], 0).fillna(0)

# ---------------------------------------------
# Nouvelle stratégie : Ichimoku + RSI50 + %R + VolumeOsc
# ---------------------------------------------
def swing_signals_strategy(df: pd.DataFrame,
                           mode: str = "Balanced",
                           len_tenkan: int = 9,
                           len_kijun: int = 26,
                           len_senkou_b: int = 52,
                           rsi_len: int = 14,
                           use_rsi_filter: bool = True,
                           wr_len: int = 14,
                           wr_recent_bars: int = 3,
                           vo_fast: int = 5,
                           vo_slow: int = 20):
    """
    Retourne (buy_now, sell_now, last_values_dict) à la dernière barre selon la logique :
      - Tendance : Ichimoku (above/below cloud + croisement Tenkan/Kijun selon le mode)
      - Filtre RSI : au-dessus/au-dessous de 50 (optionnel)
      - Timing : Williams %R (croix -80 / -20, up/down turning, ou agressif)
      - Volume : Volume Oscillator > 0 (ou assoupli selon le mode)
    """
    if df.empty:
        return False, False, {}

    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(0.0, index=df.index)

    # Ichimoku
    tenkan, kijun, spanA, spanB = ichimoku_components(h, l, len_tenkan, len_kijun, len_senkou_b)
    upperCloud = pd.concat([spanA, spanB], axis=1).max(axis=1)
    lowerCloud = pd.concat([spanA, spanB], axis=1).min(axis=1)

    aboveCloud = c > upperCloud
    belowCloud = c < lowerCloud
    bullTK = tenkan > kijun
    bearTK = tenkan < kijun

    # RSI
    rsi = rsi_wilder(c, rsi_len)

    # Williams %R
    wr = williams_r(h, l, c, wr_len)
    wr_cross_up_80 = crossover(wr, pd.Series(-80.0, index=wr.index))
    wr_cross_dn_20 = crossunder(wr, pd.Series(-20.0, index=wr.index))
    wr_up_turning  = (wr > -80) & (wr > wr.shift(1)) & (wr.shift(1) > wr.shift(2))
    wr_dn_turning  = (wr < -20) & (wr < wr.shift(1)) & (wr.shift(1) < wr.shift(2))
    wr_up_recent   = cross_recent(wr_cross_up_80, wr_recent_bars)
    wr_dn_recent   = cross_recent(wr_cross_dn_20, wr_recent_bars)

    # Volume Oscillator
    vo = volume_oscillator(v, vo_fast, vo_slow)

    # Logique par mode
    if mode == "Strict":
        longTrendOK  = aboveCloud & bullTK
        shortTrendOK = belowCloud & bearTK
        rsiBullOK    = (rsi > 50) if use_rsi_filter else pd.Series(True, index=rsi.index)
        rsiBearOK    = (rsi < 50) if use_rsi_filter else pd.Series(True, index=rsi.index)
        wrLongOK     = wr_cross_up_80
        wrShortOK    = wr_dn_recent  # = cross sous -20 récent
        voLongOK     = vo > 0
        voShortOK    = vo < 0

    elif mode == "Aggressive":
        longTrendOK  = c > kijun
        shortTrendOK = c < kijun
        rsiBullOK    = (rsi > 45) if use_rsi_filter else pd.Series(True, index=rsi.index)
        rsiBearOK    = (rsi < 55) if use_rsi_filter else pd.Series(True, index=rsi.index)
        wrLongOK     = (wr > -60) & (wr > wr.shift(1))
        wrShortOK    = (wr < -40) & (wr < wr.shift(1))
        voLongOK     = vo >= -2
        voShortOK    = vo <= 2

    else:  # Balanced (par défaut)
        longTrendOK  = (c > kijun) & (aboveCloud | (spanA > spanB))
        shortTrendOK = (c < kijun) & (belowCloud | (spanA < spanB))
        rsiBullOK    = (rsi > 48) if use_rsi_filter else pd.Series(True, index=rsi.index)
        rsiBearOK    = (rsi < 52) if use_rsi_filter else pd.Series(True, index=rsi.index)
        wrLongOK     = wr_up_recent | wr_up_turning
        wrShortOK    = wr_dn_recent | wr_dn_turning
        voLongOK     = vo >= -1
        voShortOK    = vo <= 1

    buyCond  = longTrendOK & rsiBullOK & wrLongOK  & voLongOK
    sellCond = shortTrendOK & rsiBearOK & wrShortOK & voShortOK

    buy_now  = bool(buyCond.iloc[-1])
    sell_now = bool(sellCond.iloc[-1])

    last = {
        "tenkan": float(tenkan.iloc[-1]) if not np.isnan(tenkan.iloc[-1]) else None,
        "kijun":  float(kijun.iloc[-1])  if not np.isnan(kijun.iloc[-1])  else None,
        "spanA":  float(spanA.iloc[-1])  if not np.isnan(spanA.iloc[-1])  else None,
        "spanB":  float(spanB.iloc[-1])  if not np.isnan(spanB.iloc[-1])  else None,
        "RSI":    float(rsi.iloc[-1])    if not np.isnan(rsi.iloc[-1])    else None,
        "WR":     float(wr.iloc[-1])     if not np.isnan(wr.iloc[-1])     else None,
        "VO":     float(vo.iloc[-1])     if not np.isnan(vo.iloc[-1])     else None,
        "Close":  float(c.iloc[-1])      if not np.isnan(c.iloc[-1])      else None,
    }
    return buy_now, sell_now, last

# ---------------------------------------------
# Yahoo Finance data loader
# ---------------------------------------------
@st.cache_data(show_spinner=False)
def download_bars(tickers: list[str], period: str, interval: str) -> dict:
    """
    Télécharge OHLCV pour plusieurs tickers et convertit en Heikin Ashi.
    Retourne dict[ticker] -> DataFrame(OHLCV HA).
    """
    if not tickers:
        return {}

    df = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    out: dict[str, pd.DataFrame] = {}

    if isinstance(df.columns, pd.MultiIndex):
        # Multi-ticker
        base_names = df.columns.get_level_values(0).unique()
        for t in tickers:
            if t not in base_names:
                continue
            dft = df[t].dropna(how="all")
            if dft.empty:
                continue
            # Standardize column capitalization
            dft = dft.rename(columns={c: c.capitalize() for c in dft.columns})
            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in dft.columns]
            dft = dft[keep]
            out[t] = to_heikin_ashi(dft)
    else:
        # Single-ticker
        dft = df.dropna(how="all")
        dft = dft.rename(columns={c: c.capitalize() for c in dft.columns})
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in dft.columns]
        dft = dft[keep]
        out[tickers[0]] = to_heikin_ashi(dft)

    return out

# ---------------------------------------------
# UI – Filtres & contrôles
# ---------------------------------------------
with st.spinner("Chargement de la liste S&P 500 depuis Wikipedia…"):
    sp_df, all_tickers = get_sp500_constituents()

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
with c1:
    sectors = sorted(sp_df["Sector"].dropna().unique().tolist())
    sector_sel = st.multiselect("Secteurs", sectors, [])
with c2:
    interval = st.selectbox("Intervalle", ["1d", "1h", "30m", "15m"], index=0)
with c3:
    limit = st.number_input("Nombre de tickers", min_value=5, max_value=500, value=80, step=5)
with c4:
    search = st.text_input("Recherche (ticker/nom)", "").strip().lower()

# Période par défaut selon l'intervalle
period_map = {"1d": "2y", "1h": "180d", "30m": "60d", "15m": "30d"}
period = period_map.get(interval, "2y")

if interval != "1d" and limit > 120:
    st.warning("En intraday, limite ≤ 120 tickers pour de bonnes performances.")

# --- Paramètres stratégie (sidebar) ---
st.sidebar.header("⚙️ Paramètres stratégie")
mode = st.sidebar.selectbox("Mode", ["Strict", "Balanced", "Aggressive"], index=1)
use_rsi_filter = st.sidebar.checkbox("Filtre RSI 50", value=True)
rsi_len = st.sidebar.number_input("RSI length", min_value=5, max_value=50, value=14, step=1)
wr_len = st.sidebar.number_input("Williams %R length", min_value=5, max_value=50, value=14, step=1)
wr_recent_bars = st.sidebar.number_input("Fenêtre %R récente (barres)", min_value=1, max_value=10, value=3, step=1)
vo_fast = st.sidebar.number_input("Volume Osc fast", min_value=2, max_value=50, value=5, step=1)
vo_slow = st.sidebar.number_input("Volume Osc slow", min_value=5, max_value=200, value=20, step=1)
len_tenkan = st.sidebar.number_input("Tenkan", min_value=3, max_value=50, value=9, step=1)
len_kijun = st.sidebar.number_input("Kijun", min_value=10, max_value=100, value=26, step=1)
len_senkou_b = st.sidebar.number_input("Senkou Span B", min_value=20, max_value=200, value=52, step=1)

# --- Sélection tickers ---
base = sp_df.copy()
if sector_sel:
    base = base[base["Sector"].isin(sector_sel)]
if search:
    base = base[
        base["Company"].str.lower().str.contains(search)
        | base["Symbol_yf"].str.lower().str.contains(search)
    ]

sel_tickers = base["Symbol_yf"].head(int(limit)).tolist()
st.caption(
    f"{len(sel_tickers)} tickers sélectionnés / {len(all_tickers)} au total – Heikin Ashi – Intervalle {interval}, Période {period}"
)

if not sel_tickers:
    st.info("Aucun ticker sélectionné. Ajuste les filtres.")
    st.stop()

with st.spinner("Téléchargement des chandelles (Yahoo Finance)…"):
    bars = download_bars(sel_tickers, period=period, interval=interval)

# ---------------------------------------------
# Calcul des signaux (nouvelle stratégie)
# ---------------------------------------------
results = []
for t in sel_tickers:
    dft = bars.get(t)
    if dft is None or len(dft) < max(200, len_senkou_b + 10):
        continue

    buy_now, sell_now, last = swing_signals_strategy(
        dft,
        mode=mode,
        len_tenkan=len_tenkan,
        len_kijun=len_kijun,
        len_senkou_b=len_senkou_b,
        rsi_len=rsi_len,
        use_rsi_filter=use_rsi_filter,
        wr_len=wr_len,
        wr_recent_bars=wr_recent_bars,
        vo_fast=vo_fast,
        vo_slow=vo_slow,
    )

    results.append(
        {
            "Ticker": t,
            "Company": base.loc[base["Symbol_yf"] == t, "Company"].values[0] if not base.empty else t,
            "Sector": base.loc[base["Symbol_yf"] == t, "Sector"].values[0] if not base.empty else None,
            "Buy": buy_now,
            "Sell": sell_now,
            "Close": float(last.get("Close")) if last.get("Close") is not None else None,
            "RSI": float(last.get("RSI")) if last.get("RSI") is not None else None,
            "WR": float(last.get("WR")) if last.get("WR") is not None else None,
            "VO": float(last.get("VO")) if last.get("VO") is not None else None,
        }
    )

res_df = pd.DataFrame(results)
if res_df.empty:
    st.warning("Aucun résultat (pas assez de données ou filtres trop stricts).")
    st.stop()

# ---------------------------------------------
# Affichage & export
# ---------------------------------------------
colA, colB, colC = st.columns([1, 1, 2])
with colA:
    show = st.selectbox("Afficher", ["Tous", "Buy seulement", "Sell seulement"], index=0)
with colB:
    sort_by = st.selectbox("Trier par", ["Buy", "Sell", "Close", "Ticker", "Sector"])
with colC:
    ascending = st.checkbox("Tri ascendant", value=False)

if show == "Buy seulement":
    res_view = res_df[res_df["Buy"]]
elif show == "Sell seulement":
    res_view = res_df[res_df["Sell"]]
else:
    res_view = res_df

res_view = res_view.sort_values(by=sort_by, ascending=ascending, na_position="last")
st.dataframe(res_view, use_container_width=True)

csv = res_view.to_csv(index=False).encode("utf-8")
st.download_button("💾 Télécharger les signaux (CSV)", data=csv, file_name="coach_swing_sp500_ha_signals.csv", mime="text/csv")

st.markdown(
    """
**Important :** Toutes les analyses sont calculées en **Heikin Ashi** (et non en chandeliers classiques).

**Stratégie utilisée : Ichimoku + RSI50 + Williams %R + Volume Oscillator**

- **Tendance (Ichimoku)**  
  - *Strict* : Close > nuage et Tenkan > Kijun (inverse pour short)  
  - *Balanced* : Close > Kijun et (au-dessus du nuage ou SpanA>SpanB)  
  - *Aggressive* : Close > Kijun (inverse pour short)

- **Filtre RSI (optionnel)** :  
  - Strict: >50 / <50 ; Balanced: >48 / <52 ; Aggressive: >45 / <55

- **Timing Williams %R** :  
  - Strict: croisement au-dessus de -80 (ou sous -20 pour short)  
  - Balanced: croix récente OU retournement (2 barres)  
  - Aggressive: pente positive au-dessus de -60 (ou négative sous -40)

- **Volume Oscillator (VO)** :  
  - Strict: VO>0 / VO<0 ; Balanced: ≥-1 / ≤1 ; Aggressive: ≥-2 / ≤2

Tu peux ajuster tous les paramètres dans la **sidebar**.
"""
)
