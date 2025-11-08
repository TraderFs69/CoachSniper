# streamlit_coachsniper.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import random
from typing import List

# ============================================================
#  Coach Sniper – S&P 500 (Yahoo Finance, 1D)
#  Stratégie: Ichimoku + RSI50 + Williams %R + Volume Oscillator
#  Sidebar: Mode (Strict/Balanced/Aggressive) + Filtre RSI 50
# ============================================================

st.set_page_config(page_title="Coach Sniper – Scanner S&P 500 (1D)", layout="wide")
st.title("🧭 Coach Swing – Scanner S&P 500 (1D)")

# ---------------------------------------------
# Heikin Ashi conversion
# ---------------------------------------------
def to_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit un DataFrame OHLC en Heikin Ashi (colonnes: Open, High, Low, Close)."""
    df = df.copy()
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ha_high = pd.concat([df["High"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([df["Low"],  ha_open, ha_close], axis=1).min(axis=1)

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
    Récupère la table du S&P 500 depuis Wikipedia (ou fallback CSV via secrets).
    Retourne (df, tickers_yf).
    """
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
            # IMPORTANT: éviter lxml → flavor="bs4"
            tables = pd.read_html(resp.text, flavor="bs4")
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
    return rsi.replace([np.inf, -np.inf], np.nan).fillna(0)

def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))

def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))

def cross_recent(cross: pd.Series, lookback: int = 3) -> pd.Series:
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
    return pd.Series(np.where(np.isfinite(vo), vo, 0.0), index=volume.index).fillna(0)

# ---------------------------------------------
# Stratégie (paramètres FIXES sauf Mode & Filtre RSI)
# ---------------------------------------------
RSI_LEN = 14
WR_LEN = 14
WR_RECENT = 14
VO_FAST = 5
VO_SLOW = 20
TENKAN = 9
KIJUN = 26
SENKOU_B = 52

def swing_signals_strategy(df: pd.DataFrame,
                           mode: str = "Balanced",
                           use_rsi_filter: bool = True):
    """
    Retourne (buy_now, sell_now, last_values_dict) à la dernière barre.
    Paramètres internes fixes (RSI 14, W%R 14, Fenêtre 14, VO 5/20, Ichi 9/26/52).
    """
    if df.empty:
        return False, False, {}

    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(0.0, index=df.index)

    # Ichimoku
    tenkan, kijun, spanA, spanB = ichimoku_components(h, l, TENKAN, KIJUN, SENKOU_B)
    upperCloud = pd.concat([spanA, spanB], axis=1).max(axis=1)
    lowerCloud = pd.concat([spanA, spanB], axis=1).min(axis=1)
    aboveCloud = c > upperCloud
    belowCloud = c < lowerCloud
    bullTK = tenkan > kijun
    bearTK = tenkan < kijun

    # RSI
    rsi = rsi_wilder(c, RSI_LEN)

    # Williams %R
    wr = williams_r(h, l, c, WR_LEN)
    wr_cross_up_80 = crossover(wr, pd.Series(-80.0, index=wr.index))
    wr_cross_dn_20 = crossunder(wr, pd.Series(-20.0, index=wr.index))
    wr_up_turning  = (wr > -80) & (wr > wr.shift(1)) & (wr.shift(1) > wr.shift(2))
    wr_dn_turning  = (wr < -20) & (wr < wr.shift(1)) & (wr.shift(1) < wr.shift(2))
    wr_up_recent   = cross_recent(wr_cross_up_80, WR_RECENT)
    wr_dn_recent   = cross_recent(wr_cross_dn_20, WR_RECENT)

    # Volume Oscillator
    vo = volume_oscillator(v, VO_FAST, VO_SLOW)

    # Logique par mode
    if mode == "Strict":
        longTrendOK  = aboveCloud & bullTK
        shortTrendOK = belowCloud & bearTK
        rsiBullOK    = (rsi > 50) if use_rsi_filter else pd.Series(True, index=rsi.index)
        rsiBearOK    = (rsi < 50) if use_rsi_filter else pd.Series(True, index=rsi.index)
        wrLongOK     = wr_up_recent
        wrShortOK    = wr_dn_recent
        voLongOK     = vo > 0
        voShortOK    = vo < 0

    elif mode == "Aggressive":
        longTrendOK  = c > kijun
        shortTrendOK = c < kijun
        rsiBullOK    = (rsi > 50) if use_rsi_filter else pd.Series(True, index=rsi.index)
        rsiBearOK    = (rsi < 50) if use_rsi_filter else pd.Series(True, index=rsi.index)
        wrLongOK     = (wr > -60) & (wr > wr.shift(1))
        wrShortOK    = (wr < -40) & (wr < wr.shift(1))
        voLongOK     = vo >= -2
        voShortOK    = vo <= 2

    else:  # Balanced
        longTrendOK  = (c > kijun) & (aboveCloud | (spanA > spanB))
        shortTrendOK = (c < kijun) & (belowCloud | (spanA < spanB))
        rsiBullOK    = (rsi > 50) if use_rsi_filter else pd.Series(True, index=rsi.index)
        rsiBearOK    = (rsi < 50) if use_rsi_filter else pd.Series(True, index=rsi.index)
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
# Yahoo Finance loader – anti rate-limit (par ticker)
# ---------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_one_ticker(ticker: str, period: str, interval: str, attempts: int = 6) -> pd.DataFrame | None:
    """
    Télécharge un ticker avec backoff exponentiel + jitter.
    Renvoie un DataFrame OHLCV ou None.
    """
    for k in range(attempts):
        try:
            df = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=False,  # pas de multi-threads => moins d'erreurs réseau
            )
            df = df.dropna(how="all")
            if not df.empty:
                df = df.rename(columns={c: c.capitalize() for c in df.columns})
                keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                return df[keep]
        except Exception as e:
            msg = str(e)
            # Si rate-limité, on attend plus longtemps (backoff ci-dessous)
            if "Rate limited" in msg or "Too Many Requests" in msg or "YFRateLimitError" in msg:
                pass
        # Backoff exponentiel + jitter
        sleep_s = (2 ** k) * 0.6 + random.uniform(0.0, 0.5)
        time.sleep(sleep_s)
    return None

@st.cache_data(show_spinner=False)
def download_bars(tickers: List[str], period: str, interval: str,
                  batch_size: int = 25, pause_between_calls: float = 0.6) -> dict:
    """
    Télécharge OHLCV par TICKER, en petits batches, avec pause et backoff.
    - batch_size: nb de tickers traités avant une pause supplémentaire
    - pause_between_calls: pause entre 2 tickers (anti-throttle)
    """
    out: dict[str, pd.DataFrame] = {}
    if not tickers:
        return out

    total = len(tickers)
    prog = st.progress(0.0)

    # Fallbacks de période (pour 1D seulement ici)
    period_candidates = [period, "1y", "6mo", "3mo"]

    for i, t in enumerate(tickers, 1):
        df_t = None
        for per in period_candidates:
            df_t = fetch_one_ticker(t, per, interval)
            if df_t is not None and not df_t.empty:
                break
        if df_t is not None and not df_t.empty:
            out[t] = df_t

        # Petite pause entre les tickers (anti “Too Many Requests”)
        time.sleep(pause_between_calls)

        # Pause supplémentaire entre batches
        if i % batch_size == 0:
            time.sleep(2.0 + random.uniform(0.0, 1.0))

        prog.progress(i / max(1, total))

    return out

# ---------------------------------------------
# UI – Filtres & contrôles (Intervalle 1D fixe)
# ---------------------------------------------
with st.spinner("Chargement de la liste S&P 500 depuis Wikipedia…"):
    sp_df, all_tickers = get_sp500_constituents()

c1, c2 = st.columns([1, 3])
with c1:
    sectors = sorted(sp_df["Sector"].dropna().unique().tolist())
    sector_sel = st.multiselect("Secteurs", sectors, [])
with c2:
    search = st.text_input("Recherche (ticker/nom)", "").strip().lower()

# Intervalle & période fixes
interval = "1d"
period = "2y"  # assez long pour swing

# --- Paramètres stratégie (sidebar minimaliste) ---
st.sidebar.header("⚙️ Stratégie")
mode = st.sidebar.selectbox("Mode", ["Strict", "Balanced", "Aggressive"], index=1)
use_rsi_filter = st.sidebar.checkbox("Filtre RSI 50", value=True)

# --- Options techniques utiles (anti-rate limit / rendu TV) ---
st.sidebar.header("⚙️ Options techniques")
use_heikin = st.sidebar.checkbox("Calculer en Heikin Ashi", value=True)
use_last_closed = st.sidebar.checkbox("Utiliser la dernière barre clôturée", value=True,
                                      help="Reproduit barstate.isconfirmed de TradingView")
batch_size = st.sidebar.number_input("Batch size (anti-rate limit)", 5, 100, 25, 5)
pause_between_calls = st.sidebar.number_input("Pause par ticker (sec)", 0.0, 3.0, 0.8, 0.1)

# --- Sélection tickers ---
base = sp_df.copy()
if sector_sel:
    base = base[base["Sector"].isin(sector_sel)]
if search:
    base = base[
        base["Company"].str.lower().str.contains(search)
        | base["Symbol_yf"].str.lower().str.contains(search)
    ]
limit = st.number_input("Nombre de tickers", min_value=5, max_value=500, value=60, step=5)
sel_tickers = base["Symbol_yf"].head(int(limit)).tolist()

st.caption(f"{len(sel_tickers)} tickers sélectionnés / {len(all_tickers)} – Intervalle 1D, Période {period}")

if not sel_tickers:
    st.info("Aucun ticker sélectionné. Ajuste les filtres.")
    st.stop()

with st.spinner("Téléchargement des chandelles (Yahoo Finance)…"):
    bars = download_bars(sel_tickers, period=period, interval=interval,
                         batch_size=int(batch_size), pause_between_calls=float(pause_between_calls))

valid = sum(1 for t in sel_tickers if bars.get(t) is not None and len(bars[t]) > 0)
st.caption(f"✅ Jeux de données Yahoo valides : {valid}/{len(sel_tickers)}")
if valid == 0:
    st.error("Yahoo n'a renvoyé aucune donnée. Réduis le nombre de tickers, augmente la pause, ou réessaie plus tard.")
    st.stop()

# ---------------------------------------------
# Calcul des signaux
# ---------------------------------------------
results = []
min_bars_needed = max(100, SENKOU_B + 30)  # suffisant pour SpanB(52) + marge
skipped_empty = 0
skipped_short = 0

for t in sel_tickers:
    dft_raw = bars.get(t)
    if dft_raw is None or len(dft_raw) == 0:
        skipped_empty += 1
        continue

    dft_src = to_heikin_ashi(dft_raw) if use_heikin else dft_raw
    dft_eval = dft_src.iloc[:-1] if use_last_closed and len(dft_src) > 1 else dft_src
    if len(dft_eval) < min_bars_needed:
        skipped_short += 1
        continue

    buy_now, sell_now, last = swing_signals_strategy(
        dft_eval,
        mode=mode,
        use_rsi_filter=use_rsi_filter,
    )

    last_close = float(dft_src["Close"].iloc[-1]) if len(dft_src) else None

    results.append(
        {
            "Ticker": t,
            "Company": base.loc[base["Symbol_yf"] == t, "Company"].values[0] if not base.empty else t,
            "Sector": base.loc[base["Symbol_yf"] == t, "Sector"].values[0] if not base.empty else None,
            "Buy": buy_now,
            "Sell": sell_now,
            "Close": last_close,
            "RSI": float(last.get("RSI")) if last.get("RSI") is not None else None,
            "WR": float(last.get("WR")) if last.get("WR") is not None else None,
            "VO": float(last.get("VO")) if last.get("VO") is not None else None,
        }
    )

res_df = pd.DataFrame(results)
if res_df.empty:
    st.warning(
        f"Aucun résultat. Tickers vides: {skipped_empty}, trop courts (<{min_bars_needed}): {skipped_short}. "
        "Réduis le nombre de tickers, augmente la pause par ticker, ou réessaie plus tard."
    )
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
st.download_button("💾 Télécharger (CSV)", data=csv, file_name="coach_swing_sp500_1d_signals.csv", mime="text/csv")

st.markdown(
    """
**Important :** Analyse par défaut en **Heikin Ashi** et sur la **dernière barre clôturée** (options sidebar).  
Stratégie **Ichimoku + RSI50 + Williams %R + Volume Oscillator**  
Paramètres fixes : RSI=14 · W%R=14 · Fenêtre %R=14 · VolumeOsc=5/20 · Ichimoku=9/26/52.  
Seuls **Mode** (Strict / Balanced / Aggressive) et **Filtre RSI 50** sont ajustables.
"""
)
