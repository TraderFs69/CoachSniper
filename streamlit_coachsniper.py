# streamlit_coachsniper_1d_yahoo_safe.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests, io, time, random
from typing import Dict, Tuple, List

# ==============================
# Réglages fixes (Yahoo only)
# ==============================
INTERVAL = "1d"
PERIOD   = "2y"

# Micro-chunks & backoff (anti rate-limit)
CHUNK            = 5        # <= 5 tickers par appel
BASE_SLEEP       = 3.0      # base du backoff exponentiel (sec)
MAX_BACKOFF_TRY  = 4        # 4 tentatives par batch / ticker
PAUSE_BETWEEN_OK = 1.5      # pause même quand OK (sec)

# Taille par vague (pagination UI)
DEFAULT_WAVE = 20

st.set_page_config(page_title="Coach Swing – S&P500 (Heikin Ashi, 1D • Yahoo Safe)", layout="wide")
st.title("🧭 Coach Swing – Scanner S&P 500 (Heikin Ashi, 1D • Yahoo Safe)")

# -----------------------------
# Heikin Ashi
# -----------------------------
def to_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ha_high = pd.concat([df["High"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([df["Low"],  ha_open, ha_close], axis=1).min(axis=1)
    out = df.copy()
    out["Open"], out["High"], out["Low"], out["Close"] = ha_open, ha_high, ha_low, ha_close
    return out

# -----------------------------
# Constituants S&P500
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_sp500_constituents():
    csv_url = st.secrets.get("SP500_CSV_URL")
    if csv_url:
        try:
            df = pd.read_csv(csv_url)
            if "Symbol" not in df.columns or "Security" not in df.columns:
                raise ValueError("CSV doit contenir 'Symbol' et 'Security'")
            df["Symbol_yf"] = df["Symbol"].astype(str).replace(".", "-", regex=False)
            df = df.rename(columns={
                "Security": "Company", "GICS Sector": "Sector", "GICS Sub-Industry": "SubIndustry",
                "Headquarters Location": "HQ", "Date first added": "DateAdded"
            })
            return df, df["Symbol_yf"].tolist()
        except Exception as e:
            st.warning(f"CSV fallback échec ({e}). On tente Wikipedia…)")

    headers = {"User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0; +https://streamlit.io)",
               "Accept-Language": "en-US,en;q=0.9"}
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    last_err = None
    for _ in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text), flavor="bs4")
            df = tables[0].copy()
            df["Symbol_yf"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)
            df = df.rename(columns={
                "Security": "Company", "GICS Sector": "Sector", "GICS Sub-Industry": "SubIndustry",
                "Headquarters Location": "HQ", "Date first added": "DateAdded"
            })
            return df, df["Symbol_yf"].tolist()
        except Exception as e:
            last_err = e
            time.sleep(1.0 + random.random())
    raise RuntimeError(f"Échec de récupération du S&P 500 : {last_err}")

# -----------------------------
# Indicateurs utilitaires
# -----------------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def rsi_wilder(close: pd.Series, length: int = 12) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0.0); loss = -d.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(0)

def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))

def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))

def cross_recent(cross: pd.Series, lookback: int = 3) -> pd.Series:
    out = cross.copy().astype(bool).fillna(False)
    for i in range(1, lookback+1):
        out = out | cross.shift(i).fillna(False)
    return out

def ichimoku_components(high: pd.Series, low: pd.Series, len_tenkan=9, len_kijun=26, len_senkou_b=52):
    tenkan = (high.rolling(len_tenkan).max() + low.rolling(len_tenkan).min())/2.0
    kijun  = (high.rolling(len_kijun).max()  + low.rolling(len_kijun).min()) /2.0
    spanA  = (tenkan + kijun)/2.0
    spanB  = (high.rolling(len_senkou_b).max() + low.rolling(len_senkou_b).min())/2.0
    return tenkan, kijun, spanA, spanB

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    denom = (hh - ll).replace(0, np.nan)
    wr = -100 * (hh - close) / denom
    return wr.replace([np.inf, -np.inf], np.nan).fillna(method="bfill").fillna(method="ffill")

def volume_oscillator(volume: pd.Series, fast=5, slow=20) -> pd.Series:
    ema_f = ema(volume, fast); ema_s = ema(volume, slow)
    with np.errstate(divide="ignore", invalid="ignore"):
        vo = (ema_f - ema_s) / ema_s * 100.0
    return pd.Series(np.where(np.isfinite(vo), vo, 0.0), index=volume.index).fillna(0)

# -----------------------------
# Stratégie Ichimoku (3 modes)
# -----------------------------
def coach_swing_signals(df: pd.DataFrame, mode: str = "Balanced", use_rsi50: bool = True):
    """Renvoie (buy_now, sell_now, last_dict) sur la dernière barre clôturée."""
    if df is None or df.empty:
        return False, False, {}
    data = df.iloc[:-1] if len(df) > 1 else df.copy()
    if len(data) < 82:
        return False, False, {}

    h = data["High"].astype(float)
    l = data["Low"].astype(float)
    c = data["Close"].astype(float)
    v = data["Volume"].astype(float) if "Volume" in data.columns else pd.Series(0.0, index=data.index)

    tenkan, kijun, spanA, spanB = ichimoku_components(h, l, 9, 26, 52)
    upperCloud = pd.concat([spanA, spanB], axis=1).max(axis=1)
    lowerCloud = pd.concat([spanA, spanB], axis=1).min(axis=1)
    aboveCloud = c > upperCloud
    belowCloud = c < lowerCloud
    bullTK = tenkan > kijun
    bearTK = tenkan < kijun

    rsi14 = rsi_wilder(c, 14)
    rsiBullOK = (rsi14 > 50) if use_rsi50 else pd.Series(True, index=rsi14.index)
    rsiBearOK = (rsi14 < 50) if use_rsi50 else pd.Series(True, index=rsi14.index)

    wr = williams_r(h, l, c, 14)
    wr_cross_up_80 = crossover(wr, pd.Series(-80.0, index=wr.index))
    wr_cross_dn_20 = crossunder(wr, pd.Series(-20.0, index=wr.index))
    wr_up_turning  = (wr > -80) & (wr > wr.shift(1)) & (wr.shift(1) > wr.shift(2))
    wr_dn_turning  = (wr < -20) & (wr < wr.shift(1)) & (wr.shift(1) < wr.shift(2))
    wr_up_recent   = cross_recent(wr_cross_up_80, 14)
    wr_dn_recent   = cross_recent(wr_cross_dn_20, 14)

    vo = volume_oscillator(v, 5, 20)

    if mode == "Strict":
        longTrendOK  = aboveCloud & bullTK
        shortTrendOK = belowCloud & bearTK
        wrLongOK, wrShortOK = wr_up_recent, wr_dn_recent
        voLongOK, voShortOK = vo > 0, vo < 0
    elif mode == "Aggressive":
        longTrendOK  = c > kijun
        shortTrendOK = c < kijun
        wrLongOK     = (wr > -60) & (wr > wr.shift(1))
        wrShortOK    = (wr < -40) & (wr < wr.shift(1))
        voLongOK, voShortOK = vo >= -2, vo <= 2
    else:  # Balanced
        longTrendOK  = (c > kijun) & (aboveCloud | (spanA > spanB))
        shortTrendOK = (c < kijun) & (belowCloud | (spanA < spanB))
        wrLongOK     = wr_up_recent | wr_up_turning
        wrShortOK    = wr_dn_recent | wr_dn_turning
        voLongOK, voShortOK = vo >= -1, vo <= 1

    buyCond  = longTrendOK  & rsiBullOK & wrLongOK  & voLongOK
    sellCond = shortTrendOK & rsiBearOK & wrShortOK & voShortOK

    buy_now, sell_now = bool(buyCond.iloc[-1]), bool(sellCond.iloc[-1])

    ema9 = ema(c, 9); ema20 = ema(c, 20); ema50 = ema(c, 50); ema200 = ema(c, 200)
    last = {
        "ema9": float(ema9.iloc[-1]) if len(ema9) else None,
        "ema20": float(ema20.iloc[-1]) if len(ema20) else None,
        "ema50": float(ema50.iloc[-1]) if len(ema50) else None,
        "ema200": float(ema200.iloc[-1]) if len(ema200) else None,
        "RSI": float(rsi14.iloc[-1]) if len(rsi14) else None,
        "WR": float(wr.iloc[-1]) if len(wr) else None,
        "VO": float(vo.iloc[-1]) if len(vo) else None,
    }
    return buy_now, sell_now, last

# -----------------------------
# Yahoo micro-chunks + backoff
# -----------------------------
def _call_yf(batch: List[str]):
    try:
        return yf.download(
            tickers=batch,
            period=PERIOD,
            interval=INTERVAL,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=False,   # évite les rafales internes
        )
    except Exception as e:
        if "Rate limited" in str(e) or "Too Many Requests" in str(e):
            return None
        raise

def _process_df(df, batch, out_dict):
    if df is None or df is False:
        return
    if isinstance(df.columns, pd.MultiIndex):
        base = df.columns.get_level_values(0).unique()
        for t in batch:
            if t not in base:
                continue
            dft = df[t].dropna(how="all")
            if dft.empty:
                continue
            dft = dft.rename(columns={c: c.capitalize() for c in dft.columns})
            keep = [c for c in ["Open","High","Low","Close","Volume"] if c in dft.columns]
            out_dict[t] = to_heikin_ashi(dft[keep])
    else:
        if len(batch) == 1:
            dft = df.dropna(how="all")
            if not dft.empty:
                dft = dft.rename(columns={c: c.capitalize() for c in dft.columns})
                keep = [c for c in ["Open","High","Low","Close","Volume"] if c in dft.columns]
                out_dict[batch[0]] = to_heikin_ashi(dft[keep])

@st.cache_data(show_spinner=False)
def download_bars_yahoo_safe(tickers: tuple[str, ...]) -> Dict[str, pd.DataFrame]:
    """
    Micro-chunks avec backoff exponentiel + retry individuel.
    ⚠️ Cache par liste de tickers (tuple trié) uniquement.
    """
    out: Dict[str, pd.DataFrame] = {}
    base_list = list(tickers)

    # 1) micro-chunks
    i = 0
    while i < len(base_list):
        batch = base_list[i:i+CHUNK]
        backoff = 0
        while True:
            df = _call_yf(batch)
            if df is not None and (not isinstance(df, bool)) and (not df.empty):
                _process_df(df, batch, out)
                break
            pause = BASE_SLEEP * (2 ** backoff)
            st.write(f"⏳ Yahoo backoff {pause:.0f}s (batch {i}-{i+len(batch)-1})")
            time.sleep(pause + random.random())
            backoff += 1
            if backoff >= MAX_BACKOFF_TRY:
                st.warning(f"Batch abandonné (rate limit) : {batch}")
                break
        time.sleep(PAUSE_BETWEEN_OK + random.random())
        i += CHUNK

    # 2) retry individuel sur les manquants
    missing = [t for t in base_list if t not in out]
    if missing:
        st.info(f"🔁 Retry individuel pour {len(missing)} tickers…")
    for t in missing:
        backoff = 0
        while True:
            df = _call_yf([t])
            if df is not None and (not isinstance(df, bool)) and (not df.empty):
                _process_df(df, [t], out)
                break
            pause = BASE_SLEEP * (2 ** backoff)
            time.sleep(pause + random.random())
            backoff += 1
            if backoff >= MAX_BACKOFF_TRY:
                st.error(f"❌ Échec final: {t}")
                break
        time.sleep(1.0 + random.random())

    return out

# -----------------------------
# UI – Filtres & contrôles
# -----------------------------
with st.spinner("Chargement de la liste S&P 500…"):
    sp_df, all_tickers = get_sp500_constituents()

# Dow 30 (test rapide)
DOW30 = ["AAPL","MSFT","JPM","UNH","GS","HD","MS","AMGN","CRM","MCD","CAT","HON","TRV","CVX",
         "PG","V","JNJ","BA","DIS","NKE","WMT","AXP","KO","IBM","MRK","CSCO","INTC","VZ","MMM","WBA"]

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    sectors = sorted(sp_df["Sector"].dropna().unique().tolist())
    sector_sel = st.multiselect("Secteurs", sectors, [])
with c2:
    limit = st.number_input("Nombre max de tickers", min_value=10, max_value=500, value=120, step=10)
with c3:
    search = st.text_input("Recherche (ticker/nom)", "").strip().lower()

st.sidebar.header("Stratégie")
mode = st.sidebar.selectbox("Mode", ["Balanced", "Strict", "Aggressive"], index=0)
use_rsi50 = st.sidebar.checkbox("Filtre RSI 50", value=True)

st.sidebar.header("Vague de scan (pagination)")
wave = st.sidebar.number_input("Taille de la vague (<= 20 conseillé)", 10, 60, DEFAULT_WAVE, 5)
offset = st.sidebar.number_input("Offset (départ)", 0, 500, 0, 1)
use_dow = st.sidebar.checkbox("Dow 30 (test rapide)", value=False)

st.caption(f"Intervalle: **{INTERVAL}**, Période: **{PERIOD}** — Données converties en **Heikin Ashi**")

# Filtrage tickers
if use_dow:
    base_list = [t for t in DOW30]
    base = sp_df[sp_df["Symbol_yf"].isin(base_list)].copy()
else:
    base = sp_df.copy()
    if sector_sel:
        base = base[base["Sector"].isin(sector_sel)]
    if search:
        base = base[
            base["Company"].str.lower().str.contains(search)
            | base["Symbol_yf"].str.lower().str.contains(search)
        ]
    base_list = base["Symbol_yf"].tolist()

# Limite globale
base_list = base_list[: int(limit)]
total = len(base_list)
st.caption(f"📈 Tickers filtrés: {total}")

# Pagination par vague
start = int(offset)
end = min(start + int(wave), total)
wave_list = base_list[start:end]
st.info(f"Vague: index {start} → {end-1}  |  {len(wave_list)} tickers (≤ 20 recommandé)")

# Bouton d'exécution
go = st.button("▶️ Scanner cette vague (Yahoo only)", type="primary")
if not go:
    st.stop()

# Téléchargement Yahoo (safe)
tickers_tuple = tuple(sorted(set(wave_list)))  # clé cache stable
with st.spinner("Téléchargement des chandelles (Yahoo, micro-chunks)…"):
    bars = download_bars_yahoo_safe(tickers_tuple)

valid = sum(1 for t in tickers_tuple if bars.get(t) is not None and len(bars[t]) > 0)
st.caption(f"✅ Jeux de données valides : {valid}/{len(tickers_tuple)}")
if valid == 0:
    st.error("Aucune donnée renvoyée par Yahoo pour cette vague. Diminue la vague, puis réessaie.")
    st.stop()

# -----------------------------
# Calcul des signaux
# -----------------------------
results = []
for t in tickers_tuple:
    dft = bars.get(t)
    if dft is None or len(dft) < 82:
        continue
    buy_now, sell_now, last = coach_swing_signals(dft, mode=mode, use_rsi50=use_rsi50)
    results.append({
        "Ticker": t,
        "Company": base.loc[base["Symbol_yf"] == t, "Company"].values[0] if not base.empty else t,
        "Sector":  base.loc[base["Symbol_yf"] == t, "Sector"].values[0] if not base.empty else None,
        "Buy":     buy_now,
        "Sell":    sell_now,
        "Close":   float(dft["Close"].iloc[-1]),
        "RSI":     last.get("RSI"),
        "WR":      last.get("WR"),
        "VO":      last.get("VO"),
    })

res_df = pd.DataFrame(results)
if res_df.empty:
    st.warning("Aucun résultat dans cette vague (filtres stricts ou données manquantes).")
    st.stop()

# -----------------------------
# Affichage & export
# -----------------------------
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
st.download_button("💾 Télécharger (CSV)", data=csv,
                   file_name=f"coach_swing_yahoo_1d_wave_{start}_{end-1}.csv", mime="text/csv")

st.markdown("""
**Conseils anti rate-limit (Yahoo only) :**
- Garde la **vague ≤ 20** (ex. 20, puis offset +20, etc.).
- Laisse **quelques dizaines de secondes** entre 2 vagues consécutives si des “Failed downloads” apparaissent.
- Les micro-chunks (5) + backoff exponentiel sont intégrés ; les tickers manquants sont réessayés **un par un**.
""")
