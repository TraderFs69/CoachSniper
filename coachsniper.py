# coachsniper.py
import os, io, time, random, datetime as dt, requests
from typing import Dict, Tuple, List, Optional

import streamlit as st
import pandas as pd
import numpy as np

# ==============================
# Réglages (Polygon)
# ==============================
INTERVAL = "1d"    # logique de titres & signaux = daily
YEARS    = 2       # 2 ans d'historique
ADJUSTED = True    # ajusté (splits/div)
LIMIT    = 50000   # large pour couvrir toute la plage

# Micro-chunks & backoff (réseau)
CHUNK            = 8
BASE_SLEEP       = 1.2
MAX_BACKOFF_TRY  = 4
PAUSE_BETWEEN_OK = 0.6

# Taille par vague (pagination UI)
DEFAULT_WAVE = 20

# ==============================
# Setup
# ==============================
st.set_page_config(page_title="Coach Swing – S&P500 (Heikin Ashi, 1D • Polygon)", layout="wide")
st.title("🧭 Coach Swing – Scanner S&P 500 (Heikin Ashi, 1D • Polygon)")

# Clé API Polygon (env ou secrets)
POLY = os.getenv("POLYGON_API_KEY") or st.secrets.get("POLYGON_API_KEY")
if not POLY:
    st.error("⚠️ POLYGON_API_KEY manquant. Ajoute-le dans `.env` (POLYGON_API_KEY=...) ou dans `st.secrets`.")
    st.stop()

# ==============================
# Heikin Ashi
# ==============================
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

# ==============================
# Normalisation tableau S&P 500
# ==============================
def _normalize_sp500_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise un DataFrame S&P 500 quelle que soit la forme des colonnes.
    Renvoie un df avec au minimum :
      - Symbol
      - Company (si trouvable)
      - Sector (si trouvable)
      - SubIndustry (si trouvable)
      - HQ (si trouvable)
      - DateAdded (si trouvable)
    """
    df = df_raw.copy()
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}

    # --- Symbol / Ticker ---
    symbol_col = None
    for key, col in lower_map.items():
        if "symbol" in key or "ticker" in key:
            symbol_col = col
            break
    if not symbol_col:
        raise ValueError(f"Aucune colonne de symbole trouvée dans le tableau S&P 500. Colonnes = {cols}")

    # --- Company / Security / Name ---
    company_col = None
    for key, col in lower_map.items():
        if "security" in key or "company" in key or "name" in key:
            company_col = col
            break

    # --- Sector ---
    sector_col = None
    for key, col in lower_map.items():
        if "gics sector" in key or key == "sector" or "sector" in key:
            sector_col = col
            break

    # --- Sub-Industry ---
    sub_col = None
    for key, col in lower_map.items():
        if "gics sub-industry" in key or "sub-industry" in key or "sub industry" in key:
            sub_col = col
            break

    # --- Headquarters / Location ---
    hq_col = None
    for key, col in lower_map.items():
        if "headquarters" in key or "location" in key:
            hq_col = col
            break

    # --- Date first added / Date added ---
    date_col = None
    for key, col in lower_map.items():
        if "date first added" in key or "date added" in key:
            date_col = col
            break

    rename_map = {symbol_col: "Symbol"}
    if company_col: rename_map[company_col] = "Company"
    if sector_col:  rename_map[sector_col]  = "Sector"
    if sub_col:     rename_map[sub_col]     = "SubIndustry"
    if hq_col:      rename_map[hq_col]      = "HQ"
    if date_col:    rename_map[date_col]    = "DateAdded"

    df = df.rename(columns=rename_map)
    df["Symbol"] = df["Symbol"].astype(str).str.strip()

    # Ne garder que les colonnes utiles si elles existent
    keep_cols = ["Symbol", "Company", "Sector", "SubIndustry", "HQ", "DateAdded"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    return df

# ==============================
# Constituants S&P500 (version simple style "programme qui marche")
# ==============================

@st.cache_data(show_spinner=False, ttl=60*60)
def get_sp500_constituents() -> Tuple[pd.DataFrame, List[str]]:
    path = "sp500_constituents.xlsx"

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier {path} introuvable. Ajoute-le dans ton repo GitHub (même dossier que coachsniper.py)."
        )

    df = pd.read_excel(path)

    if "Symbol" not in df.columns:
        raise ValueError("Le fichier Excel doit contenir une colonne 'Symbol'.")

    if "Company" not in df.columns:
        df["Company"] = df["Symbol"]

    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"

    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df = df[df["Symbol"] != ""]

    tickers = df["Symbol"].tolist()
    return df, tickers

# ==============================
# Indicateurs utilitaires
# ==============================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
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

# ==============================
# Stratégie Ichimoku (3 modes)
# ==============================
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

# ==============================
# Polygon — téléchargement OHLCV daily
# ==============================
def _polygon_aggs_daily(ticker: str) -> Optional[pd.DataFrame]:
    """
    Récupère 2 ans de chandelles daily via Polygon avec gestion des erreurs.
    Version PRO : robuste, retries internes, detection throttling.
    """

    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=int(YEARS * 365.25))

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"

    params = {
        "adjusted": "true" if ADJUSTED else "false",
        "sort": "asc",
        "limit": LIMIT,
        "apiKey": POLY,
    }

    retry_delays = [0.4, 0.8, 1.6, 3.2]  # Exponential backoff (PRO)
    last_error = None

    for delay in retry_delays:
        try:
            r = requests.get(url, params=params, timeout=30)

            # HTTP error (server busy, throttle, etc.)
            if r.status_code != 200:
                last_error = f"HTTP {r.status_code}"
                time.sleep(delay)
                continue

            js = r.json()
            results = js.get("results", [])

            # No results (Polygon glitch or throttling)
            if not results:
                last_error = "Empty results"
                time.sleep(delay)
                continue

            df = pd.DataFrame(results)
            df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume", "t": "ts"})
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
            df = df.set_index("ts").sort_index()

            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            out = df[keep].astype(float)

            out = to_heikin_ashi(out)
            return out

        except Exception as e:
            last_error = str(e)
            time.sleep(delay)

    # After all retries → return None
    return None

def _process_polygon_batch(batch: List[str], out_dict: Dict[str, pd.DataFrame], retry_group: List[str]):
    """
    Tente d'obtenir chaque ticker une fois en batch + place en retry_group si vide.
    """
    for t in batch:
        dft = _polygon_aggs_daily(t)
        if dft is not None and not dft.empty:
            out_dict[t] = dft
        else:
            retry_group.append(t)


@st.cache_data(show_spinner=False)
def download_bars_polygon_safe(tickers: tuple[str, ...]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Version PRO du downloader :
    - Batch processing + retry intelligent
    - Exponential backoff
    - Second pass individuelle avec pauses
    - Minimisation des tickers échoués
    """
    out: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []
    retry_group: List[str] = []

    base_list = list(tickers)

    # --- PASS 1 : batch processing ---
    i = 0
    while i < len(base_list):
        batch = base_list[i:i+CHUNK]
        _process_polygon_batch(batch, out, retry_group)
        time.sleep(0.5 + random.random())  # anti-throttle
        i += CHUNK

    # --- PASS 2 : retry individuel (lent mais robuste) ---
    if retry_group:
        second_retry = retry_group.copy()
        retry_group = []

        for t in second_retry:
            dft = _polygon_aggs_daily(t)
            if dft is not None and not dft.empty:
                out[t] = dft
            else:
                retry_group.append(t)
            time.sleep(0.7 + random.random())

    # --- PASS 3 : dernier essai (très lent mais efficace) ---
    if retry_group:
        final_round = retry_group.copy()
        retry_group = []

        for t in final_round:
            time.sleep(1.25 + random.random())
            dft = _polygon_aggs_daily(t)
            if dft is not None and not dft.empty:
                out[t] = dft
            else:
                failed.append(t)

    return out, failed

# ==============================
# UI – Filtres & contrôles
# ==============================
with st.spinner("Chargement de la liste S&P 500…"):
    sp_df, all_poly_tickers = get_sp500_constituents()  # Symbols format Polygon

# Dow 30 (test rapide)
DOW30 = ["AAPL","MSFT","JPM","UNH","GS","HD","MS","AMGN","CRM","MCD","CAT","HON","TRV","CVX",
         "PG","V","JNJ","BA","DIS","NKE","WMT","AXP","KO","IBM","MRK","CSCO","INTC","VZ","MMM","WBA"]

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    sectors = sorted(sp_df["Sector"].dropna().unique().tolist()) if "Sector" in sp_df.columns else []
    sector_sel = st.multiselect("Secteurs", sectors, [])
with c2:
    limit = st.number_input("Nombre max de tickers", min_value=10, max_value=500, value=120, step=10)
with c3:
    search = st.text_input("Recherche (ticker/nom)", "").strip().lower()

st.sidebar.header("Stratégie")
mode = st.sidebar.selectbox("Mode", ["Balanced", "Strict", "Aggressive"], index=0)
use_rsi50 = st.sidebar.checkbox("Filtre RSI 50", value=True)

st.sidebar.header("Vague de scan (pagination)")
wave = st.sidebar.number_input("Taille de la vague (≤ 20 conseillé)", 10, 60, DEFAULT_WAVE, 5)
offset = st.sidebar.number_input("Offset (départ)", 0, 500, 0, 1)
use_dow = st.sidebar.checkbox("Dow 30 (test rapide)", value=False)

st.caption(f"Source: Polygon daily ({YEARS} ans) — Données converties en **Heikin Ashi** — adjusted={ADJUSTED}")

# Filtrage tickers
if use_dow:
    base_list = [t for t in DOW30]
    base = sp_df[sp_df["Symbol"].isin(base_list)].copy()
else:
    base = sp_df.copy()
    if sector_sel and "Sector" in base.columns:
        base = base[base["Sector"].isin(sector_sel)]
    if search:
        company_col = "Company" if "Company" in base.columns else None
        mask = base["Symbol"].str.lower().str.contains(search)
        if company_col:
            mask = mask | base[company_col].str.lower().str.contains(search)
        base = base[mask]
    base_list = base["Symbol"].tolist()

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
go = st.button("▶️ Scanner cette vague (Polygon)", type="primary")
if not go:
    st.stop()

# Téléchargement Polygon (safe)
tickers_tuple = tuple(sorted(set(wave_list)))  # clé cache stable
with st.spinner("Téléchargement des chandelles (Polygon)…"):
    bars, failed = download_bars_polygon_safe(tickers_tuple)

valid = sum(1 for t in tickers_tuple if bars.get(t) is not None and len(bars[t]) > 0)
st.caption(f"✅ Jeux de données valides : {valid}/{len(tickers_tuple)}")
if failed:
    st.warning(f"⚠️ Tickers échoués (après retries): {len(failed)} — ex.: {', '.join(failed[:8])}{'…' if len(failed)>8 else ''}")

if valid == 0:
    st.error("Aucune donnée renvoyée par Polygon pour cette vague.")
    st.stop()

# ==============================
# Calcul des signaux
# ==============================
def _safe_get(series: pd.Series):
    return float(series.iloc[-1]) if series is not None and len(series) else None

results = []
for t in tickers_tuple:
    dft = bars.get(t)
    if dft is None or len(dft) < 82:
        continue
    buy_now, sell_now, last = coach_swing_signals(dft, mode=mode, use_rsi50=use_rsi50)
    results.append({
        "Ticker": t,
        "Company": base.loc[base["Symbol"] == t, "Company"].values[0] if "Company" in base.columns and not base.empty and (base["Symbol"] == t).any() else t,
        "Sector":  base.loc[base["Symbol"] == t, "Sector"].values[0] if "Sector" in base.columns and not base.empty and (base["Symbol"] == t).any() else None,
        "Buy":     buy_now,
        "Sell":    sell_now,
        "Close":   _safe_get(dft["Close"]),
        "RSI":     last.get("RSI"),
        "WR":      last.get("WR"),
        "VO":      last.get("VO"),
    })

res_df = pd.DataFrame(results)
if res_df.empty:
    st.warning("Aucun résultat dans cette vague (filtres stricts ou données manquantes).")
    st.stop()

# ==============================
# Affichage & export
# ==============================
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
                   file_name=f"coach_swing_polygon_1d_wave_{start}_{end-1}.csv", mime="text/csv")

st.markdown("""
**Notes Polygon :**
- Les quotidiens via `/v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}` sont **15 min delayed** sur ton plan — suffisant pour du daily.
- `BRK.B`, `BF.B`, etc. utilisent **le point** chez Polygon (contrairement à Yahoo qui remplace par un tiret).
- Tu peux aussi placer un fichier CSV local `sp500_constituents.csv` dans le repo avec une colonne de tickers
  si jamais Wikipédia rechange encore la structure.
""")
