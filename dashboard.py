import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Scanner Intraday Nasdaq & S&P 500",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .header-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 18px;
    }

    .header-title {
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin: 0;
    }

    .header-subtitle {
        font-size: 0.88rem;
        color: #94a3b8;
        margin: 4px 0 0 0;
    }

    .kpi-card {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 14px 16px;
    }

    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 4px;
    }

    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .kpi-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 2px;
    }

    .trend-panel {
        border-radius: 12px;
        padding: 16px 18px;
        border: 1px solid rgba(148, 163, 184, 0.14);
        background: rgba(15, 23, 42, 0.65);
        height: 100%;
    }

    .trend-panel-bull {
        border-top: 3px solid #10b981;
        background: linear-gradient(180deg, rgba(16, 185, 129, 0.09) 0%, rgba(15, 23, 42, 0.7) 100%);
    }

    .trend-panel-bear {
        border-top: 3px solid #ef4444;
        background: linear-gradient(180deg, rgba(239, 68, 68, 0.09) 0%, rgba(15, 23, 42, 0.7) 100%);
    }

    .trend-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }

    .trend-title-bull {
        color: #34d399;
        font-weight: 700;
        font-size: 1rem;
    }

    .trend-title-bear {
        color: #f87171;
        font-weight: 700;
        font-size: 1rem;
    }

    .badge-count {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        color: #e2e8f0;
    }

    .ticker-chip-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 6px;
    }

    .chip-bull {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 6px;
        background: rgba(16, 185, 129, 0.16);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    .chip-bear {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 6px;
        background: rgba(239, 68, 68, 0.16);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    .chip-empty {
        font-size: 0.85rem;
        color: #64748b;
        font-style: italic;
    }

    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.22), transparent);
        margin: 18px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

RSI_PERIOD = 14
RSI_MA_PERIOD = 20
TIMEZONE_BRT = pytz.timezone("America/Sao_Paulo")

# Universo consolidado: Top Nasdaq-100 + Top S&P 500 + SPX, SPY, QQQ, ^VIX
FALLBACK_NASDAQ_TICKERS: Tuple[str, ...] = (
    "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "MU", "AMD",
    "AVGO", "META",
    "SPX", "SPY", "QQQ", "^VIX"
)

TICKER_TO_YF: Dict[str, str] = {
    "SPX": "^GSPC",
}

INVERTED_TICKERS = {"^VIX"}


@st.cache_data(ttl=3600, show_spinner=False)
def get_all_nasdaq_stocks(exclude_etfs: bool = False, clean_symbols_only: bool = True) -> List[str]:
    return list(FALLBACK_NASDAQ_TICKERS)


def calculate_rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _extract_ticker_df(raw_data: pd.DataFrame, yf_symbol: str, total_tickers: int) -> pd.DataFrame:
    if raw_data is None or raw_data.empty:
        return pd.DataFrame()
    if total_tickers == 1:
        return raw_data.copy()
    if isinstance(raw_data.columns, pd.MultiIndex):
        if yf_symbol in raw_data.columns.get_level_values(0):
            return raw_data[yf_symbol].copy()
        if yf_symbol in raw_data.columns.get_level_values(-1):
            return raw_data.xs(yf_symbol, axis=1, level=-1).copy()
    return pd.DataFrame()


def chunk_list(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


@st.cache_data(ttl=60, show_spinner=False)
def fetch_and_process_data(
    tickers: Tuple[str, ...],
    rsi_len: int,
    ma_len: int,
    include_extended: bool,
    batch_size: int = 250,
) -> Dict[str, pd.DataFrame]:
    processed: Dict[str, pd.DataFrame] = {}
    ticker_list = list(tickers)
    batches = chunk_list(ticker_list, batch_size)

    for batch in batches:
        yf_batch = [TICKER_TO_YF.get(t, t) for t in batch]
        try:
            raw_daily = yf.download(
                tickers=yf_batch,
                period="40d",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
            )
            raw_intraday = yf.download(
                tickers=yf_batch,
                period="5d",
                interval="15m",
                group_by="ticker",
                prepost=include_extended,
                threads=True,
                progress=False,
            )
        except Exception:
            continue

        for ticker in batch:
            yf_symbol = TICKER_TO_YF.get(ticker, ticker)
            is_inverted = ticker in INVERTED_TICKERS

            try:
                df_d = _extract_ticker_df(raw_daily, yf_symbol, len(yf_batch))
                if df_d.empty or "Close" not in df_d.columns:
                    continue
                df_d = df_d.dropna(subset=["Close"])

                z_score_val = np.nan
                daily_var_pct = np.nan
                z_score_amp = np.nan

                if len(df_d) >= 2:
                    df_d["Daily_Return_Pct"] = df_d["Close"].pct_change() * 100.0
                    daily_var_pct = float(df_d["Daily_Return_Pct"].iloc[-1])

                if "High" in df_d.columns and "Low" in df_d.columns:
                    df_d["Amplitude"] = (df_d["High"] - df_d["Low"]) / df_d["Low"]

                if len(df_d) >= 21:
                    rolling_mean = df_d["Daily_Return_Pct"].rolling(20).mean()
                    rolling_std = df_d["Daily_Return_Pct"].rolling(20).std()
                    latest_ret = df_d["Daily_Return_Pct"].iloc[-1]
                    l_mean = rolling_mean.iloc[-1]
                    l_std = rolling_std.iloc[-1]

                    if pd.notna(l_std) and l_std > 0 and pd.notna(latest_ret) and pd.notna(l_mean):
                        z_score_val = float((latest_ret - l_mean) / l_std)

                    if "Amplitude" in df_d.columns:
                        amp_mean = df_d["Amplitude"].rolling(20).mean()
                        amp_std = df_d["Amplitude"].rolling(20).std()
                        latest_amp = df_d["Amplitude"].iloc[-1]
                        l_amp_mean = amp_mean.iloc[-1]
                        l_amp_std = amp_std.iloc[-1]

                        if pd.notna(l_amp_std) and l_amp_std > 0 and pd.notna(latest_amp) and pd.notna(l_amp_mean):
                            z_score_amp = float((latest_amp - l_amp_mean) / l_amp_std)

                df = _extract_ticker_df(raw_intraday, yf_symbol, len(yf_batch))
                if df.empty or "Close" not in df.columns:
                    continue
                df = df.dropna(subset=["Close"])

                if len(df) < (rsi_len + ma_len + 2):
                    continue

                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC").tz_convert(TIMEZONE_BRT)
                else:
                    df.index = df.index.tz_convert(TIMEZONE_BRT)

                rsi_series = calculate_rsi_wilder(df["Close"], period=rsi_len)
                df["RSI"] = rsi_series
                df["RSI_MA"] = df["RSI"].rolling(window=ma_len).mean()
                raw_spread = df["RSI"] - df["RSI_MA"]
                df["Spread"] = -raw_spread if is_inverted else raw_spread
                df["Acima_Media"] = (df["RSI"] < df["RSI_MA"]) if is_inverted else (df["RSI"] > df["RSI_MA"])
                df["Z_Score_20d"] = z_score_val
                df["Z_Score_Amp_20d"] = z_score_amp
                df["Var_Diaria_Pct"] = daily_var_pct

                processed[ticker] = df
            except Exception:
                continue

    return processed


def render_chips(tickers: List[str], chip_class: str, empty_text: str = "Nenhum ativo no momento") -> str:
    if not tickers:
        return f"<div class='chip-empty'>{empty_text}</div>"
    chips_html = "".join([f"<span class='{chip_class}'>{t}</span>" for t in tickers])
    return f"<div class='ticker-chip-wrap'>{chips_html}</div>"


def main() -> None:
    now_brt = datetime.now(TIMEZONE_BRT).strftime("%d/%m/%Y · %H:%M:%S BRT")

    st.markdown(
        f"""
        <div class="header-banner">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                <div>
                    <h1 class="header-title">Scanner Intraday · Nasdaq-100 & S&P 500</h1>
                    <p class="header-subtitle">Momentum RSI (15m) vs. Média Móvel · Expansão Diária & Z-Score 20D · VIX Invertido</p>
                </div>
                <span class="badge-count">🕒 {now_brt}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_nasdaq = get_all_nasdaq_stocks()

    with st.sidebar:
        st.markdown("### 🎯 Universo & Controles")
        st.caption("Top NDX + Top SPX + SPX, SPY, QQQ, ^VIX (invertido)")
        selected_tickers = all_nasdaq

        st.markdown("### ⚙️ Parâmetros Técnicos")
        rsi_len = st.number_input("Período RSI", min_value=2, max_value=50, value=RSI_PERIOD)
        ma_len = st.number_input("Média Móvel no RSI (SMA)", min_value=2, max_value=50, value=RSI_MA_PERIOD)
        include_extended = st.checkbox("Incluir Pré / After-Market", value=False)
        auto_refresh = st.checkbox("Auto-atualizar (60s)", value=False)

        st.markdown("### 📊 Visualização")
        candles_chart = st.slider("Candles no gráfico 15m", min_value=30, max_value=150, value=78)
        max_charts = st.slider("Máx. gráficos renderizados", min_value=4, max_value=24, value=14)

        if st.button("🔄 Atualizar Dados Agora", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Sincronizando cotações e calculando indicadores..."):
        data_map = fetch_and_process_data(
            tuple(selected_tickers), int(rsi_len), int(ma_len), include_extended
        )

    if not data_map:
        st.error("Não foi possível obter os dados no momento. Tente atualizar novamente.")
        return

    summary_rows = []
    acima_list = []
    abaixo_list = []
    up_daily_count = 0
    down_daily_count = 0

    rsi_col_name = f"RSI ({rsi_len})"
    ma_col_name = f"SMA RSI ({ma_len})"

    for ticker in selected_tickers:
        df = data_map.get(ticker)
        if df is None or len(df) < 2:
            continue

        last = df.iloc[-1]
        prev = df.iloc[-2]

        is_above = bool(last["Acima_Media"])
        was_above = bool(prev["Acima_Media"])
        is_inverted = ticker in INVERTED_TICKERS

        display_name = f"{ticker} (Inv)" if is_inverted else ticker

        if is_above:
            acima_list.append(display_name)
        else:
            abaixo_list.append(display_name)

        z_val = float(last["Z_Score_20d"]) if pd.notna(last
