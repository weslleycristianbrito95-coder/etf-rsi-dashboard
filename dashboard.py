from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Scanner Intraday RSI(14) vs SMA(14) - ETFs EUA",
    layout="wide",
    initial_sidebar_state="expanded",
)

TICKERS = ["QQQ", "XLK", "SMH", "IGV", "XLC", "XLY", "XLF", "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "MU", "AMD", "AUDJPY=X"]
RSI_PERIOD = 14
RSI_MA_PERIOD = 14
TIMEZONE_BRT = pytz.timezone("America/Sao_Paulo")


def calculate_rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 / (1 + rs)
    return rsi.fillna(50.0)


@st.cache_data(ttl=25, show_spinner=False)
def fetch_and_process_data(
    tickers: Tuple[str, ...],
    rsi_len: int,
    ma_len: int,
    include_extended: bool,
) -> Dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers=list(tickers),
        period="5d",
        interval="5m",
        group_by="ticker",
        prepost=include_extended,
        threads=True,
        progress=False,
    )

    processed: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            df = raw[ticker].copy() if len(tickers) > 1 else raw.copy()
            df = df.dropna(subset=["Close"])

            if df.empty or len(df) < (rsi_len + ma_len + 2):
                continue

            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert(TIMEZONE_BRT)
            else:
                df.index = df.index.tz_convert(TIMEZONE_BRT)

            df["RSI"] = calculate_rsi_wilder(df["Close"], period=rsi_len)
            df["RSI_MA"] = df["RSI"].rolling(window=ma_len).mean()
            df["Spread"] = df["RSI"] - df["RSI_MA"]
            df["Acima_Media"] = df["RSI"] > df["RSI_MA"]

            processed[ticker] = df
        except Exception as exc:
            st.warning(f"Erro ao processar {ticker}: {exc}")

    return processed


def main() -> None:
    st.title("Monitor Intraday 5m — RSI(14) vs. Média Móvel(14)")

    with st.sidebar:
        st.header("Parâmetros")
        rsi_len = st.number_input("Período RSI", min_value=2, max_value=50, value=RSI_PERIOD)
        ma_len = st.number_input("Média Móvel no RSI (SMA)", min_value=2, max_value=50, value=RSI_MA_PERIOD)
        include_extended = st.checkbox("Incluir Pré / After-Market", value=False)
        auto_refresh = st.checkbox("Auto-atualizar (30s)", value=True)
        candles_chart = st.slider("Candles visíveis no gráfico", min_value=30, max_value=150, value=78)

        if st.button("Atualizar Agora", use_container_width=True):
            st.cache_data.clear()

    data_map = fetch_and_process_data(
        tuple(TICKERS), int(rsi_len), int(ma_len), include_extended
    )

    if not data_map:
        st.error("Não foi possível obter dados do Yahoo Finance no momento. Tente novamente em instantes.")
        return

    summary_rows = []
    acima_count = 0

    for ticker in TICKERS:
        df = data_map.get(ticker)
        if df is None:
            continue

        last = df.iloc[-1]
        prev = df.iloc[-2]

        is_above = bool(last["Acima_Media"])
        was_above = bool(prev["Acima_Media"])

        if is_above:
            acima_count += 1

        if is_above and not was_above:
            evento = "CRUZOU PARA CIMA"
        elif not is_above and was_above:
            evento = "CRUZOU PARA BAIXO"
        else:
            evento = "Mantendo tendência"

        summary_rows.append(
            {
                "Ativo": ticker,
                "Último Candle (BRT)": df.index[-1].strftime("%d/%m %H:%M"),
                "Preço": round(float(last["Close"]), 2),
                "RSI (14)": round(float(last["RSI"]), 2),
                "Média RSI (14)": round(float(last["RSI_MA"]), 2),
                "Spread (RSI - MA)": round(float(last["Spread"]), 2),
                "Status": "ACIMA DA MÉDIA" if is_above else "ABAIXO DA MÉDIA",
                "Sinal Último Candle": evento,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    now_brt = datetime.now(TIMEZONE_BRT).strftime("%H:%M:%S")
    col1, col2, col3 = st.columns(3)
    col1.metric("Ativos Acima da Média (RSI > SMA)", f"{acima_count} de {len(summary_df)}")
    col2.metric("Ativos Abaixo da Média (RSI < SMA)", f"{len(summary_df) - acima_count} de {len(summary_df)}")
    col3.metric("Última Atualização (Brasília)", now_brt)

    def highlight_status(val: str) -> str:
        if val == "ACIMA DA MÉDIA":
            return "background-color: rgba(16, 185, 129, 0.25); color: #10b981; font-weight: bold;"
        if val == "ABAIXO DA MÉDIA":
            return "background-color: rgba(239, 68, 68, 0.25); color: #ef4444; font-weight: bold;"
        return ""

    def highlight_event(val: str) -> str:
        if "CIMA" in val:
            return "color: #10b981; font-weight: bold;"
        if "BAIXO" in val:
            return "color: #ef4444; font-weight: bold;"
        return "color: #888888;"

    styled_df = (
        summary_df.style
        .map(highlight_status, subset=["Status"])
        .map(highlight_event, subset=["Sinal Último Candle"])
        .format({
            "Preço": "${:.2f}",
            "RSI (14)": "{:.2f}",
            "Média RSI (14)": "{:.2f}",
            "Spread (RSI - MA)": "{:+.2f}",
        })
    )

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.subheader("Curva RSI(14) vs. Média Móvel(14) — Gráfico de 5 Minutos")

    active_tickers = [t for t in TICKERS if t in data_map]
    num_plots = len(active_tickers)
    num_cols = 2
    num_rows = (num_plots + 1) // num_cols

    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=[
            f"{t} — {'ACIMA' if data_map[t]['Acima_Media'].iloc[-1] else 'ABAIXO'}"
            for t in active_tickers
        ],
        vertical_spacing=0.04,
        horizontal_spacing=0.06,
    )

    idx = 0
    for ticker in active_tickers:
        row = (idx // num_cols) + 1
        col = (idx % num_cols) + 1
        df_plot = data_map[ticker].tail(candles_chart)
        x_labels = df_plot.index.strftime("%d/%m %H:%M")

        is_above = bool(df_plot["Acima_Media"].iloc[-1])
        rsi_color = "#10b981" if is_above else "#ef4444"

        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=df_plot["RSI"],
                mode="lines",
                name=f"{ticker} RSI",
                line=dict(color=rsi_color, width=2),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=df_plot["RSI_MA"],
                mode="lines",
                name=f"{ticker} SMA",
                line=dict(color="#f59e0b", width=1.5, dash="dot"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,255,255,0.2)", row=row, col=col)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(255,255,255,0.2)", row=row, col=col)
        fig.update_yaxes(range=[10, 90], row=row, col=col)
        fig.update_xaxes(nticks=6, row=row, col=col)

        idx += 1

    fig.update_layout(
        height=240 * num_rows,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()