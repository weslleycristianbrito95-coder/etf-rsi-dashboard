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

# Configuração da página (deve ser a primeira chamada Streamlit)
st.set_page_config(
    page_title="Scanner Intraday Nasdaq — RSI & Variação Diária",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilização CSS customizada
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
    }
    .trend-card {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .trend-up {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 5px solid #10b981;
    }
    .trend-down {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 5px solid #ef4444;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Constantes
RSI_PERIOD = 14
RSI_MA_PERIOD = 20
TIMEZONE_BRT = pytz.timezone("America/Sao_Paulo")

FALLBACK_NASDAQ_TICKERS: Tuple[str, ...] = (
    "QQQ", "MSFT", "AAPL", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA",
    "COST", "ASML", "WMT", "AMD", "PANW", "TXN", "AMAT", "MU", "LRCX", "PLTR",
    "CSCO", "INTC", "SPCX",
)


@st.cache_data(ttl=3600, show_spinner=False)
def get_all_nasdaq_stocks() -> List[str]:
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


def _extract_ticker_df(raw_data: pd.DataFrame, ticker: str, total_tickers: int) -> pd.DataFrame:
    if raw_data is None or raw_data.empty:
        return pd.DataFrame()
    if total_tickers == 1:
        return raw_data.copy()
    if isinstance(raw_data.columns, pd.MultiIndex):
        if ticker in raw_data.columns.get_level_values(0):
            return raw_data[ticker].copy()
        if ticker in raw_data.columns.get_level_values(-1):
            return raw_data.xs(ticker, axis=1, level=-1).copy()
    return pd.DataFrame()


def chunk_list(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


@st.cache_data(ttl=60, show_spinner=False)
def fetch_and_process_data(
    tickers: Tuple[str, ...],
    rsi_len: int,
    ma_len: int,
    include_extended: bool,
    batch_size: int = 50,
) -> Dict[str, pd.DataFrame]:
    processed: Dict[str, pd.DataFrame] = {}
    ticker_list = list(tickers)
    batches = chunk_list(ticker_list, batch_size)

    for batch in batches:
        try:
            raw_daily = yf.download(
                tickers=batch,
                period="40d",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
            )
            raw_intraday = yf.download(
                tickers=batch,
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
            try:
                df_d = _extract_ticker_df(raw_daily, ticker, len(batch))
                if df_d.empty or "Close" not in df_d.columns:
                    continue
                df_d = df_d.dropna(subset=["Close"])

                z_score_val = np.nan
                daily_var_pct = np.nan

                if len(df_d) >= 2:
                    df_d["Daily_Return_Pct"] = df_d["Close"].pct_change() * 100.0
                    daily_var_pct = float(df_d["Daily_Return_Pct"].iloc[-1])

                if len(df_d) >= 21:
                    rolling_mean = df_d["Daily_Return_Pct"].rolling(20).mean()
                    rolling_std = df_d["Daily_Return_Pct"].rolling(20).std()

                    latest_ret = df_d["Daily_Return_Pct"].iloc[-1]
                    l_mean = rolling_mean.iloc[-1]
                    l_std = rolling_std.iloc[-1]

                    if pd.notna(l_std) and l_std > 0 and pd.notna(latest_ret) and pd.notna(l_mean):
                        z_score_val = float((latest_ret - l_mean) / l_std)

                df = _extract_ticker_df(raw_intraday, ticker, len(batch))
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
                df["Spread"] = df["RSI"] - df["RSI_MA"]
                df["Acima_Media"] = df["RSI"] > df["RSI_MA"]
                df["Z_Score_20d"] = z_score_val
                df["Var_Diaria_Pct"] = daily_var_pct

                processed[ticker] = df
            except Exception:
                continue

    return processed


def main() -> None:
    st.title("SCANNER PORTFÓLIO NASDAQ — CONFIGURAÇÃO")

    all_nasdaq = get_all_nasdaq_stocks()

    with st.sidebar:
        st.header("Universo Escolhido")
        custom_input = st.text_area(
            "Ativos monitorados (separados por vírgula)",
            value=", ".join(all_nasdaq),
            height=110,
        )
        selected_tickers = [
            t.strip().upper() for t in custom_input.split(",") if t.strip()
        ]
        if not selected_tickers:
            selected_tickers = all_nasdaq

        st.header("Parâmetros Técnicos")
        rsi_len = st.number_input("Período RSI", min_value=2, max_value=50, value=RSI_PERIOD)
        ma_len = st.number_input("Média Móvel no RSI (SMA)", min_value=2, max_value=50, value=RSI_MA_PERIOD)
        include_extended = st.checkbox("Incluir Pré / After-Market", value=False)
        auto_refresh = st.checkbox("Auto-atualizar (60s)", value=False)
        candles_chart = st.slider("Candles visíveis no gráfico", min_value=30, max_value=150, value=78)
        max_charts = st.slider("Máx. gráficos 15m renderizados", min_value=4, max_value=60, value=20)

        if st.button("Atualizar Agora", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Baixando e calculando indicadores..."):
        data_map = fetch_and_process_data(
            tuple(selected_tickers), int(rsi_len), int(ma_len), include_extended
        )

    if not data_map:
        st.error("Não foi possível obter os dados no momento. Verifique os tickers ou tente novamente.")
        return

    summary_rows = []
    acima_list = []
    abaixo_list = []
    up_daily_count = 0
    down_daily_count = 0

    rsi_col_name = f"RSI ({rsi_len})"
    ma_col_name = f"Média RSI ({ma_len})"

    for ticker in selected_tickers:
        df = data_map.get(ticker)
        if df is None or len(df) < 2:
            continue

        last = df.iloc[-1]
        prev = df.iloc[-2]

        is_above = bool(last["Acima_Media"])
        was_above = bool(prev["Acima_Media"])

        if is_above:
            acima_list.append(ticker)
        else:
            abaixo_list.append(ticker)

        z_val = float(last["Z_Score_20d"]) if pd.notna(last["Z_Score_20d"]) else np.nan
        var_diaria = float(last["Var_Diaria_Pct"]) if pd.notna(last["Var_Diaria_Pct"]) else np.nan

        if pd.notna(var_diaria):
            if var_diaria > 0:
                up_daily_count += 1
            elif var_diaria < 0:
                down_daily_count += 1

        if is_above and not was_above:
            evento = "CRUZOU PARA CIMA"
        elif not is_above and was_above:
            evento = "CRUZOU PARA BAIXO"
        else:
            evento = "Mantendo tendência"

        summary_rows.append(
            {
                "Ativo": ticker,
                "Último Candle": df.index[-1].strftime("%d/%m %H:%M"),
                "Preço": round(float(last["Close"]), 2),
                "Var. Diária (%)": var_diaria,
                rsi_col_name: round(float(last["RSI"]), 2),
                ma_col_name: round(float(last["RSI_MA"]), 2),
                "Spread (RSI - MA)": round(float(last["Spread"]), 2),
                "Z-Score (20d)": z_val,
                "Status": "ACIMA DA MÉDIA" if is_above else "ABAIXO DA MÉDIA",
                "Sinal Último Candle": evento,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    total_ativos = len(summary_df)
    now_brt = datetime.now(TIMEZONE_BRT).strftime("%H:%M:%S")

    # Cálculo da balança de força (Alta vs. Baixa no RSI)
    pct_acima = (len(acima_list) / max(total_ativos, 1)) * 100.0
    pct_abaixo = 100.0 - pct_acima if total_ativos > 0 else 0.0
    spread_medio = float(summary_df["Spread (RSI - MA)"].mean()) if not summary_df.empty else 0.0

    top_alta = (
        summary_df[summary_df["Spread (RSI - MA)"] > 0]
        .sort_values(by="Spread (RSI - MA)", ascending=False)
        .head(4)["Ativo"]
        .tolist()
    )
    top_baixa = (
        summary_df[summary_df["Spread (RSI - MA)"] < 0]
        .sort_values(by="Spread (RSI - MA)", ascending=True)
        .head(4)["Ativo"]
        .tolist()
    )

    st.markdown("### ⚖️ Balança de Força — Alta vs. Baixa no RSI")

    st.markdown(
        f"""
        <div style="margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; font-weight: 700; margin-bottom: 6px;">
                <span style="color: #10b981;">🟢 ALTA (RSI &gt; Média): {len(acima_list)} ({pct_acima:.1f}%)</span>
                <span style="color: #9ca3af;">Spread Médio: {spread_medio:+.2f}</span>
                <span style="color: #ef4444;">🔴 BAIXA (RSI &lt; Média): {len(abaixo_list)} ({pct_abaixo:.1f}%)</span>
            </div>
            <div style="width: 100%; height: 22px; background-color: rgba(239, 68, 68, 0.35); border-radius: 999px; overflow: hidden; display: flex; border: 1px solid rgba(255,255,255,0.12);">
                <div style="width: {pct_acima:.1f}%; background: linear-gradient(90deg, #059669, #10b981); transition: width 0.3s ease;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_up, c_down = st.columns(2)

    with c_up:
        destaque_alta = ", ".join(top_alta) if top_alta else "—"
        st.markdown(
            f"""<div class="trend-card trend-up">
            <h4 style='color: #10b981; margin: 0;'>🔥 Pressão de Alta: {len(acima_list)} Ativos ({pct_acima:.1f}%)</h4>
            <p style='font-size: 0.9rem; color: #d1fae5; margin: 6px 0 4px 0;'><strong>Maior força (Spread +):</strong> {destaque_alta}</p>
            <p style='font-size: 1.05rem; font-weight: bold; margin: 6px 0 0 0;'>{', '.join(acima_list) if acima_list else 'Nenhum'}</p>
            </div>""",
            unsafe_allow_html=True,
        )

    with c_down:
        destaque_baixa = ", ".join(top_baixa) if top_baixa else "—"
        st.markdown(
            f"""<div class="trend-card trend-down">
            <h4 style='color: #ef4444; margin: 0;'>❄️ Pressão de Baixa: {len(abaixo_list)} Ativos ({pct_abaixo:.1f}%)</h4>
            <p style='font-size: 0.9rem; color: #fee2e2; margin: 6px 0 4px 0;'><strong>Maior pressão (Spread -):</strong> {destaque_baixa}</p>
            <p style='font-size: 1.05rem; font-weight: bold; margin: 6px 0 0 0;'>{', '.join(abaixo_list) if abaixo_list else 'Nenhum'}</p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ações Analisadas", f"{total_ativos}")
    col2.metric("RSI > SMA (15m)", f"{len(acima_list)} ({pct_acima:.1f}%)")
    col3.metric("Variação Diária (Altas/Baixas)", f"{up_daily_count} ↑ / {down_daily_count} ↓")
    col4.metric("Hora da Atualização (BRT)", now_brt)

    # Filtros da tabela
    f1, f2, f3 = st.columns([2, 2, 2])
    with f1:
        busca_ticker = st.text_input("Filtrar por ticker (ex.: NVDA, AAPL, TSLA)", value="").strip().upper()
    with f2:
        ordenar_por = st.selectbox(
            "Ordenar tabela por",
            options=["Z-Score (20d)", "Var. Diária (%)", "Spread (RSI - MA)", "Ativo"],
            index=1,
        )
    with f3:
        ordem_desc = st.checkbox("Ordem decrescente", value=True)

    filtered_df = summary_df.copy()
    if busca_ticker:
        filtered_df = filtered_df[filtered_df["Ativo"].str.contains(busca_ticker, na=False)]

    if ordenar_por in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=ordenar_por, ascending=not ordem_desc, na_position="last")

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

    def highlight_zscore(val: float) -> str:
        if pd.isna(val):
            return ""
        if val > 1.5:
            return "background-color: rgba(16, 185, 129, 0.22); color: #10b981; font-weight: bold;"
        if val < -1.5:
            return "background-color: rgba(239, 68, 68, 0.22); color: #ef4444; font-weight: bold;"
        return ""

    def highlight_pct(val: float) -> str:
        if pd.isna(val):
            return ""
        if val > 0:
            return "color: #10b981; font-weight: bold;"
        if val < 0:
            return "color: #ef4444; font-weight: bold;"
        return ""

    styled_df = (
        filtered_df.style
        .map(highlight_status, subset=["Status"])
        .map(highlight_event, subset=["Sinal Último Candle"])
        .map(highlight_zscore, subset=["Z-Score (20d)"])
        .map(highlight_pct, subset=["Var. Diária (%)"])
        .format(
            {
                "Preço": "${:.2f}",
                "Var. Diária (%)": "{:+.2f}%",
                rsi_col_name: "{:.2f}",
                ma_col_name: "{:.2f}",
                "Spread (RSI - MA)": "{:+.2f}",
                "Z-Score (20d)": "{:+.2f}",
            },
            na_rep="-",
        )
    )

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.subheader(f"Curva RSI({rsi_len}) vs. Média Móvel ({ma_len}) — Gráfico de 15 Minutos")

    chart_candidates = filtered_df["Ativo"].tolist()[:max_charts]
    active_tickers = [t for t in chart_candidates if t in data_map]

    if active_tickers:
        num_plots = len(active_tickers)
        num_cols = 2
        num_rows = (num_plots + 1) // num_cols

        subplot_titles_list = []
        for t in active_tickers:
            is_abv = bool(data_map[t]["Acima_Media"].iloc[-1])
            z_t = data_map[t]["Z_Score_20d"].iloc[-1]
            z_str = f"Z: {z_t:+.2f}" if pd.notna(z_t) else "Z: -"
            label = "ACIMA" if is_abv else "ABAIXO"
            subplot_titles_list.append(f"{t} — {label} ({z_str})")

        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=subplot_titles_list,
            vertical_spacing=0.05,
            horizontal_spacing=0.06,
        )

        for idx, ticker in enumerate(active_tickers):
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

        fig.update_layout(
            height=240 * num_rows,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

    if auto_refresh:
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
