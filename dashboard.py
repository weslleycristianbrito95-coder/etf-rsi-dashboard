from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

RSI_PERIOD = 14
RSI_MA_PERIOD = 20
TIMEZONE_BRT = pytz.timezone("America/Sao_Paulo")

DEFAULT_TICKERS: Tuple[str, ...] = (
    "QQQ", "MSFT", "AAPL", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA", "COST",
    "ASML", "WMT", "AMD", "PANW", "TXN", "AMAT", "MU", "LRCX", "PLTR", "CSCO",
    "INTC", "SPCX",
)


@dataclass(frozen=True)
class ScannerConfig:
    tickers: Tuple[str, ...] = DEFAULT_TICKERS
    rsi_len: int = RSI_PERIOD
    ma_len: int = RSI_MA_PERIOD
    include_extended: bool = False
    batch_size: int = 50


def calculate_rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _extract_ticker_df(raw_data: Optional[pd.DataFrame], ticker: str, total_tickers: int) -> pd.DataFrame:
    if raw_data is None or raw_data.empty:
        return pd.DataFrame()
    if total_tickers == 1:
        return raw_data.copy()
    if isinstance(raw_data.columns, pd.MultiIndex):
        lvl0 = raw_data.columns.get_level_values(0)
        lvl1 = raw_data.columns.get_level_values(-1)
        if ticker in lvl0:
            return raw_data[ticker].copy()
        if ticker in lvl1:
            return raw_data.xs(ticker, axis=1, level=-1).copy()
    return pd.DataFrame()


def chunk_list(items: List[str], size: int) -> List[List[str]]:
    if size <= 0:
        raise ValueError("batch_size deve ser maior que zero.")
    return [items[i : i + size] for i in range(0, len(items), size)]


def fetch_and_process_data(config: ScannerConfig) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Retorna (data_map, warnings).
    Não engole erros silenciosamente: registra avisos quando um batch ou ticker falha.
    """
    processed: Dict[str, pd.DataFrame] = {}
    warnings: List[str] = []
    ticker_list = [t.strip().upper() for t in config.tickers if t.strip()]
    batches = chunk_list(ticker_list, config.batch_size)

    for batch in batches:
        try:
            raw_daily = yf.download(
                tickers=batch,
                period="40d",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=False,
            )
            raw_intraday = yf.download(
                tickers=batch,
                period="5d",
                interval="15m",
                group_by="ticker",
                prepost=config.include_extended,
                threads=True,
                progress=False,
                auto_adjust=False,
            )
        except Exception as exc:
            warnings.append(f"Falha no download do lote {batch[:3]}...: {exc}")
            continue

        for ticker in batch:
            try:
                df_d = _extract_ticker_df(raw_daily, ticker, len(batch))
                if df_d.empty or "Close" not in df_d.columns:
                    warnings.append(f"{ticker}: sem dados diários válidos.")
                    continue
                df_d = df_d.dropna(subset=["Close"])

                z_score_val = np.nan
                daily_var_pct = np.nan

                if len(df_d) >= 2:
                    df_d = df_d.copy()
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
                    warnings.append(f"{ticker}: sem dados intraday de 15m.")
                    continue
                df = df.dropna(subset=["Close"]).copy()

                min_bars = config.rsi_len + config.ma_len + 2
                if len(df) < min_bars:
                    warnings.append(f"{ticker}: histórico intraday insuficiente ({len(df)} < {min_bars} candles).")
                    continue

                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC").tz_convert(TIMEZONE_BRT)
                else:
                    df.index = df.index.tz_convert(TIMEZONE_BRT)

                df["RSI"] = calculate_rsi_wilder(df["Close"], period=config.rsi_len)
                df["RSI_MA"] = df["RSI"].rolling(window=config.ma_len).mean()
                df["Spread"] = df["RSI"] - df["RSI_MA"]
                df["Acima_Media"] = df["RSI"] > df["RSI_MA"]
                df["Z_Score_20d"] = z_score_val
                df["Var_Diaria_Pct"] = daily_var_pct

                processed[ticker] = df
            except Exception as exc:
                warnings.append(f"{ticker}: erro ao processar séries ({exc}).")

    return processed, warnings


def save_snapshot(data_map: Dict[str, pd.DataFrame], output_path: Path, config: ScannerConfig) -> None:
    """Salva um snapshot JSON leve para consumo rápido via GitHub App / Actions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_brt": datetime.now(TIMEZONE_BRT).isoformat(),
        "config": {
            "rsi_len": config.rsi_len,
            "ma_len": config.ma_len,
            "include_extended": config.include_extended,
        },
        "tickers": {},
    }
    for ticker, df in data_map.items():
        tail_df = df.tail(160).copy()
        records = []
        for idx, row in tail_df.iterrows():
            records.append(
                {
                    "timestamp": idx.isoformat(),
                    "Close": None if pd.isna(row["Close"]) else float(row["Close"]),
                    "RSI": None if pd.isna(row["RSI"]) else float(row["RSI"]),
                    "RSI_MA": None if pd.isna(row["RSI_MA"]) else float(row["RSI_MA"]),
                    "Spread": None if pd.isna(row["Spread"]) else float(row["Spread"]),
                    "Acima_Media": bool(row["Acima_Media"]),
                    "Z_Score_20d": None if pd.isna(row["Z_Score_20d"]) else float(row["Z_Score_20d"]),
                    "Var_Diaria_Pct": None if pd.isna(row["Var_Diaria_Pct"]) else float(row["Var_Diaria_Pct"]),
                }
            )
        payload["tickers"][ticker] = records

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_snapshot(snapshot_path: Path) -> Tuple[Dict[str, pd.DataFrame], Optional[str]]:
    """Carrega snapshot salvo no repositório, se existir."""
    if not snapshot_path.exists():
        return {}, None

    raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
    generated_at = raw.get("generated_at_brt")
    result: Dict[str, pd.DataFrame] = {}

    for ticker, rows in raw.get("tickers", {}).items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        if df.index.tz is None:
            df.index = df.index.tz_localize(TIMEZONE_BRT)
        else:
            df.index = df.index.tz_convert(TIMEZONE_BRT)
        result[ticker] = df

    return result, generated_at


if __name__ == "__main__":
    cfg = ScannerConfig()
    data, warns = fetch_and_process_data(cfg)
    save_snapshot(data, Path("data/scanner_latest.json"), cfg)
    print(f"Processados {len(data)} ativos. Avisos: {len(warns)}")
    for w in warns:
        print(" -", w)
