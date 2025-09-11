"""
QuickTrend 24/7 — Señales en TIEMPO REAL para ETFs/Acciones (Baja Volatilidad)
- Ideal para apuestas binarias de corto plazo (1-5 minutos)
- Usa SPY, TLT, AAPL, MSFT, etc.
- Datos en vivo vía Alpaca WebSocket
"""

import os
import json
import time
from datetime import datetime
import logging
import threading

import streamlit as st
import pandas as pd
import numpy as np

from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient

# Config
CRED_FILE = "alpaca_credentials.json"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(layout="wide", page_title="QuickTrend Binario", page_icon="🎯")

# -----------------------
# CREDENCIALES
# -----------------------
def save_credentials(key: str, secret: str):
    data = {"ALPACA_API_KEY": key.strip(), "ALPACA_API_SECRET": secret.strip()}
    try:
        with open(CRED_FILE, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False

def load_credentials():
    if os.path.exists(CRED_FILE):
        try:
            with open(CRED_FILE, "r") as f:
                return json.load(f)
        except:
            return None
    return None

def setup_credentials():
    creds = load_credentials()
    if creds and creds.get("ALPACA_API_KEY") and creds.get("ALPACA_API_SECRET"):
        return creds

    st.title("🎯 QuickTrend Binario - Setup")
    with st.form("form"):
        key = st.text_input("API Key", type="password")
        secret = st.text_input("API Secret", type="password")
        if st.form_submit_button("💾 Guardar y Conectar"):
            if save_credentials(key, secret):
                st.rerun()
    st.stop()

# -----------------------
# INDICADORES
# -----------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = np.where(ma_down > 1e-9, ma_up / ma_down, 100)
    return 100 - (100 / (1 + rs))

def get_signal(close_series):
    if len(close_series) < 20:
        return None

    try:
        ema8 = ema(close_series, 8).iloc[-1]
        ema21 = ema(close_series, 21).iloc[-1]
        rsi_val = rsi(close_series).iloc[-1]

        score = 0
        if ema8 > ema21: score += 30
        else: score -= 30
        if rsi_val < 30: score += 25  # Sobreventa → subida probable
        elif rsi_val > 70: score -= 25  # Sobrecompra → bajada probable

        direction = "🟢 SUBIR" if score > 0 else "🔴 BAJAR"
        confidence = min(95, max(5, 50 + abs(score) * 0.4))

        return {
            "direction": direction,
            "confidence": confidence,
            "rsi": rsi_val,
            "price": close_series.iloc[-1],
            "updated": datetime.now()
        }
    except Exception as e:
        logging.error(f"Error: {e}")
        return None

# -----------------------
# STREAMING PARA STOCKS/ETFs
# -----------------------
class StockLiveData:
    def __init__(self, key, secret, symbols):
        self.symbols = symbols
        self.data = {s: [] for s in symbols}
        self.signals = {}
        self.lock = threading.Lock()

        self.stream = StockDataStream(key, secret)
        for symbol in symbols:
            self.stream.subscribe_bars(self.on_bar, symbol)

    async def on_bar(self, bar):
        with self.lock:
            self.data[bar.symbol].append(bar.close)
            if len(self.data[bar.symbol]) > 50:
                self.data[bar.symbol] = self.data[bar.symbol][-50:]

            if len(self.data[bar.symbol]) >= 20:
                series = pd.Series(self.data[bar.symbol])
                signal = get_signal(series)
                if signal:
                    self.signals[bar.symbol] = signal

    def start(self):
        def run():
            self.stream.run()
        threading.Thread(target=run, daemon=True).start()

    def get_signals(self):
        with self.lock:
            return self.signals.copy()

# -----------------------
# ACTIVOS RECOMENDADOS (Baja Volatilidad + Alta Probabilidad)
# -----------------------
SYMBOLS = [
    "SPY",   # S&P 500 - Alta liquidez, patrones claros
    "TLT",   # Bonos a largo plazo - Baja volatilidad
    "IEF",   # Bonos 7-10 años - Movimientos suaves
    "AAPL",  # Apple - Líquida, técnicamente predecible
    "MSFT",  # Microsoft - Estable, buena para RSI
    "JNJ",   # Johnson & Johnson - Muy estable
    "DIA"    # Dow Jones - Menos volátil que SPY
]

# -----------------------
# MAIN APP
# -----------------------
def main():
    creds = setup_credentials()
    key = creds["ALPACA_API_KEY"]
    secret = creds["ALPACA_API_SECRET"]

    st.title("🎯 QuickTrend Binario - Señales en Vivo")
    st.markdown("### Activos de baja volatilidad: SPY, TLT, AAPL, IEF, etc.")

    if 'stream' not in st.session_state:
        with st.spinner("🔌 Conectando..."):
            st.session_state.stream = StockLiveData(key, secret, SYMBOLS)
            st.session_state.stream.start()
            st.success("✅ ¡Conectado! Recibiendo datos en vivo...")

    placeholder = st.empty()

    while True:
        signals = st.session_state.stream.get_signals()

        with placeholder.container():
            st.markdown(f"**Actualizado:** {datetime.now().strftime('%H:%M:%S')}")

            if not signals:
                st.info("⏳ Esperando datos...")
            else:
                cols = st.columns(2)
                for i, (symbol, sig) in enumerate(signals.items()):
                    with cols[i % 2]:
                        is_up = "SUBIR" in sig["direction"]
                        color = "#00C851" if is_up else "#FF4444"
                        emoji = "📈" if is_up else "📉"

                        st.markdown(f"""
                        <div style="background:{'#E8F5E8' if is_up else '#FFF0F0'}; padding:20px; border-radius:12px; border-left:5px solid {color}; margin-bottom:20px;">
                            <h3 style="margin:0; color:{color}">{emoji} {symbol}</h3>
                            <p style="font-size:1.3em; font-weight:bold;">${sig['price']:.2f}</p>
                            <p><b>Señal:</b> {sig['direction']}</p>
                            <p><b>Confianza:</b> {sig['confidence']:.1f}%</p>
                            <p><b>RSI:</b> {sig['rsi']:.1f}</p>
                        </div>
                        """, unsafe_allow_html=True)

        time.sleep(1)

if __name__ == "__main__":
    main()
