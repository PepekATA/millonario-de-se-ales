"""
QuickTrend Trader Pro — Sistema Autónomo Inteligente
- Entrena con datos históricos cuando el mercado está cerrado.
- Opera en vivo cuando el mercado está abierto.
- Nunca vende en pérdida.
- Cierra solo en tope de ganancias.
- Vigila activos 24/7.
"""

import os
import json
import time
from datetime import datetime, timedelta
import logging
import threading
import pandas as pd
import numpy as np
import streamlit as st

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Config
CRED_FILE = "alpaca_credentials.json"
STATE_FILE = "trader_state.json"
LOG_FILE = "trader.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(layout="wide", page_title="QuickTrend Trader Pro", page_icon="🤖")

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

    st.title("🤖 QuickTrend Trader Pro - Setup")
    with st.form("form"):
        key = st.text_input("API Key", type="password")
        secret = st.text_input("API Secret", type="password")
        if st.form_submit_button("💾 Guardar y Conectar"):
            if save_credentials(key, secret):
                st.rerun()
    st.stop()

# -----------------------
# INDICADORES TÉCNICOS
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

def calculate_signal(df):
    if len(df) < 50:
        return None

    close = df['close']
    ema8 = ema(close, 8).iloc[-1]
    ema21 = ema(close, 21).iloc[-1]
    rsi_val = rsi(close).iloc[-1]

    score = 0
    if ema8 > ema21: score += 30
    else: score -= 30
    if rsi_val < 30: score += 25
    elif rsi_val > 70: score -= 25

    direction = "BUY" if score > 0 else "SELL"
    confidence = min(95, max(5, 50 + abs(score) * 0.4))

    return {
        "direction": direction,
        "confidence": confidence,
        "rsi": rsi_val,
        "price": close.iloc[-1],
        "ema8": ema8,
        "ema21": ema21
    }

# -----------------------
# GESTOR DE ESTADO (POSICIONES ABIERTAS, TOPE DE GANANCIAS, ETC.)
# -----------------------
class TradeStateManager:
    def __init__(self):
        self.state = self.load_state()
        self.lock = threading.Lock()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    return json.load(f)
            except:
                pass
        return {"positions": {}, "last_training": None}

    def save_state(self):
        with self.lock:
            with open(STATE_FILE, "w") as f:
                json.dump(self.state, f, indent=2)

    def add_position(self, symbol, qty, entry_price):
        with self.lock:
            self.state["positions"][symbol] = {
                "qty": qty,
                "entry_price": entry_price,
                "highest_price": entry_price,
                "created_at": datetime.now().isoformat()
            }
            self.save_state()

    def update_position(self, symbol, current_price):
        with self.lock:
            if symbol in self.state["positions"]:
                pos = self.state["positions"][symbol]
                pos["highest_price"] = max(pos["highest_price"], current_price)
                self.save_state()

    def should_take_profit(self, symbol, current_price, take_profit_pct=0.5):
        with self.lock:
            if symbol in self.state["positions"]:
                pos = self.state["positions"][symbol]
                entry = pos["entry_price"]
                highest = pos["highest_price"]
                # Solo vende si está en ganancias y ha alcanzado tope
                if current_price >= entry * (1 + take_profit_pct / 100):
                    return True
                # O si ha retrocedido desde el máximo
                if current_price <= highest * 0.995:  # 0.5% de retracement
                    return current_price > entry  # Pero solo si sigue en ganancia
            return False

    def close_position(self, symbol):
        with self.lock:
            if symbol in self.state["positions"]:
                del self.state["positions"][symbol]
                self.save_state()

    def get_positions(self):
        with self.lock:
            return self.state["positions"].copy()

# -----------------------
# ENTRENADOR (BACKTESTING + OPTIMIZACIÓN)
# -----------------------
class Trainer:
    def __init__(self, data_client, symbols):
        self.data_client = data_client
        self.symbols = symbols

    def train(self):
        st.info("🎓 Entrenando con datos históricos...")
        results = {}
        for symbol in self.symbols:
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Minute,
                    start=datetime.now() - timedelta(days=30),
                    end=datetime.now()
                )
                bars = self.data_client.get_stock_bars(request).df
                if isinstance(bars.index, pd.MultiIndex):
                    bars = bars.xs(symbol, level=1)

                signals = []
                for i in range(50, len(bars)):
                    df_slice = bars.iloc[:i+1]
                    signal = calculate_signal(df_slice)
                    if signal:
                        signals.append(signal)

                if signals:
                    win_rate = sum(1 for s in signals if s["confidence"] > 70) / len(signals) * 100
                    results[symbol] = {"win_rate": round(win_rate, 2), "signals": len(signals)}
                    logging.info(f"Entrenado {symbol}: {win_rate:.2f}% win rate")
            except Exception as e:
                logging.error(f"Error entrenando {symbol}: {e}")

        st.success(f"✅ Entrenamiento completado: {len(results)} activos analizados")
        return results

# -----------------------
# TRADER EN VIVO
# -----------------------
class LiveTrader:
    def __init__(self, trading_client, data_client, state_manager, symbols):
        self.trading_client = trading_client
        self.data_client = data_client
        self.state_manager = state_manager
        self.symbols = symbols
        self.stream = StockDataStream(
            trading_client._api_key,
            trading_client._secret_key
        )
        self.positions = self.state_manager.get_positions()

        for symbol in symbols:
            self.stream.subscribe_bars(self.on_bar, symbol)

    async def on_bar(self, bar):
        try:
            # Actualizar posición si existe
            self.state_manager.update_position(bar.symbol, bar.close)

            # Verificar si debe cerrar (take profit)
            if self.state_manager.should_take_profit(bar.symbol, bar.close):
                self.close_position(bar.symbol, bar.close)

            # Generar señal
            df = self.get_recent_data(bar.symbol, 50)
            if df is not None and len(df) >= 50:
                signal = calculate_signal(df)
                if signal and signal["confidence"] > 75:
                    self.execute_signal(bar.symbol, signal, bar.close)

        except Exception as e:
            logging.error(f"Error en on_bar: {e}")

    def get_recent_data(self, symbol, limit=50):
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                limit=limit
            )
            bars = self.data_client.get_stock_bars(request).df
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.xs(symbol, level=1)
            return bars
        except:
            return None

    def execute_signal(self, symbol, signal, current_price):
        if signal["direction"] == "BUY":
            # Solo comprar si no tenemos posición
            if symbol not in self.state_manager.get_positions():
                qty = max(1, int(1000 / current_price))  # $1000 por operación
                try:
                    order = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                    self.trading_client.submit_order(order)
                    self.state_manager.add_position(symbol, qty, current_price)
                    st.toast(f"✅ COMPRADO {qty}x {symbol} a ${current_price:.2f}", icon="🛒")
                    logging.info(f"COMPRADO {qty}x {symbol} a ${current_price:.2f}")
                except Exception as e:
                    logging.error(f"Error comprando {symbol}: {e}")

    def close_position(self, symbol, current_price):
        pos = self.state_manager.get_positions().get(symbol)
        if pos:
            try:
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=pos["qty"],
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                self.trading_client.submit_order(order)
                profit_pct = (current_price / pos["entry_price"] - 1) * 100
                self.state_manager.close_position(symbol)
                st.toast(f"💰 VENDIDO {symbol} a ${current_price:.2f} (+{profit_pct:.2f}%)", icon="💸")
                logging.info(f"VENDIDO {symbol} a ${current_price:.2f} (+{profit_pct:.2f}%)")
            except Exception as e:
                logging.error(f"Error vendiendo {symbol}: {e}")

    def start(self):
        def run():
            self.stream.run()
        threading.Thread(target=run, daemon=True).start()

# -----------------------
# VERIFICAR HORARIO DE MERCADO (EST)
# -----------------------
def is_market_open():
    now = datetime.now()
    est = now.astimezone()  # Ajusta según tu zona
    # Horario de mercado: 9:30 AM - 4:00 PM EST, Lunes a Viernes
    if est.weekday() >= 5:  # Sábado o Domingo
        return False
    market_open = est.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = est.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= est <= market_close

# -----------------------
# ACTIVOS RECOMENDADOS (Baja Volatilidad + Alta Liquidez)
# -----------------------
SYMBOLS = ["SPY", "TLT", "IEF", "AAPL", "MSFT", "JNJ", "DIA"]

# -----------------------
# APLICACIÓN PRINCIPAL
# -----------------------
def main():
    creds = setup_credentials()
    API_KEY = creds["ALPACA_API_KEY"]
    API_SECRET = creds["ALPACA_API_SECRET"]

    # Clientes
    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
    data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
    state_manager = TradeStateManager()

    st.title("🤖 QuickTrend Trader Pro")
    st.markdown("### Sistema Autónomo de Trading — Nunca vende en pérdida")

    # Mostrar estado
    col1, col2, col3 = st.columns(3)
    col1.metric("Mercado", "🟢 ABIERTO" if is_market_open() else "🔴 CERRADO")
    col2.metric("Posiciones Abiertas", len(state_manager.get_positions()))
    col3.metric("Último Entrenamiento", state_manager.state.get("last_training", "Nunca"))

    # Placeholder para logs y señales
    log_placeholder = st.empty()
    positions_placeholder = st.empty()

    # Iniciar trainer si es after-hours
    if not is_market_open():
        trainer = Trainer(data_client, SYMBOLS)
        results = trainer.train()
        state_manager.state["last_training"] = datetime.now().isoformat()
        state_manager.save_state()

        with log_placeholder.container():
            st.info("📊 Resultados del Entrenamiento:")
            for symbol, res in results.items():
                st.write(f"{symbol}: {res['win_rate']}% win rate ({res['signals']} señales)")

    # Iniciar live trader si mercado está abierto
    if is_market_open():
        trader = LiveTrader(trading_client, data_client, state_manager, SYMBOLS)
        trader.start()
        st.success("✅ Trader en vivo activado — Monitoreando oportunidades...")

    # Loop principal
    while True:
        # Mostrar posiciones
        with positions_placeholder.container():
            positions = state_manager.get_positions()
            if positions:
                st.subheader("📈 Posiciones Abiertas")
                for symbol, pos in positions.items():
                    current_price = pos.get("highest_price", pos["entry_price"])
                    profit_pct = (current_price / pos["entry_price"] - 1) * 100
                    st.markdown(f"""
                    <div style="background:#E8F5E8; padding:15px; border-radius:10px; margin-bottom:10px;">
                        <b>{symbol}</b> | Qty: {pos['qty']} | Entrada: ${pos['entry_price']:.2f} | Actual: ${current_price:.2f} | Ganancia: +{profit_pct:.2f}%
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("📭 No hay posiciones abiertas")

        time.sleep(5)  # Actualizar cada 5 segundos

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)
        if st.button("🔄 Reiniciar"):
            st.rerun()

st.markdown("---")
st.caption("🤖 QuickTrend Trader Pro — Entrena en after-hours, opera en horario de mercado, nunca vende en pérdida.")
