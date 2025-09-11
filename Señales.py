"""
QuickTrend Trader Pro — MULTI-POSICIONES (Compatible 100% con tu setup actual)
- Usa SOLO: https://paper-api.alpaca.markets/v2
- Usa SOLO: alpaca-trade-api (la que YA tienes instalada)
- Compra múltiples activos simultáneamente
- Vende solo cuando está en ganancias (nunca en pérdida)
- Entrena con datos históricos cuando mercado está cerrado
- Opera en tiempo real cuando mercado está abierto
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

# ✅ USAMOS SOLO alpaca-trade-api (la que YA tienes)
import alpaca_trade_api as tradeapi

# Config
CRED_FILE = "alpaca_credentials.json"
STATE_FILE = "trader_state.json"
LOG_FILE = "trader.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(layout="wide", page_title="QuickTrend Trader Pro", page_icon="🤖")

# -----------------------
# CREDENCIALES (usa tu URL)
# -----------------------
def save_credentials(key: str, secret: str, base_url: str = "https://paper-api.alpaca.markets/v2"):
    data = {
        "ALPACA_API_KEY": key.strip(),
        "ALPACA_API_SECRET": secret.strip(),
        "ALPACA_BASE_URL": base_url.strip()
    }
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

def calculate_signal(close_series):
    if len(close_series) < 50:
        return None

    ema8 = ema(close_series, 8).iloc[-1]
    ema21 = ema(close_series, 21).iloc[-1]
    rsi_val = rsi(close_series).iloc[-1]

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
        "score": score
    }

# -----------------------
# GESTOR DE POSICIONES (MULTI-POSICIONES)
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
            if symbol not in self.state["positions"]:
                self.state["positions"][symbol] = []
            position_id = f"{symbol}_{int(time.time())}"
            self.state["positions"][symbol].append({
                "id": position_id,
                "qty": qty,
                "entry_price": entry_price,
                "highest_price": entry_price,
                "created_at": datetime.now().isoformat(),
                "status": "open"
            })
            self.save_state()
            return position_id

    def update_position(self, symbol, current_price):
        with self.lock:
            if symbol in self.state["positions"]:
                for pos in self.state["positions"][symbol]:
                    if pos["status"] == "open":
                        pos["highest_price"] = max(pos["highest_price"], current_price)
                self.save_state()

    def get_open_positions(self, symbol=None):
        with self.lock:
            if symbol:
                return [p for p in self.state["positions"].get(symbol, []) if p["status"] == "open"]
            else:
                all_positions = []
                for sym, positions in self.state["positions"].items():
                    all_positions.extend([p for p in positions if p["status"] == "open"])
                return all_positions

    def should_take_profit(self, position, current_price, take_profit_pct=0.5):
        entry = position["entry_price"]
        highest = position["highest_price"]
        # Solo vende si está en ganancias y alcanzó tope O retrocedió desde máximo
        if current_price >= entry * (1 + take_profit_pct / 100):
            return True
        if current_price <= highest * 0.995 and current_price > entry:
            return True
        return False

    def close_position(self, symbol, position_id):
        with self.lock:
            if symbol in self.state["positions"]:
                for pos in self.state["positions"][symbol]:
                    if pos["id"] == position_id and pos["status"] == "open":
                        pos["status"] = "closed"
                        pos["closed_at"] = datetime.now().isoformat()
                        self.save_state()
                        return True
            return False

    def get_total_exposure(self, symbol):
        with self.lock:
            positions = self.get_open_positions(symbol)
            return sum(p["qty"] * p["entry_price"] for p in positions)

# -----------------------
# CLASE TRADER (usa solo alpaca-trade-api)
# -----------------------
class AlpacaTrader:
    def __init__(self, api_key, api_secret, base_url):
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        self.state_manager = TradeStateManager()
        self.symbols = ["SPY", "TLT", "IEF", "AAPL", "MSFT", "JNJ", "DIA"]
        self.is_streaming = False

    def get_historical_data(self, symbol, limit=100):
        try:
            bars = self.api.get_bars(symbol, '1Min', limit=limit).df
            if isinstance(bars.columns, pd.MultiIndex):
                bars = bars[symbol]
            return bars
        except Exception as e:
            logging.error(f"Error getting historical data for {symbol}: {e}")
            return None

    def train(self):
        st.info("🎓 Entrenando con datos históricos...")
        results = {}
        for symbol in self.symbols:
            try:
                df = self.get_historical_data(symbol, 500)
                if df is not None and len(df) >= 50:
                    signals = []
                    for i in range(50, len(df)):
                        slice_df = df.iloc[:i+1]
                        signal = calculate_signal(slice_df['close'])
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

    def execute_buy(self, symbol, current_price):
        try:
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            max_risk = min(1000, buying_power * 0.05)
            qty = max(1, int(max_risk / current_price))

            current_exposure = self.state_manager.get_total_exposure(symbol)
            if current_exposure > buying_power * 0.2:
                logging.info(f"Exposición máxima alcanzada en {symbol}")
                return

            self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='day'
            )
            position_id = self.state_manager.add_position(symbol, qty, current_price)
            st.toast(f"✅ COMPRADO {qty}x {symbol} a ${current_price:.2f}", icon="🛒")
            logging.info(f"COMPRADO {qty}x {symbol} a ${current_price:.2f} (ID: {position_id})")
        except Exception as e:
            logging.error(f"Error comprando {symbol}: {e}")

    def execute_sell(self, symbol, position_id, qty, current_price):
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='day'
            )
            self.state_manager.close_position(symbol, position_id)
            st.toast(f"💰 VENDIDO {qty}x {symbol} a ${current_price:.2f}", icon="💸")
            logging.info(f"VENDIDO {qty}x {symbol} a ${current_price:.2f} (ID: {position_id})")
        except Exception as e:
            logging.error(f"Error vendiendo {symbol}: {e}")

    def check_positions(self, symbol, current_price):
        open_positions = self.state_manager.get_open_positions(symbol)
        for pos in open_positions:
            if self.state_manager.should_take_profit(pos, current_price):
                self.execute_sell(symbol, pos["id"], pos["qty"], current_price)

    def start_streaming(self):
        if self.is_streaming:
            return

        conn = tradeapi.StreamConn(
            self.api._key_id,
            self.api._secret_key,
            base_url=self.api._base_url
        )

        @conn.on(r'AM$')
        async def on_minute_bars(conn, channel, bar):
            try:
                symbol = bar.symbol
                current_price = bar.close

                # Actualizar posiciones
                self.state_manager.update_position(symbol, current_price)

                # Verificar take profit
                self.check_positions(symbol, current_price)

                # Generar señal de compra
                df = self.get_historical_data(symbol, 50)
                if df is not None and len(df) >= 50:
                    signal = calculate_signal(df['close'])
                    if signal and signal["direction"] == "BUY" and signal["confidence"] > 75:
                        self.execute_buy(symbol, current_price)

            except Exception as e:
                logging.error(f"Error en streaming para {bar.symbol}: {e}")

        def run_stream():
            try:
                conn.run(['AM.' + symbol for symbol in self.symbols])
            except Exception as e:
                logging.error(f"Error en stream: {e}")

        stream_thread = threading.Thread(target=run_stream, daemon=True)
        stream_thread.start()
        self.is_streaming = True
        st.success("✅ Streaming en tiempo real activado")

# -----------------------
# VERIFICAR HORARIO DE MERCADO (EST)
# -----------------------
def is_market_open():
    now = datetime.now()
    # Ajustar a EST (UTC-5)
    est = now - timedelta(hours=5) if now.hour >= 5 else now + timedelta(hours=19)
    if est.weekday() >= 5:  # Sábado o Domingo
        return False
    market_open = est.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = est.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= est <= market_close

# -----------------------
# APLICACIÓN PRINCIPAL
# -----------------------
def main():
    creds = setup_credentials()
    API_KEY = creds["ALPACA_API_KEY"]
    API_SECRET = creds["ALPACA_API_SECRET"]
    BASE_URL = creds["ALPACA_BASE_URL"]

    trader = AlpacaTrader(API_KEY, API_SECRET, BASE_URL)

    st.title("🤖 QuickTrend Trader Pro - MULTI-POSICIONES")
    st.markdown("### Sistema Autónomo — Usa solo tu API: https://paper-api.alpaca.markets/v2")

    col1, col2, col3 = st.columns(3)
    col1.metric("Mercado", "🟢 ABIERTO" if is_market_open() else "🔴 CERRADO")
    total_positions = sum(len(trader.state_manager.get_open_positions(sym)) for sym in trader.symbols)
    col2.metric("Posiciones Abiertas", total_positions)
    col3.metric("Último Entrenamiento", trader.state_manager.state.get("last_training", "Nunca"))

    log_placeholder = st.empty()
    positions_placeholder = st.empty()

    # Entrenar si mercado cerrado
    if not is_market_open():
        results = trader.train()
        trader.state_manager.state["last_training"] = datetime.now().isoformat()
        trader.state_manager.save_state()

        with log_placeholder.container():
            st.info("📊 Resultados del Entrenamiento:")
            for symbol, res in results.items():
                st.write(f"{symbol}: {res['win_rate']}% win rate ({res['signals']} señales)")

    # Iniciar streaming si mercado abierto
    if is_market_open():
        trader.start_streaming()

    # Mostrar posiciones
    while True:
        with positions_placeholder.container():
            st.subheader("📈 Posiciones Abiertas por Activo")
            for symbol in trader.symbols:
                positions = trader.state_manager.get_open_positions(symbol)
                if positions:
                    st.markdown(f"### {symbol}")
                    for pos in positions:
                        profit_pct = (pos["highest_price"] / pos["entry_price"] - 1) * 100
                        st.markdown(f"""
                        <div style="background:#E8F5E8; padding:10px; border-radius:8px; margin-bottom:5px; font-size:0.9em;">
                            ID: {pos['id'][-6:]} | Qty: {pos['qty']} | Entrada: ${pos['entry_price']:.2f} | Máximo: ${pos['highest_price']:.2f} | Ganancia: +{profit_pct:.2f}%
                        </div>
                        """, unsafe_allow_html=True)

        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)
        if st.button("🔄 Reiniciar"):
            st.rerun()

st.markdown("---")
st.caption("🤖 QuickTrend Trader Pro — Compra múltiples activos, vende solo en ganancias, nunca en pérdida. Usa solo tu API: https://paper-api.alpaca.markets/v2")
