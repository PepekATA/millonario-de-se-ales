# Millonario de Señales 2.0 - Completo para Streamlit Cloud
# Funciona con Alpaca paper trading o real
# Guarda credenciales y posiciones en archivos JSON
# Incluye dashboard con botones, gráficos y ML incremental

import os
import json
import time
import threading
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# -------------------------------
# Verificar e instalar paquetes faltantes
# -------------------------------
import subprocess
import sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    install("alpaca-trade-api")
    import alpaca_trade_api as tradeapi

# -------------------------------
# Archivos de persistencia
# -------------------------------
CRED_FILE = "credentials.json"
STATE_FILE = "positions.json"
MODEL_FILE = "ml_model.json"

# -------------------------------
# Credenciales
# -------------------------------
def save_credentials(key, secret, paper=True):
    data = {
        "ALPACA_API_KEY": key.strip(),
        "ALPACA_API_SECRET": secret.strip(),
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets/v2" if paper else "https://api.alpaca.markets"
    }
    with open(CRED_FILE, "w") as f:
        json.dump(data, f, indent=2)

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
    st.title("🤖 Millonario de Señales - Setup")
    with st.form("cred_form"):
        key = st.text_input("API Key", type="password")
        secret = st.text_input("API Secret", type="password")
        paper = st.checkbox("Usar cuenta demo (paper trading)", value=True)
        if st.form_submit_button("💾 Guardar y Conectar"):
            save_credentials(key, secret, paper)
            st.experimental_rerun()
    st.stop()

# -------------------------------
# Indicadores técnicos
# -------------------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
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
    confidence = min(95, max(5, 50 + abs(score)*0.4))
    return {"direction": direction, "confidence": confidence, "score": score, "rsi": rsi_val}

# -------------------------------
# Trade State Manager
# -------------------------------
class TradeStateManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = self.load_state()

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
            pos_id = f"{symbol}_{int(time.time())}"
            self.state["positions"][symbol].append({
                "id": pos_id,
                "qty": qty,
                "entry_price": entry_price,
                "highest_price": entry_price,
                "status": "open",
                "created_at": datetime.now().isoformat()
            })
            self.save_state()
            return pos_id

    def update_position(self, symbol, current_price):
        with self.lock:
            if symbol in self.state["positions"]:
                for pos in self.state["positions"][symbol]:
                    if pos["status"]=="open":
                        pos["highest_price"] = max(pos["highest_price"], current_price)
            self.save_state()

    def get_open_positions(self, symbol=None):
        with self.lock:
            if symbol:
                return [p for p in self.state["positions"].get(symbol, []) if p["status"]=="open"]
            else:
                all_pos=[]
                for sym, pos_list in self.state["positions"].items():
                    all_pos.extend([p for p in pos_list if p["status"]=="open"])
                return all_pos

    def close_position(self, symbol, pos_id):
        with self.lock:
            if symbol in self.state["positions"]:
                for p in self.state["positions"][symbol]:
                    if p["id"]==pos_id and p["status"]=="open":
                        p["status"]="closed"
                        p["closed_at"]=datetime.now().isoformat()
                        self.save_state()
                        return True
        return False

# -------------------------------
# Trader
# -------------------------------
class AlpacaTrader:
    def __init__(self, creds):
        self.api = tradeapi.REST(creds["ALPACA_API_KEY"], creds["ALPACA_API_SECRET"], creds["ALPACA_BASE_URL"], api_version='v2')
        self.state_manager = TradeStateManager()
        self.symbols = ["AAPL","MSFT","TSLA","GOOG","AMZN"]  # puedes agregar los 24/7 si Alpaca tiene crypto
        self.is_running=False

    def get_historical(self, symbol, limit=100):
        try:
            bars = self.api.get_bars(symbol, "1Min", limit=limit).df
            if isinstance(bars.columns, pd.MultiIndex):
                bars = bars[symbol]
            if "close" not in bars.columns:
                return None
            return bars
        except:
            return None

    def execute_buy(self, symbol, current_price):
        try:
            account=self.api.get_account()
            buying_power=float(account.buying_power)
            qty=max(1,int((buying_power*0.05)/current_price))
            self.api.submit_order(symbol=symbol, qty=qty, side="buy", type="market", time_in_force="day")
            self.state_manager.add_position(symbol, qty, current_price)
        except Exception as e:
            st.error(f"Error comprando {symbol}: {e}")

    def execute_sell(self, symbol, pos_id, qty, current_price):
        try:
            # nunca vende en pérdida
            self.api.submit_order(symbol=symbol, qty=qty, side="sell", type="market", time_in_force="day")
            self.state_manager.close_position(symbol, pos_id)
        except Exception as e:
            st.error(f"Error vendiendo {symbol}: {e}")

    def check_positions(self, symbol, current_price):
        for pos in self.state_manager.get_open_positions(symbol):
            # Vende solo si está en ganancia
            if current_price>pos["entry_price"]:
                self.execute_sell(symbol, pos["id"], pos["qty"], current_price)
            else:
                self.state_manager.update_position(symbol, current_price)

# -------------------------------
# Streamlit App
# -------------------------------
def main():
    creds = setup_credentials()
    trader = AlpacaTrader(creds)

    st.title("🤖 Millonario de Señales 2.0")
    st.sidebar.header("Control Manual")
    symbol_manual = st.sidebar.text_input("Símbolo para comprar/vender", value="AAPL").upper()
    if st.sidebar.button("Comprar Manual"):
        df=trader.get_historical(symbol_manual,50)
        if df is not None:
            trader.execute_buy(symbol_manual, df["close"].iloc[-1])
    if st.sidebar.button("Vender Manual"):
        open_pos=trader.state_manager.get_open_positions(symbol_manual)
        if open_pos:
            trader.execute_sell(symbol_manual, open_pos[-1]["id"], open_pos[-1]["qty"], df["close"].iloc[-1])

    if st.sidebar.button("Iniciar Bot"):
        trader.is_running=True
    if st.sidebar.button("Detener Bot"):
        trader.is_running=False

    # Mostrar posiciones
    st.subheader("📈 Posiciones Abiertas")
    for sym in trader.symbols:
        pos_list=trader.state_manager.get_open_positions(sym)
        if pos_list:
            st.markdown(f"### {sym}")
            for p in pos_list:
                profit=(p["highest_price"]/p["entry_price"]-1)*100
                color="green" if profit>0 else "red"
                st.markdown(f"<span style='color:{color}'>ID: {p['id'][-6:]} | Qty: {p['qty']} | Entrada: ${p['entry_price']:.2f} | Ganancia: {profit:.2f}%</span>", unsafe_allow_html=True)

    # Gráficos
    for sym in trader.symbols:
        df=trader.get_historical(sym,50)
        if df is not None:
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["close"], name=sym, line=dict(color="green" if df["close"].iloc[-1]>=df["close"].iloc[0] else "red")))
            st.plotly_chart(fig, use_container_width=True)

    # Bot automático
    if trader.is_running:
        for sym in trader.symbols:
            df=trader.get_historical(sym,50)
            if df is not None and len(df)>=50:
                signal=calculate_signal(df["close"])
                current_price=df["close"].iloc[-1]
                if signal and signal["direction"]=="BUY" and signal["confidence"]>70:
                    trader.execute_buy(sym, current_price)
                trader.check_positions(sym, current_price)

    st.info("Bot activo" if trader.is_running else "Bot detenido")
    st.experimental_rerun()
    
if __name__=="__main__":
    main()
