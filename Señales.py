# millonario_streamlit.py
import os
import json
import time
import logging
from datetime import datetime
import threading

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import alpaca_trade_api as tradeapi

# -----------------------
# Configuración
# -----------------------
CRED_FILE = "alpaca_credentials.json"
STATE_FILE = "trader_state.json"
LOG_FILE = "trader.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------
# Manejo de Credenciales
# -----------------------
def save_credentials(key, secret, base_url):
    data = {"ALPACA_API_KEY": key.strip(),
            "ALPACA_API_SECRET": secret.strip(),
            "ALPACA_BASE_URL": base_url.strip()}
    with open(CRED_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_credentials():
    if os.path.exists(CRED_FILE):
        try:
            with open(CRED_FILE, "r") as f:
                data = f.read().strip()
                if not data:
                    return None
                return json.loads(data)
        except json.JSONDecodeError:
            return None
    return None

def setup_credentials():
    creds = load_credentials()
    if creds:
        return creds
    st.title("Configuración de Alpaca")
    with st.form("form_credentials"):
        key = st.text_input("API Key", type="password")
        secret = st.text_input("API Secret", type="password")
        base_url = st.selectbox("Tipo de cuenta", ["https://paper-api.alpaca.markets/v2",
                                                   "https://api.alpaca.markets"], index=0)
        if st.form_submit_button("Guardar"):
            save_credentials(key, secret, base_url)
            st.experimental_rerun()
    st.stop()

# -----------------------
# Indicadores
# -----------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = np.where(ma_down>1e-9, ma_up/ma_down, 100)
    return 100 - (100/(1+rs))

def calculate_signal(close_series):
    if len(close_series)<50: return None
    ema8 = ema(close_series,8).iloc[-1]
    ema21 = ema(close_series,21).iloc[-1]
    rsi_val = rsi(close_series).iloc[-1]
    score = 0
    if ema8>ema21: score+=30
    else: score-=30
    if rsi_val<30: score+=25
    elif rsi_val>70: score-=25
    direction = "BUY" if score>0 else "SELL"
    confidence = min(95,max(5,50+abs(score)*0.4))
    return {"direction":direction,"confidence":confidence,"rsi":rsi_val,"score":score}

# -----------------------
# Estado de Posiciones
# -----------------------
class TradeStateManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = self.load_state()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE,"r") as f:
                    data = f.read().strip()
                    if not data:
                        return {"positions":{}, "last_training":None}
                    return json.loads(data)
            except:
                return {"positions":{}, "last_training":None}
        return {"positions":{}, "last_training":None}

    def save_state(self):
        with self.lock:
            with open(STATE_FILE,"w") as f:
                json.dump(self.state,f,indent=2)

    def add_position(self,symbol,qty,entry_price):
        with self.lock:
            if symbol not in self.state["positions"]:
                self.state["positions"][symbol]=[]
            pos_id=f"{symbol}_{int(time.time())}"
            self.state["positions"][symbol].append({
                "id":pos_id,
                "qty":qty,
                "entry_price":entry_price,
                "highest_price":entry_price,
                "status":"open",
                "created_at":datetime.now().isoformat()
            })
            self.save_state()
            return pos_id

    def update_position(self,symbol,current_price):
        with self.lock:
            for pos in self.state["positions"].get(symbol,[]):
                if pos["status"]=="open":
                    pos["highest_price"]=max(pos["highest_price"],current_price)
            self.save_state()

    def get_open_positions(self,symbol=None):
        with self.lock:
            if symbol:
                return [p for p in self.state["positions"].get(symbol,[]) if p["status"]=="open"]
            else:
                all_pos=[]
                for sym,positions in self.state["positions"].items():
                    all_pos.extend([p for p in positions if p["status"]=="open"])
                return all_pos

    def should_take_profit(self,pos,current_price,take_profit_pct=0.5):
        # Vende solo si está en ganancia
        entry = pos["entry_price"]
        if current_price>=entry*(1+take_profit_pct/100):
            return True
        return False

    def close_position(self,symbol,pos_id):
        with self.lock:
            for pos in self.state["positions"].get(symbol,[]):
                if pos["id"]==pos_id and pos["status"]=="open":
                    pos["status"]="closed"
                    pos["closed_at"]=datetime.now().isoformat()
                    self.save_state()
                    return True
            return False

# -----------------------
# Trader Alpaca
# -----------------------
class AlpacaTrader:
    def __init__(self,api_key,api_secret,base_url):
        self.api=tradeapi.REST(api_key,api_secret,base_url,api_version='v2')
        self.state_manager=TradeStateManager()
        self.symbols=["SPY","AAPL","MSFT","TSLA"]  # ej. activos 24/7, puedes ajustar
        self.is_running=False

    def get_historical_data(self,symbol,limit=100):
        try:
            bars=self.api.get_bars(symbol,'1Min',limit=limit).df
            if bars.empty: return None
            if isinstance(bars.columns,pd.MultiIndex):
                bars=bars[symbol]
            if "close" not in bars.columns: return None
            return bars
        except Exception as e:
            logging.error(f"Error datos históricos {symbol}: {e}")
            return None

    def execute_buy(self,symbol,current_price):
        # Compra demo con cantidad fija para ejemplo
        qty=max(1,int(100/current_price))
        self.state_manager.add_position(symbol,qty,current_price)
        logging.info(f"Comprado {qty} {symbol} a {current_price}")

    def execute_sell(self,symbol,pos_id,qty,current_price):
        self.state_manager.close_position(symbol,pos_id)
        logging.info(f"Vendido {qty} {symbol} a {current_price}")

# -----------------------
# Streamlit App
# -----------------------
def main():
    creds=setup_credentials()
    trader=AlpacaTrader(creds["ALPACA_API_KEY"],creds["ALPACA_API_SECRET"],creds["ALPACA_BASE_URL"])

    st.title("🤖 Millonario de Señales")

    # Botones manuales
    st.sidebar.header("Control Manual")
    symbol_input=st.sidebar.text_input("Símbolo para comprar/vender","AAPL")
    if st.sidebar.button("Comprar Manual"):
        df=trader.get_historical_data(symbol_input,50)
        if df is not None:
            trader.execute_buy(symbol_input,df["close"].iloc[-1])
    if st.sidebar.button("Vender Manual"):
        pos=trader.state_manager.get_open_positions(symbol_input)
        if pos:
            trader.execute_sell(symbol_input,pos[0]["id"],pos[0]["qty"],pos[0]["highest_price"])

    # Mostrar posiciones abiertas
    st.subheader("📈 Posiciones Abiertas")
    for sym in trader.symbols:
        pos_list=trader.state_manager.get_open_positions(sym)
        if pos_list:
            df=trader.get_historical_data(sym,50)
            if df is not None:
                trend_color="green" if df["close"].iloc[-1]>df["close"].iloc[0] else "red"
                fig=go.Figure(go.Candlestick(x=df.index,
                                             open=df["open"],high=df["high"],
                                             low=df["low"],close=df["close"]))
                st.plotly_chart(fig,use_container_width=True)
            for p in pos_list:
                profit_pct=(p["highest_price"]/p["entry_price"]-1)*100
                st.markdown(f"{sym} | Qty: {p['qty']} | Entrada: ${p['entry_price']:.2f} | Ganancia: {profit_pct:.2f}%")

if __name__=="__main__":
    main()
