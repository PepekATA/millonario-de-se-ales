# Millonario de Señales Crypto 3.1
# Funciona con Alpaca Crypto (SOLUSD, BTCUSD, ETHUSD, AVAXUSD)
# ML incremental, backtesting simple, gráficos, control manual

import os, json, time, threading
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import subprocess, sys

# Instalar Alpaca si falta
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
try:
    import alpaca_trade_api as tradeapi
except ImportError:
    install("alpaca-trade-api")
    import alpaca_trade_api as tradeapi

# -------------------------
# Archivos de persistencia
# -------------------------
CRED_FILE="credentials.json"
STATE_FILE="positions.json"
MODEL_FILE="ml_model.json"

# -------------------------
# Manejo de credenciales
# -------------------------
def save_credentials(key, secret, paper=True):
    data = {"ALPACA_API_KEY":key.strip(),
            "ALPACA_API_SECRET":secret.strip(),
            "ALPACA_BASE_URL":"https://paper-api.alpaca.markets/v2" if paper else "https://api.alpaca.markets"}
    with open(CRED_FILE,"w") as f: json.dump(data,f,indent=2)

def load_credentials():
    if os.path.exists(CRED_FILE):
        try: return json.load(open(CRED_FILE,"r"))
        except: return None
    return None

def setup_credentials():
    creds=load_credentials()
    if creds and creds.get("ALPACA_API_KEY") and creds.get("ALPACA_API_SECRET"): return creds
    st.title("🤖 Millonario de Señales Crypto - Setup")
    with st.form("cred_form"):
        key = st.text_input("API Key", type="password")
        secret = st.text_input("API Secret", type="password")
        paper = st.checkbox("Usar cuenta demo (paper trading)", value=True)
        if st.form_submit_button("💾 Guardar y Conectar"):
            save_credentials(key, secret, paper)
            st.experimental_rerun()
    st.stop()

# -------------------------
# ML incremental simple
# -------------------------
class MLModel:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            try: return json.load(open(MODEL_FILE,"r"))
            except: pass
        return {}

    def save_model(self):
        with open(MODEL_FILE,"w") as f: json.dump(self.model,f,indent=2)

    def update(self, symbol, direction):
        if symbol not in self.model: self.model[symbol]={'up':0,'down':0}
        self.model[symbol][direction]+=1
        self.save_model()

    def predict(self, symbol):
        if symbol not in self.model: return 0.5
        total=self.model[symbol]['up']+self.model[symbol]['down']
        if total==0: return 0.5
        return self.model[symbol]['up']/total

# -------------------------
# Trade State Manager
# -------------------------
class TradeStateManager:
    def __init__(self):
        self.lock=threading.Lock()
        self.state=self.load_state()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try: return json.load(open(STATE_FILE,"r"))
            except: pass
        return {"positions":{}}

    def save_state(self):
        with self.lock: json.dump(self.state,open(STATE_FILE,"w"),indent=2)

    def add_position(self,symbol,qty,entry_price):
        with self.lock:
            if symbol not in self.state["positions"]: self.state["positions"][symbol]=[]
            pos_id=f"{symbol}_{int(time.time())}"
            self.state["positions"][symbol].append({"id":pos_id,"qty":qty,"entry_price":entry_price,
                                                    "highest_price":entry_price,"status":"open",
                                                    "created_at":datetime.now().isoformat()})
            self.save_state()
            return pos_id

    def update_position(self,symbol,current_price):
        with self.lock:
            if symbol in self.state["positions"]:
                for pos in self.state["positions"][symbol]:
                    if pos["status"]=="open":
                        pos["highest_price"]=max(pos["highest_price"],current_price)
            self.save_state()

    def get_open_positions(self,symbol=None):
        with self.lock:
            if symbol: return [p for p in self.state["positions"].get(symbol,[]) if p["status"]=="open"]
            all_pos=[]
            for sym,pos_list in self.state["positions"].items():
                all_pos.extend([p for p in pos_list if p["status"]=="open"])
            return all_pos

    def close_position(self,symbol,pos_id):
        with self.lock:
            if symbol in self.state["positions"]:
                for p in self.state["positions"][symbol]:
                    if p["id"]==pos_id and p["status"]=="open":
                        p["status"]="closed"
                        p["closed_at"]=datetime.now().isoformat()
                        self.save_state()
                        return True
        return False

# -------------------------
# Trader Crypto
# -------------------------
class CryptoTrader:
    def __init__(self,creds,ml_model):
        self.api=tradeapi.REST(creds["ALPACA_API_KEY"],creds["ALPACA_API_SECRET"],creds["ALPACA_BASE_URL"],api_version='v2')
        self.state_manager=TradeStateManager()
        self.ml_model=ml_model
        self.symbols=["SOLUSD","BTCUSD","ETHUSD","AVAXUSD"]
        self.is_running=False
        self.capital_fraction=0.1 # 10% por activo

    def get_historical(self,symbol,limit=50):
        try:
            bars=self.api.get_crypto_bars(symbol,"1Min").df
            bars=bars[bars['exchange']=='CBSE'] # tomar un exchange estable
            bars=bars.tail(limit)
            if bars.empty: return None
            return bars
        except: return None

    def execute_buy(self,symbol,current_price):
        try:
            account=self.api.get_account()
            buying_power=float(account.cash)
            qty=max(0.0001,(buying_power*self.capital_fraction)/current_price)
            self.api.submit_order(symbol=symbol,qty=qty,side="buy",type="market",time_in_force="day")
            self.state_manager.add_position(symbol,qty,current_price)
        except Exception as e: st.error(f"Error comprando {symbol}: {e}")

    def execute_sell(self,symbol,pos_id,qty,current_price):
        try:
            self.api.submit_order(symbol=symbol,qty=qty,side="sell",type="market",time_in_force="day")
            self.state_manager.close_position(symbol,pos_id)
        except Exception as e: st.error(f"Error vendiendo {symbol}: {e}")

    def check_positions(self,symbol,current_price):
        for pos in self.state_manager.get_open_positions(symbol):
            # vender si sube 0.3% como micro-ganancia
            profit=(current_price/pos["entry_price"]-1)
            if profit>0.003: self.execute_sell(symbol,pos["id"],pos["qty"],current_price)
            else: self.state_manager.update_position(symbol,current_price)

    def backtest_micro_gain(self,symbol,close_series):
        last_price=close_series.iloc[-1]
        predicted_up=self.ml_model.predict(symbol)
        return predicted_up>0.6 # compra si chance >60%

# -------------------------
# Streamlit App
# -------------------------
def main():
    creds=setup_credentials()
    ml_model=MLModel()
    trader=CryptoTrader(creds,ml_model)

    st.title("🤖 Millonario de Señales Crypto 3.1")
    st.sidebar.header("Control Manual")
    symbol_manual=st.sidebar.text_input("Símbolo","SOLUSD").upper()
    if st.sidebar.button("Comprar Manual"):
        df=trader.get_historical(symbol_manual,50)
        if df is not None: trader.execute_buy(symbol_manual,df["close"].iloc[-1])
    if st.sidebar.button("Vender Manual"):
        df=trader.get_historical(symbol_manual,50)
        open_pos=trader.state_manager.get_open_positions(symbol_manual)
        if open_pos: trader.execute_sell(symbol_manual,open_pos[-1]["id"],open_pos[-1]["qty"],df["close"].iloc[-1])
    if st.sidebar.button("Iniciar Bot"): trader.is_running=True
    if st.sidebar.button("Detener Bot"): trader.is_running=False

    # Posiciones
    st.subheader("📈 Posiciones Abiertas")
    for sym in trader.symbols:
        pos_list=trader.state_manager.get_open_positions(sym)
        if pos_list:
            st.markdown(f"### {sym}")
            for p in pos_list:
                profit=(p["highest_price"]/p["entry_price"]-1)*100
                color="green" if profit>0 else "red"
                st.markdown(f"<span style='color:{color}'>ID:{p['id'][-6:]} Qty:{p['qty']:.6f} Entrada:${p['entry_price']:.2f} Ganancia:{profit:.2f}%</span>",unsafe_allow_html=True)

    # Graficos
    for sym in trader.symbols:
        df=trader.get_historical(sym,50)
        if df is not None:
            fig=go.Figure()
            fig.add_trace(go.Candlestick(x=df.index,open=df.open,high=df.high,low=df.low,close=df.close))
            st.plotly_chart(fig,use_container_width=True)

    # Bot auto-trade (thread)
    def auto_trade():
        while trader.is_running:
            for sym in trader.symbols:
                df=trader.get_historical(sym,50)
                if df is None: continue
                last_close=df["close"].iloc[-1]
                # actualizar posiciones
                trader.check_positions(sym,last_close)
                # predicción micro-trade
                if trader.backtest_micro_gain(sym,df["close"]):
                    trader.execute_buy(sym,last_close)
            time.sleep(15)
    if trader.is_running: threading.Thread(target=auto_trade,daemon=True).start()

if __name__=="__main__":
    main()
