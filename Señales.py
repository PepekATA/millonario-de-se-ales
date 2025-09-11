# -*- coding: utf-8 -*-
import os, json, time, threading
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import alpaca_trade_api as tradeapi

# -----------------------
# Archivos de configuración
# -----------------------
CRED_FILE = "alpaca_credentials.json"
STATE_FILE = "trader_state.json"

# -----------------------
# Manejo de credenciales
# -----------------------
def save_credentials(key, secret, base_url):
    data = {
        "ALPACA_API_KEY": key.strip(),
        "ALPACA_API_SECRET": secret.strip(),
        "ALPACA_BASE_URL": base_url.strip()
    }
    with open(CRED_FILE,"w") as f:
        json.dump(data,f,indent=2)

def load_credentials():
    if os.path.exists(CRED_FILE):
        try:
            with open(CRED_FILE,"r") as f:
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
        account_type = st.radio("Tipo de cuenta", ["Demo", "Real"])
        if st.form_submit_button("💾 Guardar y Conectar"):
            base_url = "https://paper-api.alpaca.markets/v2" if account_type=="Demo" else "https://api.alpaca.markets"
            save_credentials(key, secret, base_url)
            st.experimental_rerun()
    st.stop()

# -----------------------
# Indicadores simples
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
    return 100-(100/(1+rs))

def calculate_signal(close_series):
    if len(close_series)<50: return None
    ema8 = ema(close_series,8).iloc[-1]
    ema21 = ema(close_series,21).iloc[-1]
    rsi_val = rsi(close_series).iloc[-1]
    score=0
    if ema8>ema21: score+=30
    else: score-=30
    if rsi_val<30: score+=25
    elif rsi_val>70: score-=25
    direction="BUY" if score>0 else "WAIT"
    confidence=min(95,max(5,50+abs(score)*0.4))
    return {"direction":direction,"confidence":confidence,"rsi":rsi_val,"score":score}

# -----------------------
# Gestión de posiciones
# -----------------------
class TradeStateManager:
    def __init__(self):
        self.state = self.load_state()
        self.lock = threading.Lock()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE,"r") as f:
                    return json.load(f)
            except:
                pass
        return {"positions":{},"last_training":None}

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
                "created_at":datetime.now().isoformat(),
                "status":"open"
            })
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
            if symbol:
                return [p for p in self.state["positions"].get(symbol,[]) if p["status"]=="open"]
            else:
                all_positions=[]
                for sym,positions in self.state["positions"].items():
                    all_positions.extend([p for p in positions if p["status"]=="open"])
                return all_positions

    def should_take_profit(self,position,current_price,take_profit_pct=0.5):
        entry = position["entry_price"]
        highest = position["highest_price"]
        if current_price >= entry*(1+take_profit_pct/100):
            return True
        if current_price >= entry*1.001 and current_price < highest*0.995:
            return True
        return False

    def close_position(self,symbol,position_id):
        with self.lock:
            if symbol in self.state["positions"]:
                for pos in self.state["positions"][symbol]:
                    if pos["id"]==position_id and pos["status"]=="open":
                        pos["status"]="closed"
                        pos["closed_at"]=datetime.now().isoformat()
                        self.save_state()
                        return True
            return False

    def get_total_exposure(self,symbol):
        with self.lock:
            positions = self.get_open_positions(symbol)
            return sum(p["qty"]*p["entry_price"] for p in positions)

# -----------------------
# Trader
# -----------------------
class AlpacaTrader:
    def __init__(self,api_key,api_secret,base_url):
        self.api=tradeapi.REST(api_key,api_secret,base_url,api_version='v2')
        self.state_manager=TradeStateManager()
        # Criptos 24/7
        self.symbols=["BTCUSD","ETHUSD","LTCUSD","BCHUSD","DOGEUSD"]
        self.running=False

    def get_historical_data(self,symbol,limit=100):
        try:
            bars=self.api.get_bars(symbol,"1Min",limit=limit).df
            # Manejar MultiIndex
            if isinstance(bars.columns,pd.MultiIndex):
                bars = bars[symbol]
            return bars
        except:
            return None

    def execute_buy(self,symbol,current_price):
        try:
            account=self.api.get_account()
            buying_power=float(account.buying_power)
            max_risk=min(1000,buying_power*0.05)
            qty=max(1,int(max_risk/current_price))
            if self.state_manager.get_total_exposure(symbol)>buying_power*0.2:
                return
            self.api.submit_order(symbol=symbol,qty=qty,side='buy',type='market',time_in_force='day')
            self.state_manager.add_position(symbol,qty,current_price)
        except:
            pass

    def execute_sell(self,symbol,position_id,qty,current_price):
        try:
            self.api.submit_order(symbol=symbol,qty=qty,side='sell',type='market',time_in_force='day')
            self.state_manager.close_position(symbol,position_id)
        except:
            pass

    def check_positions(self,symbol,current_price):
        open_positions=self.state_manager.get_open_positions(symbol)
        for pos in open_positions:
            if self.state_manager.should_take_profit(pos,current_price):
                self.execute_sell(symbol,pos["id"],pos["qty"],current_price)

    def start_bot(self):
        self.running=True
        def loop():
            while self.running:
                for symbol in self.symbols:
                    df=self.get_historical_data(symbol,50)
                    if df is not None and len(df)>=50:
                        current_price=df["close"].iloc[-1]
                        signal=calculate_signal(df["close"])
                        if signal and signal["direction"]=="BUY" and signal["confidence"]>70:
                            self.execute_buy(symbol,current_price)
                        self.check_positions(symbol,current_price)
                time.sleep(10)
        self.thread=threading.Thread(target=loop,daemon=True)
        self.thread.start()

    def stop_bot(self):
        self.running=False
        if hasattr(self,"thread") and self.thread.is_alive():
            self.thread.join()

# -----------------------
# Interfaz Streamlit
# -----------------------
def main():
    creds = setup_credentials()
    trader = AlpacaTrader(creds["ALPACA_API_KEY"],creds["ALPACA_API_SECRET"],creds["ALPACA_BASE_URL"])

    st.title("🤖 QuickTrend Trader Pro - 24/7 Cripto")
    col1,col2=st.columns(2)
    if col1.button("🚦 Iniciar Bot"):
        trader.start_bot()
        st.success("Bot iniciado!")
    if col2.button("🛑 Detener Bot"):
        trader.stop_bot()
        st.warning("Bot detenido!")

    st.subheader("📊 Gráficos y Posiciones")
    for symbol in trader.symbols:
        df=trader.get_historical_data(symbol,50)
        if df is not None:
            current_price=df["close"].iloc[-1]
            signal=calculate_signal(df["close"])
            color='green' if signal and signal["direction"]=="BUY" else 'red'
            st.markdown(f"### {symbol} - Precio: ${current_price:.2f}")
            st.line_chart(df["close"],use_container_width=True)
            open_positions=trader.state_manager.get_open_positions(symbol)
            for pos in open_positions:
                profit_pct=(current_price/pos["entry_price"]-1)*100
                st.markdown(f"ID: {pos['id'][-6:]} | Qty: {pos['qty']} | Entrada: ${pos['entry_price']:.2f} | Ganancia: {profit_pct:.2f}%",unsafe_allow_html=True)

if __name__=="__main__":
    main()
