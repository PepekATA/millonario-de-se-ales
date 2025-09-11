# =======================================================
# QuickTrend Trader Pro - Todo en 1
# =======================================================

import sys
import subprocess

# ---------------------------
# Función para instalar dependencias automáticamente
# ---------------------------
def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = __import__(package)

# Instalar librerías necesarias
for pkg in ['alpaca_trade_api', 'pandas', 'numpy', 'matplotlib', 'streamlit', 'sklearn', 'tensorflow']:
    install_and_import(pkg)

# ---------------------------
# Librerías importadas
# ---------------------------
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import json
import time
import threading
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# =======================================================
# CONFIGURACIÓN
# =======================================================
API_KEY = "TU_API_KEY"
API_SECRET = "TU_API_SECRET"
BASE_URL = "https://paper-api.alpaca.markets"
POSITIONS_FILE = "positions.json"

# Activos 24/7 en Alpaca Crypto
ASSETS = ['BTC/USD', 'ETH/USD', 'DOGE/USD', 'SOL/USD', 'UNI/USD']

# Inicializar API Alpaca
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# =======================================================
# FUNCIONES DE PERSISTENCIA
# =======================================================
def load_positions():
    try:
        with open(POSITIONS_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_positions(positions):
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)

positions = load_positions()

# =======================================================
# FUNCIONES DE IA
# =======================================================
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(60,1)),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = create_model()

def predict_trend(prices):
    if len(prices) < 60:
        return 0
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(np.array(prices).reshape(-1,1))
    X = []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i,0])
    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[1],1))
    pred = model.predict(X, verbose=0)
    return pred[-1][0]

# =======================================================
# FUNCIONES DE TRADING
# =======================================================
def get_historical(symbol, limit=200):
    bars = api.get_crypto_bars(symbol.replace('/',''), '1Min', limit=limit).df
    bars = bars[bars['exchange']=='CBSE']
    return bars

def get_balance():
    try:
        return float(api.get_account().cash)
    except:
        return 0

def allocate_capital():
    balance = get_balance()
    allocation = balance / len(ASSETS)
    return allocation

def buy(symbol, qty):
    try:
        api.submit_order(symbol.replace('/',''), qty, 'buy', 'market', 'gtc')
        positions[symbol] = positions.get(symbol,[])
        positions[symbol].append({'qty':qty,'entry_time':datetime.now().isoformat(),'entry_price':get_last_price(symbol)})
        save_positions(positions)
    except Exception as e:
        st.warning(f"Error comprando {symbol}: {e}")

def sell(symbol, qty, index):
    try:
        api.submit_order(symbol.replace('/',''), qty, 'sell', 'market', 'gtc')
        positions[symbol][index]['exit_time'] = datetime.now().isoformat()
        positions[symbol][index]['exit_price'] = get_last_price(symbol)
        save_positions(positions)
    except Exception as e:
        st.warning(f"Error vendiendo {symbol}: {e}")

def get_last_price(symbol):
    bars = get_historical(symbol, limit=1)
    return bars['close'].iloc[-1]

def monitor_positions():
    while bot_running[0]:
        for symbol in ASSETS:
            if symbol in positions:
                for i,pos in enumerate(positions[symbol]):
                    if 'exit_price' not in pos:
                        current = get_last_price(symbol)
                        if current > pos['entry_price']:
                            sell(symbol,pos['qty'],i)
        time.sleep(5)

# =======================================================
# INTERFAZ STREAMLIT
# =======================================================
st.title("🤖 QuickTrend Trader Pro - 24/7 Crypto")
st.sidebar.header("Controles del Bot")

bot_running = [False]

def start_bot():
    bot_running[0] = True
    threading.Thread(target=monitor_positions,daemon=True).start()
    
def stop_bot():
    bot_running[0] = False

if st.sidebar.button("Iniciar Bot"):
    start_bot()
if st.sidebar.button("Detener Bot"):
    stop_bot()

st.sidebar.header("Trading Manual")
for symbol in ASSETS:
    col1, col2 = st.sidebar.columns(2)
    if col1.button(f"Comprar {symbol}"):
        buy(symbol,1)
    if col2.button(f"Vender {symbol}"):
        if symbol in positions:
            for i,pos in enumerate(positions[symbol]):
                if 'exit_price' not in pos:
                    sell(symbol,pos['qty'],i)
                    break

st.header("📊 Estado Actual")
for symbol in ASSETS:
    prices = get_historical(symbol)['close'][-50:].tolist()
    trend = predict_trend(prices)
    color = 'green' if trend>prices[-1] else 'red'
    st.markdown(f"### {symbol} - Último: {prices[-1]:.2f} - Tendencia: {'Subiendo' if color=='green' else 'Bajando'}")
    fig, ax = plt.subplots()
    ax.plot(prices,color=color)
    st.pyplot(fig)
