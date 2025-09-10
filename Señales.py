"""
QuickTrend 24/7 — Predictor de Tendencias Forex/Crypto
- Guarda credenciales permanentemente (solo pide una vez)
- Muestra SOLO activos 24/7 con predicciones visuales
- Timeframes múltiples con probabilidades y duraciones
- Gráficos en tiempo real con colores predictivos
"""

import os
import json
import time
from datetime import datetime, timedelta
import math
import concurrent.futures
from typing import Dict, List, Optional
import logging

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Configurar logging
logging.basicConfig(filename='app_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Alpaca client imports (compatibility)
try:
    import alpaca_trade_api as tradeapi
    ALPACA_LIB = "tradeapi"
except Exception:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        ALPACA_LIB = "alpaca-py"
    except Exception:
        ALPACA_LIB = None

# -----------------------
# CONFIGURACIÓN INICIAL
# -----------------------
CRED_FILE = "alpaca_credentials.json"
CACHE_FILE = "market_cache.json"

st.set_page_config(
    layout="wide", 
    page_title="QuickTrend 24/7 Predictor", 
    initial_sidebar_state="expanded",
    page_icon="🔮"
)

# -----------------------
# SISTEMA DE CREDENCIALES PERMANENTE
# -----------------------
def save_credentials(key: str, secret: str, base_url: str = "https://paper-api.alpaca.markets"):
    """Guarda credenciales de forma permanente"""
    # Sanitizar base_url
    base_url = base_url.strip()
    data = {
        "ALPACA_API_KEY": key,
        "ALPACA_API_SECRET": secret,
        "ALPACA_BASE_URL": base_url,
        "saved_at": datetime.now().isoformat()
    }
    try:
        with open(CRED_FILE, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error guardando credenciales: {e}")
        logging.error(f"Error saving credentials: {e}")
        return False

def load_credentials():
    """Carga credenciales guardadas"""
    if os.path.exists(CRED_FILE):
        try:
            with open(CRED_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading credentials: {e}")
            return None
    return None

def setup_credentials():
    """Sistema de setup de credenciales (solo una vez)"""
    creds = load_credentials()
    
    if creds and creds.get("ALPACA_API_KEY") and creds.get("ALPACA_API_SECRET"):
        return creds
    
    # Primera vez - pedir credenciales
    st.title("🔮 QuickTrend 24/7 - Configuración Inicial")
    st.markdown("### Configuración de API Alpaca (Solo una vez)")
    
    with st.form("credentials_form"):
        st.markdown("Introduce tus credenciales de Alpaca Markets:")
        col1, col2 = st.columns(2)
        
        with col1:
            api_key = st.text_input("API Key", type="password", help="Tu API Key de Alpaca")
        with col2:
            api_secret = st.text_input("API Secret", type="password", help="Tu API Secret de Alpaca")
        
        base_url = st.selectbox(
            "Entorno",
            ["https://paper-api.alpaca.markets", "https://api.alpaca.markets"],
            help="Paper trading (recomendado para pruebas)"
        )
        
        submitted = st.form_submit_button("💾 Guardar Credenciales")
        
        if submitted:
            if api_key and api_secret:
                if save_credentials(api_key.strip(), api_secret.strip(), base_url):
                    st.success("✅ Credenciales guardadas correctamente!")
                    st.info("🔄 Recargando aplicación...")
                    time.sleep(2)
                    st.rerun()
            else:
                st.error("❌ Por favor introduce ambas credenciales")
    
    st.stop()

# -----------------------
# CLIENTE ALPACA
# -----------------------
def create_alpaca_client(creds):
    """Crea cliente Alpaca con manejo de errores"""
    try:
        api_key = creds["ALPACA_API_KEY"]
        api_secret = creds["ALPACA_API_SECRET"]
        base_url = creds.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").strip()  # Sanitizado
        
        if ALPACA_LIB == "tradeapi":
            return tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        elif ALPACA_LIB == "alpaca-py":
            return StockHistoricalDataClient(api_key, api_secret)
        else:
            st.error("❌ Instala 'alpaca-trade-api' o 'alpaca-py'")
            return None
    except Exception as e:
        st.error(f"❌ Error conectando con Alpaca: {e}")
        logging.error(f"Alpaca client error: {e}")
        return None

# -----------------------
# INDICADORES TÉCNICOS AVANZADOS
# -----------------------
def ema(series, span):
    """Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    """RSI con manejo robusto de casos extremos"""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    
    # Evitar división por cero
    rs = np.where(ma_down > 1e-9, ma_up / ma_down, 100)
    return 100 - (100 / (1 + rs))

def bollinger_bands(series, period=20, std_dev=2):
    """Bandas de Bollinger"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def macd(series, fast=12, slow=26, signal=9):
    """MACD"""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# -----------------------
# SISTEMA DE PREDICCIÓN AVANZADO
# -----------------------
def advanced_prediction_system(close_series, volume_series=None):
    """
    Sistema de predicción multi-indicador con timeframes
    Retorna predicciones detalladas con probabilidades
    """
    if close_series is None or len(close_series) < 50:
        return create_empty_prediction()
    
    try:
        # Indicadores técnicos
        ema_8 = ema(close_series, 8).iloc[-1]
        ema_21 = ema(close_series, 21).iloc[-1]
        ema_50 = ema(close_series, 50).iloc[-1] if len(close_series) >= 50 else ema_21
        
        rsi_val = rsi(close_series).iloc[-1]
        
        # MACD
        macd_line, signal_line, histogram = macd(close_series)
        macd_signal = "BUY" if histogram.iloc[-1] > 0 else "SELL"
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = bollinger_bands(close_series)
        bb_position = (close_series.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        
        # Momentum y tendencia
        price_change_1 = (close_series.iloc[-1] / close_series.iloc[-2] - 1) * 100
        price_change_5 = (close_series.iloc[-1] / close_series.iloc[-6] - 1) * 100 if len(close_series) >= 6 else price_change_1
        price_change_20 = (close_series.iloc[-1] / close_series.iloc[-21] - 1) * 100 if len(close_series) >= 21 else price_change_5
        
        # Volatilidad
        volatility = close_series.pct_change().std() * np.sqrt(1440)  # volatilidad diaria
        
        # Sistema de scoring
        trend_score = calculate_trend_score(ema_8, ema_21, ema_50, rsi_val, bb_position, macd_signal)
        
        # Predicciones por timeframe
        timeframe_predictions = {}
        timeframes = ["30s", "1m", "2m", "5m", "10m", "15m", "1h", "2h", "1d", "2d"]
        
        for tf in timeframes:
            prediction = predict_timeframe(trend_score, volatility, tf, price_change_1, price_change_5, price_change_20)
            timeframe_predictions[tf] = prediction
        
        return {
            "trend_score": trend_score,
            "direction": "SUBIENDO" if trend_score > 0 else "BAJANDO",
            "strength": abs(trend_score),
            "rsi": round(rsi_val, 1),
            "volatility": round(volatility * 100, 2),
            "price_change_1m": round(price_change_1, 3),
            "bb_position": round(bb_position * 100, 1),
            "timeframe_predictions": timeframe_predictions,
            "confidence": calculate_confidence(trend_score, volatility, rsi_val),
            "current_price": close_series.iloc[-1]
        }
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return create_empty_prediction()

def calculate_trend_score(ema_8, ema_21, ema_50, rsi_val, bb_position, macd_signal):
    """Calcula score de tendencia combinando múltiples indicadores"""
    score = 0
    
    # EMA trend
    if ema_8 > ema_21:
        score += 25
    if ema_21 > ema_50:
        score += 15
    if ema_8 > ema_50:
        score += 10
    
    # RSI
    if rsi_val > 70:
        score -= 20  # Sobrecompra
    elif rsi_val < 30:
        score += 20  # Sobreventa
    elif 40 <= rsi_val <= 60:
        score += 5   # Zona neutral
    
    # Bollinger position
    if bb_position > 0.8:
        score -= 15  # Cerca del límite superior
    elif bb_position < 0.2:
        score += 15  # Cerca del límite inferior
    
    # MACD
    if macd_signal == "BUY":
        score += 10
    else:
        score -= 10
    
    # Normalizar a -100, 100
    return max(-100, min(100, score))

def predict_timeframe(trend_score, volatility, timeframe, price_change_1, price_change_5, price_change_20):
    """Predice movimiento para un timeframe específico"""
    base_probability = 50 + (trend_score * 0.4)
    
    # Ajustes por timeframe
    tf_multipliers = {
        "30s": 0.3, "1m": 0.5, "2m": 0.7, "5m": 1.0, 
        "10m": 1.2, "15m": 1.4, "1h": 1.8, "2h": 2.0, 
        "1d": 2.5, "2d": 3.0
    }
    
    multiplier = tf_multipliers.get(timeframe, 1.0)
    
    # Ajustar por volatilidad
    vol_adjustment = volatility * 50 * multiplier
    
    # Ajustar por momentum reciente
    momentum_adjustment = 0
    if timeframe in ["30s", "1m", "2m"]:
        momentum_adjustment = price_change_1 * 2
    elif timeframe in ["5m", "10m", "15m"]:
        momentum_adjustment = price_change_5 * 1.5
    else:
        momentum_adjustment = price_change_20 * 1.0
    
    final_probability = base_probability + momentum_adjustment
    final_probability = max(5, min(95, final_probability))
    
    # Duración estimada (en minutos)
    duration_map = {
        "30s": 0.5, "1m": 1, "2m": 2, "5m": 5, "10m": 10,
        "15m": 15, "1h": 60, "2h": 120, "1d": 1440, "2d": 2880
    }
    duration = duration_map.get(timeframe, 5)
    
    # Ajustar duración por volatilidad
    if volatility > 0.02:  # Alta volatilidad
        duration *= 0.7
    elif volatility < 0.005:  # Baja volatilidad
        duration *= 1.5
    
    return {
        "probability": round(final_probability, 1),
        "duration_minutes": round(duration, 1),
        "direction": "SUBIENDO" if final_probability > 50 else "BAJANDO",
        "strength": abs(final_probability - 50) / 50 * 100
    }

def calculate_confidence(trend_score, volatility, rsi_val):
    """Calcula confianza general en las predicciones"""
    confidence = 50
    
    # Mayor confianza con trend score fuerte
    confidence += abs(trend_score) * 0.3
    
    # Menor confianza con alta volatilidad
    confidence -= volatility * 500
    
    # Ajuste por RSI extremos
    if rsi_val > 80 or rsi_val < 20:
        confidence += 10
    
    return max(20, min(95, round(confidence, 1)))

def create_empty_prediction():
    """Crea predicción vacía para casos de error"""
    return {
        "trend_score": 0,
        "direction": "N/D",
        "strength": 0,
        "rsi": 50,
        "volatility": 0,
        "price_change_1m": 0,
        "bb_position": 50,
        "timeframe_predictions": {},
        "confidence": 0,
        "current_price": 0
    }

# -----------------------
# OBTENER ACTIVOS 24/7
# -----------------------
@st.cache_data(ttl=3600)  # Cache por 1 hora
def get_24_7_assets(_client):  # 👈 USO DE _client PARA EVITAR HASHING
    """Obtiene solo activos que operan 24/7 (Forex + Crypto)"""
    assets_24_7 = []
    
    try:
        if ALPACA_LIB == "tradeapi":
            all_assets = _client.list_assets()
            for asset in all_assets:
                asset_class = getattr(asset, 'asset_class', '') or getattr(asset, 'class', '')
                status = getattr(asset, 'status', '')
                symbol = getattr(asset, 'symbol', '')
                
                # Solo crypto y forex (24/7)
                if (asset_class.lower() in ['crypto', 'fx'] and 
                    status == 'active' and 
                    symbol):
                    assets_24_7.append({
                        'symbol': symbol,
                        'class': asset_class,
                        'name': getattr(asset, 'name', symbol)
                    })
        
        # Fallback con activos conocidos 24/7
        if not assets_24_7:
            fallback_assets = [
                {'symbol': 'EURUSD', 'class': 'fx', 'name': 'Euro/US Dollar'},
                {'symbol': 'GBPUSD', 'class': 'fx', 'name': 'British Pound/US Dollar'},
                {'symbol': 'USDJPY', 'class': 'fx', 'name': 'US Dollar/Japanese Yen'},
                {'symbol': 'USDCHF', 'class': 'fx', 'name': 'US Dollar/Swiss Franc'},
                {'symbol': 'AUDUSD', 'class': 'fx', 'name': 'Australian Dollar/US Dollar'},
                {'symbol': 'USDCAD', 'class': 'fx', 'name': 'US Dollar/Canadian Dollar'},
                {'symbol': 'NZDUSD', 'class': 'fx', 'name': 'New Zealand Dollar/US Dollar'},
                {'symbol': 'BTCUSD', 'class': 'crypto', 'name': 'Bitcoin/US Dollar'},
                {'symbol': 'ETHUSD', 'class': 'crypto', 'name': 'Ethereum/US Dollar'},
                {'symbol': 'LTCUSD', 'class': 'crypto', 'name': 'Litecoin/US Dollar'},
            ]
            assets_24_7.extend(fallback_assets)
            
    except Exception as e:
        st.warning(f"Error obteniendo activos: {e}")
        logging.error(f"Error fetching assets: {e}")
        # Return fallback
        return [
            {'symbol': 'EURUSD', 'class': 'fx', 'name': 'Euro/US Dollar'},
            {'symbol': 'GBPUSD', 'class': 'fx', 'name': 'British Pound/US Dollar'},
            {'symbol': 'BTCUSD', 'class': 'crypto', 'name': 'Bitcoin/US Dollar'},
        ]
    
    return assets_24_7

# -----------------------
# OBTENER DATOS DE PRECIOS (CON CACHÉ)
# -----------------------
@st.cache_data(ttl=300)  # Cache por 5 minutos
def fetch_price_data_cached(_client, symbol, timeframe="1Min", limit=500):
    """Obtiene datos de precios con manejo robusto de errores (caché)"""
    try:
        if ALPACA_LIB == "tradeapi":
            bars = _client.get_bars(symbol, timeframe, limit=limit).df
            
            # Manejar MultiIndex
            if isinstance(bars.columns, pd.MultiIndex):
                if symbol in bars.columns.levels[0]:
                    bars = bars.xs(symbol, axis=1, level=0)
            
            # Renombrar columnas
            bars = bars.rename(columns={
                'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
            })
            
            # Convertir a numérico
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in bars.columns:
                    bars[col] = pd.to_numeric(bars[col], errors='coerce')
            
            return bars
            
    except Exception as e:
        logging.error(f"Error fetching price data for {symbol}: {e}")
        return None
    
    return None

def fetch_price_data(client, symbol, timeframe="1Min", limit=500):
    """Wrapper sin caché (para uso interno si se necesita fresco)"""
    return fetch_price_data_cached(client, symbol, timeframe, limit)

# -----------------------
# PROCESAMIENTO DE PREDICCIONES
# -----------------------
def process_asset_prediction(client, asset_info):
    """Procesa predicción para un activo específico"""
    symbol = asset_info['symbol']
    
    try:
        # Obtener datos de precios
        price_data = fetch_price_data_cached(client, symbol, "1Min", 200)
        
        if price_data is None or price_data.empty:
            # Intentar con timeframe más largo
            price_data = fetch_price_data_cached(client, symbol, "5Min", 200)
        
        if price_data is None or price_data.empty:
            return create_error_result(symbol, "Sin datos")
        
        # Generar predicción
        volume_data = price_data.get('volume', None)
        prediction = advanced_prediction_system(price_data['close'], volume_data)
        
        return {
            'symbol': symbol,
            'name': asset_info.get('name', symbol),
            'class': asset_info.get('class', ''),
            'prediction': prediction,
            'current_price': prediction['current_price'],
            'has_data': True,
            'last_update': datetime.now()
        }
        
    except Exception as e:
        logging.error(f"Error processing {symbol}: {e}")
        return create_error_result(symbol, str(e))

def create_error_result(symbol, error_msg):
    """Crea resultado de error para un símbolo"""
    return {
        'symbol': symbol,
        'name': symbol,
        'class': '',
        'prediction': create_empty_prediction(),
        'current_price': 0,
        'has_data': False,
        'error': error_msg,
        'last_update': datetime.now()
    }

# -----------------------
# FUNCIONES AUXILIARES
# -----------------------
def format_duration(minutes):
    """Formatea duración en formato legible"""
    if minutes < 1:
        return f"{int(minutes * 60)}s"
    elif minutes < 60:
        return f"{int(minutes)}m"
    elif minutes < 1440:
        hours = int(minutes / 60)
        return f"{hours}h"
    else:
        days = int(minutes / 1440)
        return f"{days}d"

def get_market_status():
    """Obtiene estado actual del mercado"""
    now = datetime.now()
    if now.weekday() < 5:  # Lunes a Viernes
        return "🟢 Mercado Abierto (Alta Liquidez)"
    else:  # Fin de semana
        return "🟡 Fin de Semana (Liquidez Reducida)"

def calculate_risk_score(prediction):
    """Calcula score de riesgo basado en predicción"""
    volatility = prediction.get('volatility', 0)
    confidence = prediction.get('confidence', 50)
    risk = (volatility * 100) + ((100 - confidence) * 0.5)
    if risk < 20:
        return "🟢 Bajo"
    elif risk < 50:
        return "🟡 Medio"
    else:
        return "🔴 Alto"

# -----------------------
# APLICACIÓN PRINCIPAL
# -----------------------
def main():
    # Setup de credenciales
    creds = setup_credentials()
    
    # Crear cliente
    client = create_alpaca_client(creds)
    if client is None:
        st.error("❌ No se pudo conectar con Alpaca")
        if st.button("🗑️ Eliminar credenciales y reconfigurar"):
            if os.path.exists(CRED_FILE):
                os.remove(CRED_FILE)
            st.rerun()
        st.stop()
    
    # Header principal
    st.title("🔮 QuickTrend 24/7 - Predictor en Tiempo Real")
    st.markdown("**Predicciones automáticas para activos 24/7 (Forex & Crypto)**")
    st.caption(f"Última actualización: {datetime.now().strftime('%H:%M:%S')}")

    # Sidebar de configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        auto_refresh = st.checkbox("🔄 Auto-refresh", value=False)
        refresh_interval = st.slider("Intervalo (segundos)", 5, 60, 15)
        max_assets = st.slider("Máx. activos mostrados", 10, 100, 30)
        
        show_confidence = st.checkbox("Mostrar confianza", value=True)
        show_detailed_timeframes = st.checkbox("Mostrar todos los timeframes", value=True)
        
        # Dark mode toggle
        if st.checkbox("🌙 Modo Oscuro", value=False):
            st.markdown("""
            <style>
                .stApp { background-color: #1E1E1E; color: white; }
                .stMarkdown, .stText, .stMetric { color: white; }
                .stButton>button { background-color: #007BFF; color: white; }
            </style>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        if st.button("🗑️ Reconfigurar API"):
            if os.path.exists(CRED_FILE):
                os.remove(CRED_FILE)
            st.rerun()
    
    # Obtener activos 24/7
    with st.spinner("📡 Obteniendo activos 24/7..."):
        assets_24_7 = get_24_7_assets(client)  # 👈 PASAMOS client normal, la función usa _client
        assets_24_7 = assets_24_7[:max_assets]  # Limitar cantidad
    
    st.success(f"✅ {len(assets_24_7)} activos 24/7 encontrados")
    
    # Skeleton loading mientras se procesan predicciones
    if not auto_refresh:
        st.info("⏳ Generando predicciones...")

    # Procesar predicciones
    with st.spinner("🔮 Generando predicciones..."):
        progress_bar = st.progress(0)
        results = []
        
        for i, asset in enumerate(assets_24_7):
            result = process_asset_prediction(client, asset)
            results.append(result)
            progress_bar.progress((i + 1) / len(assets_24_7))
            time.sleep(0.1)  # evitar rate limits
    
    # Mostrar resultados
    display_predictions(results, show_confidence, show_detailed_timeframes)
    
    # Gráfico detallado
    display_detailed_chart(client, results)
    
    # Estadísticas
    display_performance_stats(results)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

def display_predictions(results, show_confidence, show_detailed_timeframes):
    """Muestra las predicciones en formato visual"""
    
    valid_results = [r for r in results if r['has_data']]
    
    if not valid_results:
        st.warning("⚠️ No se pudieron obtener predicciones")
        return
    
    valid_results.sort(key=lambda x: x['prediction']['confidence'], reverse=True)
    
    st.markdown("## 📊 Predicciones en Tiempo Real")
    
    cols = st.columns(3)
    
    for i, result in enumerate(valid_results):
        col_idx = i % 3
        with cols[col_idx]:
            display_prediction_card(result, show_confidence, show_detailed_timeframes)

def display_prediction_card(result, show_confidence, show_detailed_timeframes):
    """Muestra una tarjeta de predicción individual"""
    
    symbol = result['symbol']
    prediction = result['prediction']
    direction = prediction['direction']
    
    if direction == "SUBIENDO":
        color = "#00C851"
        emoji = "📈"
        bg_color = "#E8F5E8"
    elif direction == "BAJANDO":
        color = "#FF4444"
        emoji = "📉"
        bg_color = "#FFF0F0"
    else:
        color = "#6C757D"
        emoji = "➖"
        bg_color = "#F8F9FA"
    
    with st.container():
        st.markdown(f"""
        <div style="
            background-color: {bg_color};
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid {color};
            margin-bottom: 15px;
        ">
            <h4 style="margin: 0; color: {color};">
                {emoji} {symbol}
            </h4>
            <p style="margin: 5px 0; font-size: 0.9em; color: #666;">
                {result['name']} ({result['class'].upper()})
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Tendencia",
                direction,
                f"{prediction['strength']:.1f}% fuerza"
            )
            
            if show_confidence:
                st.metric(
                    "Confianza",
                    f"{prediction['confidence']:.1f}%",
                    help="Basado en consistencia de indicadores y volatilidad"
                )
        
        with col2:
            st.metric(
                "RSI",
                f"{prediction['rsi']:.1f}",
                f"Vol: {prediction['volatility']:.2f}%",
                help=">70 sobrecomprado, <30 sobrevendido"
            )
            
            st.metric(
                "Precio",
                f"{prediction['current_price']:.5f}" if prediction['current_price'] else "N/D"
            )
        
        if show_detailed_timeframes and prediction['timeframe_predictions']:
            st.markdown("**⏱️ Predicciones por Tiempo:**")
            tf_data = []
            for tf, pred in prediction['timeframe_predictions'].items():
                tf_data.append({
                    'Tiempo': tf,
                    'Dirección': pred['direction'],
                    'Probabilidad': f"{pred['probability']:.1f}%",
                    'Duración': format_duration(pred['duration_minutes'])
                })
            if tf_data:
                df_tf = pd.DataFrame(tf_data)
                st.dataframe(df_tf, use_container_width=True, hide_index=True)

def display_detailed_chart(client, results):
    """Muestra gráfico detallado del activo seleccionado"""
    
    valid_results = [r for r in results if r['has_data']]
    if not valid_results:
        return
    
    st.markdown("## 📈 Análisis Técnico Detallado")
    
    symbols = [r['symbol'] for r in valid_results]
    selected_symbol = st.selectbox("Selecciona activo para análisis:", symbols)
    
    if not selected_symbol:
        return
    
    selected_result = next((r for r in valid_results if r['symbol'] == selected_symbol), None)
    if not selected_result:
        return
    
    with st.spinner(f"📊 Cargando datos de {selected_symbol}..."):
        price_data = fetch_price_data_cached(client, selected_symbol, "5Min", 300)
        
        if price_data is None or price_data.empty:
            st.warning("No se pudieron cargar datos para el gráfico")
            return
        
        price_data['EMA8'] = ema(price_data['close'], 8)
        price_data['EMA21'] = ema(price_data['close'], 21)
        price_data['RSI'] = rsi(price_data['close'])
        
        create_technical_chart(price_data, selected_result)

def create_technical_chart(price_data, result):
    """Crea gráfico técnico interactivo"""
    
    symbol = result['symbol']
    prediction = result['prediction']
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{symbol} - Precio y EMAs', 'RSI'),
        vertical_spacing=0.1,
        row_weights=[0.7, 0.3]
    )
    
    fig.add_trace(go.Candlestick(
        x=price_data.index,
        open=price_data['open'],
        high=price_data['high'],
        low=price_data['low'],
        close=price_data['close'],
        name=symbol
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=price_data.index, y=price_data['EMA8'], name='EMA 8', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=price_data.index, y=price_data['EMA21'], name='EMA 21', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=price_data.index, y=price_data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    if prediction['direction'] == "SUBIENDO":
        bg_color = "rgba(0, 200, 81, 0.1)"
    elif prediction['direction'] == "BAJANDO":
        bg_color = "rgba(255, 68, 68, 0.1)"
    else:
        bg_color = "rgba(108, 117, 125, 0.1)"
    
    fig.update_layout(
        height=700,
        title=f"📈 {symbol} - Análisis Técnico en Tiempo Real",
        xaxis_title="Tiempo",
        plot_bgcolor=bg_color,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Precio", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Resumen Técnico")
        st.write(f"**Dirección:** {prediction['direction']}")
        st.write(f"**Fuerza:** {prediction['strength']:.1f}%")
        st.write(f"**Confianza:** {prediction['confidence']:.1f}%")
    with col2:
        st.markdown("### 🎯 Indicadores Clave")
        st.write(f"**RSI:** {prediction['rsi']:.1f}")
        st.write(f"**Volatilidad:** {prediction['volatility']:.2f}%")
        st.write(f"**Cambio 1m:** {prediction['price_change_1m']:.3f}%")
    with col3:
        st.markdown("### ⏱️ Predicciones Rápidas")
        if prediction['timeframe_predictions']:
            short_term = ['30s', '1m', '2m', '5m']
            for tf in short_term:
                if tf in prediction['timeframe_predictions']:
                    pred = prediction['timeframe_predictions'][tf]
                    emoji = "📈" if pred['direction'] == "SUBIENDO" else "📉"
                    st.write(f"**{tf}:** {emoji} {pred['probability']:.1f}% ({format_duration(pred['duration_minutes'])})")

def display_performance_stats(results):
    """Muestra estadísticas de rendimiento del sistema"""
    
    valid_results = [r for r in results if r['has_data']]
    if not valid_results:
        return
    
    st.markdown("## 📊 Estadísticas del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Distribución de Señales")
        directions = [r['prediction']['direction'] for r in valid_results]
        direction_counts = pd.Series(directions).value_counts()
        fig_pie = go.Figure(data=[go.Pie(labels=direction_counts.index, values=direction_counts.values, hole=.3, marker_colors=['#00C851', '#FF4444', '#6C757D'])])
        fig_pie.update_layout(title="Direcciones", height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Distribución de Confianza")
        confidences = [r['prediction']['confidence'] for r in valid_results]
        fig_hist = go.Figure(data=[go.Histogram(x=confidences, nbinsx=10, marker_color='rgba(0, 200, 81, 0.7)')])
        fig_hist.update_layout(title="Confianza (%)", xaxis_title="%", yaxis_title="Activos", height=300)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("### 🏆 Top 10 por Confianza")
    top_performers = sorted(valid_results, key=lambda x: x['prediction']['confidence'], reverse=True)[:10]
    top_data = [{
        'Símbolo': r['symbol'],
        'Dirección': r['prediction']['direction'],
        'Confianza (%)': f"{r['prediction']['confidence']:.1f}",
        'Fuerza (%)': f"{r['prediction']['strength']:.1f}",
        'RSI': f"{r['prediction']['rsi']:.1f}"
    } for r in top_performers]
    st.dataframe(pd.DataFrame(top_data), use_container_width=True, hide_index=True)

# -----------------------
# EJECUTAR APLICACIÓN
# -----------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Error en la aplicación: {e}")
        logging.exception("Application crashed")
        if st.button("🔄 Reiniciar Aplicación"):
            st.rerun()
        with st.expander("🔧 Información de Debug"):
            st.code(f"Error: {str(e)}")
            st.code(f"Biblioteca Alpaca: {ALPACA_LIB}")
            st.code(f"Archivo de credenciales existe: {os.path.exists(CRED_FILE)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🔮 QuickTrend 24/7 - Predictor Automático de Tendencias<br>
        ⚠️ Advertencia: Las predicciones son estimaciones algorítmicas. No constituyen consejo financiero.
    </div>
    """, 
    unsafe_allow_html=True
)
