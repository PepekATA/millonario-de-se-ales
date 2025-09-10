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

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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
        return False

def load_credentials():
    """Carga credenciales guardadas"""
    if os.path.exists(CRED_FILE):
        try:
            with open(CRED_FILE, "r") as f:
                return json.load(f)
        except Exception:
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
        base_url = creds.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        
        if ALPACA_LIB == "tradeapi":
            return tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        elif ALPACA_LIB == "alpaca-py":
            return StockHistoricalDataClient(api_key, api_secret)
        else:
            st.error("❌ Instala 'alpaca-trade-api' o 'alpaca-py'")
            return None
    except Exception as e:
        st.error(f"❌ Error conectando con Alpaca: {e}")
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
def get_24_7_assets(client):
    """Obtiene solo activos que operan 24/7 (Forex + Crypto)"""
    assets_24_7 = []
    
    try:
        if ALPACA_LIB == "tradeapi":
            all_assets = client.list_assets()
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
        # Return fallback
        return [
            {'symbol': 'EURUSD', 'class': 'fx', 'name': 'Euro/US Dollar'},
            {'symbol': 'GBPUSD', 'class': 'fx', 'name': 'British Pound/US Dollar'},
            {'symbol': 'BTCUSD', 'class': 'crypto', 'name': 'Bitcoin/US Dollar'},
        ]
    
    return assets_24_7

# -----------------------
# OBTENER DATOS DE PRECIOS
# -----------------------
def fetch_price_data(client, symbol, timeframe="1Min", limit=500):
    """Obtiene datos de precios con manejo robusto de errores"""
    try:
        if ALPACA_LIB == "tradeapi":
            bars = client.get_bars(symbol, timeframe, limit=limit).df
            
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
        return None
    
    return None

# -----------------------
# PROCESAMIENTO DE PREDICCIONES
# -----------------------
def process_asset_prediction(client, asset_info):
    """Procesa predicción para un activo específico"""
    symbol = asset_info['symbol']
    
    try:
        # Obtener datos de precios
        price_data = fetch_price_data(client, symbol, "1Min", 200)
        
        if price_data is None or price_data.empty:
            # Intentar con timeframe más largo
            price_data = fetch_price_data(client, symbol, "5Min", 200)
        
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
    
    # Sidebar de configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        auto_refresh = st.checkbox("🔄 Auto-refresh", value=False)
        refresh_interval = st.slider("Intervalo (segundos)", 5, 60, 15)
        max_assets = st.slider("Máx. activos mostrados", 10, 100, 30)
        
        show_confidence = st.checkbox("Mostrar confianza", value=True)
        show_detailed_timeframes = st.checkbox("Mostrar todos los timeframes", value=True)
        
        st.markdown("---")
        if st.button("🗑️ Reconfigurar API"):
            if os.path.exists(CRED_FILE):
                os.remove(CRED_FILE)
            st.rerun()
    
    # Obtener activos 24/7
    with st.spinner("📡 Obteniendo activos 24/7..."):
        assets_24_7 = get_24_7_assets(client)
        assets_24_7 = assets_24_7[:max_assets]  # Limitar cantidad
    
    st.success(f"✅ {len(assets_24_7)} activos 24/7 encontrados")
    
    # Contenedor principal para resultados
    results_container = st.container()
    
    # Procesar predicciones
    with st.spinner("🔮 Generando predicciones..."):
        progress_bar = st.progress(0)
        results = []
        
        # Procesamiento secuencial para evitar rate limits
        for i, asset in enumerate(assets_24_7):
            result = process_asset_prediction(client, asset)
            results.append(result)
            progress_bar.progress((i + 1) / len(assets_24_7))
            
            # Pequeño delay para evitar rate limiting
            time.sleep(0.1)
    
    # Mostrar resultados
    display_predictions(results, show_confidence, show_detailed_timeframes)
    
    # Gráfico detallado
    display_detailed_chart(client, results)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

def display_predictions(results, show_confidence, show_detailed_timeframes):
    """Muestra las predicciones en formato visual"""
    
    # Filtrar solo resultados con datos
    valid_results = [r for r in results if r['has_data']]
    
    if not valid_results:
        st.warning("⚠️ No se pudieron obtener predicciones")
        return
    
    # Ordenar por confianza/fuerza
    valid_results.sort(key=lambda x: x['prediction']['confidence'], reverse=True)
    
    st.markdown("## 📊 Predicciones en Tiempo Real")
    
    # Vista en tarjetas
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
    
    # Color según dirección
    if direction == "SUBIENDO":
        color = "#00C851"  # Verde
        emoji = "📈"
        bg_color = "#E8F5E8"
    elif direction == "BAJANDO":
        color = "#FF4444"  # Rojo
        emoji = "📉"
        bg_color = "#FFF0F0"
    else:
        color = "#6C757D"  # Gris
        emoji = "➖"
        bg_color = "#F8F9FA"
    
    # Crear tarjeta
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
        
        # Información principal
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
                    f"{prediction['confidence']:.1f}%"
                )
        
        with col2:
            st.metric(
                "RSI",
                f"{prediction['rsi']:.1f}",
                f"Volatilidad: {prediction['volatility']:.2f}%"
            )
            
            st.metric(
                "Precio Actual",
                f"{prediction['current_price']:.5f}" if prediction['current_price'] else "N/D"
            )
        
        # Predicciones por timeframe
        if show_detailed_timeframes and prediction['timeframe_predictions']:
            st.markdown("**Predicciones por Tiempo:**")
            
            tf_data = []
            for tf, pred in prediction['timeframe_predictions'].items():
                tf_data.append({
                    'Tiempo': tf,
                    'Dirección': pred['direction'],
                    'Probabilidad': f"{pred['probability']:.1f}%",
                    'Duración': f"{pred['duration_minutes']:.1f}m"
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
    
    # Selector de activo
    symbols = [r['symbol'] for r in valid_results]
    selected_symbol = st.selectbox("Selecciona activo para análisis:", symbols)
    
    if not selected_symbol:
        return
    
    # Encontrar resultado seleccionado
    selected_result = next((r for r in valid_results if r['symbol'] == selected_symbol), None)
    if not selected_result:
        return
    
    # Obtener datos históricos para gráfico
    with st.spinner(f"📊 Cargando datos de {selected_symbol}..."):
        price_data = fetch_price_data(client, selected_symbol, "5Min", 300)
        
        if price_data is None or price_data.empty:
            st.warning("No se pudieron cargar datos para el gráfico")
            return
        
        # Calcular indicadores para el gráfico
        price_data['EMA8'] = ema(price_data['close'], 8)
        price_data['EMA21'] = ema(price_data['close'], 21)
        price_data['RSI'] = rsi(price_data['close'])
        
        # Crear gráfico
        create_technical_chart(price_data, selected_result)

def create_technical_chart(price_data, result):
    """Crea gráfico técnico interactivo"""
    
    symbol = result['symbol']
    prediction = result['prediction']
    
    # Crear subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{symbol} - Precio y EMAs', 'RSI'),
        vertical_spacing=0.1,
        row_weights=[0.7, 0.3]
    )
    
    # Gráfico principal - Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=price_data.index,
            open=price_data['open'],
            high=price_data['high'],
            low=price_data['low'],
            close=price_data['close'],
            name=symbol
        ),
        row=1, col=1
    )
    
    # EMAs
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['EMA8'],
            name='EMA 8',
            line=dict(color='orange', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['EMA21'],
            name='EMA 21',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['RSI'],
            name='RSI',
            line=dict(color='purple', width=2)
        ),
        row=2, col=1
    )
    
    # Líneas de referencia RSI
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    # Color de fondo según predicción
    if prediction['direction'] == "SUBIENDO":
        bg_color = "rgba(0, 200, 81, 0.1)"
    elif prediction['direction'] == "BAJANDO":
        bg_color = "rgba(255, 68, 68, 0.1)"
    else:
        bg_color = "rgba(108, 117, 125, 0.1)"
    
    # Configurar layout
    fig.update_layout(
        height=700,
        title=f"📈 {symbol} - Análisis Técnico en Tiempo Real",
        xaxis_title="Tiempo",
        plot_bgcolor=bg_color,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Configurar ejes Y
    fig.update_yaxes(title_text="Precio", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Información adicional del análisis
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
            # Mostrar predicciones de corto plazo
            short_term = ['30s', '1m', '2m', '5m']
            for tf in short_term:
                if tf in prediction['timeframe_predictions']:
                    pred = prediction['timeframe_predictions'][tf]
                    direction_emoji = "📈" if pred['direction'] == "SUBIENDO" else "📉"
                    st.write(f"**{tf}:** {direction_emoji} {pred['probability']:.1f}% ({pred['duration_minutes']:.1f}m)")

# -----------------------
# SISTEMA DE MONITOREO EN TIEMPO REAL
# -----------------------
def create_live_monitoring_dashboard(client, assets):
    """Crea dashboard de monitoreo en tiempo real"""
    
    st.markdown("## 🔴 LIVE - Monitor de Señales")
    
    # Contenedor para métricas en tiempo real
    metrics_container = st.container()
    
    # Contenedor para alertas
    alerts_container = st.container()
    
    # Obtener datos en tiempo real
    live_data = {}
    strong_signals = []
    
    for asset in assets[:10]:  # Limitar a 10 para rendimiento
        try:
            result = process_asset_prediction(client, asset)
            if result['has_data']:
                live_data[asset['symbol']] = result
                
                # Detectar señales fuertes
                prediction = result['prediction']
                if prediction['confidence'] > 80 and prediction['strength'] > 70:
                    strong_signals.append({
                        'symbol': asset['symbol'],
                        'direction': prediction['direction'],
                        'confidence': prediction['confidence'],
                        'strength': prediction['strength']
                    })
        except:
            continue
    
    # Mostrar métricas en tiempo real
    with metrics_container:
        if live_data:
            st.markdown("### ⚡ Métricas en Tiempo Real")
            
            cols = st.columns(4)
            
            # Total activos monitoreados
            with cols[0]:
                st.metric(
                    "🎯 Activos Activos",
                    len(live_data),
                    f"de {len(assets)} totales"
                )
            
            # Señales alcistas
            bullish_count = sum(1 for data in live_data.values() 
                              if data['prediction']['direction'] == "SUBIENDO")
            with cols[1]:
                st.metric(
                    "📈 Señales Alcistas",
                    bullish_count,
                    f"{(bullish_count/len(live_data)*100):.1f}%" if live_data else "0%"
                )
            
            # Señales bajistas
            bearish_count = len(live_data) - bullish_count
            with cols[2]:
                st.metric(
                    "📉 Señales Bajistas",
                    bearish_count,
                    f"{(bearish_count/len(live_data)*100):.1f}%" if live_data else "0%"
                )
            
            # Confianza promedio
            avg_confidence = sum(data['prediction']['confidence'] 
                               for data in live_data.values()) / len(live_data) if live_data else 0
            with cols[3]:
                st.metric(
                    "🎲 Confianza Promedio",
                    f"{avg_confidence:.1f}%",
                    "Alta" if avg_confidence > 70 else ("Media" if avg_confidence > 50 else "Baja")
                )
    
    # Mostrar alertas de señales fuertes
    with alerts_container:
        if strong_signals:
            st.markdown("### 🚨 Alertas de Señales Fuertes")
            
            for signal in strong_signals[:5]:  # Máximo 5 alertas
                direction_color = "#00C851" if signal['direction'] == "SUBIENDO" else "#FF4444"
                direction_emoji = "📈" if signal['direction'] == "SUBIENDO" else "📉"
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(90deg, {direction_color}15, transparent);
                    border-left: 4px solid {direction_color};
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 5px;
                ">
                    <strong>{direction_emoji} {signal['symbol']}</strong> - {signal['direction']} 
                    | Confianza: <strong>{signal['confidence']:.1f}%</strong> 
                    | Fuerza: <strong>{signal['strength']:.1f}%</strong>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("🔍 No hay señales fuertes en este momento")

# -----------------------
# SISTEMA DE ALERTAS Y NOTIFICACIONES
# -----------------------
def setup_alert_system():
    """Configura sistema de alertas personalizables"""
    
    with st.sidebar.expander("🔔 Configurar Alertas"):
        st.markdown("**Alertas Automáticas**")
        
        alert_confidence_threshold = st.slider(
            "Umbral de confianza (%)",
            50, 95, 80
        )
        
        alert_strength_threshold = st.slider(
            "Umbral de fuerza (%)",
            50, 100, 70
        )
        
        alert_symbols = st.multiselect(
            "Símbolos a monitorear",
            ["EURUSD", "GBPUSD", "BTCUSD", "ETHUSD"],
            default=["EURUSD", "BTCUSD"]
        )
        
        enable_sound_alerts = st.checkbox("🔊 Alertas sonoras", value=False)
        
        return {
            'confidence_threshold': alert_confidence_threshold,
            'strength_threshold': alert_strength_threshold,
            'symbols': alert_symbols,
            'sound_enabled': enable_sound_alerts
        }

# -----------------------
# EXPORTACIÓN DE DATOS
# -----------------------
def export_predictions_data(results):
    """Permite exportar datos de predicciones"""
    
    if not results:
        return
    
    with st.sidebar.expander("💾 Exportar Datos"):
        st.markdown("**Descargar Predicciones**")
        
        # Preparar datos para exportación
        export_data = []
        for result in results:
            if result['has_data']:
                pred = result['prediction']
                
                # Datos básicos
                row = {
                    'Símbolo': result['symbol'],
                    'Nombre': result['name'],
                    'Clase': result['class'],
                    'Dirección': pred['direction'],
                    'Confianza_%': pred['confidence'],
                    'Fuerza_%': pred['strength'],
                    'RSI': pred['rsi'],
                    'Volatilidad_%': pred['volatility'],
                    'Precio_Actual': pred['current_price'],
                    'Timestamp': result['last_update'].strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Agregar predicciones por timeframe
                for tf, tf_pred in pred.get('timeframe_predictions', {}).items():
                    row[f'{tf}_Probabilidad_%'] = tf_pred['probability']
                    row[f'{tf}_Duración_min'] = tf_pred['duration_minutes']
                
                export_data.append(row)
        
        if export_data:
            df_export = pd.DataFrame(export_data)
            
            # Botón de descarga CSV
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="📄 Descargar CSV",
                data=csv,
                file_name=f"quicktrend_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Botón de descarga JSON
            json_data = df_export.to_json(orient='records', indent=2)
            st.download_button(
                label="📋 Descargar JSON",
                data=json_data,
                file_name=f"quicktrend_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# -----------------------
# ESTADÍSTICAS Y RENDIMIENTO
# -----------------------
def display_performance_stats(results):
    """Muestra estadísticas de rendimiento del sistema"""
    
    if not results:
        return
    
    st.markdown("## 📊 Estadísticas del Sistema")
    
    valid_results = [r for r in results if r['has_data']]
    
    if not valid_results:
        st.warning("No hay datos suficientes para estadísticas")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Distribución de Señales")
        
        # Contar direcciones
        directions = [r['prediction']['direction'] for r in valid_results]
        direction_counts = pd.Series(directions).value_counts()
        
        # Crear gráfico de pie
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=direction_counts.index,
                values=direction_counts.values,
                hole=.3,
                marker_colors=['#00C851', '#FF4444', '#6C757D']
            )
        ])
        
        fig_pie.update_layout(
            title="Distribución de Direcciones",
            height=300
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Distribución de Confianza")
        
        # Histograma de confianza
        confidences = [r['prediction']['confidence'] for r in valid_results]
        
        fig_hist = go.Figure(data=[
            go.Histogram(
                x=confidences,
                nbinsx=10,
                marker_color='rgba(0, 200, 81, 0.7)'
            )
        ])
        
        fig_hist.update_layout(
            title="Distribución de Confianza",
            xaxis_title="Confianza (%)",
            yaxis_title="Cantidad de Activos",
            height=300
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Tabla de top performers
    st.markdown("### 🏆 Top Performers (Mayor Confianza)")
    
    # Ordenar por confianza
    top_performers = sorted(valid_results, key=lambda x: x['prediction']['confidence'], reverse=True)[:10]
    
    top_data = []
    for result in top_performers:
        pred = result['prediction']
        top_data.append({
            'Símbolo': result['symbol'],
            'Dirección': pred['direction'],
            'Confianza (%)': f"{pred['confidence']:.1f}",
            'Fuerza (%)': f"{pred['strength']:.1f}",
            'RSI': f"{pred['rsi']:.1f}"
        })
    
    df_top = pd.DataFrame(top_data)
    st.dataframe(df_top, use_container_width=True, hide_index=True)

# -----------------------
# EJECUTAR APLICACIÓN
# -----------------------
if __name__ == "__main__":
    try:
        # Configurar alertas
        alert_config = setup_alert_system()
        
        # Ejecutar aplicación principal
        main()
        
        # Mostrar estadísticas al final
        # (se ejecutaría después de obtener resultados en main)
        
    except Exception as e:
        st.error(f"❌ Error en la aplicación: {e}")
        
        # Botón de reinicio en caso de error crítico
        if st.button("🔄 Reiniciar Aplicación"):
            st.rerun()
        
        # Información de debug
        with st.expander("🔧 Información de Debug"):
            st.code(f"Error: {str(e)}")
            st.code(f"Biblioteca Alpaca: {ALPACA_LIB}")
            st.code(f"Archivo de credenciales existe: {os.path.exists(CRED_FILE)}")

# -----------------------
# FUNCIONES AUXILIARES ADICIONALES
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
    
    # Los mercados forex y crypto operan 24/7, pero con diferentes volúmenes
    if now.weekday() < 5:  # Lunes a Viernes
        return "🟢 Mercado Abierto (Alta Liquidez)"
    else:  # Fin de semana
        return "🟡 Fin de Semana (Liquidez Reducida)"

def calculate_risk_score(prediction):
    """Calcula score de riesgo basado en predicción"""
    volatility = prediction.get('volatility', 0)
    confidence = prediction.get('confidence', 50)
    
    # Mayor volatilidad = mayor riesgo
    # Menor confianza = mayor riesgo
    risk = (volatility * 100) + ((100 - confidence) * 0.5)
    
    if risk < 20:
        return "🟢 Bajo"
    elif risk < 50:
        return "🟡 Medio"
    else:
        return "🔴 Alto"

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
