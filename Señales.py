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
import logging

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Configurar logging
logging.basicConfig(filename='app_errors.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Forzar uso de alpaca-py
try:
    from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_LIB = "alpaca-py"
except Exception as e:
    st.error("❌ Error: Debes instalar 'alpaca-py'. Ejecuta: pip install alpaca-py")
    logging.error(f"Import error: {e}")
    ALPACA_LIB = None

# -----------------------
# CONFIGURACIÓN INICIAL
# -----------------------
CRED_FILE = "alpaca_credentials.json"

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
# CLIENTE ALPACA (SOLO alpaca-py)
# -----------------------
def create_alpaca_client(creds):
    """Crea cliente Alpaca usando alpaca-py"""
    try:
        api_key = creds["ALPACA_API_KEY"]
        api_secret = creds["ALPACA_API_SECRET"]

        # Solo usamos StockHistoricalDataClient como base
        client = StockHistoricalDataClient(api_key, api_secret)
        # Guardamos keys para crypto
        client._api_key = api_key
        client._secret_key = api_secret
        return client
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
        ema_8 = ema(close_series, 8).iloc[-1]
        ema_21 = ema(close_series, 21).iloc[-1]
        ema_50 = ema(close_series, 50).iloc[-1] if len(close_series) >= 50 else ema_21

        rsi_val = rsi(close_series).iloc[-1]

        macd_line, signal_line, histogram = macd(close_series)
        macd_signal = "BUY" if histogram.iloc[-1] > 0 else "SELL"

        bb_upper, bb_middle, bb_lower = bollinger_bands(close_series)
        bb_position = (close_series.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

        price_change_1 = (close_series.iloc[-1] / close_series.iloc[-2] - 1) * 100
        price_change_5 = (close_series.iloc[-1] / close_series.iloc[-6] - 1) * 100 if len(close_series) >= 6 else price_change_1
        price_change_20 = (close_series.iloc[-1] / close_series.iloc[-21] - 1) * 100 if len(close_series) >= 21 else price_change_5

        volatility = close_series.pct_change().std() * np.sqrt(1440)

        trend_score = calculate_trend_score(ema_8, ema_21, ema_50, rsi_val, bb_position, macd_signal)

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
    score = 0
    if ema_8 > ema_21: score += 25
    if ema_21 > ema_50: score += 15
    if ema_8 > ema_50: score += 10
    if rsi_val > 70: score -= 20
    elif rsi_val < 30: score += 20
    elif 40 <= rsi_val <= 60: score += 5
    if bb_position > 0.8: score -= 15
    elif bb_position < 0.2: score += 15
    if macd_signal == "BUY": score += 10
    else: score -= 10
    return max(-100, min(100, score))

def predict_timeframe(trend_score, volatility, timeframe, price_change_1, price_change_5, price_change_20):
    base_probability = 50 + (trend_score * 0.4)
    tf_multipliers = {
        "30s": 0.3, "1m": 0.5, "2m": 0.7, "5m": 1.0,
        "10m": 1.2, "15m": 1.4, "1h": 1.8, "2h": 2.0,
        "1d": 2.5, "2d": 3.0
    }
    multiplier = tf_multipliers.get(timeframe, 1.0)
    momentum_adjustment = 0
    if timeframe in ["30s", "1m", "2m"]:
        momentum_adjustment = price_change_1 * 2
    elif timeframe in ["5m", "10m", "15m"]:
        momentum_adjustment = price_change_5 * 1.5
    else:
        momentum_adjustment = price_change_20 * 1.0

    final_probability = base_probability + momentum_adjustment
    final_probability = max(5, min(95, final_probability))

    duration_map = {
        "30s": 0.5, "1m": 1, "2m": 2, "5m": 5, "10m": 10,
        "15m": 15, "1h": 60, "2h": 120, "1d": 1440, "2d": 2880
    }
    duration = duration_map.get(timeframe, 5)

    if volatility > 0.02:
        duration *= 0.7
    elif volatility < 0.005:
        duration *= 1.5

    return {
        "probability": round(final_probability, 1),
        "duration_minutes": round(duration, 1),
        "direction": "SUBIENDO" if final_probability > 50 else "BAJANDO",
        "strength": abs(final_probability - 50) / 50 * 100
    }

def calculate_confidence(trend_score, volatility, rsi_val):
    confidence = 50 + abs(trend_score) * 0.3 - volatility * 500
    if rsi_val > 80 or rsi_val < 20:
        confidence += 10
    return max(20, min(95, round(confidence, 1)))

def create_empty_prediction():
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
@st.cache_data(ttl=3600)
def get_24_7_assets(_client):
    """Obtiene solo activos que operan 24/7 (Forex + Crypto)"""
    fallback_assets = [
        {'symbol': 'BTC/USD', 'class': 'crypto', 'name': 'Bitcoin/US Dollar'},
        {'symbol': 'ETH/USD', 'class': 'crypto', 'name': 'Ethereum/US Dollar'},
        {'symbol': 'EUR/USD', 'class': 'fx', 'name': 'Euro/US Dollar'},
        {'symbol': 'GBP/USD', 'class': 'fx', 'name': 'British Pound/US Dollar'},
        {'symbol': 'USD/JPY', 'class': 'fx', 'name': 'US Dollar/Japanese Yen'},
        {'symbol': 'USD/CHF', 'class': 'fx', 'name': 'US Dollar/Swiss Franc'},
        {'symbol': 'AUD/USD', 'class': 'fx', 'name': 'Australian Dollar/US Dollar'},
        {'symbol': 'USD/CAD', 'class': 'fx', 'name': 'US Dollar/Canadian Dollar'},
        {'symbol': 'NZD/USD', 'class': 'fx', 'name': 'New Zealand Dollar/US Dollar'},
        {'symbol': 'LTC/USD', 'class': 'crypto', 'name': 'Litecoin/US Dollar'},
    ]
    return fallback_assets

# -----------------------
# OBTENER DATOS DE PRECIOS (CON alpaca-py)
# -----------------------
@st.cache_data(ttl=300)
def fetch_price_data_cached(api_key, api_secret, symbol, timeframe="1Min", limit=500):
    """Obtiene datos de precios usando alpaca-py"""
    try:
        # Determinar si es crypto o forex
        is_crypto = "/USD" in symbol and any(c in symbol for c in ["BTC", "ETH", "LTC"])

        tf_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame.Minute * 5,
            "15Min": TimeFrame.Minute * 15,
            "1H": TimeFrame.Hour,
            "1D": TimeFrame.Day,
        }
        tf = tf_map.get(timeframe, TimeFrame.Minute)

        if is_crypto:
            client = CryptoHistoricalDataClient(api_key, api_secret)
            request_params = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                limit=limit
            )
            bars = client.get_crypto_bars(request_params)
        else:
            # Para Forex, usar formato correcto: "EUR/USD" -> "EURUSD"
            clean_symbol = symbol.replace("/", "")
            client = StockHistoricalDataClient(api_key, api_secret)
            request_params = StockBarsRequest(
                symbol_or_symbols=clean_symbol,
                timeframe=tf,
                limit=limit
            )
            bars = client.get_stock_bars(request_params)

        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(0)

        # Asegurar columnas numéricas
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    except Exception as e:
        logging.error(f"Error fetching {symbol}: {e}")
        st.warning(f"⚠️ {symbol}: {str(e)[:80]}...")
        return None

# -----------------------
# PROCESAMIENTO DE PREDICCIONES
# -----------------------
def process_asset_prediction(api_key, api_secret, asset_info):
    """Procesa predicción para un activo específico"""
    symbol = asset_info['symbol']

    try:
        # Intentar con timeframe de 1 minuto
        price_data = fetch_price_data_cached(api_key, api_secret, symbol, "1Min", 200)

        if price_data is None or len(price_data) < 50:
            # Intentar con 5 minutos
            price_data = fetch_price_data_cached(api_key, api_secret, symbol, "5Min", 200)

        if price_data is None or len(price_data) < 50:
            # Datos simulados como fallback
            st.warning(f"Usando datos simulados para {symbol}")
            dates = pd.date_range(end=datetime.now(), periods=100, freq='5T')
            np.random.seed(abs(hash(symbol)) % (2**32))
            base_price = 100 if "USD" in symbol else 1.2
            noise = np.cumsum(np.random.randn(100) * 0.01 * base_price)
            prices = base_price + noise
            price_data = pd.DataFrame({
                'open': prices * 0.998,
                'high': prices * 1.002,
                'low': prices * 0.995,
                'close': prices,
                'volume': np.random.randint(1000, 10000, 100)
            }, index=dates)

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
        st.error(f"❌ {symbol}: {str(e)}")
        return create_error_result(symbol, str(e))

def create_error_result(symbol, error_msg):
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
    if minutes < 1:
        return f"{int(minutes * 60)}s"
    elif minutes < 60:
        return f"{int(minutes)}m"
    elif minutes < 1440:
        return f"{int(minutes/60)}h"
    else:
        return f"{int(minutes/1440)}d"

# -----------------------
# APLICACIÓN PRINCIPAL
# -----------------------
def main():
    # Setup de credenciales
    creds = setup_credentials()

    # Extraer keys para usarlas directamente (evitar pasar cliente a cache)
    API_KEY = creds["ALPACA_API_KEY"]
    API_SECRET = creds["ALPACA_API_SECRET"]

    # Header principal
    st.title("🔮 QuickTrend 24/7 - Predictor en Tiempo Real")
    st.markdown("**Predicciones automáticas para activos 24/7 (Forex & Crypto)**")
    st.caption(f"Última actualización: {datetime.now().strftime('%H:%M:%S')}")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        auto_refresh = st.checkbox("🔄 Auto-refresh", value=True)
        refresh_interval = st.slider("Intervalo (segundos)", 5, 60, 15)
        max_assets = st.slider("Máx. activos mostrados", 5, 50, 20)
        show_confidence = st.checkbox("Mostrar confianza", value=True)
        show_detailed_timeframes = st.checkbox("Mostrar timeframes", value=False)

        if st.checkbox("🌙 Modo Oscuro", value=False):
            st.markdown("""
            <style>
                .stApp { background-color: #1E1E1E; color: white; }
                .stMarkdown, .stText, .stMetric { color: white; }
            </style>
            """, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🗑️ Reconfigurar API"):
            if os.path.exists(CRED_FILE):
                os.remove(CRED_FILE)
            st.rerun()

    # Obtener activos
    with st.spinner("📡 Obteniendo activos 24/7..."):
        assets_24_7 = get_24_7_assets(None)  # No necesitamos cliente aquí
        assets_24_7 = assets_24_7[:max_assets]

    st.success(f"✅ {len(assets_24_7)} activos 24/7 encontrados")

    # Procesar predicciones
    with st.spinner("🔮 Generando predicciones..."):
        results = []
        progress_bar = st.progress(0)

        for i, asset in enumerate(assets_24_7):
            result = process_asset_prediction(API_KEY, API_SECRET, asset)
            results.append(result)
            progress_bar.progress((i + 1) / len(assets_24_7))
            time.sleep(0.5)  # Evitar rate limits

    # Mostrar resultados
    valid_results = [r for r in results if r['has_data']]
    if not valid_results:
        st.warning("⚠️ No se pudieron obtener predicciones reales. Mostrando modo demo.")
        # Generar datos de demo
        demo_results = []
        for asset in assets_24_7[:5]:
            demo_pred = {
                "trend_score": np.random.randint(-80, 80),
                "direction": "SUBIENDO" if np.random.random() > 0.5 else "BAJANDO",
                "strength": np.random.randint(30, 90),
                "rsi": np.random.randint(20, 80),
                "volatility": np.random.uniform(0.5, 3.0),
                "price_change_1m": np.random.uniform(-2, 2),
                "bb_position": np.random.uniform(20, 80),
                "timeframe_predictions": {
                    tf: {
                        "probability": np.random.uniform(40, 90),
                        "duration_minutes": np.random.uniform(1, 120),
                        "direction": "SUBIENDO" if np.random.random() > 0.5 else "BAJANDO",
                        "strength": np.random.uniform(30, 100)
                    } for tf in ["1m", "5m", "15m"]
                },
                "confidence": np.random.uniform(40, 95),
                "current_price": np.random.uniform(0.8, 20000)
            }
            demo_results.append({
                'symbol': asset['symbol'],
                'name': asset['name'],
                'class': asset['class'],
                'prediction': demo_pred,
                'current_price': demo_pred['current_price'],
                'has_data': True,
                'last_update': datetime.now()
            })
        display_predictions(demo_results, show_confidence, show_detailed_timeframes)
    else:
        display_predictions(valid_results, show_confidence, show_detailed_timeframes)

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

def display_predictions(results, show_confidence, show_detailed_timeframes):
    """Muestra las predicciones en formato visual"""
    valid_results = [r for r in results if r['has_data']]
    if not valid_results:
        st.warning("⚠️ No hay predicciones válidas")
        return

    valid_results.sort(key=lambda x: x['prediction']['confidence'], reverse=True)
    st.markdown("## 📊 Predicciones en Tiempo Real")

    cols = st.columns(2)
    for i, result in enumerate(valid_results):
        with cols[i % 2]:
            display_prediction_card(result, show_confidence, show_detailed_timeframes)

def display_prediction_card(result, show_confidence, show_detailed_timeframes):
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
        st.metric("Tendencia", direction, f"{prediction['strength']:.1f}%")
        if show_confidence:
            st.metric("Confianza", f"{prediction['confidence']:.1f}%")
    with col2:
        st.metric("RSI", f"{prediction['rsi']:.1f}", f"Vol: {prediction['volatility']:.2f}%")
        st.metric("Precio", f"{prediction['current_price']:.4f}")

    if show_detailed_timeframes and prediction['timeframe_predictions']:
        st.markdown("**⏱️ Próximos movimientos:**")
        for tf in ["1m", "5m", "15m"]:
            if tf in prediction['timeframe_predictions']:
                pred = prediction['timeframe_predictions'][tf]
                emoji = "📈" if pred['direction'] == "SUBIENDO" else "📉"
                st.caption(f"{tf}: {emoji} {pred['probability']:.1f}% ({format_duration(pred['duration_minutes'])})")

# -----------------------
# EJECUTAR APLICACIÓN
# -----------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Error crítico: {e}")
        logging.exception("Error crítico")
        if st.button("🔄 Reiniciar"):
            st.rerun()

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
