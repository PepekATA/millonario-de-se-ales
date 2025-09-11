# Millonario de Señales Crypto 4.0
# Sistema mejorado con señales visuales, botones de trading y operación 24/7
# Compatible con Alpaca Crypto (SOLUSD, BTCUSD, ETHUSD, AVAXUSD)

import os, json, time, threading
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess, sys

# Instalar dependencias
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

try:
    import alpaca_trade_api as tradeapi
    import talib
except ImportError:
    install("alpaca-trade-api")
    install("TA-Lib")
    import alpaca_trade_api as tradeapi
    import talib

# -------------------------
# Archivos de persistencia
# -------------------------
CRED_FILE = "credentials.json"
STATE_FILE = "positions.json"
MODEL_FILE = "ml_model.json"
SIGNALS_FILE = "trading_signals.json"

# -------------------------
# Indicadores Técnicos y Señales
# -------------------------
class TechnicalAnalyzer:
    def __init__(self):
        pass
    
    def calculate_signals(self, df):
        """Calcula múltiples indicadores técnicos"""
        if len(df) < 20:
            return df
        
        # RSI
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'].values)
        
        # Medias móviles
        df['sma_20'] = talib.SMA(df['close'].values, timeperiod=20)
        df['ema_12'] = talib.EMA(df['close'].values, timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'].values, timeperiod=26)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'].values)
        
        # Señales de compra/venta
        df['buy_signal'] = self.generate_buy_signals(df)
        df['sell_signal'] = self.generate_sell_signals(df)
        
        return df
    
    def generate_buy_signals(self, df):
        """Genera señales de compra basadas en múltiples indicadores"""
        signals = pd.Series(False, index=df.index)
        
        # RSI oversold
        rsi_signal = df['rsi'] < 30
        
        # MACD cruce alcista
        macd_signal = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        
        # Precio toca banda inferior de Bollinger
        bb_signal = df['close'] <= df['bb_lower']
        
        # EMA cruce alcista
        ema_signal = (df['ema_12'] > df['ema_26']) & (df['ema_12'].shift(1) <= df['ema_26'].shift(1))
        
        # Combinar señales (al menos 2 de 4)
        signal_count = rsi_signal.astype(int) + macd_signal.astype(int) + bb_signal.astype(int) + ema_signal.astype(int)
        signals = signal_count >= 2
        
        return signals
    
    def generate_sell_signals(self, df):
        """Genera señales de venta"""
        signals = pd.Series(False, index=df.index)
        
        # RSI overbought
        rsi_signal = df['rsi'] > 70
        
        # MACD cruce bajista
        macd_signal = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # Precio toca banda superior de Bollinger
        bb_signal = df['close'] >= df['bb_upper']
        
        # EMA cruce bajista
        ema_signal = (df['ema_12'] < df['ema_26']) & (df['ema_12'].shift(1) >= df['ema_26'].shift(1))
        
        signal_count = rsi_signal.astype(int) + macd_signal.astype(int) + bb_signal.astype(int) + ema_signal.astype(int)
        signals = signal_count >= 2
        
        return signals

# -------------------------
# Sistema de Señales Mejorado
# -------------------------
class SignalManager:
    def __init__(self):
        self.signals = self.load_signals()
        
    def load_signals(self):
        if os.path.exists(SIGNALS_FILE):
            try:
                return json.load(open(SIGNALS_FILE, "r"))
            except:
                pass
        return {"manual_signals": {}, "auto_signals": {}}
    
    def save_signals(self):
        with open(SIGNALS_FILE, "w") as f:
            json.dump(self.signals, f, indent=2)
    
    def add_manual_signal(self, symbol, signal_type, strength=1.0):
        """Añade señal manual de subida o bajada"""
        timestamp = datetime.now().isoformat()
        if symbol not in self.signals["manual_signals"]:
            self.signals["manual_signals"][symbol] = []
        
        self.signals["manual_signals"][symbol].append({
            "type": signal_type,  # "up" o "down"
            "strength": strength,
            "timestamp": timestamp
        })
        self.save_signals()
    
    def get_recent_signals(self, symbol, hours=1):
        """Obtiene señales recientes para un símbolo"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = []
        
        if symbol in self.signals["manual_signals"]:
            for signal in self.signals["manual_signals"][symbol]:
                signal_time = datetime.fromisoformat(signal["timestamp"])
                if signal_time > cutoff:
                    recent.append(signal)
        
        return recent

# -------------------------
# ML Model Mejorado
# -------------------------
class EnhancedMLModel:
    def __init__(self):
        self.model = self.load_model()
        self.analyzer = TechnicalAnalyzer()

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            try:
                return json.load(open(MODEL_FILE, "r"))
            except:
                pass
        return {}

    def save_model(self):
        with open(MODEL_FILE, "w") as f:
            json.dump(self.model, f, indent=2)

    def update(self, symbol, direction, strength=1.0):
        if symbol not in self.model:
            self.model[symbol] = {'up': 0, 'down': 0, 'accuracy': 0.5}
        
        self.model[symbol][direction] += strength
        self.save_model()

    def predict_with_technical(self, symbol, df):
        """Predicción combinando ML y análisis técnico"""
        if len(df) < 20:
            return 0.5, "Datos insuficientes"
        
        # Análisis técnico
        df = self.analyzer.calculate_signals(df)
        
        # Señal técnica
        technical_score = 0.5
        if df['buy_signal'].iloc[-1]:
            technical_score = 0.8
        elif df['sell_signal'].iloc[-1]:
            technical_score = 0.2
        
        # ML score
        ml_score = self.predict(symbol)
        
        # Combinar scores
        final_score = (technical_score * 0.6) + (ml_score * 0.4)
        
        return final_score, f"Técnico: {technical_score:.2f}, ML: {ml_score:.2f}"

    def predict(self, symbol):
        if symbol not in self.model:
            return 0.5
        
        total = self.model[symbol]['up'] + self.model[symbol]['down']
        if total == 0:
            return 0.5
        
        return self.model[symbol]['up'] / total

# -------------------------
# Enhanced Crypto Trader
# -------------------------
class EnhancedCryptoTrader:
    def __init__(self, creds, ml_model, signal_manager):
        self.api = tradeapi.REST(creds["ALPACA_API_KEY"], creds["ALPACA_API_SECRET"], 
                                creds["ALPACA_BASE_URL"], api_version='v2')
        self.state_manager = TradeStateManager()
        self.ml_model = ml_model
        self.signal_manager = signal_manager
        self.symbols = ["SOLUSD", "BTCUSD", "ETHUSD", "AVAXUSD"]
        self.is_running = False
        self.capital_fraction = 0.05  # 5% por activo para ser más conservador
        self.analyzer = TechnicalAnalyzer()

    def get_historical(self, symbol, limit=100):
        """Obtiene datos históricos con más períodos para indicadores"""
        try:
            bars = self.api.get_crypto_bars(symbol, "1Min").df
            bars = bars[bars['exchange'] == 'CBSE']
            bars = bars.tail(limit)
            if bars.empty:
                return None
            return bars
        except Exception as e:
            st.error(f"Error obteniendo datos para {symbol}: {e}")
            return None

    def execute_smart_buy(self, symbol, current_price, signal_strength=1.0):
        """Compra inteligente basada en fuerza de señal"""
        try:
            account = self.api.get_account()
            buying_power = float(account.cash)
            
            # Ajustar cantidad según fuerza de señal
            base_qty = (buying_power * self.capital_fraction) / current_price
            adjusted_qty = max(0.0001, base_qty * signal_strength)
            
            self.api.submit_order(
                symbol=symbol,
                qty=adjusted_qty,
                side="buy",
                type="market",
                time_in_force="day"
            )
            
            pos_id = self.state_manager.add_position(symbol, adjusted_qty, current_price)
            self.ml_model.update(symbol, "up", signal_strength)
            
            return pos_id
            
        except Exception as e:
            st.error(f"Error comprando {symbol}: {e}")
            return None

    def execute_smart_sell(self, symbol, pos_id, qty, current_price, signal_strength=1.0):
        """Venta inteligente"""
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side="sell",
                type="market",
                time_in_force="day"
            )
            
            self.state_manager.close_position(symbol, pos_id)
            self.ml_model.update(symbol, "down", signal_strength)
            
        except Exception as e:
            st.error(f"Error vendiendo {symbol}: {e}")

    def should_trade_now(self, symbol):
        """Determina si debe operar basado en señales múltiples"""
        # Obtener datos
        df = self.get_historical(symbol, 100)
        if df is None:
            return False, 0.5, "Sin datos"
        
        # Predicción con análisis técnico
        score, reason = self.ml_model.predict_with_technical(symbol, df)
        
        # Verificar señales manuales recientes
        recent_signals = self.signal_manager.get_recent_signals(symbol, hours=1)
        manual_boost = 0
        for signal in recent_signals:
            if signal["type"] == "up":
                manual_boost += 0.1 * signal["strength"]
            else:
                manual_boost -= 0.1 * signal["strength"]
        
        final_score = min(0.95, max(0.05, score + manual_boost))
        
        return True, final_score, f"{reason} + Manual: {manual_boost:.2f}"

# -------------------------
# Manejo de estados (sin cambios)
# -------------------------
class TradeStateManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = self.load_state()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                return json.load(open(STATE_FILE, "r"))
            except:
                pass
        return {"positions": {}}

    def save_state(self):
        with self.lock:
            json.dump(self.state, open(STATE_FILE, "w"), indent=2)

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
                    if pos["status"] == "open":
                        pos["highest_price"] = max(pos["highest_price"], current_price)
            self.save_state()

    def get_open_positions(self, symbol=None):
        with self.lock:
            if symbol:
                return [p for p in self.state["positions"].get(symbol, []) if p["status"] == "open"]
            
            all_pos = []
            for sym, pos_list in self.state["positions"].items():
                all_pos.extend([p for p in pos_list if p["status"] == "open"])
            return all_pos

    def close_position(self, symbol, pos_id):
        with self.lock:
            if symbol in self.state["positions"]:
                for p in self.state["positions"][symbol]:
                    if p["id"] == pos_id and p["status"] == "open":
                        p["status"] = "closed"
                        p["closed_at"] = datetime.now().isoformat()
                        self.save_state()
                        return True
        return False

# -------------------------
# Funciones de credenciales (sin cambios)
# -------------------------
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
            return json.load(open(CRED_FILE, "r"))
        except:
            return None
    return None

def setup_credentials():
    creds = load_credentials()
    if creds and creds.get("ALPACA_API_KEY") and creds.get("ALPACA_API_SECRET"):
        return creds
    
    st.title("🤖 Millonario de Señales Crypto - Setup")
    with st.form("cred_form"):
        key = st.text_input("API Key", type="password")
        secret = st.text_input("API Secret", type="password")
        paper = st.checkbox("Usar cuenta demo (paper trading)", value=True)
        
        if st.form_submit_button("💾 Guardar y Conectar"):
            save_credentials(key, secret, paper)
            st.rerun()
    st.stop()

# -------------------------
# Función para crear gráficos avanzados
# -------------------------
def create_advanced_chart(df, symbol):
    """Crea gráfico avanzado con indicadores y señales"""
    if df is None or len(df) < 20:
        return None
    
    # Aplicar análisis técnico
    analyzer = TechnicalAnalyzer()
    df = analyzer.calculate_signals(df)
    
    # Crear subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} - Precio y Señales', 'RSI', 'MACD'),
        row_width=[0.2, 0.7, 0.1]
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Precio'
    ), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index, y=df['bb_upper'],
        line=dict(color='rgba(250,0,0,0.5)'),
        name='BB Superior'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['bb_lower'],
        line=dict(color='rgba(0,250,0,0.5)'),
        name='BB Inferior',
        fill='tonexty'
    ), row=1, col=1)
    
    # EMAs
    fig.add_trace(go.Scatter(
        x=df.index, y=df['ema_12'],
        line=dict(color='blue', width=1),
        name='EMA 12'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['ema_26'],
        line=dict(color='orange', width=1),
        name='EMA 26'
    ), row=1, col=1)
    
    # Señales de compra
    buy_signals = df[df['buy_signal'] == True]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['low'] * 0.999,
            mode='markers',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            name='Señal Compra'
        ), row=1, col=1)
    
    # Señales de venta
    sell_signals = df[df['sell_signal'] == True]
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['high'] * 1.001,
            mode='markers',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            name='Señal Venta'
        ), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df['rsi'],
        line=dict(color='purple'),
        name='RSI'
    ), row=2, col=1)
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(
        x=df.index, y=df['macd'],
        line=dict(color='blue'),
        name='MACD'
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['macd_signal'],
        line=dict(color='red'),
        name='Señal MACD'
    ), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
    
    return fig

# -------------------------
# Streamlit App Principal
# -------------------------
def main():
    st.set_page_config(page_title="Crypto Millionaire 4.0", layout="wide")
    
    creds = setup_credentials()
    ml_model = EnhancedMLModel()
    signal_manager = SignalManager()
    trader = EnhancedCryptoTrader(creds, ml_model, signal_manager)

    # Header
    st.title("🚀 Millonario de Señales Crypto 4.0")
    st.markdown("### Sistema Avanzado de Trading 24/7 con IA")
    
    # Estado del bot
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🟢 INICIAR BOT 24/7"):
            trader.is_running = True
            st.success("¡Bot iniciado!")
            
    with col2:
        if st.button("🔴 DETENER BOT"):
            trader.is_running = False
            st.warning("Bot detenido")
            
    with col3:
        status = "🟢 ACTIVO" if trader.is_running else "🔴 INACTIVO"
        st.metric("Estado del Bot", status)

    # Panel de señales manuales
    st.sidebar.header("🎯 Señales Manuales")
    symbol_manual = st.sidebar.selectbox("Símbolo", trader.symbols)
    
    col_up, col_down = st.sidebar.columns(2)
    with col_up:
        if st.button("📈 SEÑAL SUBIDA", type="primary"):
            signal_manager.add_manual_signal(symbol_manual, "up", 1.0)
            st.success(f"Señal de subida para {symbol_manual}")
            
    with col_down:
        if st.button("📉 SEÑAL BAJADA", type="secondary"):
            signal_manager.add_manual_signal(symbol_manual, "down", 1.0)
            st.success(f"Señal de bajada para {symbol_manual}")

    # Controles manuales de trading
    st.sidebar.subheader("🎮 Control Manual")
    if st.sidebar.button("💰 COMPRAR MANUAL"):
        df = trader.get_historical(symbol_manual, 100)
        if df is not None:
            pos_id = trader.execute_smart_buy(symbol_manual, df["close"].iloc[-1], 1.0)
            if pos_id:
                st.sidebar.success(f"Compra ejecutada: {pos_id}")

    if st.sidebar.button("💸 VENDER MANUAL"):
        df = trader.get_historical(symbol_manual, 100)
        open_pos = trader.state_manager.get_open_positions(symbol_manual)
        if open_pos and df is not None:
            trader.execute_smart_sell(symbol_manual, open_pos[-1]["id"], 
                                   open_pos[-1]["qty"], df["close"].iloc[-1], 1.0)
            st.sidebar.success("Venta ejecutada")

    # Dashboard principal
    for symbol in trader.symbols:
        st.subheader(f"📊 {symbol}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Obtener datos y análisis
        df = trader.get_historical(symbol, 100)
        if df is not None:
            can_trade, score, reason = trader.should_trade_now(symbol)
            current_price = df["close"].iloc[-1]
            price_change = ((current_price - df["close"].iloc[-2]) / df["close"].iloc[-2]) * 100
            
            with col1:
                st.metric("Precio Actual", f"${current_price:.4f}", f"{price_change:.2f}%")
                
            with col2:
                color = "normal" if 0.4 <= score <= 0.6 else ("inverse" if score > 0.6 else "off")
                st.metric("Señal IA", f"{score:.1%}", help=reason, delta_color=color)
                
            with col3:
                recent_signals = signal_manager.get_recent_signals(symbol, hours=1)
                manual_count = len(recent_signals)
                st.metric("Señales Manuales (1h)", manual_count)
                
            with col4:
                open_positions = trader.state_manager.get_open_positions(symbol)
                total_qty = sum(p["qty"] for p in open_positions)
                st.metric("Posiciones Abiertas", len(open_positions), f"Qty: {total_qty:.6f}")

            # Gráfico avanzado
            chart = create_advanced_chart(df.copy(), symbol)
            if chart:
                st.plotly_chart(chart, use_container_width=True)

            # Mostrar posiciones detalladas
            if open_positions:
                st.markdown("#### Posiciones Detalladas")
                for p in open_positions:
                    profit = ((current_price / p["entry_price"]) - 1) * 100
                    color = "green" if profit > 0 else "red"
                    
                    st.markdown(f"""
                    <div style='background-color: rgba(0,0,0,0.1); padding: 10px; border-radius: 5px; margin: 5px 0;'>
                        <span style='color: {color}; font-weight: bold;'>
                            ID: {p['id'][-8:]} | Qty: {p['qty']:.6f} | 
                            Entrada: ${p['entry_price']:.4f} | 
                            P&L: {profit:.2f}% | 
                            Máximo: ${p['highest_price']:.4f}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

    # Bot de trading automático 24/7
    def auto_trading_bot():
        """Bot que opera 24/7 con múltiples estrategias"""
        while trader.is_running:
            try:
                for symbol in trader.symbols:
                    can_trade, score, reason = trader.should_trade_now(symbol)
                    
                    if not can_trade:
                        continue
                        
                    df = trader.get_historical(symbol, 100)
                    if df is None:
                        continue
                        
                    current_price = df["close"].iloc[-1]
                    open_positions = trader.state_manager.get_open_positions(symbol)
                    
                    # Gestión de posiciones existentes
                    for pos in open_positions:
                        profit = (current_price / pos["entry_price"]) - 1
                        
                        # Take profit más agresivo
                        if profit > 0.008:  # 0.8% ganancia
                            trader.execute_smart_sell(symbol, pos["id"], pos["qty"], current_price, 1.0)
                        
                        # Stop loss dinámico
                        elif profit < -0.012:  # 1.2% pérdida
                            trader.execute_smart_sell(symbol, pos["id"], pos["qty"], current_price, 0.5)
                    
                    # Nuevas compras basadas en señales fuertes
                    if score > 0.75 and len(open_positions) < 3:  # Máximo 3 posiciones por activo
                        trader.execute_smart_buy(symbol, current_price, score)
                    
                    # Actualizar posiciones
                    trader.state_manager.update_position(symbol, current_price)
                
                time.sleep(10)  # Verificar cada 10 segundos
                
            except Exception as e:
                print(f"Error en bot: {e}")
                time.sleep(30)  # Esperar más tiempo si hay error

    # Iniciar bot en thread separado si está activado
    if trader.is_running and 'bot_thread' not in st.session_state:
        st.session_state.bot_thread = threading.Thread(target=auto_trading_bot, daemon=True)
        st.session_state.bot_thread.start()

    # Auto-refresh cada 30 segundos
    time.sleep(5)
    st.rerun()

if __name__ == "__main__":
    main()
