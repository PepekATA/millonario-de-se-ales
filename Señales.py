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
except ImportError:
    install("alpaca-trade-api")
    import alpaca_trade_api as tradeapi

# -------------------------
# Archivos de persistencia
# -------------------------
CRED_FILE = "credentials.json"
STATE_FILE = "positions.json"
MODEL_FILE = "ml_model.json"
SIGNALS_FILE = "trading_signals.json"

# -------------------------
# Indicadores Técnicos Manuales (sin TA-Lib)
# -------------------------
class TechnicalAnalyzer:
    def __init__(self):
        pass
    
    def calculate_rsi(self, prices, period=14):
        """Calcula RSI manualmente"""
        if len(prices) < period + 1:
            return pd.Series([50] * len(prices), index=prices.index)
        
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcula MACD manualmente"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        
        return macd, macd_signal, macd_hist
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calcula Bandas de Bollinger manualmente"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def calculate_signals(self, df):
        """Calcula múltiples indicadores técnicos"""
        if len(df) < 30:
            return df
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(df['close'])
        
        # Medias móviles
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'])
        
        # Señales de compra/venta
        df['buy_signal'] = self.generate_buy_signals(df)
        df['sell_signal'] = self.generate_sell_signals(df)
        
        # Volatilidad
        df['volatility'] = df['close'].rolling(window=20).std()
        
        return df
    
    def generate_buy_signals(self, df):
        """Genera señales de compra basadas en múltiples indicadores"""
        signals = pd.Series(False, index=df.index)
        
        if len(df) < 30:
            return signals
        
        # RSI oversold
        rsi_signal = df['rsi'] < 35
        
        # MACD cruce alcista
        macd_signal = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        
        # Precio cerca de banda inferior
        bb_signal = df['close'] <= (df['bb_lower'] * 1.005)
        
        # EMA cruce alcista
        ema_signal = (df['ema_12'] > df['ema_26']) & (df['ema_12'].shift(1) <= df['ema_26'].shift(1))
        
        # Volumen (simulado con volatilidad)
        vol_signal = df['volatility'] > df['volatility'].rolling(window=10).mean()
        
        # Combinar señales (al menos 2 de 5)
        signal_count = (rsi_signal.astype(int) + macd_signal.astype(int) + 
                       bb_signal.astype(int) + ema_signal.astype(int) + vol_signal.astype(int))
        signals = signal_count >= 2
        
        return signals
    
    def generate_sell_signals(self, df):
        """Genera señales de venta"""
        signals = pd.Series(False, index=df.index)
        
        if len(df) < 30:
            return signals
        
        # RSI overbought
        rsi_signal = df['rsi'] > 65
        
        # MACD cruce bajista
        macd_signal = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # Precio cerca de banda superior
        bb_signal = df['close'] >= (df['bb_upper'] * 0.995)
        
        # EMA cruce bajista
        ema_signal = (df['ema_12'] < df['ema_26']) & (df['ema_12'].shift(1) >= df['ema_26'].shift(1))
        
        # Volatilidad alta (posible reversión)
        vol_signal = df['volatility'] > df['volatility'].rolling(window=20).quantile(0.8)
        
        signal_count = (rsi_signal.astype(int) + macd_signal.astype(int) + 
                       bb_signal.astype(int) + ema_signal.astype(int) + vol_signal.astype(int))
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
                try:
                    signal_time = datetime.fromisoformat(signal["timestamp"])
                    if signal_time > cutoff:
                        recent.append(signal)
                except:
                    continue
        
        return recent
    
    def get_signal_strength(self, symbol):
        """Calcula fuerza de señal combinada"""
        recent = self.get_recent_signals(symbol, hours=2)
        up_strength = sum(s["strength"] for s in recent if s["type"] == "up")
        down_strength = sum(s["strength"] for s in recent if s["type"] == "down")
        
        net_strength = up_strength - down_strength
        return max(-1, min(1, net_strength / 3))  # Normalizar entre -1 y 1

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
            self.model[symbol] = {'up': 0, 'down': 0, 'total_trades': 0}
        
        self.model[symbol][direction] += strength
        self.model[symbol]['total_trades'] += 1
        self.save_model()

    def predict_with_technical(self, symbol, df, signal_manager):
        """Predicción combinando ML y análisis técnico"""
        if len(df) < 30:
            return 0.5, "📊 Datos insuficientes para análisis"
        
        # Análisis técnico
        df = self.analyzer.calculate_signals(df)
        
        # Señal técnica básica
        technical_score = 0.5
        tech_reason = "neutro"
        
        last_row = df.iloc[-1]
        
        if last_row['buy_signal']:
            technical_score = 0.75
            tech_reason = "compra técnica"
        elif last_row['sell_signal']:
            technical_score = 0.25
            tech_reason = "venta técnica"
        else:
            # Análisis de tendencia
            if last_row['ema_12'] > last_row['ema_26'] and last_row['rsi'] < 60:
                technical_score = 0.65
                tech_reason = "tendencia alcista"
            elif last_row['ema_12'] < last_row['ema_26'] and last_row['rsi'] > 40:
                technical_score = 0.35
                tech_reason = "tendencia bajista"
        
        # ML score
        ml_score = self.predict(symbol)
        
        # Señales manuales
        manual_strength = signal_manager.get_signal_strength(symbol)
        manual_score = 0.5 + (manual_strength * 0.3)
        
        # Combinar scores
        final_score = (technical_score * 0.5) + (ml_score * 0.3) + (manual_score * 0.2)
        final_score = max(0.05, min(0.95, final_score))
        
        reason = f"🔬 {tech_reason} | 🤖 ML:{ml_score:.2f} | 👤 Manual:{manual_score:.2f}"
        
        return final_score, reason

    def predict(self, symbol):
        if symbol not in self.model:
            return 0.5
        
        total = self.model[symbol]['up'] + self.model[symbol]['down']
        if total == 0:
            return 0.5
        
        return self.model[symbol]['up'] / total

# -------------------------
# Manejo de estados
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
        return {"positions": {}, "daily_stats": {}}

    def save_state(self):
        with self.lock:
            json.dump(self.state, open(STATE_FILE, "w"), indent=2)

    def add_position(self, symbol, qty, entry_price):
        with self.lock:
            if symbol not in self.state["positions"]:
                self.state["positions"][symbol] = []
            
            pos_id = f"{symbol}_{int(time.time())}"
            position = {
                "id": pos_id,
                "qty": qty,
                "entry_price": entry_price,
                "highest_price": entry_price,
                "lowest_price": entry_price,
                "status": "open",
                "created_at": datetime.now().isoformat(),
                "stop_loss": entry_price * 0.988,  # 1.2% stop loss
                "take_profit": entry_price * 1.008  # 0.8% take profit
            }
            
            self.state["positions"][symbol].append(position)
            self.save_state()
            return pos_id

    def update_position(self, symbol, current_price):
        with self.lock:
            if symbol in self.state["positions"]:
                for pos in self.state["positions"][symbol]:
                    if pos["status"] == "open":
                        pos["highest_price"] = max(pos["highest_price"], current_price)
                        pos["lowest_price"] = min(pos["lowest_price"], current_price)
                        
                        # Trailing stop dinámico
                        if current_price > pos["entry_price"] * 1.015:  # Si sube 1.5%
                            pos["stop_loss"] = max(pos["stop_loss"], current_price * 0.995)  # Ajustar stop
            self.save_state()

    def get_open_positions(self, symbol=None):
        with self.lock:
            if symbol:
                return [p for p in self.state["positions"].get(symbol, []) if p["status"] == "open"]
            
            all_pos = []
            for sym, pos_list in self.state["positions"].items():
                all_pos.extend([p for p in pos_list if p["status"] == "open"])
            return all_pos

    def close_position(self, symbol, pos_id, exit_price=None):
        with self.lock:
            if symbol in self.state["positions"]:
                for p in self.state["positions"][symbol]:
                    if p["id"] == pos_id and p["status"] == "open":
                        p["status"] = "closed"
                        p["closed_at"] = datetime.now().isoformat()
                        if exit_price:
                            p["exit_price"] = exit_price
                            p["pnl"] = ((exit_price / p["entry_price"]) - 1) * 100
                        self.save_state()
                        return True
        return False

    def get_daily_pnl(self):
        """Calcula P&L del día"""
        today = datetime.now().date()
        total_pnl = 0
        
        with self.lock:
            for symbol, positions in self.state["positions"].items():
                for pos in positions:
                    if pos["status"] == "closed" and "exit_price" in pos:
                        closed_date = datetime.fromisoformat(pos["closed_at"]).date()
                        if closed_date == today:
                            total_pnl += pos.get("pnl", 0)
        
        return total_pnl

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
        # Expanded list of symbols for more trading options
        self.symbols = [
            "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD",
            "ADA/USD", "DOGE/USD", "MATIC/USD", "XRP/USD",
            "LINK/USD", "SHIB/USD"
        ]
        self.is_running = False
        self.capital_fraction = 0.05
        self.analyzer = TechnicalAnalyzer()
        self.last_trade_time = {}

    def validate_symbol(self, symbol):
        """Valida si el símbolo es tradable"""
        try:
            asset = self.api.get_asset(symbol)
            return asset.tradable
        except Exception as e:
            st.error(f"Invalid symbol {symbol}: {e}")
            return False

    def get_account_info(self):
        """Obtiene información de la cuenta"""
        try:
            account = self.api.get_account()
            return {
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "day_trade_count": int(account.day_trade_count)
            }
        except Exception as e:
            st.error(f"Error obtaining account info: {e}")
            return {"cash": 0, "portfolio_value": 0, "buying_power": 0, "day_trade_count": 0}

    def get_historical(self, symbol, limit=100):
        """Obtiene datos históricos"""
        try:
            if not self.validate_symbol(symbol):
                return None
            bars = self.api.get_crypto_bars(symbol, "1Min", limit=limit).df
            if bars.empty:
                st.error(f"No data returned for {symbol}")
                return None
            
            # Filtrar por exchange si es necesario (Coinbase)
            if 'exchange' in bars.columns:
                bars = bars[bars['exchange'] == 'CBSE']
            
            bars = bars.tail(limit)
            return bars
        except Exception as e:
            st.error(f"Error obteniendo datos para {symbol}: {e}")
            return None

    def can_trade(self, symbol):
        """Verifica si puede hacer trading"""
        now = time.time()
        last_trade = self.last_trade_time.get(symbol, 0)
        
        # Limitar frecuencia de trading (mínimo 2 minutos entre trades)
        if now - last_trade < 120:
            return False
            
        # Verificar posiciones abiertas
        open_positions = self.state_manager.get_open_positions(symbol)
        if len(open_positions) >= 2:  # Máximo 2 posiciones por símbolo
            return False
            
        return True

    def execute_smart_buy(self, symbol, current_price, signal_strength=1.0):
        """Compra inteligente basada en fuerza de señal"""
        if not self.can_trade(symbol):
            return None
            
        try:
            account_info = self.get_account_info()
            buying_power = account_info["cash"]
            
            if buying_power < 10:  # Mínimo $10 para operar
                return None
            
            # Ajustar cantidad según fuerza de señal
            base_qty = (buying_power * self.capital_fraction) / current_price
            adjusted_qty = max(0.0001, base_qty * min(signal_strength, 1.5))
            
            # Ejecutar orden
            order = self.api.submit_order(
                symbol=symbol,
                qty=adjusted_qty,
                side="buy",
                type="market",
                time_in_force="day"
            )
            
            pos_id = self.state_manager.add_position(symbol, adjusted_qty, current_price)
            self.ml_model.update(symbol, "up", signal_strength)
            self.last_trade_time[symbol] = time.time()
            
            return pos_id
            
        except Exception as e:
            st.error(f"Error comprando {symbol}: {e}")
            return None

    def execute_smart_sell(self, symbol, pos_id, qty, current_price, signal_strength=1.0):
        """Venta inteligente"""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side="sell",
                type="market",
                time_in_force="day"
            )
            
            self.state_manager.close_position(symbol, pos_id, current_price)
            self.ml_model.update(symbol, "down", signal_strength)
            self.last_trade_time[symbol] = time.time()
            
            return True
            
        except Exception as e:
            st.error(f"Error vendiendo {symbol}: {e}")
            return False

    def manage_positions(self, symbol, current_price):
        """Gestión automática de posiciones"""
        open_positions = self.state_manager.get_open_positions(symbol)
        
        for pos in open_positions:
            # Calcular P&L
            pnl = (current_price / pos["entry_price"]) - 1
            
            # Take Profit
            if current_price >= pos.get("take_profit", pos["entry_price"] * 1.008):
                self.execute_smart_sell(symbol, pos["id"], pos["qty"], current_price, 1.0)
                continue
            
            # Stop Loss
            if current_price <= pos.get("stop_loss", pos["entry_price"] * 0.988):
                self.execute_smart_sell(symbol, pos["id"], pos["qty"], current_price, 0.5)
                continue
            
            # Actualizar posición
            self.state_manager.update_position(symbol, current_price)

    def should_trade_now(self, symbol):
        """Determina si debe operar"""
        df = self.get_historical(symbol, 100)
        if df is None or len(df) < 30:
            return False, 0.5, "Sin datos suficientes"
        
        # Obtener predicción
        score, reason = self.ml_model.predict_with_technical(symbol, df, self.signal_manager)
        
        return True, score, reason

# -------------------------
# Funciones de credenciales
# -------------------------
def save_credentials(key, secret, paper=True):
    data = {
        "ALPACA_API_KEY": key.strip(),
        "ALPACA_API_SECRET": secret.strip(),
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
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
    st.info("🔑 Necesitas una cuenta de Alpaca Markets para usar este bot")
    
    with st.form("cred_form"):
        st.markdown("### Configuración de API")
        key = st.text_input("🔐 Alpaca API Key", type="password", help="Tu clave API de Alpaca")
        secret = st.text_input("🔒 Alpaca API Secret", type="password", help="Tu clave secreta de Alpaca")
        paper = st.checkbox("📄 Usar cuenta demo (Paper Trading)", value=True, 
                           help="Recomendado para pruebas")
        
        if st.form_submit_button("💾 Guardar y Conectar", type="primary"):
            if key.strip() and secret.strip():
                save_credentials(key, secret, paper)
                st.success("✅ Credenciales guardadas. Reiniciando...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Por favor ingresa tanto la API Key como el Secret")
    
    with st.expander("ℹ️ ¿Cómo obtener las credenciales?"):
        st.markdown("""
        1. Ve a [Alpaca Markets](https://alpaca.markets)
        2. Crea una cuenta gratuita
        3. Ve a la sección de API
        4. Genera tu API Key y Secret
        5. Para pruebas, usa Paper Trading
        """)
    
    st.stop()

# -------------------------
# Función para crear gráficos avanzados
# -------------------------
def create_advanced_chart(df, symbol):
    """Crea gráfico avanzado con indicadores y señales"""
    if df is None or len(df) < 20:
        return None
    
    try:
        # Aplicar análisis técnico
        analyzer = TechnicalAnalyzer()
        df = analyzer.calculate_signals(df)
        
        # Crear subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} - Precio y Señales', 'RSI (14)', 'MACD'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick principal
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Precio',
            increasing_line_color='green',
            decreasing_line_color='red'
        ), row=1, col=1)
        
        # Bollinger Bands si existen
        if 'bb_upper' in df.columns and not df['bb_upper'].isna().all():
            fig.add_trace(go.Scatter(
                x=df.index, y=df['bb_upper'],
                line=dict(color='rgba(255,0,0,0.3)', width=1),
                name='BB Superior',
                showlegend=False
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index, y=df['bb_lower'],
                line=dict(color='rgba(0,255,0,0.3)', width=1),
                name='BB Inferior',
                fill='tonexty',
                fillcolor='rgba(0,100,255,0.1)',
                showlegend=False
            ), row=1, col=1)
        
        # EMAs
        if 'ema_12' in df.columns and not df['ema_12'].isna().all():
            fig.add_trace(go.Scatter(
                x=df.index, y=df['ema_12'],
                line=dict(color='blue', width=1),
                name='EMA 12'
            ), row=1, col=1)
        
        if 'ema_26' in df.columns and not df['ema_26'].isna().all():
            fig.add_trace(go.Scatter(
                x=df.index, y=df['ema_26'],
                line=dict(color='orange', width=1),
                name='EMA 26'
            ), row=1, col=1)
        
        # Señales de compra
        if 'buy_signal' in df.columns:
            buy_signals = df[df['buy_signal'] == True]
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['low'] * 0.998,
                    mode='markers',
                    marker=dict(color='lime', size=12, symbol='triangle-up'),
                    name='🟢 Compra'
                ), row=1, col=1)
        
        # Señales de venta
        if 'sell_signal' in df.columns:
            sell_signals = df[df['sell_signal'] == True]
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['high'] * 1.002,
                    mode='markers',
                    marker=dict(color='red', size=12, symbol='triangle-down'),
                    name='🔴 Venta'
                ), row=1, col=1)
        
        # RSI
        if 'rsi' in df.columns and not df['rsi'].isna().all():
            fig.add_trace(go.Scatter(
                x=df.index, y=df['rsi'],
                line=dict(color='purple', width=2),
                name='RSI'
            ), row=2, col=1)
            
            # Líneas de referencia RSI
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         annotation_text="Sobrecompra", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                         annotation_text="Sobreventa", row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        # MACD
        if 'macd' in df.columns and not df['macd'].isna().all():
            fig.add_trace(go.Scatter(
                x=df.index, y=df['macd'],
                line=dict(color='blue', width=2),
                name='MACD'
            ), row=3, col=1)
            
            if 'macd_signal' in df.columns and not df['macd_signal'].isna().all():
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['macd_signal'],
                    line=dict(color='red', width=1),
                    name='Señal'
                ), row=3, col=1)
            
            if 'macd_hist' in df.columns and not df['macd_hist'].isna().all():
                colors = ['green' if x >= 0 else 'red' for x in df['macd_hist']]
                fig.add_trace(go.Bar(
                    x=df.index, y=df['macd_hist'],
                    name='Histograma',
                    marker_color=colors,
                    opacity=0.6
                ), row=3, col=1)
        
        # Configuración del layout
        fig.update_layout(
            height=800,
            title=f"📊 {symbol} - Análisis Técnico Completo",
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            font=dict(size=10),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Actualizar ejes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
        
    except Exception as e:
        st.error(f"Error creando gráfico: {e}")
        return None

# -------------------------
# Funciones de utilidad UI
# -------------------------
def display_position_card(pos, current_price, symbol):
    """Muestra tarjeta de posición individual"""
    entry_price = pos["entry_price"]
    qty = pos["qty"]
    pnl = ((current_price / entry_price) - 1) * 100
    value = qty * current_price
    
    # Colores basados en P&L
    if pnl > 0:
        color = "🟢"
        bg_color = "rgba(0,255,0,0.1)"
    else:
        color = "🔴"
        bg_color = "rgba(255,0,0,0.1)"
    
    # Tiempo desde apertura
    created = datetime.fromisoformat(pos["created_at"])
    time_open = datetime.now() - created
    hours_open = int(time_open.total_seconds() / 3600)
    
    st.markdown(f"""
    <div style='background: {bg_color}; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid {"green" if pnl > 0 else "red"}'>
        <h4>{color} {symbol} - ID: {pos['id'][-8:]}</h4>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <strong>Cantidad:</strong> {qty:.6f}<br>
                <strong>Precio Entrada:</strong> ${entry_price:.4f}<br>
                <strong>Precio Actual:</strong> ${current_price:.4f}
            </div>
            <div style='text-align: right;'>
                <strong>P&L:</strong> <span style='color: {"green" if pnl > 0 else "red"}; font-size: 18px;'>{pnl:+.2f}%</span><br>
                <strong>Valor:</strong> ${value:.2f}<br>
                <strong>Tiempo:</strong> {hours_open}h
            </div>
        </div>
        <div style='margin-top: 10px;'>
            <small>🔝 Máximo: ${pos.get("highest_price", entry_price):.4f} | 
            🔻 Mínimo: ${pos.get("lowest_price", entry_price):.4f}</small>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_trading_stats(trader):
    """Muestra estadísticas de trading"""
    account_info = trader.get_account_info()
    daily_pnl = trader.state_manager.get_daily_pnl()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Capital", f"${account_info['cash']:.2f}")
    
    with col2:
        st.metric("📈 Portafolio", f"${account_info['portfolio_value']:.2f}")
    
    with col3:
        pnl_color = "normal" if daily_pnl == 0 else ("normal" if daily_pnl > 0 else "inverse")
        st.metric("📊 P&L Diario", f"{daily_pnl:+.2f}%", delta_color=pnl_color)
    
    with col4:
        total_positions = len(trader.state_manager.get_open_positions())
        st.metric("🎯 Posiciones", total_positions)

# -------------------------
# Bot de Trading 24/7
# -------------------------
def create_trading_bot(trader):
    """Crea y ejecuta el bot de trading automático"""
    
    def trading_loop():
        """Loop principal del bot"""
        while trader.is_running:
            try:
                for symbol in trader.symbols:
                    if not trader.is_running:
                        break
                    
                    # Obtener datos actuales
                    df = trader.get_historical(symbol, 100)
                    if df is None or len(df) < 30:
                        continue
                    
                    current_price = df["close"].iloc[-1]
                    
                    # Gestionar posiciones existentes
                    trader.manage_positions(symbol, current_price)
                    
                    # Evaluar nuevas oportunidades
                    can_trade, score, reason = trader.should_trade_now(symbol)
                    
                    if not can_trade:
                        continue
                    
                    # Decisiones de trading basadas en score
                    if score > 0.75 and trader.can_trade(symbol):
                        # Señal fuerte de compra
                        trader.execute_smart_buy(symbol, current_price, score)
                        st.success(f"🟢 Compra automática: {symbol} - Score: {score:.2f}")
                    
                    elif score < 0.25:
                        # Señal fuerte de venta - cerrar posiciones
                        open_positions = trader.state_manager.get_open_positions(symbol)
                        for pos in open_positions[:1]:  # Cerrar solo una posición
                            if trader.execute_smart_sell(symbol, pos["id"], pos["qty"], current_price, 1-score):
                                st.warning(f"🔴 Venta automática: {symbol} - Score: {score:.2f}")
                
                # Pausa entre ciclos
                time.sleep(15)  # Verificar cada 15 segundos
                
            except Exception as e:
                st.error(f"❌ Error en bot: {e}")
                time.sleep(30)  # Pausa más larga si hay error
    
    # Ejecutar bot en thread separado
    if trader.is_running:
        if 'bot_thread' not in st.session_state or not st.session_state.bot_thread.is_alive():
            st.session_state.bot_thread = threading.Thread(target=trading_loop, daemon=True)
            st.session_state.bot_thread.start()

# -------------------------
# Streamlit App Principal
# -------------------------
def main():
    st.set_page_config(
        page_title="Crypto Millionaire Bot 4.0", 
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🚀"
    )
    
    # CSS personalizado
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 20px;
        }
        .metric-card {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .signal-button {
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Inicializar componentes
    creds = setup_credentials()
    ml_model = EnhancedMLModel()
    signal_manager = SignalManager()
    trader = EnhancedCryptoTrader(creds, ml_model, signal_manager)

    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🚀 MILLONARIO DE SEÑALES CRYPTO 4.0</h1>
        <h3>Sistema Avanzado de Trading Automático 24/7 con Inteligencia Artificial</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Control del Bot
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🟢 INICIAR BOT 24/7", type="primary", use_container_width=True):
            trader.is_running = True
            st.session_state.bot_status = "running"
            st.success("🤖 Bot iniciado y operando!")
            st.balloons()
    
    with col2:
        if st.button("🔴 DETENER BOT", type="secondary", use_container_width=True):
            trader.is_running = False
            st.session_state.bot_status = "stopped"
            st.warning("⏹️ Bot detenido")
    
    with col3:
        status = "🟢 ACTIVO" if trader.is_running else "🔴 INACTIVO"
        color = "green" if trader.is_running else "red"
        st.markdown(f"<h3 style='color: {color}; text-align: center;'>{status}</h3>", unsafe_allow_html=True)
    
    with col4:
        if st.button("🔄 ACTUALIZAR", use_container_width=True):
            st.rerun()

    # Ejecutar bot si está activo
    if trader.is_running:
        create_trading_bot(trader)

    # Panel lateral de control
    st.sidebar.markdown("## 🎛️ PANEL DE CONTROL")
    
    # Selector de símbolo
    symbol_manual = st.sidebar.selectbox(
        "🎯 Seleccionar Activo", 
        trader.symbols,
        help="Elige el activo para señales manuales"
    )
    
    # Botones de señales manuales
    st.sidebar.markdown("### 📊 SEÑALES MANUALES")
    
    col_up, col_down = st.sidebar.columns(2)
    
    with col_up:
        if st.button("📈 SEÑAL\nSUBIDA", key="signal_up", use_container_width=True):
            signal_manager.add_manual_signal(symbol_manual, "up", 1.0)
            st.sidebar.success(f"📈 Señal ALCISTA para {symbol_manual}")
    
    with col_down:
        if st.button("📉 SEÑAL\nBAJADA", key="signal_down", use_container_width=True):
            signal_manager.add_manual_signal(symbol_manual, "down", 1.0)
            st.sidebar.success(f"📉 Señal BAJISTA para {symbol_manual}")
    
    # Fuerza de señal
    signal_strength = st.sidebar.slider(
        "💪 Fuerza de Señal", 
        min_value=0.1, 
        max_value=2.0, 
        value=1.0, 
        step=0.1,
        help="Intensidad de la señal manual"
    )
    
    # Controles manuales de trading
    st.sidebar.markdown("### 🎮 TRADING MANUAL")
    
    if st.sidebar.button("💰 COMPRAR AHORA", type="primary", use_container_width=True):
        df = trader.get_historical(symbol_manual, 100)
        if df is not None:
            current_price = df["close"].iloc[-1]
            pos_id = trader.execute_smart_buy(symbol_manual, current_price, signal_strength)
            if pos_id:
                st.sidebar.success(f"✅ Compra ejecutada: {symbol_manual}\nID: {pos_id[-8:]}")
            else:
                st.sidebar.error("❌ No se pudo ejecutar la compra")
    
    if st.sidebar.button("💸 VENDER TODO", type="secondary", use_container_width=True):
        open_positions = trader.state_manager.get_open_positions(symbol_manual)
        if open_positions:
            df = trader.get_historical(symbol_manual, 100)
            if df is not None:
                current_price = df["close"].iloc[-1]
                sold_count = 0
                for pos in open_positions:
                    if trader.execute_smart_sell(symbol_manual, pos["id"], pos["qty"], current_price, signal_strength):
                        sold_count += 1
                st.sidebar.success(f"✅ {sold_count} posiciones vendidas")
        else:
            st.sidebar.info("ℹ️ No hay posiciones abiertas")

    # Estadísticas principales
    st.markdown("## 📊 ESTADÍSTICAS DE TRADING")
    display_trading_stats(trader)
    
    # Separador
    st.markdown("---")

    # Dashboard por símbolo
    for symbol in trader.symbols:
        with st.expander(f"📈 {symbol} - Análisis Completo", expanded=False):
            
            # Obtener datos y métricas
            df = trader.get_historical(symbol, 100)
            if df is None:
                st.error(f"❌ No se pudieron obtener datos para {symbol}")
                continue
            
            current_price = df["close"].iloc[-1]
            can_trade, score, reason = trader.should_trade_now(symbol)
            open_positions = trader.state_manager.get_open_positions(symbol)
            
            # Métricas principales
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                price_change = ((current_price - df["close"].iloc[-2]) / df["close"].iloc[-2]) * 100
                st.metric(
                    "💲 Precio", 
                    f"${current_price:.4f}", 
                    f"{price_change:+.2f}%",
                    delta_color="normal"
                )
            
            with col2:
                # Color del score basado en valor
                if score > 0.7:
                    score_color = "🟢"
                elif score < 0.3:
                    score_color = "🔴"
                else:
                    score_color = "🟡"
                
                st.metric(
                    "🤖 Señal IA", 
                    f"{score_color} {score:.1%}",
                    help=reason
                )
            
            with col3:
                recent_signals = signal_manager.get_recent_signals(symbol, hours=2)
                manual_strength = signal_manager.get_signal_strength(symbol)
                manual_color = "🟢" if manual_strength > 0 else ("🔴" if manual_strength < 0 else "⚪")
                
                st.metric(
                    "👤 Señales Manuales", 
                    f"{manual_color} {len(recent_signals)}",
                    f"Fuerza: {manual_strength:+.1f}"
                )
            
            with col4:
                total_qty = sum(p["qty"] for p in open_positions)
                total_value = total_qty * current_price
                
                st.metric(
                    "💼 Posiciones", 
                    len(open_positions),
                    f"${total_value:.2f}"
                )
            
            with col5:
                if open_positions:
                    avg_pnl = sum(((current_price / p["entry_price"]) - 1) * 100 for p in open_positions) / len(open_positions)
                    pnl_color = "normal" if avg_pnl > 0 else "inverse"
                    st.metric(
                        "📊 P&L Promedio", 
                        f"{avg_pnl:+.2f}%",
                        delta_color=pnl_color
                    )
                else:
                    st.metric("📊 P&L Promedio", "0.00%")
            
            # Gráfico avanzado
            chart = create_advanced_chart(df.copy(), symbol)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Posiciones detalladas
            if open_positions:
                st.markdown(f"#### 💼 Posiciones Abiertas en {symbol}")
                for pos in open_positions:
                    display_position_card(pos, current_price, symbol)
            
            # Señales recientes
            recent_signals = signal_manager.get_recent_signals(symbol, hours=6)
            if recent_signals:
                st.markdown(f"#### 🔔 Señales Recientes ({symbol})")
                for i, signal in enumerate(recent_signals[-3:]):  # Mostrar últimas 3
                    time_ago = datetime.now() - datetime.fromisoformat(signal["timestamp"])
                    hours_ago = int(time_ago.total_seconds() / 3600)
                    minutes_ago = int((time_ago.total_seconds() % 3600) / 60)
                    
                    signal_icon = "📈" if signal["type"] == "up" else "📉"
                    signal_text = "ALCISTA" if signal["type"] == "up" else "BAJISTA"
                    
                    st.info(f"{signal_icon} **Señal {signal_text}** - Hace {hours_ago}h {minutes_ago}m - Fuerza: {signal['strength']}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; margin-top: 20px;'>
        <h4>🚀 Millonario de Señales Crypto 4.0</h4>
        <p>Sistema de Trading Automático con IA • Desarrollado para maximizar ganancias 24/7</p>
        <p><em>⚠️ El trading de criptomonedas conlleva riesgos. Opera con responsabilidad.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh cada 30 segundos cuando el bot está activo
    if trader.is_running:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
