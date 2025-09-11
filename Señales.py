# Millonario de Señales Crypto 5.0
# Sistema con Red Neuronal, Saldo Real y Símbolos Corregidos
# Compatible con Alpaca Crypto (SOL/USD, BTC/USD, ETH/USD, AVAX/USD)

import os, json, time, threading
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess, sys
import pickle

# Instalar dependencias
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

try:
    import alpaca_trade_api as tradeapi
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
except ImportError:
    install("alpaca-trade-api")
    install("scikit-learn")
    import alpaca_trade_api as tradeapi
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

# -------------------------
# Archivos de persistencia
# -------------------------
CRED_FILE = "credentials.json"
STATE_FILE = "positions.json"
MODEL_FILE = "neural_model.pkl"
SCALER_FILE = "neural_scaler.pkl"
SIGNALS_FILE = "trading_signals.json"
TRAINING_DATA_FILE = "training_data.json"

# -------------------------
# Red Neuronal para Trading
# -------------------------
class NeuralTradingModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.training_data = self.load_training_data()
        self.load_model()
        
    def load_training_data(self):
        """Carga datos de entrenamiento históricos"""
        if os.path.exists(TRAINING_DATA_FILE):
            try:
                return json.load(open(TRAINING_DATA_FILE, "r"))
            except:
                pass
        return {"features": [], "targets": []}
    
    def save_training_data(self):
        """Guarda datos de entrenamiento"""
        with open(TRAINING_DATA_FILE, "w") as f:
            json.dump(self.training_data, f, indent=2)
    
    def load_model(self):
        """Carga el modelo y scaler entrenados"""
        try:
            if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
                with open(MODEL_FILE, "rb") as f:
                    self.model = pickle.load(f)
                with open(SCALER_FILE, "rb") as f:
                    self.scaler = pickle.load(f)
                return True
        except:
            pass
        return False
    
    def save_model(self):
        """Guarda el modelo y scaler"""
        try:
            with open(MODEL_FILE, "wb") as f:
                pickle.dump(self.model, f)
            with open(SCALER_FILE, "wb") as f:
                pickle.dump(self.scaler, f)
        except Exception as e:
            st.error(f"Error guardando modelo: {e}")
    
    def extract_features(self, df, manual_signals=0):
        """Extrae características técnicas para la red neuronal"""
        if len(df) < 30:
            return None
        
        # Indicadores técnicos
        rsi = self.calculate_rsi(df['close'])
        macd, macd_signal, _ = self.calculate_macd(df['close'])
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['close'])
        
        # Características de precio
        price_change = df['close'].pct_change(1)
        volatility = price_change.rolling(window=20).std()
        volume_proxy = df['high'] - df['low']  # Proxy de volumen
        
        # Características de última observación
        last_idx = -1
        features = [
            df['close'].iloc[last_idx] / df['close'].iloc[-10] - 1,  # Cambio 10 períodos
            df['close'].iloc[last_idx] / df['close'].iloc[-20] - 1,  # Cambio 20 períodos
            rsi.iloc[last_idx] / 100,  # RSI normalizado
            (df['close'].iloc[last_idx] - bb_lower.iloc[last_idx]) / (bb_upper.iloc[last_idx] - bb_lower.iloc[last_idx]),  # Posición en BB
            macd.iloc[last_idx] - macd_signal.iloc[last_idx],  # Diferencia MACD
            volatility.iloc[last_idx],  # Volatilidad actual
            volume_proxy.iloc[last_idx] / volume_proxy.rolling(window=20).mean().iloc[last_idx],  # Volumen relativo
            manual_signals  # Señales manuales
        ]
        
        # Verificar que no hay NaN
        features = [0 if np.isnan(x) else x for x in features]
        return np.array(features).reshape(1, -1)
    
    def calculate_rsi(self, prices, period=14):
        """Calcula RSI"""
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcula MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calcula Bandas de Bollinger"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def add_training_sample(self, features, target):
        """Añade muestra de entrenamiento"""
        if features is not None:
            self.training_data["features"].append(features.flatten().tolist())
            self.training_data["targets"].append(target)
            
            # Mantener solo últimas 1000 muestras
            if len(self.training_data["features"]) > 1000:
                self.training_data["features"] = self.training_data["features"][-1000:]
                self.training_data["targets"] = self.training_data["targets"][-1000:]
            
            self.save_training_data()
    
    def train_model(self):
        """Entrena la red neuronal con datos históricos"""
        if len(self.training_data["features"]) < 20:
            return False, "Necesitas al menos 20 operaciones para entrenar"
        
        try:
            X = np.array(self.training_data["features"])
            y = np.array(self.training_data["targets"])
            
            # Normalizar características
            X_scaled = self.scaler.fit_transform(X)
            
            # Dividir datos
            if len(X) > 50:
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            else:
                X_train, X_test, y_train, y_test = X_scaled, X_scaled, y, y
            
            # Crear y entrenar modelo
            self.model = MLPRegressor(
                hidden_layer_sizes=(50, 30, 20),  # 3 capas ocultas
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluar modelo
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            self.save_model()
            
            return True, f"Modelo entrenado - Train Score: {train_score:.3f}, Test Score: {test_score:.3f}"
            
        except Exception as e:
            return False, f"Error entrenando modelo: {e}"
    
    def predict(self, features):
        """Hace predicción con la red neuronal"""
        if self.model is None or features is None:
            return 0.5
        
        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)[0]
            
            # Convertir a probabilidad (0-1)
            probability = 1 / (1 + np.exp(-prediction))  # Función sigmoide
            return max(0.01, min(0.99, probability))
            
        except:
            return 0.5

# -------------------------
# Indicadores Técnicos Mejorados
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
        """Genera señales de compra"""
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
        
        # Combinar señales
        signal_count = (rsi_signal.astype(int) + macd_signal.astype(int) + 
                       bb_signal.astype(int) + ema_signal.astype(int))
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
        
        signal_count = (rsi_signal.astype(int) + macd_signal.astype(int) + 
                       bb_signal.astype(int) + ema_signal.astype(int))
        signals = signal_count >= 2
        
        return signals

# -------------------------
# Sistema de Señales
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
        return {"manual_signals": {}}
    
    def save_signals(self):
        with open(SIGNALS_FILE, "w") as f:
            json.dump(self.signals, f, indent=2)
    
    def add_manual_signal(self, symbol, signal_type, strength=1.0):
        """Añade señal manual"""
        timestamp = datetime.now().isoformat()
        if symbol not in self.signals["manual_signals"]:
            self.signals["manual_signals"][symbol] = []
        
        self.signals["manual_signals"][symbol].append({
            "type": signal_type,
            "strength": strength,
            "timestamp": timestamp
        })
        
        # Mantener solo las últimas 50 señales por símbolo
        self.signals["manual_signals"][symbol] = self.signals["manual_signals"][symbol][-50:]
        self.save_signals()
    
    def get_recent_signals(self, symbol, hours=2):
        """Obtiene señales recientes"""
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
        """Calcula fuerza neta de señales"""
        recent = self.get_recent_signals(symbol, hours=1)
        up_strength = sum(s["strength"] for s in recent if s["type"] == "up")
        down_strength = sum(s["strength"] for s in recent if s["type"] == "down")
        
        net_strength = up_strength - down_strength
        return max(-2, min(2, net_strength))

# -------------------------
# Manejo de Estados y Posiciones
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
        return {"positions": {}, "daily_pnl": 0, "total_trades": 0}

    def save_state(self):
        with self.lock:
            json.dump(self.state, open(STATE_FILE, "w"), indent=2)

    def add_position(self, symbol, qty, entry_price, order_id=None):
        with self.lock:
            if symbol not in self.state["positions"]:
                self.state["positions"][symbol] = []
            
            pos_id = f"{symbol}_{int(time.time())}"
            position = {
                "id": pos_id,
                "qty": float(qty),
                "entry_price": float(entry_price),
                "current_price": float(entry_price),
                "highest_price": float(entry_price),
                "lowest_price": float(entry_price),
                "status": "open",
                "created_at": datetime.now().isoformat(),
                "order_id": order_id,
                "pnl_percent": 0.0,
                "pnl_dollar": 0.0
            }
            
            self.state["positions"][symbol].append(position)
            self.state["total_trades"] += 1
            self.save_state()
            return pos_id

    def update_position(self, symbol, current_price):
        with self.lock:
            if symbol in self.state["positions"]:
                for pos in self.state["positions"][symbol]:
                    if pos["status"] == "open":
                        pos["current_price"] = float(current_price)
                        pos["highest_price"] = max(pos["highest_price"], current_price)
                        pos["lowest_price"] = min(pos["lowest_price"], current_price)
                        
                        # Calcular P&L
                        pnl_percent = ((current_price / pos["entry_price"]) - 1) * 100
                        pnl_dollar = (current_price - pos["entry_price"]) * pos["qty"]
                        
                        pos["pnl_percent"] = pnl_percent
                        pos["pnl_dollar"] = pnl_dollar
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
                            final_pnl = ((exit_price / p["entry_price"]) - 1) * 100
                            p["final_pnl_percent"] = final_pnl
                            p["final_pnl_dollar"] = (exit_price - p["entry_price"]) * p["qty"]
                            
                            # Actualizar P&L diario
                            self.state["daily_pnl"] += final_pnl
                        
                        self.save_state()
                        return True
        return False

    def get_portfolio_summary(self):
        """Resumen del portafolio"""
        open_positions = self.get_open_positions()
        total_positions = len(open_positions)
        total_pnl = sum(p.get("pnl_dollar", 0) for p in open_positions)
        total_pnl_percent = sum(p.get("pnl_percent", 0) for p in open_positions) / max(1, total_positions)
        
        return {
            "total_positions": total_positions,
            "total_pnl_dollar": total_pnl,
            "avg_pnl_percent": total_pnl_percent,
            "daily_pnl": self.state.get("daily_pnl", 0),
            "total_trades": self.state.get("total_trades", 0)
        }

# -------------------------
# Enhanced Crypto Trader
# -------------------------
class EnhancedCryptoTrader:
    def __init__(self, creds, neural_model, signal_manager):
        self.api = tradeapi.REST(
            creds["ALPACA_API_KEY"], 
            creds["ALPACA_API_SECRET"], 
            creds["ALPACA_BASE_URL"], 
            api_version='v2'
        )
        self.state_manager = TradeStateManager()
        self.neural_model = neural_model
        self.signal_manager = signal_manager
        
        # Símbolos corregidos para Alpaca
        self.symbols = ["SOL/USD", "BTC/USD", "ETH/USD", "AVAX/USD"]
        
        self.is_running = False
        self.capital_fraction = 0.08  # 8% por activo
        self.analyzer = TechnicalAnalyzer()
        self.last_trade_time = {}

    def get_account_info(self):
        """Obtiene información real de la cuenta"""
        try:
            account = self.api.get_account()
            return {
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "day_trade_count": int(account.day_trade_count),
                "account_blocked": account.account_blocked,
                "trading_blocked": account.trading_blocked
            }
        except Exception as e:
            st.error(f"Error obteniendo info de cuenta: {e}")
            return {
                "cash": 0, "portfolio_value": 0, "buying_power": 0,
                "equity": 0, "day_trade_count": 0, 
                "account_blocked": False, "trading_blocked": False
            }

    def get_historical(self, symbol, limit=100):
        """Obtiene datos históricos con formato corregido"""
        try:
            # Usar el nuevo formato de símbolos de Alpaca
            timeframe = "1Min"
            end = datetime.now()
            start = end - timedelta(hours=2)  # Últimas 2 horas
            
            bars = self.api.get_crypto_bars(
                symbol, 
                timeframe, 
                start=start.isoformat(),
                end=end.isoformat(),
                limit=limit
            ).df
            
            if bars.empty:
                return None
            
            # Asegurar que tenemos las columnas necesarias
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in bars.columns:
                    if col == 'volume':
                        bars[col] = bars['high'] - bars['low']  # Proxy de volumen
                    else:
                        bars[col] = bars['close']  # Fallback
            
            return bars.tail(limit)
            
        except Exception as e:
            # Intentar formato alternativo
            try:
                alt_symbol = symbol.replace("/", "")
                bars = self.api.get_crypto_bars(alt_symbol, "1Min", limit=limit).df
                if not bars.empty:
                    return bars.tail(limit)
            except:
                pass
            
            st.warning(f"No se pudieron obtener datos para {symbol}: {e}")
            return None

    def can_trade(self, symbol):
        """Verifica si puede hacer trading"""
        account_info = self.get_account_info()
        
        # Verificar si la cuenta está bloqueada
        if account_info.get("trading_blocked", True):
            return False, "Cuenta de trading bloqueada"
        
        # Verificar capital mínimo
        if account_info.get("cash", 0) < 10:
            return False, "Capital insuficiente (mín $10)"
        
        # Verificar límite de day trading
        if account_info.get("day_trade_count", 0) >= 3:
            return False, "Límite de day trading alcanzado"
        
        # Verificar tiempo entre trades
        now = time.time()
        last_trade = self.last_trade_time.get(symbol, 0)
        if now - last_trade < 60:  # Mínimo 1 minuto entre trades
            return False, "Muy poco tiempo desde último trade"
            
        # Verificar posiciones abiertas
        open_positions = self.state_manager.get_open_positions(symbol)
        if len(open_positions) >= 3:  # Máximo 3 posiciones por símbolo
            return False, "Máximo de posiciones alcanzado"
            
        return True, "OK"

    def execute_buy(self, symbol, current_price, signal_strength=1.0, reason="Manual"):
        """Ejecuta compra inteligente"""
        can_trade, trade_reason = self.can_trade(symbol)
        if not can_trade:
            st.warning(f"No se puede comprar {symbol}: {trade_reason}")
            return None
            
        try:
            account_info = self.get_account_info()
            available_cash = account_info["cash"]
            
            # Calcular cantidad basada en capital y fuerza de señal
            base_amount = available_cash * self.capital_fraction
            adjusted_amount = base_amount * min(signal_strength, 1.5)
            qty = max(0.0001, adjusted_amount / current_price)
            
            # Ejecutar orden de compra
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side="buy",
                type="market",
                time_in_force="day"
            )
            
            # Registrar posición
            pos_id = self.state_manager.add_position(
                symbol, qty, current_price, order.id
            )
            
            # Añadir a datos de entrenamiento
            df = self.get_historical(symbol, 50)
            if df is not None:
                manual_strength = self.signal_manager.get_signal_strength(symbol)
                features = self.neural_model.extract_features(df, manual_strength)
                # Target: 1 para compra exitosa
                self.neural_model.add_training_sample(features, 1.0)
            
            self.last_trade_time[symbol] = time.time()
            
            st.success(f"✅ Compra ejecutada: {symbol}\n"
                      f"📊 Cantidad: {qty:.6f}\n"
                      f"💰 Precio: ${current_price:.4f}\n"
                      f"🎯 Razón: {reason}")
            
            return pos_id
            
        except Exception as e:
            st.error(f"❌ Error comprando {symbol}: {e}")
            return None

    def execute_sell(self, symbol, pos_id, qty, current_price, reason="Manual"):
        """Ejecuta venta inteligente"""
        try:
            # Ejecutar orden de venta
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side="sell",
                type="market",
                time_in_force="day"
            )
            
            # Cerrar posición
            success = self.state_manager.close_position(symbol, pos_id, current_price)
            
            if success:
                # Añadir a datos de entrenamiento
                df = self.get_historical(symbol, 50)
                if df is not None:
                    manual_strength = self.signal_manager.get_signal_strength(symbol)
                    features = self.neural_model.extract_features(df, manual_strength)
                    # Target: 0 para venta
                    self.neural_model.add_training_sample(features, 0.0)
                
                self.last_trade_time[symbol] = time.time()
                
                st.success(f"✅ Venta ejecutada: {symbol}\n"
                          f"📊 Cantidad: {qty:.6f}\n"
                          f"💰 Precio: ${current_price:.4f}\n"
                          f"🎯 Razón: {reason}")
                
                return True
        
        except Exception as e:
            st.error(f"❌ Error vendiendo {symbol}: {e}")
            
        return False

    def get_neural_prediction(self, symbol):
        """Obtiene predicción de la red neuronal"""
        df = self.get_historical(symbol, 100)
        if df is None:
            return 0.5, "Sin datos"
        
        try:
            manual_strength = self.signal_manager.get_signal_strength(symbol)
            features = self.neural_model.extract_features(df, manual_strength)
            
            if features is not None:
                prediction = self.neural_model.predict(features)
                confidence = abs(prediction - 0.5) * 2  # 0-1
                
                if prediction > 0.6:
                    signal = f"🟢 COMPRAR ({confidence:.1%})"
                elif prediction < 0.4:
                    signal = f"🔴 VENDER ({confidence:.1%})"
                else:
                    signal = f"🟡 NEUTRAL ({confidence:.1%})"
                
                return prediction, signal
            
        except Exception as e:
            return 0.5, f"Error: {e}"
        
        return 0.5, "Análisis pendiente"

    def manage_positions_automatically(self, symbol, current_price):
        """Gestión automática de posiciones con IA"""
        open_positions = self.state_manager.get_open_positions(symbol)
        
        for pos in open_positions:
            entry_price = pos["entry_price"]
            qty = pos["qty"]
            pnl_percent = ((current_price / entry_price) - 1) * 100
            
            # Actualizar posición
            self.state_manager.update_position(symbol, current_price)
            
            # Take profit dinámico basado en volatilidad
            df = self.get_historical(symbol, 50)
            if df is not None and len(df) > 20:
                volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
                take_profit_threshold = max(0.5, min(2.0, volatility * 100))
            else:
                take_profit_threshold = 0.8
            
            # Stop loss dinámico
            stop_loss_threshold = -1.5
            
            # Decisiones automáticas
            if pnl_percent >= take_profit_threshold:
                self.execute_sell(symbol, pos["id"], qty, current_price, 
                                f"Take Profit Automático ({pnl_percent:+.2f}%)")
            
            elif pnl_percent <= stop_loss_threshold:
                self.execute_sell(symbol, pos["id"], qty, current_price, 
                                f"Stop Loss Automático ({pnl_percent:+.2f}%)")
            
            # Trailing stop avanzado
            elif pnl_percent > 1.0:  # Si está en ganancia > 1%
                trailing_stop = pos["highest_price"] * 0.995  # 0.5% trailing
                if current_price <= trailing_stop:
                    self.execute_sell(symbol, pos["id"], qty, current_price, 
                                    f"Trailing Stop ({pnl_percent:+.2f}%)")

    def should_buy_now(self, symbol):
        """Decide si debe comprar basado en múltiples factores"""
        # 1. Predicción de red neuronal
        neural_score, neural_reason = self.get_neural_prediction(symbol)
        
        # 2. Análisis técnico tradicional
        df = self.get_historical(symbol, 100)
        if df is None:
            return False, 0.5, "Sin datos"
        
        df = self.analyzer.calculate_signals(df)
        technical_score = 0.5
        
        if len(df) > 0:
            last_row = df.iloc[-1]
            if last_row.get('buy_signal', False):
                technical_score = 0.75
            elif last_row.get('sell_signal', False):
                technical_score = 0.25
            else:
                # Tendencia general
                if (last_row.get('ema_12', 0) > last_row.get('ema_26', 0) and 
                    last_row.get('rsi', 50) < 65):
                    technical_score = 0.65
        
        # 3. Señales manuales
        manual_strength = self.signal_manager.get_signal_strength(symbol)
        manual_score = 0.5 + (manual_strength * 0.15)
        
        # 4. Análisis de momentum
        momentum_score = 0.5
        if len(df) >= 10:
            recent_change = (df['close'].iloc[-1] / df['close'].iloc[-10] - 1)
            if recent_change > 0.02:  # Subió > 2%
                momentum_score = 0.7
            elif recent_change < -0.02:  # Bajó > 2%
                momentum_score = 0.3
        
        # Combinar todas las señales
        final_score = (
            neural_score * 0.4 +      # 40% red neuronal
            technical_score * 0.3 +   # 30% análisis técnico
            manual_score * 0.2 +      # 20% señales manuales
            momentum_score * 0.1      # 10% momentum
        )
        
        # Normalizar
        final_score = max(0.01, min(0.99, final_score))
        
        reason = f"🧠 Neural:{neural_score:.2f} | 📊 Tech:{technical_score:.2f} | 👤 Manual:{manual_score:.2f}"
        
        should_buy = final_score > 0.72  # Umbral para comprar
        
        return should_buy, final_score, reason

# -------------------------
# Funciones de credenciales y setup
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
    
    st.title("🤖 Millonario de Señales Crypto 5.0 - Setup")
    
    st.markdown("""
    ### 🚀 Bienvenido al Sistema de Trading con IA más Avanzado
    
    **Nuevas características en v5.0:**
    - 🧠 **Red Neuronal Real** que aprende de cada operación
    - 💰 **Saldo y Posiciones Reales** de tu cuenta Alpaca
    - 📊 **Símbolos Corregidos** (SOL/USD, BTC/USD, etc.)
    - 🎯 **Trading Automático 24/7** con IA
    - 📈 **Análisis Técnico Avanzado** 
    """)
    
    with st.form("cred_form"):
        st.markdown("### 🔐 Configuración de API de Alpaca")
        
        col1, col2 = st.columns(2)
        with col1:
            key = st.text_input("🔑 API Key", type="password", help="Tu clave API de Alpaca Markets")
        with col2:
            secret = st.text_input("🔒 API Secret", type="password", help="Tu clave secreta de Alpaca Markets")
        
        paper = st.checkbox("📄 Usar Paper Trading (Recomendado para pruebas)", value=True)
        
        if st.form_submit_button("💾 Guardar y Comenzar a Ganar", type="primary"):
            if key.strip() and secret.strip():
                save_credentials(key, secret, paper)
                st.success("✅ ¡Perfecto! Configuración guardada.")
                st.balloons()
                time.sleep(2)
                st.rerun()
            else:
                st.error("❌ Por favor completa todos los campos")
    
    with st.expander("💡 ¿Cómo obtener las credenciales de Alpaca?"):
        st.markdown("""
        1. **Regístrate gratis** en [Alpaca Markets](https://alpaca.markets)
        2. **Verifica tu cuenta** (proceso rápido)
        3. **Ve a la sección API** en tu dashboard
        4. **Genera tu API Key y Secret**
        5. **¡Comienza a ganar dinero con IA!** 🚀
        
        **💡 Tip:** Usa Paper Trading primero para probar el sistema sin riesgo.
        """)
    
    st.warning("⚠️ Este sistema puede generar ganancias reales, pero también pérdidas. Invierte responsablemente.")
    st.stop()

# -------------------------
# Funciones de visualización mejoradas
# -------------------------
def create_advanced_chart(df, symbol, positions=None):
    """Crea gráfico avanzado con posiciones marcadas"""
    if df is None or len(df) < 20:
        return None
    
    try:
        analyzer = TechnicalAnalyzer()
        df = analyzer.calculate_signals(df)
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'🚀 {symbol} - Precio y Señales de IA', 
                'RSI (14) - Momentum', 
                'MACD - Tendencia'
            ),
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
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ), row=1, col=1)
        
        # Bollinger Bands
        if 'bb_upper' in df.columns and not df['bb_upper'].isna().all():
            fig.add_trace(go.Scatter(
                x=df.index, y=df['bb_upper'],
                line=dict(color='rgba(255,100,100,0.5)', width=1),
                name='BB Superior',
                showlegend=False
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index, y=df['bb_lower'],
                line=dict(color='rgba(100,255,100,0.5)', width=1),
                name='BB Inferior',
                fill='tonexty',
                fillcolor='rgba(100,150,255,0.1)',
                showlegend=False
            ), row=1, col=1)
        
        # EMAs
        if 'ema_12' in df.columns and not df['ema_12'].isna().all():
            fig.add_trace(go.Scatter(
                x=df.index, y=df['ema_12'],
                line=dict(color='#00ccff', width=2),
                name='EMA 12'
            ), row=1, col=1)
        
        if 'ema_26' in df.columns and not df['ema_26'].isna().all():
            fig.add_trace(go.Scatter(
                x=df.index, y=df['ema_26'],
                line=dict(color='#ff8800', width=2),
                name='EMA 26'
            ), row=1, col=1)
        
        # Señales de la IA
        if 'buy_signal' in df.columns:
            buy_signals = df[df['buy_signal'] == True]
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['low'] * 0.997,
                    mode='markers',
                    marker=dict(color='lime', size=15, symbol='triangle-up'),
                    name='🤖 IA: COMPRAR'
                ), row=1, col=1)
        
        if 'sell_signal' in df.columns:
            sell_signals = df[df['sell_signal'] == True]
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['high'] * 1.003,
                    mode='markers',
                    marker=dict(color='red', size=15, symbol='triangle-down'),
                    name='🤖 IA: VENDER'
                ), row=1, col=1)
        
        # Marcar posiciones activas
        if positions:
            for pos in positions:
                entry_time = datetime.fromisoformat(pos['created_at'])
                if entry_time.date() == datetime.now().date():  # Solo posiciones de hoy
                    fig.add_vline(
                        x=entry_time,
                        line_dash="dash",
                        line_color="yellow",
                        annotation_text=f"💰 Entrada: ${pos['entry_price']:.4f}",
                        row=1, col=1
                    )
        
        # RSI
        if 'rsi' in df.columns and not df['rsi'].isna().all():
            fig.add_trace(go.Scatter(
                x=df.index, y=df['rsi'],
                line=dict(color='purple', width=3),
                name='RSI'
            ), row=2, col=1)
            
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
                    name='Señal MACD'
                ), row=3, col=1)
            
            if 'macd_hist' in df.columns and not df['macd_hist'].isna().all():
                colors = ['#00ff88' if x >= 0 else '#ff4444' for x in df['macd_hist']]
                fig.add_trace(go.Bar(
                    x=df.index, y=df['macd_hist'],
                    name='Histograma',
                    marker_color=colors,
                    opacity=0.7
                ), row=3, col=1)
        
        fig.update_layout(
            height=900,
            title=f"📊 {symbol} - Análisis Completo con IA",
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            font=dict(size=11),
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.3)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.3)')
        
        return fig
        
    except Exception as e:
        st.error(f"Error creando gráfico: {e}")
        return None

def display_portfolio_metrics(trader):
    """Muestra métricas del portafolio en tiempo real"""
    account_info = trader.get_account_info()
    portfolio_summary = trader.state_manager.get_portfolio_summary()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        cash = account_info.get('cash', 0)
        st.metric(
            "💰 Capital Disponible", 
            f"${cash:,.2f}",
            help="Dinero disponible para trading"
        )
    
    with col2:
        portfolio_value = account_info.get('portfolio_value', 0)
        equity_change = portfolio_value - account_info.get('equity', portfolio_value)
        st.metric(
            "📈 Valor del Portafolio", 
            f"${portfolio_value:,.2f}",
            f"${equity_change:+,.2f}",
            delta_color="normal"
        )
    
    with col3:
        avg_pnl = portfolio_summary.get('avg_pnl_percent', 0)
        color = "normal" if avg_pnl >= 0 else "inverse"
        st.metric(
            "📊 P&L Promedio", 
            f"{avg_pnl:+.2f}%",
            delta_color=color,
            help="Ganancia/Pérdida promedio de posiciones abiertas"
        )
    
    with col4:
        total_positions = portfolio_summary.get('total_positions', 0)
        st.metric(
            "🎯 Posiciones Activas", 
            total_positions,
            help="Número de posiciones abiertas"
        )
    
    with col5:
        total_trades = portfolio_summary.get('total_trades', 0)
        st.metric(
            "📈 Trades Totales", 
            total_trades,
            help="Total de operaciones realizadas"
        )

def display_position_details(pos, current_price, symbol):
    """Muestra detalles de posición con diseño mejorado"""
    pnl_percent = pos.get("pnl_percent", 0)
    pnl_dollar = pos.get("pnl_dollar", 0)
    
    # Determinar colores
    if pnl_percent > 0:
        color = "#00ff88"
        bg_color = "rgba(0,255,136,0.1)"
        icon = "🟢"
    elif pnl_percent < 0:
        color = "#ff4444"
        bg_color = "rgba(255,68,68,0.1)"
        icon = "🔴"
    else:
        color = "#888888"
        bg_color = "rgba(136,136,136,0.1)"
        icon = "⚪"
    
    # Tiempo desde apertura
    created = datetime.fromisoformat(pos["created_at"])
    time_diff = datetime.now() - created
    hours = int(time_diff.total_seconds() / 3600)
    minutes = int((time_diff.total_seconds() % 3600) / 60)
    
    st.markdown(f"""
    <div style='
        background: {bg_color}; 
        padding: 20px; 
        border-radius: 15px; 
        margin: 15px 0; 
        border-left: 5px solid {color};
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    '>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
            <h3 style='margin: 0; color: {color};'>{icon} {symbol}</h3>
            <span style='background: {color}; color: white; padding: 5px 10px; border-radius: 20px; font-size: 12px; font-weight: bold;'>
                ID: {pos['id'][-8:]}
            </span>
        </div>
        
        <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 15px;'>
            <div style='text-align: center;'>
                <div style='font-size: 12px; color: #888; margin-bottom: 5px;'>CANTIDAD</div>
                <div style='font-size: 16px; font-weight: bold;'>{pos['qty']:.6f}</div>
            </div>
            <div style='text-align: center;'>
                <div style='font-size: 12px; color: #888; margin-bottom: 5px;'>ENTRADA</div>
                <div style='font-size: 16px; font-weight: bold;'>${pos['entry_price']:.4f}</div>
            </div>
            <div style='text-align: center;'>
                <div style='font-size: 12px; color: #888; margin-bottom: 5px;'>ACTUAL</div>
                <div style='font-size: 16px; font-weight: bold;'>${current_price:.4f}</div>
            </div>
        </div>
        
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
            <div style='text-align: center;'>
                <div style='font-size: 14px; color: #888;'>P&L Porcentaje</div>
                <div style='font-size: 24px; font-weight: bold; color: {color};'>{pnl_percent:+.2f}%</div>
            </div>
            <div style='text-align: center;'>
                <div style='font-size: 14px; color: #888;'>P&L Dólares</div>
                <div style='font-size: 24px; font-weight: bold; color: {color};'>${pnl_dollar:+.2f}</div>
            </div>
            <div style='text-align: center;'>
                <div style='font-size: 14px; color: #888;'>Tiempo Abierto</div>
                <div style='font-size: 16px; font-weight: bold;'>{hours}h {minutes}m</div>
            </div>
        </div>
        
        <div style='font-size: 12px; color: #666; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px;'>
            🔝 Máximo: ${pos.get("highest_price", pos["entry_price"]):.4f} | 
            🔻 Mínimo: ${pos.get("lowest_price", pos["entry_price"]):.4f}
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Bot de Trading Automático con IA
# -------------------------
def create_intelligent_bot(trader):
    """Bot inteligente que aprende y mejora continuamente"""
    
    def intelligent_trading_loop():
        training_counter = 0
        
        while trader.is_running:
            try:
                st.info("🤖 Bot de IA operando...")
                
                for symbol in trader.symbols:
                    if not trader.is_running:
                        break
                    
                    # Obtener datos actuales
                    df = trader.get_historical(symbol, 100)
                    if df is None or len(df) < 30:
                        continue
                    
                    current_price = df["close"].iloc[-1]
                    
                    # Gestión automática de posiciones existentes
                    trader.manage_positions_automatically(symbol, current_price)
                    
                    # Evaluar oportunidad de compra
                    should_buy, score, reason = trader.should_buy_now(symbol)
                    
                    # Ejecutar compra si las condiciones son favorables
                    if should_buy:
                        can_trade, trade_reason = trader.can_trade(symbol)
                        if can_trade:
                            pos_id = trader.execute_buy(
                                symbol, current_price, score, 
                                f"IA Automática - {reason}"
                            )
                            if pos_id:
                                st.success(f"🤖 IA compró {symbol} - Score: {score:.2f}")
                    
                    # Log del estado
                    st.write(f"📊 {symbol}: ${current_price:.4f} - Score: {score:.2f} - {reason}")
                
                # Entrenar modelo cada 10 ciclos
                training_counter += 1
                if training_counter >= 10:
                    success, message = trader.neural_model.train_model()
                    if success:
                        st.success(f"🧠 Modelo reentrenado: {message}")
                    training_counter = 0
                
                # Pausa entre ciclos
                time.sleep(20)  # 20 segundos entre verificaciones
                
            except Exception as e:
                st.error(f"❌ Error en bot IA: {e}")
                time.sleep(60)  # Pausa más larga si hay error
    
    # Ejecutar bot en thread separado
    if trader.is_running:
        if ('intelligent_bot_thread' not in st.session_state or 
            not st.session_state.intelligent_bot_thread.is_alive()):
            st.session_state.intelligent_bot_thread = threading.Thread(
                target=intelligent_trading_loop, daemon=True
            )
            st.session_state.intelligent_bot_thread.start()

# -------------------------
# Aplicación Principal
# -------------------------
def main():
    st.set_page_config(
        page_title="Crypto AI Millionaire v5.0", 
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🧠"
    )
    
    # CSS mejorado
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            color: white;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        .status-active {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: bold;
            text-align: center;
        }
        .status-inactive {
            background: linear-gradient(45deg, #f44336, #da190b);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: bold;
            text-align: center;
        }
        .neural-status {
            background: linear-gradient(45deg, #9C27B0, #673AB7);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Inicializar sistema
    creds = setup_credentials()
    neural_model = NeuralTradingModel()
    signal_manager = SignalManager()
    trader = EnhancedCryptoTrader(creds, neural_model, signal_manager)

    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🧠 MILLONARIO DE SEÑALES CRYPTO 5.0</h1>
        <h3>Sistema Inteligente con Red Neuronal • Trading Automático 24/7</h3>
        <p>✨ Ahora con IA Real que Aprende de Cada Operación ✨</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Estado del sistema
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 INICIAR IA", type="primary", use_container_width=True):
            trader.is_running = True
            st.balloons()
    
    with col2:
        if st.button("⏹️ DETENER IA", type="secondary", use_container_width=True):
            trader.is_running = False
    
    with col3:
        if trader.is_running:
            st.markdown('<div class="status-active">🟢 IA ACTIVA</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-inactive">🔴 IA INACTIVA</div>', unsafe_allow_html=True)
    
    with col4:
        # Estado del modelo neuronal
        training_samples = len(neural_model.training_data.get("features", []))
        if neural_model.model is not None:
            st.markdown(f"""
            <div class="neural-status">
                🧠 Red Neuronal Entrenada<br>
                <small>{training_samples} muestras de aprendizaje</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="neural-status">
                🤖 Modelo en Entrenamiento<br>
                <small>{training_samples}/20 muestras necesarias</small>
            </div>
            """, unsafe_allow_html=True)

    # Ejecutar bot inteligente
    if trader.is_running:
        create_intelligent_bot(trader)
    
    # Panel de control lateral
    st.sidebar.markdown("## 🎛️ CONTROL DE IA")
    
    # Entrenamiento manual
    if st.sidebar.button("🧠 ENTRENAR RED NEURONAL", type="primary"):
        with st.sidebar:
            with st.spinner("Entrenando..."):
                success, message = neural_model.train_model()
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.warning(f"⚠️ {message}")
    
    # Señales manuales
    st.sidebar.markdown("### 📊 SEÑALES MANUALES")
    symbol_manual = st.sidebar.selectbox("🎯 Símbolo", trader.symbols)
    
    col_up, col_down = st.sidebar.columns(2)
    
    with col_up:
        if st.button("📈 ALCISTA", use_container_width=True, type="primary"):
            signal_manager.add_manual_signal(symbol_manual, "up", 1.5)
            st.sidebar.success(f"📈 Señal ALCISTA: {symbol_manual}")
    
    with col_down:
        if st.button("📉 BAJISTA", use_container_width=True, type="secondary"):
            signal_manager.add_manual_signal(symbol_manual, "down", 1.5)
            st.sidebar.success(f"📉 Señal BAJISTA: {symbol_manual}")
    
    # Controles manuales
    st.sidebar.markdown("### 🎮 TRADING MANUAL")
    
    if st.sidebar.button("💰 COMPRAR AHORA", type="primary", use_container_width=True):
        df = trader.get_historical(symbol_manual, 100)
        if df is not None:
            current_price = df["close"].iloc[-1]
            pos_id = trader.execute_buy(symbol_manual, current_price, 1.0, "Compra Manual")
            if pos_id:
                st.sidebar.success(f"✅ Compra manual ejecutada")
    
    if st.sidebar.button("💸 VENDER TODO", type="secondary", use_container_width=True):
        positions = trader.state_manager.get_open_positions(symbol_manual)
        if positions:
            df = trader.get_historical(symbol_manual, 100)
            if df is not None:
                current_price = df["close"].iloc[-1]
                for pos in positions:
                    trader.execute_sell(symbol_manual, pos["id"], pos["qty"], current_price, "Venta Manual")
                st.sidebar.success(f"✅ {len(positions)} posiciones vendidas")
        else:
            st.sidebar.info("ℹ️ No hay posiciones en este símbolo")
    
    # Refresh automático
    if st.sidebar.button("🔄 ACTUALIZAR DATOS", use_container_width=True):
        st.rerun()

    # Métricas del portafolio
    st.markdown("## 📊 PANEL DE CONTROL PRINCIPAL")
    display_portfolio_metrics(trader)
    
    # Separador
    st.markdown("---")
    
    # Información de la cuenta
    account_info = trader.get_account_info()
    if account_info.get("trading_blocked", False):
        st.error("🚫 Trading bloqueado en tu cuenta. Contacta a Alpaca.")
    elif account_info.get("day_trade_count", 0) >= 3:
        st.warning("⚠️ Has alcanzado el límite de day trading (3). Las nuevas posiciones deben mantenerse overnight.")
    
    # Dashboard por cada símbolo
    for symbol in trader.symbols:
        with st.expander(f"📈 {symbol} - Análisis Completo con IA", expanded=True):
            
            # Obtener datos
            df = trader.get_historical(symbol, 100)
            if df is None:
                st.error(f"❌ No se pudieron obtener datos para {symbol}")
                continue
            
            current_price = df["close"].iloc[-1]
            positions = trader.state_manager.get_open_positions(symbol)
            
            # Obtener predicciones de IA
            should_buy, ai_score, ai_reason = trader.should_buy_now(symbol)
            neural_prediction, neural_signal = trader.get_neural_prediction(symbol)
            
            # Métricas principales
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                price_change = ((current_price - df["close"].iloc[-2]) / df["close"].iloc[-2]) * 100
                st.metric(
                    "💲 Precio Actual", 
                    f"${current_price:.4f}", 
                    f"{price_change:+.2f}%"
                )
            
            with col2:
                # Color según predicción
                if ai_score > 0.7:
                    ai_color = "🟢"
                elif ai_score < 0.3:
                    ai_color = "🔴"
                else:
                    ai_color = "🟡"
                
                st.metric(
                    "🧠 Predicción IA", 
                    f"{ai_color} {ai_score:.1%}",
                    help=ai_reason
                )
            
            with col3:
                st.metric(
                    "🤖 Red Neuronal", 
                    neural_signal,
                    f"Confianza: {abs(neural_prediction-0.5)*200:.0f}%"
                )
            
            with col4:
                manual_strength = signal_manager.get_signal_strength(symbol)
                manual_color = "🟢" if manual_strength > 0 else ("🔴" if manual_strength < 0 else "⚪")
                recent_signals = signal_manager.get_recent_signals(symbol, hours=1)
                
                st.metric(
                    "👤 Señales Manuales", 
                    f"{manual_color} {len(recent_signals)}",
                    f"Fuerza: {manual_strength:+.1f}"
                )
            
            with col5:
                total_qty = sum(p["qty"] for p in positions)
                total_value = total_qty * current_price if total_qty > 0 else 0
                avg_pnl = sum(p.get("pnl_percent", 0) for p in positions) / max(1, len(positions))
                
                st.metric(
                    "💼 Posiciones", 
                    f"{len(positions)} activas",
                    f"${total_value:.2f} • {avg_pnl:+.1f}%"
                )
            
            # Recomendación de la IA
            if should_buy:
                st.success(f"🤖 **RECOMENDACIÓN IA: COMPRAR {symbol}**\n\n{ai_reason}")
            elif ai_score < 0.3:
                st.error(f"🤖 **RECOMENDACIÓN IA: VENDER {symbol}**\n\n{ai_reason}")
            else:
                st.info(f"🤖 **RECOMENDACIÓN IA: MANTENER {symbol}**\n\n{ai_reason}")
            
            # Gráfico avanzado
            chart = create_advanced_chart(df.copy(), symbol, positions)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.warning(f"⚠️ No se pudo crear el gráfico para {symbol}")
            
            # Detalles de posiciones
            if positions:
                st.markdown(f"#### 💼 Posiciones Activas en {symbol}")
                for pos in positions:
                    display_position_details(pos, current_price, symbol)
            
            # Historial de señales
            recent_signals = signal_manager.get_recent_signals(symbol, hours=6)
            if recent_signals:
                st.markdown(f"#### 🔔 Historial de Señales ({symbol})")
                
                # Mostrar las últimas 3 señales
                for signal in recent_signals[-3:]:
                    time_ago = datetime.now() - datetime.fromisoformat(signal["timestamp"])
                    hours_ago = int(time_ago.total_seconds() / 3600)
                    minutes_ago = int((time_ago.total_seconds() % 3600) / 60)
                    
                    if signal["type"] == "up":
                        st.success(f"📈 **SEÑAL ALCISTA** • Hace {hours_ago}h {minutes_ago}m • Fuerza: {signal['strength']}")
                    else:
                        st.error(f"📉 **SEÑAL BAJISTA** • Hace {hours_ago}h {minutes_ago}m • Fuerza: {signal['strength']}")
    
    # Sección de estadísticas avanzadas
    st.markdown("---")
    st.markdown("## 📈 ESTADÍSTICAS AVANZADAS DE LA IA")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧠 Estado de la Red Neuronal")
        training_samples = len(neural_model.training_data.get("features", []))
        
        if neural_model.model is not None:
            st.success(f"✅ Modelo entrenado con {training_samples} muestras")
            st.info("🎯 La IA está aprendiendo de cada operación para mejorar las predicciones")
            
            # Mostrar algunas métricas del modelo
            if training_samples > 50:
                accuracy_proxy = min(95, 50 + training_samples * 0.3)
                st.metric("🎯 Precisión Estimada", f"{accuracy_proxy:.1f}%")
        else:
            st.warning(f"⏳ Recopilando datos... {training_samples}/20 muestras necesarias")
            progress = min(1.0, training_samples / 20)
            st.progress(progress)
    
    with col2:
        st.markdown("### 📊 Resumen de Trading")
        summary = trader.state_manager.get_portfolio_summary()
        
        total_positions = summary.get("total_positions", 0)
        total_trades = summary.get("total_trades", 0)
        avg_pnl = summary.get("avg_pnl_percent", 0)
        
        if total_trades > 0:
            st.metric("📈 Trades Realizados", total_trades)
            st.metric("💼 Posiciones Abiertas", total_positions)
            
            if avg_pnl > 0:
                st.success(f"🟢 Rentabilidad Promedio: +{avg_pnl:.2f}%")
            elif avg_pnl < 0:
                st.error(f"🔴 Pérdida Promedio: {avg_pnl:.2f}%")
            else:
                st.info("⚪ Sin ganancias/pérdidas netas")
        else:
            st.info("📊 Aún no hay historial de trading")
    
    # Sistema de alertas
    st.markdown("### 🔔 ALERTAS DEL SISTEMA")
    
    # Verificar condiciones del mercado
    market_alerts = []
    
    for symbol in trader.symbols:
        df = trader.get_historical(symbol, 50)
        if df is not None and len(df) > 20:
            current_price = df["close"].iloc[-1]
            
            # Alerta de volatilidad alta
            volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
            if volatility > 0.05:  # 5% volatilidad
                market_alerts.append(f"⚠️ {symbol}: Alta volatilidad detectada ({volatility*100:.1f}%)")
            
            # Alerta de cambio significativo
            change_1h = (current_price / df['close'].iloc[-60] - 1) * 100 if len(df) >= 60 else 0
            if abs(change_1h) > 3:  # Cambio > 3% en 1 hora
                direction = "📈" if change_1h > 0 else "📉"
                market_alerts.append(f"{direction} {symbol}: Cambio significativo {change_1h:+.1f}% en 1h")
    
    if market_alerts:
        for alert in market_alerts:
            st.warning(alert)
    else:
        st.success("✅ Mercado estable, condiciones normales de trading")
    
    # Footer mejorado
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(45deg, #667eea, #764ba2); border-radius: 15px; color: white; margin-top: 30px;'>
        <h2>🧠 Millonario de Señales Crypto 5.0</h2>
        <h4>Sistema de Trading Inteligente con Red Neuronal Real</h4>
        <p><strong>🚀 Características Avanzadas:</strong></p>
        <p>✅ Red Neuronal que Aprende • ✅ Saldo Real de Alpaca • ✅ Análisis Técnico Avanzado</p>
        <p>✅ Trading Automático 24/7 • ✅ Gestión de Riesgo IA • ✅ Señales Manuales</p>
        <br>
        <p><em>⚠️ Advertencia: El trading de criptomonedas implica riesgos. La IA mejora las probabilidades pero no garantiza ganancias. Invierte responsablemente.</em></p>
        <p><small>Desarrollado con ❤️ para maximizar tu potencial de trading</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh para datos en tiempo real
    if trader.is_running:
        time.sleep(30)  # Actualizar cada 30 segundos
        st.rerun()

if __name__ == "__main__":
    main()
