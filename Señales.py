"""
QuickTrend Trader Pro — Versión GIP (Nunca vende en pérdida)
- Guarda todo en ./GIP_data (modelos, estado, logs).
- Usa SOLO alpaca-trade-api.
- Multi-posiciones, aprendizaje automático online (incremental).
- Nunca vender en pérdida. Vende solo si hay ganancia y la IA indica caída o TP alcanzado.
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import streamlit as st

import alpaca_trade_api as tradeapi

# ML
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# -----------------------
# CONFIG DIRECTORIOS
# -----------------------
BASE_SAVE_DIR = os.environ.get("GIP_DIR", "./GIP_data")
CRED_FILE = os.path.join(BASE_SAVE_DIR, "alpaca_credentials.json")
STATE_FILE = os.path.join(BASE_SAVE_DIR, "trader_state.json")
MODEL_FILE = os.path.join(BASE_SAVE_DIR, "ml_model.pkl")
SCALER_FILE = os.path.join(BASE_SAVE_DIR, "ml_scaler.pkl")
LOG_FILE = os.path.join(BASE_SAVE_DIR, "trader.log")

os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# Logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(layout="wide", page_title="QuickTrend Trader Pro - GIP", page_icon="🤖")

# -----------------------
# UTILIDADES: INDICADORES
# -----------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    # Espera df con columna 'close'
    f = pd.DataFrame(index=df.index)
    f['close'] = df['close']
    f['ema8'] = ema(df['close'], 8)
    f['ema21'] = ema(df['close'], 21)
    f['rsi14'] = rsi(df['close'], 14)
    f['mom_5'] = df['close'].pct_change(5)
    f['vol_5'] = df['volume'].rolling(5).mean().fillna(0)
    f['ema_diff'] = f['ema8'] - f['ema21']
    f = f.dropna()
    return f

# -----------------------
# ESTADO Y MANEJO POSICIONES
# -----------------------
class TradeStateManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error cargando state: {e}")
        # Estructura inicial
        return {
            "positions": {},       # symbol -> list of positions
            "last_training": None,
            "model_trained_until": None,
            "cash_buckets": [100,200,300,500],
            "history": []
        }

    def save_state(self):
        with self.lock:
            with open(STATE_FILE, "w") as f:
                json.dump(self.state, f, indent=2, default=str)

    def add_position(self, symbol: str, qty: int, entry_price: float) -> str:
        with self.lock:
            if symbol not in self.state["positions"]:
                self.state["positions"][symbol] = []
            position_id = f"{symbol}_{int(time.time())}"
            pos = {
                "id": position_id,
                "qty": qty,
                "entry_price": float(entry_price),
                "highest_price": float(entry_price),
                "created_at": datetime.utcnow().isoformat(),
                "status": "open"
            }
            self.state["positions"][symbol].append(pos)
            self.save_state()
            return position_id

    def update_position(self, symbol: str, current_price: float):
        with self.lock:
            for pos in self.state["positions"].get(symbol, []):
                if pos["status"] == "open":
                    pos["highest_price"] = max(pos["highest_price"], float(current_price))
            self.save_state()

    def close_position(self, symbol: str, position_id: str):
        with self.lock:
            for pos in self.state["positions"].get(symbol, []):
                if pos["id"] == position_id and pos["status"] == "open":
                    pos["status"] = "closed"
                    pos["closed_at"] = datetime.utcnow().isoformat()
            self.save_state()

    def get_open_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        with self.lock:
            if symbol:
                return [p for p in self.state["positions"].get(symbol, []) if p["status"] == "open"]
            else:
                res = []
                for s, lst in self.state["positions"].items():
                    res.extend([p for p in lst if p["status"] == "open"])
                return res

    def get_total_exposure(self, symbol: str) -> float:
        with self.lock:
            positions = self.get_open_positions(symbol)
            return sum(p["qty"] * p["entry_price"] for p in positions)

    def log_trade_history(self, entry: Dict[str, Any]):
        with self.lock:
            self.state["history"].append(entry)
            # limit history size
            if len(self.state["history"]) > 5000:
                self.state["history"] = self.state["history"][-5000:]
            self.save_state()

# -----------------------
# ML: MODELO INCREMENTAL (SGDClassifier)
# -----------------------
class OnlineModel:
    def __init__(self, model_file=MODEL_FILE):
        self.model_file = model_file
        # Pipeline: scaler + sgd
        self.pipeline = None
        if os.path.exists(self.model_file):
            try:
                self.pipeline = joblib.load(self.model_file)
                logging.info("Modelo cargado desde disco.")
            except Exception as e:
                logging.error(f"Error cargando modelo: {e}")
                self.pipeline = None
        if self.pipeline is None:
            # initialize a new incremental model (binary classification: 1 = subir, 0 = bajar/estancado)
            sgd = SGDClassifier(loss='log', max_iter=1000, tol=1e-3)
            scaler = StandardScaler()
            # We'll wrap in Pipeline but need to call partial_fit on the classifier inside pipeline;
            # for simplicity, store scaler and classifier separately
            self.scaler = scaler
            self.clf = sgd
            # We will track if clf has been partially fitted
            self.is_fitted = False
            self._save_pipeline()

    def _save_pipeline(self):
        try:
            # Save scaler + clf together
            tmp = {"scaler": self.scaler, "clf": self.clf, "is_fitted": getattr(self, "is_fitted", False)}
            joblib.dump(tmp, self.model_file)
            logging.info("Modelo guardado en disco.")
        except Exception as e:
            logging.error(f"Error guardando modelo: {e}")

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        # X: 2D array, y: labels (0/1)
        try:
            # fit scaler incrementally by partial standardization: we will fit scaler on batch (approximate)
            if not getattr(self, "scaler_fitted", False):
                self.scaler.fit(X)
                self.scaler_fitted = True
            else:
                # update mean/var roughly by refit with concat - acceptable for our online setting
                # This is a simple approach: re-fit scaler on combined stored data might be better but bigger.
                self.scaler.fit(X)
            Xs = self.scaler.transform(X)
            if not self.is_fitted:
                # partial_fit requires classes param on first call
                self.clf.partial_fit(Xs, y, classes=np.array([0,1]))
                self.is_fitted = True
            else:
                self.clf.partial_fit(Xs, y)
            self._save_pipeline()
        except Exception as e:
            logging.error(f"Error en partial_fit: {e}")

    def predict_proba(self, X: np.ndarray) -> float:
        try:
            if not self.is_fitted:
                return 0.5
            Xs = self.scaler.transform(X.reshape(1, -1))
            proba = self.clf.predict_proba(Xs)[0][1]
            return float(proba)
        except Exception as e:
            logging.error(f"Error predict_proba: {e}")
            return 0.5

# -----------------------
# TRADER PRINCIPAL
# -----------------------
class AlpacaTrader:
    def __init__(self, api_key: str, api_secret: str, base_url: str, symbols: List[str], state_manager: TradeStateManager, online_model: OnlineModel):
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        self.symbols = symbols
        self.state_manager = state_manager
        self.online_model = online_model
        self.is_streaming = False
        # allocation: divide cash buckets across symbols evenly
        self.cash_buckets = state_manager.state.get("cash_buckets", [100,200,300,500])
        self.alloc_per_symbol = self._compute_alloc_per_symbol()
        self.stream_thread = None

    def _compute_alloc_per_symbol(self) -> Dict[str, float]:
        total = sum(self.cash_buckets)
        alloc = {}
        share = total / max(1, len(self.symbols))
        for s in self.symbols:
            alloc[s] = share
        return alloc

    def get_historical_data(self, symbol: str, limit: int=500, timeframe: str='1Min') -> pd.DataFrame:
        try:
            bars = self.api.get_bars(symbol, timeframe, limit=limit).df
            # bars may have multiindex columns in some versions
            if isinstance(bars.columns, pd.MultiIndex):
                # try to select symbol
                if symbol in bars.columns.levels[0]:
                    bars = bars[symbol]
            # Ensure columns: open,high,low,close,volume
            bars = bars.reset_index().set_index('timestamp')
            bars.index = pd.to_datetime(bars.index)
            bars = bars.rename(columns={c: c.lower() for c in bars.columns})
            return bars
        except Exception as e:
            logging.error(f"Error get_historical_data {symbol}: {e}")
            return None

    def decide_and_buy(self, symbol: str, current_price: float):
        # Never buy if buying_power low; decide qty from allocation bucket
        try:
            # allocation per symbol
            alloc = self.alloc_per_symbol.get(symbol, sum(self.cash_buckets)/len(self.symbols))
            qty = max(1, int(alloc / max(0.0001, current_price)))
            # Avoid over-exposure: if current total exposure > 30% of account buying power for symbol -> skip
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            current_exposure = self.state_manager.get_total_exposure(symbol)
            if current_exposure > buying_power * 0.3:
                logging.info(f"Exposición por símbolo alta en {symbol}. Saltando compra.")
                return

            # Submit market buy
            self.api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='day')
            position_id = self.state_manager.add_position(symbol, qty, current_price)
            self.state_manager.log_trade_history({
                "time": datetime.utcnow().isoformat(),
                "action": "buy",
                "symbol": symbol,
                "qty": qty,
                "price": current_price,
                "id": position_id
            })
            st.toast(f"🛒 COMPRADO {qty} {symbol} a ${current_price:.2f}", icon="🛒")
            logging.info(f"Comprado {qty} {symbol} a {current_price} (id {position_id})")
        except Exception as e:
            logging.error(f"Error execute_buy {symbol}: {e}")

    def execute_sell(self, symbol: str, position: Dict[str, Any], current_price: float):
        # IMPORTANT: This method will NEVER sell at a loss. We enforce current_price > entry_price.
        try:
            if current_price <= position["entry_price"]:
                logging.info(f"Refusing to sell {symbol} at loss (cur {current_price} <= entry {position['entry_price']})")
                return False
            # submit market sell
            self.api.submit_order(symbol=symbol, qty=position["qty"], side='sell', type='market', time_in_force='day')
            self.state_manager.close_position(symbol, position["id"])
            self.state_manager.log_trade_history({
                "time": datetime.utcnow().isoformat(),
                "action": "sell",
                "symbol": symbol,
                "qty": position["qty"],
                "price": current_price,
                "id": position["id"]
            })
            st.toast(f"💰 VENDIDO {position['qty']} {symbol} a ${current_price:.2f}", icon="💸")
            logging.info(f"Vendido {position['qty']} {symbol} a {current_price} (id {position['id']})")
            return True
        except Exception as e:
            logging.error(f"Error execute_sell {symbol}: {e}")
            return False

    def check_and_manage_positions(self, symbol: str, current_price: float):
        # Verifica todas las posiciones abiertas y decide si vender (siempre > entry)
        open_positions = self.state_manager.get_open_positions(symbol)
        for pos in open_positions:
            # actualizar highest
            self.state_manager.update_position(symbol, current_price)
            entry = pos["entry_price"]
            highest = pos.get("highest_price", entry)
            # Si el Pct de ganancia supera X, o si la IA predice caída fuerte, vender (si no hay pérdida).
            profit_pct = (current_price / entry - 1) * 100
            # condiciones de venta:
            take_profit_pct = 0.5  # conserva tu preferencia: 0.5% TP como en original (puedes cambiar)
            # 1) si supera TP y luego retrocede desde máximo (trailing)
            if current_price >= entry * (1 + take_profit_pct / 100):
                # Si la IA predice bajada o cayó desde máximo -> vender
                feat_df = self.get_recent_features(symbol, lookback=50)
                if feat_df is not None and not feat_df.empty:
                    x = feat_df.iloc[-1].values
                    proba_up = self.online_model.predict_proba(x)
                    # Si probabilidad de subir baja (ej. < 0.55) o estamos retrocediendo desde máximo -> vender
                    if proba_up < 0.55 or current_price <= highest * 0.995:
                        self.execute_sell(symbol, pos, current_price)
                else:
                    # sin features, vender por TP (si está en ganancia)
                    self.execute_sell(symbol, pos, current_price)
            else:
                # Si la IA indica fuerte caída inminente pero precio sigue > entry -> vende (respeta regla)
                feat_df = self.get_recent_features(symbol, lookback=50)
                if feat_df is not None and not feat_df.empty:
                    x = feat_df.iloc[-1].values
                    proba_up = self.online_model.predict_proba(x)
                    if proba_up < 0.3 and current_price > entry:
                        # vendemos por señal de caída (pero nunca a pérdida)
                        self.execute_sell(symbol, pos, current_price)

    def get_recent_features(self, symbol: str, lookback: int = 100) -> pd.DataFrame:
        df = self.get_historical_data(symbol, limit=lookback)
        if df is None or len(df) < 30:
            return None
        feats = make_features(df)
        # use selected columns
        selected = feats[['ema_diff','rsi14','mom_5','vol_5']].dropna()
        return selected

    def train_model_offline(self):
        # Entrenamiento supervisado a partir de datos históricos:
        # - Para cada barra t, construir features y etiqueta = 1 si precio t+N sube > X% en horizonte, 0 otherwise.
        logging.info("Iniciando entrenamiento offline (historical)")
        X_batches = []
        y_batches = []
        horizon = 10  # barras futuras a considerar (ej: 10 minutos)
        threshold = 0.002  # 0.2% futuro para considerar subida (puedes ajustar)
        for symbol in self.symbols:
            df = self.get_historical_data(symbol, limit=800)
            if df is None or len(df) < 50:
                continue
            feats = make_features(df)
            closes = feats['close']
            # crear etiquetas
            future = closes.shift(-horizon)
            ret = (future - closes) / closes
            labels = (ret > threshold).astype(int)
            # alinear X e y
            X = feats[['ema_diff','rsi14','mom_5','vol_5']].values
            y = labels.values
            # remover nan al final
            valid = ~np.isnan(y)
            X = X[valid]
            y = y[valid]
            if len(X) > 50:
                X_batches.append(X)
                y_batches.append(y)
        # Concatenate and partial_fit en batchs
        if X_batches:
            X_all = np.vstack(X_batches)
            y_all = np.concatenate(y_batches)
            # Hacer entrenamiento incremental en mini-batches para no agotar memoria
            batch_size = 1024
            for i in range(0, len(y_all), batch_size):
                Xb = X_all[i:i+batch_size]
                yb = y_all[i:i+batch_size]
                self.online_model.partial_fit(Xb, yb)
            # marcar entrenamiento
            self.state_manager.state["last_training"] = datetime.utcnow().isoformat()
            self.state_manager.save_state()
            logging.info("Entrenamiento offline completado.")
            return True
        logging.info("No se pudo entrenar (datos insuficientes).")
        return False

    # -----------------------
    # Streaming / loop principal (trabaja 24x7 en hilo)
    # -----------------------
    def start_background_loop(self):
        if self.is_streaming:
            return
        self.is_streaming = True

        def loop():
            logging.info("Iniciando background loop 24x7")
            while self.is_streaming:
                try:
                    for symbol in self.symbols:
                        # obtener último precio de mercado
                        try:
                            quote = self.api.get_latest_trade(symbol)
                            current_price = float(quote.price)
                        except Exception:
                            # fallback: barras 1 minuto
                            df = self.get_historical_data(symbol, limit=1)
                            if df is None or df.empty:
                                continue
                            current_price = float(df['close'].iloc[-1])

                        # actualizar posiciones internas
                        self.state_manager.update_position(symbol, current_price)

                        # Chequear posiciones abiertas y decidir venta si corresponde
                        self.check_and_manage_positions(symbol, current_price)

                        # Obtener features y evaluar compra
                        feats = self.get_recent_features(symbol, lookback=80)
                        if feats is not None and not feats.empty:
                            x = feats.iloc[-1].values
                            proba_up = self.online_model.predict_proba(x)
                            # Condiciones de compra:
                            # - no sobreexposición
                            # - probabilidad de subir alta (>0.7)
                            # - si símbolo está en retroceso (ej: ema_diff < 0) espera comprar más barato: -> si proba_up > 0.75 comprar
                            if proba_up > 0.75:
                                # Heurística: si precio actual > entry of open positions, y la IA predice subida fuerte -> NO vender; si no hay posiciones -> comprar
                                open_pos = self.state_manager.get_open_positions(symbol)
                                if not open_pos:
                                    self.decide_and_buy(symbol, current_price)
                                else:
                                    # si hay posiciones en ganancia y proba_up alta -> aguantar (no vender)
                                    pass
                            else:
                                # si proba_up baja pero precio está > entry y en ganancia -> evaluar venta en check_and_manage_positions
                                pass

                        # esperar un poco por símbolo para no golpear API
                        time.sleep(0.5)

                    # pequeño descanso entre ciclos completos
                    time.sleep(1)
                except Exception as e:
                    logging.error(f"Error en background loop: {e}")
                    time.sleep(5)

        self.stream_thread = threading.Thread(target=loop, daemon=True)
        self.stream_thread.start()

    def stop_background_loop(self):
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2)

# -----------------------
# UTIL: HORARIO MERCADO (EST)
# -----------------------
def is_market_open() -> bool:
    now_utc = datetime.utcnow()
    # EST = UTC-5 (no ajuste DST preciso aquí; para producción usar pytz/zoneinfo)
    est = now_utc - timedelta(hours=5)
    if est.weekday() >= 5:
        return False
    market_open = est.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = est.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= est <= market_close

# -----------------------
# UI + MAIN
# -----------------------
def save_credentials(key: str, secret: str, base_url: str = "https://paper-api.alpaca.markets"):
    data = {"ALPACA_API_KEY": key.strip(), "ALPACA_API_SECRET": secret.strip(), "ALPACA_BASE_URL": base_url.strip()}
    with open(CRED_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return True

def load_credentials():
    if os.path.exists(CRED_FILE):
        with open(CRED_FILE, "r") as f:
            return json.load(f)
    return None

def setup_credentials_ui():
    creds = load_credentials()
    if creds and creds.get("ALPACA_API_KEY") and creds.get("ALPACA_API_SECRET"):
        return creds
    st.title("🤖 QuickTrend Trader Pro - Setup (GIP)")
    with st.form("creds"):
        key = st.text_input("API Key", type="password")
        secret = st.text_input("API Secret", type="password")
        base_url = st.text_input("Base URL (paper):", value="https://paper-api.alpaca.markets")
        if st.form_submit_button("Guardar credenciales"):
            save_credentials(key, secret, base_url)
            st.success("Credenciales guardadas. Recarga la página.")
            st.stop()
    st.stop()

def main():
    creds = setup_credentials_ui()
    API_KEY = creds["ALPACA_API_KEY"]
    API_SECRET = creds["ALPACA_API_SECRET"]
    BASE_URL = creds.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    # Símbolos por defecto (puedes editarlos en UI)
    default_symbols = ["SPY", "AAPL", "MSFT", "QQQ", "TLT", "IEF"]
    st.title("🤖 QuickTrend Trader Pro — GIP (Nunca vende en pérdida)")
    col_top = st.columns([2,2,2,2])
    market_status = "🟢 ABIERTO" if is_market_open() else "🔴 CERRADO"
    col_top[0].metric("Mercado", market_status)
    last_train = "Nunca"
    state_manager = TradeStateManager()
    if state_manager.state.get("last_training"):
        last_train = state_manager.state["last_training"]
    col_top[1].metric("Último entrenamiento", last_train)
    col_top[2].metric("Modelo guardado en", MODEL_FILE)
    col_top[3].metric("Estado guardado en", STATE_FILE)

    # Sidebar: parámetros
    with st.sidebar:
        st.header("Configuración")
        symbols = st.text_area("Símbolos (coma separados)", value=",".join(default_symbols))
        symbols_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        cash_buckets_text = st.text_input("Buckets de efectivo (coma):", value="100,200,300,500")
        try:
            cash_buckets = [float(x.strip()) for x in cash_buckets_text.split(",") if x.strip()]
            state_manager.state["cash_buckets"] = cash_buckets
        except:
            cash_buckets = state_manager.state.get("cash_buckets", [100,200,300,500])
        if st.button("Guardar estado actual"):
            state_manager.save_state()
            st.success("Estado guardado en disco.")

    # Inicializar modelo y trader
    online_model = OnlineModel()
    trader = AlpacaTrader(API_KEY, API_SECRET, BASE_URL, symbols_list, state_manager, online_model)

    # Pintar área de control
    st.subheader("Controles")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Entrenar modelo (offline)"):
        with st.spinner("Entrenando con datos históricos..."):
            ok = trader.train_model_offline()
            if ok:
                st.success("Entrenamiento offline completado.")
            else:
                st.warning("No se completó (datos insuficientes).")

    if c2.button("Iniciar vigilancia 24x7 (background)"):
        trader.start_background_loop()
        st.success("Vigilancia activa en background (24x7).")

    if c3.button("Detener vigilancia"):
        trader.stop_background_loop()
        st.warning("Vigilancia detenida.")

    if c4.button("Forzar ciclo: chequear posiciones ahora"):
        # Ejecuta un ciclo manual rápido
        for sym in symbols_list:
            df = trader.get_historical_data(sym, limit=10)
            if df is None or df.empty:
                continue
            current_price = float(df['close'].iloc[-1])
            trader.check_and_manage_positions(sym, current_price)
        st.success("Ciclo manual ejecutado.")

    # Mostrar posiciones abiertas
    st.subheader("Posiciones abiertas")
    total_positions = sum(len(state_manager.get_open_positions(s)) for s in symbols_list)
    st.write(f"Total posiciones abiertas: {total_positions}")

    for sym in symbols_list:
        positions = state_manager.get_open_positions(sym)
        if positions:
            st.markdown(f"### {sym}")
            for pos in positions:
                profit_pct = (pos.get("highest_price", pos["entry_price"]) / pos["entry_price"] - 1) * 100
                st.markdown(f"ID: {pos['id']} | Qty: {pos['qty']} | Entrada: ${pos['entry_price']:.2f} | Máximo: ${pos['highest_price']:.2f} | Ganancia: +{profit_pct:.2f}%")

    # Historial (últimos 30)
    st.subheader("Historial de operaciones (últimas 30)")
    hist = state_manager.state.get("history", [])[-30:]
    st.table(pd.DataFrame(hist[::-1]))

    st.caption(f"Todos los archivos se guardan en: {BASE_SAVE_DIR}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Error en la aplicación: {e}")
        logging.exception(e)
