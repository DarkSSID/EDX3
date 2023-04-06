import logging
import threading
import asyncio
from ccxt import NetworkError, InsufficientFunds, ExchangeError
import time
import sys
import re
from bs4 import BeautifulSoup
import aiohttp
import csv
from datetime import datetime
import numpy as np
import pandas as pd
import talib as ta
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Nadam
from sklearn.preprocessing import MinMaxScaler
import requests
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json
import websockets
from typing import Dict, List, Tuple
import random
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ctypes
import llama_cpp
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.ensemble import GradientBoostingRegressor
from fbprophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Charger la bibliothèque partagée compilée
llama_dll = ctypes.CDLL('llama_interface.dll')  # votre bibliothèque partagée LlaMa.cpp provenant de ('https://github.com/ggerganov/llama.cpp/find/master') 

# Configuration
config = {
    "OPPORTUNITY_THRESHOLD": 0.05,
    "TRADE_AMOUNT": 1000,
    "API_URL": 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin%2Cethereum%2Cdogecoin&vs_currencies=usd',
    "API_CALL_INTERVAL": 60,
    "STOP_LOSS_PERCENTAGE": 0.05,
    "CSV_FILENAME": "trading_data.csv",
    "WEBSOCKET_URI": 'wss://ws-feed.pro.coinbase.com',
    "PRODUCT_IDS": ['BTC-USD', 'ETH-USD', 'DOGE-USD'],
    "TRANSACTION_DATA_FILENAME": "transaction_data.csv",
}

logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

class TradingDataManager:
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.current_prices = {}
        self.arbitrage_opportunities = {}

    def train_llama_model(self, vader_sentiments, trading_signals):
        vader_sentiments_c = (ctypes.c_double * len(vader_sentiments))(*vader_sentiments)
        trading_signals_c = (ctypes.c_double * len(trading_signals))(*trading_signals)
        llama_dll.llama_train(vader_sentiments_c, trading_signals_c, len(vader_sentiments))

    def predict_trading_signal(self, vader_sentiment):
        vader_sentiment_c = (ctypes.c_double * 1)(vader_sentiment)
        return llama_dll.llama_predict(vader_sentiment_c)


        # Initialiser le modèle LlaMa
    num_inputs = 2  # Nombre d'entrées (scores de sentiment VADER) - Positif et Négatif (Haussier ou Baissier)
    num_outputs = 2  # Nombre de sorties (signaux de trading) - Positif et Négatif (Haussier ou Baissier)
    num_layers = 3  # Nombre de couches dans le modèle LlaMa
    num_neurons = 69  # Nombre de neurones par couche

    llama_dll.llama_initialize(num_inputs, num_outputs, num_layers, num_neurons)

        # Entraîner le modèle LlaMa avec les données historiques
        # Remplacez ces données par les scores de sentiment VADER et les signaux de trading correspondants de votre projet
    vader_sentiments = np.array([0.1, 0.2, -0.1, -0.3])
    trading_signals = np.array([1, 1, -1, -1])

    num_samples = len(vader_sentiments)

        # Entraîner le modèle LlaMa
    llama_dll.llama_train.restype = None
    llama_dll.llama_train.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), ctypes.c_int]
    llama_dll.llama_train(vader_sentiments, trading_signals, num_samples)

        # Initialiser l'analyseur de sentiment VADER
    analyzer = SentimentIntensityAnalyzer()

        # Analyser les sentiments en temps réel (remplacez 'text' par le texte des nouvelles financières ou des tweets pertinents)
    text = "Le marché est en hausse grâce aux bonnes nouvelles sur l'économie."
    sentiment_score = analyzer.polarity_scores(text)['compound']

        # Prédire les signaux de trading à partir des scores de sentiment VADER
    llama_dll.llama_predict.restype = ctypes.c_double
    llama_dll.llama_predict.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')]

    test_sentiment = np.array([sentiment_score])
    predicted_signal = llama_dll.llama_predict(test_sentiment)

    print("Sentiment VADER:", test_sentiment)

    def strategy_with_llama(self, sentiment_score, existing_signal):
    # Transformer le score de sentiment en un tableau numpy
        test_sentiment = np.array([sentiment_score])
    
    # Utiliser le modèle LlaMa pour prédire le signal de trading en fonction du score de sentiment VADER
        predicted_signal = self.predict_trading_signal(test_sentiment[0])
    
    # Ajuster le signal existant avec le signal prédit par le modèle LlaMa
        adjusted_signal = existing_signal * predicted_signal
    
    # Retourner le signal ajusté
        return adjusted_signal

    async def get_sentiment_data(self, url: str) -> List[str]:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    soup = BeautifulSoup(await response.text(), "html.parser")
                    data = []

                    for element in soup.find_all("div", class_="container"):
                        text = element.get_text()
                        cleaned_text = re.sub("\s+", " ", text)
                        data.append(cleaned_text)

                    return data
                else:
                    print(f"Error getting sentiment data: {response.status}")
                    sys.exit(1)

    def analyze_sentiments_real_time(self, text):
        # Code pour intégrer Llama.cpp et utiliser son analyse des sentiments
        result = llama_cpp.analyze_sentiment(text)  # Remplacez par le code d'intégration de Llama.cpp
        adjusted_signal = self.strategy_with_llama(sentiment_score, result)
        return adjusted_signal

    async def connect_websocket(self, websocket_url: str):
        self.websocket = await websockets.connect(websocket_url)
        await self.handle_messages()

    async def handle_messages(self):
        while True:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)

                if 'exchange_fees' in data:
                    self.exchange_fees = data['exchange_fees']

                if 'exchange_transaction_times' in data:
                    self.exchange_transaction_times = data['exchange_transaction_times']
            except websockets.ConnectionClosed:
                break

    def get_exchange_fees(self):
         return self.exchange_fees

    def get_exchange_transaction_times(self):
        return self.exchange_transaction_times

    async def get_arbitrage_opportunities(self):
        uri = self.config["WEBSOCKET_URI"]

        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps({
             "type": "subscribe",
                "channels": [{"name": "ticker", "product_ids": self.config["PRODUCT_IDS"]}]
            }))

            while True:
                response = await websocket.recv()
                data = json.loads(response)

                if data['type'] == 'ticker':
                    product_id = data['product_id']
                    price = float(data['price'])

                    # Update the current price
                    self.current_prices[product_id] = price

                    # Calculate arbitrage opportunities based on current prices and the OPPORTUNITY_THRESHOLD
                    opportunities = {}
                    for coin1, price1 in self.current_prices.items():
                        for coin2, price2 in self.current_prices.items():
                            if coin1 != coin2:
                                arbitrage_margin = abs(price1 - price2) / min(price1, price2)
                            if arbitrage_margin > self.config["OPPORTUNITY_THRESHOLD"]:
                               opportunities[f"{coin1}-{coin2}"] = arbitrage_margin

                    self.arbitrage_opportunities = opportunities

    async def get_historical_prices(self) -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, requests.get, self.config["API_URL"])
        historical_prices = pd.DataFrame(response.json())
        return historical_prices

    def compute_momentum(self, historical_prices: pd.DataFrame, period: int = 14) -> pd.Series:
        momentum = ta.MOM(historical_prices.close, timeperiod=period)
        return momentum

    def compute_rsi(self, historical_prices: pd.DataFrame, period: int = 14) -> pd.Series:
        rsi = ta.RSI(historical_prices.close, timeperiod=period)
        return rsi

    def compute_macd(self, historical_prices: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        macd, macd_signal, macd_hist = ta.MACD(historical_prices.close, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
        return macd, macd_signal, macd_hist

    def compute_fibonacci_levels(self, historical_prices: pd.DataFrame) -> pd.Series:
        high_prices = historical_prices.high
        low_prices = historical_prices.low
        fibonacci_levels = []

        for index in range(len(high_prices)):
            if index == 0:
                fibonacci_levels.append(np.nan)
                continue

            high = high_prices[index - 1]
            low = low_prices[index - 1]

            level_0 = low
            level_1 = low + 0.236 * (high - low)
            level_2 = low + 0.382 * (high - low)
            level_3 = low + 0.5 * (high - low)
            level_4 = low + 0.618 * (high - low)
            level_5 = low + 0.786 * (high - low)
            level_6 = high

        fib_levels = [level_0, level_1, level_2, level_3, level_4, level_5, level_6]
        fibonacci_levels.append(fib_levels)

        return pd.Series(fibonacci_levels, index=historical_prices.index)

    def create_dataset(self, dataset: pd.DataFrame, look_back=60):
        X, y = [], []
        for i in range(look_back, len(dataset)): X.append(dataset[i - look_back:i, 0])
        y.append(dataset[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def predict_prices(self, market_data):
    # Prétraiter les données de marché
        preprocessed_data = self.preprocess_data(market_data)

    # Obtenir les prédictions des modèles de base
        arima_predictions = self.arima_model.predict_prices(preprocessed_data)
        gbm_predictions = self.gbm_model.predict_prices(preprocessed_data)
        prophet_predictions = self.prophet_model.predict_prices(preprocessed_data)

    # Préparer les données pour le modèle de stacking
        stacked_predictions = np.column_stack((arima_predictions, gbm_predictions, prophet_predictions))

    # Obtenir les prédictions du modèle de stacking
        final_predictions = self.stacking_model.predict_stacking_model(stacked_predictions)

        return final_predictions

    def get_atr(self, historical_prices: pd.DataFrame, period: int = 14) -> float:
        high_prices = historical_prices['high'].values
        low_prices = historical_prices['low'].values
        close_prices = historical_prices['close'].values

        atr = ta.ATR(high_prices, low_prices, close_prices, timeperiod=period)
        return atr[-1]

    def estimate_transaction_time(self, transaction_data: pd.DataFrame, transaction_type: str) -> float:
    # Estimate transaction time based on historical transaction data
        filtered_data = transaction_data[transaction_data['type'] == transaction_type]
        average_time = filtered_data['time'].mean()
        return average_time

    def get_transaction_data(self) -> pd.DataFrame:
        transaction_data = pd.read_csv(self.config["TRANSACTION_DATA_FILENAME"])
        return transaction_data

    def update_transaction_data(self, transaction_data: pd.DataFrame):
        transaction_data.to_csv(self.config["TRANSACTION_DATA_FILENAME"], index=False)

    def compute_covariance_matrix(self, historical_prices: pd.DataFrame) -> np.ndarray:
        # Calculate the log returns of historical prices
        log_returns = np.log(historical_prices / historical_prices.shift(1))

        # Calculate the covariance matrix of log returns
        covariance_matrix = log_returns.cov()

        return covariance_matrix.values

    def select_crypto_pairs(self, covariance_matrix: np.ndarray, num_pairs: int = 3) -> List[Tuple[str, str]]:
        # Find the pairs with the lowest covariance
        pairs = []
        num_coins = len(self.config["PRODUCT_IDS"])
        for i in range(num_coins):
            for j in range(i + 1, num_coins):
                pairs.append(((self.config["PRODUCT_IDS"][i], self.config["PRODUCT_IDS"][j]), covariance_matrix[i, j]))

        pairs.sort(key=lambda x: x[1])
        selected_pairs = [pair[0] for pair in pairs[:num_pairs]]

        return selected_pairs

class CombinedTradingBot:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_manager = TradingDataManager(config)
        self.initial_capital = 0
        self.current_capital = 0
        self.active_trading_bot = ActiveTradingBot(self.config, self.data_manager)
        self.passive_trading_bot = PassiveTradingBot(self.config, self.data_manager)
        self.arbitrage_trading_bot = ArbitrageTradingBot(self.config, self.data_manager)
        self.user_interface = UserInterface(self.config, self)
        self.trading_models = []
        self.stacking_model = None
        self.real_time_sentiment_score = 0
        self.train_models()
        self.auto_mode = config["AUTO_MODE"]

    def analyze_sentiments(self, sentiment_data):
        sentiment_scores = []
        vader_analyzer = SentimentIntensityAnalyzer()

        for text in sentiment_data:
           result = vader_analyzer.polarity_scores(text)
           sentiment_scores.append(result["compound"])

        return np.mean(sentiment_scores)


    def compute_indicators(self, historical_prices):
        fibonacci_levels = self.data_manager.compute_fibonacci_levels(historical_prices)
        momentum = self.data_manager.compute_momentum(historical_prices)
        rsi = self.data_manager.compute_rsi(historical_prices)
        macd = self.data_manager.compute_macd(historical_prices)
        return fibonacci_levels, momentum, rsi, macd

    def preprocess_data(self):
        X_train, y_train, X_val, y_val = self.data_manager.preprocess_data()

        # Add sentiment analysis data (offline)
        sentiment_scores_offline = sentiment_scores
        for i in range(len(X_train)):
        sentiment_scores_offline.append(self.analyze_sentiment(X_train[i]))
        X_train = np.concatenate((X_train, np.array(sentiment_scores_offline).reshape(-1, 1)), axis=1)

        sentiment_scores_offline = sentiment_scores
        for i in range(len(X_val)):
        sentiment_scores_offline.append(self.analyze_sentiment(X_val[i]))
        X_val = np.concatenate((X_val, np.array(sentiment_scores_offline).reshape(-1, 1)), axis=1)

        # Add real-time sentiment analysis data
        real_time_sentiment_scores = [self.real_time_sentiment_score] * len(X_train)
        X_train = np.concatenate((X_train, np.array(real_time_sentiment_scores).reshape(-1, 1)), axis=1)

        real_time_sentiment_scores = [self.real_time_sentiment_score] * len(X_val)
        X_val = np.concatenate((X_val, np.array(real_time_sentiment_scores).reshape(-1, 1)), axis=1)

        # Add technical indicators
        historical_prices = self.data_manager.get_historical_prices()
        fibonacci_levels, momentum, rsi, macd = self.compute_indicators(historical_prices)

        # Ensure that the shapes are correct before concatenating
        X_train = np.concatenate((X_train, fibonacci_levels[:-1], momentum[:-1].reshape(-1, 1), rsi[:-1].reshape(-1, 1), macd[:-1].reshape(-1, 1)), axis=1)
        X_val = np.concatenate((X_val, fibonacci_levels[-1], momentum[-1].reshape(-1, 1), rsi[-1].reshape(-1, 1), macd[-1].reshape(-1, 1)), axis=1)

        return X_train, y_train, X_val, y_val

    def train_models(self):
        preprocessed_data = self.preprocess_data()

        # Entraîner les modèles de base
        self.arima_model.train_arima_model(preprocessed_data)
        self.gbm_model.train_gbm_model(preprocessed_data)
        self.prophet_model.train_prophet_model(preprocessed_data)

        # Obtenir les prédictions des modèles de base pour l'ensemble d'entraînement
        arima_predictions = self.arima_model.predict_prices(preprocessed_data)
        gbm_predictions = self.gbm_model.predict_prices(preprocessed_data)
        prophet_predictions = self.prophet_model.predict_prices(preprocessed_data)

        # Préparer les données pour le modèle de stacking
        stacked_predictions = np.column_stack((arima_predictions, gbm_predictions, prophet_predictions))
        target = self.get_target_variable(preprocessed_data)

        # Entraîner le modèle de stacking
        self.stacking_model.train_stacking_model(stacked_predictions, target)

    def update_investment_plan(self):
        self.active_trading_bot.update_investment_plan()
        self.passive_trading_bot.update_investment_plan()
        self.arbitrage_trading_bot.update_investment_plan()

        # Ajuster le capital investi en fonction de la performance du portefeuille
        total_portfolio_value = self.active_trading_bot.portfolio_value + self.passive_trading_bot.portfolio_value + self.arbitrage_trading_bot.portfolio_value
        capital_ratio = total_portfolio_value / self.initial_capital

        if capital_ratio >= 1.05:
            increase_ratio = capital_ratio - 1
            self.active_trading_bot.increase_investment(increase_ratio)
            self.passive_trading_bot.increase_investment(increase_ratio)
            self.arbitrage_trading_bot.increase_investment(increase_ratio)
            self.initial_capital = total_portfolio_value

    def train_stacking_model(self):
        # Get the preprocessed data
        X_train, y_train, X_val, y_val = self.data_manager.preprocess_data()

        # Train the base models
        for i in range(self.config["NUM_BASE_MODELS"]):
            self.train_model()

        # Predict on the validation set using the base models
        y_val_predictions = []
        for model in self.trading_models:
            y_val_predictions.append(model.predict(X_val))

        # Reshape the predictions to match the shape of the validation set
        y_val_predictions = np.concatenate(y_val_predictions, axis=1)

        # Train the stacking model using the base models' predictions as features
        self.stacking_model = Sequential()
        self.stacking_model.add(Dense(64, activation='relu', input_shape=(y_val_predictions.shape[1],)))
        self.stacking_model.add(Dropout(0.2))
        self.stacking_model.add(Dense(1))

        self.stacking_model.compile(optimizer='adam', loss='mse')
        self.stacking_model.fit(y_val_predictions, y_val, epochs=self.config["NUM_EPOCHS"], 
                                batch_size=self.config["BATCH_SIZE"], verbose=1)

 
    def select_optimal_pairs(self, num_pairs=10, liquidity_weight=0.4, volatility_weight=0.3, spread_weight=0.2, covariance_weight=0.1, strategy='active'):
        all_pairs = self.data_manager.get_all_pairs()
        pair_scores = {}
        covariance_matrix = self.data_manager.get_covariance_matrix()

        for pair in all_pairs:
            ticker = self.data_manager.get_ticker(pair)
            liquidity = ticker['24h_volume']

            historical_prices = self.data_manager.get_historical_prices(pair)
            returns = np.diff(np.log(historical_prices))
            volatility = np.std(returns)

            spread = ticker['ask_price'] - ticker['bid_price']

            i, j = self.data_manager.get_pair_indices(pair)
            covariance = covariance_matrix[i, j]

            # Utiliser les prédictions du modèle de stacking pour estimer les prix futurs
            stacked_prediction = self.data_manager.stacking_model.predict_stacked_prices(pair)

            if strategy == 'active':
                score = liquidity_weight * liquidity - volatility_weight * volatility - spread_weight * spread - covariance_weight * covariance + stacked_prediction
            else:  # stratégie passive
                score = liquidity_weight * liquidity + volatility_weight * volatility - spread_weight * spread - covariance_weight * covariance + stacked_prediction

            pair_scores[pair] = score

        sorted_pairs = sorted(pair_scores, key=pair_scores.get, reverse=True)
        optimal_pairs = sorted_pairs[:num_pairs]

        return optimal_pairs

    def analyze_market(self):
        # Récupérer la matrice de covariance
        covariance_matrix = self.data_manager.get_covariance_matrix()

        # Sélectionner les paires optimales en fonction de la covariance et d'autres critères
        selected_pairs = self.select_optimal_pairs(covariance_matrix)

        # Analyser les sentiments du marché
        market_sentiment = self.analyze_sentiments() 

        for pair in selected_pairs:
           historical_prices = self.get_historical_prices(pair)

        # Calculer les indicateurs techniques
           fibonacci_levels, momentum, rsi, macd = self.compute_indicators(historical_prices)

        # Utiliser les indicateurs techniques et les sentiments du marché pour prendre une décision d'achat ou de vente
           buy_signal = (
           rsi < 30 and
           historical_prices[-1] < fibonacci_levels[0.618] and
           macd[-1] > 0 and
           self.sentiment_score > 0 and  # Utilisez le sentiment hors ligne (VADER)
           self.real_time_sentiment_score > 0  # Utilisez le sentiment en temps réel (LlaMa.cpp)
           )

           sell_signal = (
           rsi > 70 and
           historical_prices[-1] > fibonacci_levels[0.382] and
           macd[-1] < 0 and
           self.sentiment_score < 0 and  # Utilisez le sentiment hors ligne (VADER)
           self.real_time_sentiment_score < 0  # Utilisez le sentiment en temps réel (LlaMa.cpp)
           )

        # Prendre en compte les seuils dynamiques et la gestion des risques pour les transactions
        if buy_signal and self.check_risk_management("buy"):
            self.update_dynamic_thresholds("buy")
            self.execute_trade(pair, "buy")
        elif sell_signal and self.check_risk_management("sell"):
            self.update_dynamic_thresholds("sell")
            self.execute_trade(pair, "sell")

    def create_investment_plan(self):
        # Élaborer un plan d'investissement
        # Inclure des objectifs clairs, des délais et des niveaux de risque acceptables
        # Intégrer la stratégie d'arbitrage dans le plan d'investissement en utilisant les bénéfices générés pour investir dans le trading actif et passif
        self.active_trading_bot.update_investment_plan()
        self.passive_trading_bot.update_investment_plan()
        self.arbitrage_trading_bot.update_investment_plan()

    def execute_active_investment(self):
        # Mettre en œuvre la stratégie d'investissement actif
        self.active_trading_bot.execute_trades()

    def execute_passive_investment(self):
        # Mettre en œuvre la stratégie d'investissement passif
        self.passive_trading_bot.execute_trades()

    def execute_arbitrage_investment(self):
        # Mettre en œuvre la stratégie d'arbitrage
        self.arbitrage_trading_bot.execute_trades()

    def reevaluate_portfolio(self):
        # Surveiller les performances des investissements
        # Réévaluer le portefeuille et ajuster l'approche si nécessaire
        self.active_trading_bot.update_portfolio()
        self.passive_trading_bot.update_portfolio()

    def check_risk_management(self, trade):
        # Vérifiez si le trade satisfait aux exigences de gestion des risques
        # Implémentez la logique pour déterminer si le trade respecte vos critères de gestion des risques
        return trade['risk'] <= self.config['MAX_RISK']

    def find_arbitrage_opportunities(self, exchange_prices):
        # Trouver les opportunités d'arbitrage en fonction des prix des différentes bourses
        # Implémentez la logique pour identifier les opportunités d'arbitrage en fonction des prix des différentes bourses
        opportunities = []

        for pair, prices in exchange_prices.items():
            spread = max(prices) - min(prices)
            if spread >= self.base_decision_threshold['arbitrage']:
                opportunities.append((pair, spread))

        return opportunities

    def is_trade_profitable(self, pair, predicted_price, synthesized_sentiments):
        current_price = self.data_manager.get_current_price(pair)
        expected_profit = predicted_price - current_price

        base_decision_threshold = self.config["TRADE_DECISION_THRESHOLD"]
        adjusted_decision_threshold = base_decision_threshold * (1 + synthesized_sentiments)

        return expected_profit >= adjusted_decision_threshold

    async def start_websocket(self):
        tdm = self.data_manager
        await tdm.connect_websocket("wss://your-websocket-url")

    def init_user_interface(self):
        self.root = tk.Tk()
        self.root.title("Crypto Trading Bot")
        self.root.geometry("800x600")

    # Champ pour entrer le capital initial
        self.capital_label = tk.Label(self.root, text="Le capital initial:")
        self.capital_label.grid(column=0, row=4, padx=20, pady=10, sticky=tk.W)

        self.capital_entry = tk.Entry(self.root)
        self.capital_entry.grid(column=1, row=4, padx=20, pady=10, sticky=tk.W)

        self.capital_button = tk.Button(self.root, text="Définir le capitale", command=self.set_initial_capital)
        self.capital_button.grid(column=2, row=4, padx=20, pady=10, sticky=tk.W)

    # Bouton pour lancer le bot
        self.launch_button = tk.Button(self.root, text="Lancer le bot", command=self.launch_bot)
        self.launch_button.grid(column=0, row=5, padx=20, pady=20, sticky=tk.W)

    # Graphique pour afficher les opportunités d'arbitrage
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.plot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().grid(column=1, row=0, rowspan=4, padx=20, pady=20)

    # Zone de texte pour afficher les logs
        self.log_text = tk.Text(self.root, wrap=tk.WORD, height=10)
        self.log_text.grid(column=0, row=3, padx=20, pady=20)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def launch_bot(self):
    # Exécutez la méthode `run` dans un nouveau thread
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
    # Interface graphique
        self.init_user_interface()

    # Démarrez le websocket
        asyncio.run(self.start_websocket())

        while True:
        # Mettez à jour l'interface utilisateur avec les données actuelles
            self.update()

            is_active_investment_favorable = self.analyze_market()
            self.real_time_sentiment_score = self.get_real_time_sentiment()
            self.create_investment_plan()

            if is_active_investment_favorable:
                self.execute_active_investment()
            else:
                self.execute_passive_investment()

            self.execute_arbitrage_investment()
            self.execute_trade()

            self.reevaluate_portfolio()
            time.sleep(self.config["UPDATE_INTERVAL"])

    def optimize_portfolio(self, covariance_matrix: pd.DataFrame, returns: pd.Series) -> Tuple[pd.Series, float]:
        num_assets = len(returns)
        weights = np.random.dirichlet(np.ones(num_assets), size=1)
        portfolio_return = np.dot(weights, returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility
        return weights, sharpe_ratio

    def update(self):
        # Get the latest arbitrage opportunities
        arbitrage_opportunities = self.data_manager.arbitrage_opportunities

        # Update the interface
        self.plot.clear()
        self.plot.bar(arbitrage_opportunities.keys(), arbitrage_opportunities.values())
        self.canvas.draw()

        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        for key, value in arbitrage_opportunities.items():
            self.log_text.insert(tk.END, f"{key}: {value}\n")

        # Calculate the covariance matrix
        historical_prices = self.data_manager.get_historical_prices()
        covariance_matrix = self.data_manager.compute_covariance_matrix(historical_prices)
        returns = self.data_manager.compute_returns(historical_prices)

        # Optimize the portfolio
        optimal_weights, optimal_sharpe_ratio = self.optimize_portfolio(covariance_matrix, returns)

        # Use the optimal weights to adjust your trading strategy
        selected_pairs = self.data_manager.select_crypto_pairs(covariance_matrix, optimal_weights)

    async def execute_fast_limit_order(self, exchange, pair, order_type, price=None, amount=None):
        try:
            if order_type == "buy":
                # Placez un ordre d'achat à cours limité
                order = await exchange.create_limit_buy_order(pair, amount, price)
            elif order_type == "sell":
                # Placez un ordre de vente à cours limité
                order = await exchange.create_limit_sell_order(pair, amount, price)
            else:
                raise ValueError(f"Invalid order type: {order_type}")

            return order

        except NetworkError as e:
            print(f"NetworkError: {e}")
            # Vous pouvez décider de réessayer la demande ici ou de gérer l'erreur différemment
        except InsufficientFunds as e:
            print(f"InsufficientFunds: {e}")
            # Vous pouvez décider de gérer l'erreur différemment, par exemple en ajustant la taille de la commande
        except ExchangeError as e:
            print(f"ExchangeError: {e}")
            # Vous pouvez décider de gérer l'erreur différemment en fonction de l'erreur spécifique de l'échange

        return None

    # Execute trades
    def execute_trade(self):

    # Obtenez les données de marché, les prédictions de prix et les sentiments
        market_data = self.data_manager.get_market_data()
        price_predictions = self.predict_prices(market_data)
        market_sentiments = self.analyze_sentiments(market_data)

    # Utilisez la méthode analyze_sentiments_real_time pour analyser les sentiments en temps réel
        # Vous devrez peut-être adapter cette partie en fonction de la structure de vos données de marché (par exemple, en choisissant un texte pertinent)
        market_text = "Votre texte ici"  # À remplacer par le texte pertinent extrait de market_data
        synthesized_sentiments = self.analyze_sentiments_real_time(market_text)

        # Mettez à jour les seuils de décision en fonction des sentiments du marché
        self.strategy_with_llama(synthesized_sentiments)

        # Identifiez les opportunités d'arbitrage et intégrez-les dans votre stratégie de trading
        exchange_prices = self.data_manager.get_exchange_prices()
        arbitrage_opportunities = self.find_arbitrage_opportunities(exchange_prices)

        if arbitrage_opportunities:
        # Exécutez les transactions d'arbitrage si des opportunités sont identifiées
            for opportunity in arbitrage_opportunities:
               pair, spread = opportunity

            # Obtenez les frais de transaction et les temps de transaction pour la paire sur les différentes bourses
               exchange_fees = self.data_manager.get_exchange_fees(pair)
               exchange_transaction_times = self.data_manager.get_exchange_transaction_times(pair)

            # Calculez la marge bénéficiaire nette en tenant compte des frais de transaction et des variations de prix possibles
               net_profit_margin = spread - sum(exchange_fees.values()) - self.config["PRICE_VARIATION_MARGIN"]

            # Vérifiez si la marge bénéficiaire nette est acceptable et si le temps de transaction est conforme aux critères de gestion des risques
               base_arbitrage_decision_threshold = self.config["ARBITRAGE_DECISION_THRESHOLD"]
               adjusted_arbitrage_decision_threshold = base_arbitrage_decision_threshold * (1 + sentiments_real_time)

            if net_profit_margin >= adjusted_arbitrage_decision_threshold and self.check_risk_management(pair, exchange_transaction_times):
            # Déterminez les échanges et les prix d'achat et de vente pour les transactions d'arbitrage
                exchange_buy, buy_price = min(exchange_prices[pair].items(), key=lambda x: x[1])
                exchange_sell, sell_price = max(exchange_prices[pair].items(), key=lambda x: x[1])


            # Déterminez la quantité à acheter / vendre en fonction de la marge bénéficiaire nette et des prix d'achat et de vente
                amount = net_profit_margin / (sell_price - buy_price)

            # Exécutez la transaction d'arbitrage de manière optimale
                asyncio.run(self.execute_fast_limit_order(exchange_buy, pair, "buy", buy_price, amount))
                asyncio.run(self.execute_fast_limit_order(exchange_sell, pair, "sell", sell_price, amount))

    # Parcourez les prédictions de prix et évaluez les opportunités de trading
            for pair, predicted_price in price_predictions.items():

        # Évaluez la rentabilité des transactions avant de les exécuter
                if self.is_trade_profitable(pair, predicted_price, synthesized_sentiments):
            # Vérifiez la gestion des risques avant d'exécuter une transaction
                   if self.check_risk_management(pair, predicted_price):
                        asyncio.run(self.execute_fast_limit_order(exchange, pair, "buy", buy_price, amount))

if __name__ == "__main__":
config = load_config("config.json")
bot = CombinedTradingBot(config)
bot.run()
