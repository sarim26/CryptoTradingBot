"""
Configuration file for the crypto trading bot.
All trading parameters and settings are defined here.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Exchange Configuration
EXCHANGE_NAME = 'binance'
API_KEY = os.getenv('BINANCE_API_KEY', '')
API_SECRET = os.getenv('BINANCE_API_SECRET', '')

# Trading Mode: 'paper' for simulation, 'live' for real trading
TRADING_MODE = os.getenv('TRADING_MODE', 'paper')

# Paper Trading Configuration
INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', 10000))  # Starting balance in USDT

# Trading Strategy Parameters
BUY_DROP_PERCENTAGE = 1  # Buy when price drops by this percentage
SELL_INCREASE_PERCENTAGE = 2.5  # Sell when price increases by this percentage

# Volatility-Based Risk Management
ENABLE_VOLATILITY_CEILING = True  # Enable upper price limit based on volatility
SOL_VOLATILITY_CEILING = 220.0  # Maximum price to buy SOL (based on recent volatility)
VOLATILITY_TRACKING_PERIOD = 24  # Hours to track for volatility calculation

# Trading Amount Settings
TRADE_PERCENTAGE = 10  # Percentage of available balance to use per trade (10% = 0.1 of balance)

# Data Fetching
PRICE_CHECK_INTERVAL = 4  # Seconds between price checks

# Default Trading Pair
DEFAULT_SYMBOL = 'BTC/USDT'  # You can change this to any supported pair

# Platform Fees Configuration
BINANCE_BUY_FEE_PERCENT = 0.1  # Binance trading fee for buy orders (0.1%)
BINANCE_SELL_FEE_PERCENT = 0.1  # Binance trading fee for sell orders (0.1%)
ENABLE_PLATFORM_FEES = True  # Set to False to disable platform fee calculations

# Tax Configuration
TAX_COUNTRY = 'UK'  # UK, US, INDIA, or NONE (for no tax calculations)
ENABLE_TAX_CALCULATIONS = True  # Set to False to disable tax calculations

# Technical Indicators
ENABLE_RSI = True  # Enable RSI filter/signals
RSI_PERIOD = int(os.getenv('RSI_PERIOD', 10))  # RSI lookback period
RSI_TIMEFRAME = os.getenv('RSI_TIMEFRAME', '1m')  # OHLCV timeframe for RSI
RSI_OVERSOLD = float(os.getenv('RSI_OVERSOLD', 30))  # Buy zone threshold
RSI_OVERBOUGHT = float(os.getenv('RSI_OVERBOUGHT', 70))  # Sell zone threshold

# Robust Mean (Outlier-resistant) Settings
ENABLE_ROBUST_MEAN = True
ROBUST_MEAN_LOOKBACK_HOURS = int(os.getenv('ROBUST_MEAN_LOOKBACK_HOURS', 6))
ROBUST_MEAN_TIMEFRAME = os.getenv('ROBUST_MEAN_TIMEFRAME', '1m')
ROBUST_MEAN_USE_PRICE = os.getenv('ROBUST_MEAN_USE_PRICE', 'close')  # 'close' | 'hl2' | 'hlc3' | 'ohlc4'
ROBUST_MEAN_REFRESH_SECONDS = int(os.getenv('ROBUST_MEAN_REFRESH_SECONDS', 10))  # 10s or 600s
ROBUST_MEAN_METHOD = os.getenv('ROBUST_MEAN_METHOD', 'iqr')  # 'iqr' | 'zscore' | 'mad'
ROBUST_MEAN_ZSCORE = float(os.getenv('ROBUST_MEAN_ZSCORE', 3.0))
ROBUST_MEAN_IQR_K = float(os.getenv('ROBUST_MEAN_IQR_K', 1.5))
ROBUST_MEAN_MAD_K = float(os.getenv('ROBUST_MEAN_MAD_K', 3.5))

# ML-Based Buy Decision Settings
ENABLE_ML_BUY_DECISION = True  # Enable ML-based buy percentage calculation
ML_LOOKBACK_HOURS = int(os.getenv('ML_LOOKBACK_HOURS', 720))  # Hours of historical data for ML analysis
ML_TIMEFRAME = os.getenv('ML_TIMEFRAME', '5m')  # Timeframe for ML data collection
ML_MIN_BUY_PERCENTAGE = float(os.getenv('ML_MIN_BUY_PERCENTAGE', 0.5))  # Minimum buy percentage (0.5%)
ML_MAX_BUY_PERCENTAGE = float(os.getenv('ML_MAX_BUY_PERCENTAGE', 3.0))  # Maximum buy percentage (3.0%)
ML_CONFIDENCE_THRESHOLD = float(os.getenv('ML_CONFIDENCE_THRESHOLD', 0.4))  # Minimum confidence for ML decision
ML_CACHE_DURATION_SECONDS = int(os.getenv('ML_CACHE_DURATION_SECONDS', 30))  # How long to cache ML predictions
ML_FALLBACK_TO_CONFIG = True  # Fall back to config BUY_DROP_PERCENTAGE if ML fails

