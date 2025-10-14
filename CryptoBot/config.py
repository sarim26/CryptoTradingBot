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
BUY_DROP_PERCENTAGE = 1.5  # Buy when price drops by this percentage
SELL_INCREASE_PERCENTAGE = 2.5  # Sell when price increases by this percentage

# Trading Amount Settings
TRADE_PERCENTAGE = 10  # Percentage of available balance to use per trade (10% = 0.1 of balance)

# Data Fetching
PRICE_CHECK_INTERVAL = 5  # Seconds between price checks

# Default Trading Pair
DEFAULT_SYMBOL = 'BTC/USDT'  # You can change this to any supported pair

# Platform Fees Configuration
BINANCE_BUY_FEE_PERCENT = 0.1  # Binance trading fee for buy orders (0.1%)
BINANCE_SELL_FEE_PERCENT = 0.1  # Binance trading fee for sell orders (0.1%)
ENABLE_PLATFORM_FEES = True  # Set to False to disable platform fee calculations

# Tax Configuration
TAX_COUNTRY = 'UK'  # UK, US, INDIA, or NONE (for no tax calculations)
ENABLE_TAX_CALCULATIONS = True  # Set to False to disable tax calculations

