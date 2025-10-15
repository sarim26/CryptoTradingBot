"""
Exchange Connector Module
Handles all communication with the cryptocurrency exchange API.
"""

import ccxt
from typing import Optional, Dict
import config


class ExchangeConnector:
    """
    Connects to a cryptocurrency exchange and provides methods to fetch market data.
    """
    
    def __init__(self, exchange_name: str = config.EXCHANGE_NAME):
        """
        Initialize connection to the exchange.
        
        Args:
            exchange_name: Name of the exchange (e.g., 'binance', 'coinbase')
        """
        self.exchange_name = exchange_name
        self.exchange = None
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """
        Initialize the exchange object using ccxt library.
        In paper trading mode, no API keys are needed for public data.
        """
        try:
            # Create exchange instance
            exchange_class = getattr(ccxt, self.exchange_name)
            
            if config.TRADING_MODE == 'live' and config.API_KEY and config.API_SECRET:
                # Live trading mode with API keys
                self.exchange = exchange_class({
                    'apiKey': config.API_KEY,
                    'secret': config.API_SECRET,
                    'enableRateLimit': True,  # Respect exchange rate limits
                })
                print(f"Connected to {self.exchange_name} in LIVE mode")
            else:
                # Paper trading mode - no API keys needed for public data
                self.exchange = exchange_class({
                    'enableRateLimit': True,
                })
                print(f"Connected to {self.exchange_name} in PAPER TRADING mode")
            
            # Load markets
            self.exchange.load_markets()
            
        except Exception as e:
            print(f"Error initializing exchange: {e}")
            raise
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Fetch the current price for a trading pair.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
        
        Returns:
            Current price as float, or None if error occurs
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']  # Last traded price
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            return None
    
    def get_ticker_info(self, symbol: str) -> Optional[Dict]:
        """
        Fetch detailed ticker information including bid, ask, high, low, volume.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
        
        Returns:
            Dictionary with ticker information, or None if error occurs
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'high': ticker['high'],
                'low': ticker['low'],
                'volume': ticker['baseVolume'],
            }
        except Exception as e:
            print(f"Error fetching ticker info for {symbol}: {e}")
            return None
    
    def get_available_symbols(self, quote_currency: str = 'USDT') -> list:
        """
        Get list of available trading pairs for a specific quote currency.
        
        Args:
            quote_currency: Quote currency to filter by (e.g., 'USDT', 'USD')
        
        Returns:
            List of trading pair symbols
        """
        try:
            markets = self.exchange.load_markets()
            symbols = [
                symbol for symbol in markets.keys()
                if quote_currency in symbol and markets[symbol]['active']
            ]
            return sorted(symbols)
        except Exception as e:
            print(f"Error fetching available symbols: {e}")
            return []
    
    def place_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None):
        """
        Place an order on the exchange (only works in LIVE mode).
        In paper trading mode, this method won't actually place real orders.
        
        Args:
            symbol: Trading pair symbol
            order_type: 'market' or 'limit'
            side: 'buy' or 'sell'
            amount: Amount to trade
            price: Price for limit orders (optional for market orders)
        
        Returns:
            Order information or None
        """
        if config.TRADING_MODE != 'live':
            print("Order not placed - Paper trading mode is active")
            return None
        
        try:
            if order_type == 'market':
                order = self.exchange.create_market_order(symbol, side, amount)
            elif order_type == 'limit':
                order = self.exchange.create_limit_order(symbol, side, amount, price)
            else:
                print(f"Invalid order type: {order_type}")
                return None
            
            return order
        except Exception as e:
            print(f"Error placing order: {e}")
            return None

    def get_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 200):
        """
        Fetch OHLCV candles for a symbol and timeframe.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Exchange timeframe string (e.g., '1m', '5m', '1h')
            limit: Number of candles to fetch

        Returns:
            List of candles: [timestamp, open, high, low, close, volume]
        """
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol} {timeframe}: {e}")
            return []

