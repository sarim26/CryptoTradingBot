"""
Trading Strategy Module
Implements the buy/sell logic based on price movements.
"""

from typing import Optional, Dict, List
import config


class TradingStrategy:
    """
    Implements a simple percentage-based trading strategy:
    - Buy when price drops by X% from last buy/reference price
    - Sell when price increases by Y% from buy price
    """
    
    def __init__(self, 
                 buy_drop_percent: float = config.BUY_DROP_PERCENTAGE,
                 sell_increase_percent: float = config.SELL_INCREASE_PERCENTAGE,
                 enable_rsi: bool = getattr(config, 'ENABLE_RSI', True),
                 rsi_period: int = getattr(config, 'RSI_PERIOD', 14),
                 rsi_timeframe: str = getattr(config, 'RSI_TIMEFRAME', '1m'),
                 rsi_oversold: float = getattr(config, 'RSI_OVERSOLD', 30.0),
                 rsi_overbought: float = getattr(config, 'RSI_OVERBOUGHT', 70.0)):
        """
        Initialize the trading strategy.
        
        Args:
            buy_drop_percent: Percentage drop to trigger buy signal
            sell_increase_percent: Percentage increase to trigger sell signal
        """
        self.buy_drop_percent = buy_drop_percent
        self.sell_increase_percent = sell_increase_percent
        self.enable_rsi = enable_rsi
        self.rsi_period = rsi_period
        self.rsi_timeframe = rsi_timeframe
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        # Track reference prices for each symbol
        self.reference_prices: Dict[str, float] = {}
        
        print(f"\n{'='*60}")
        print(f"Trading Strategy Initialized")
        print(f"{'='*60}")
        print(f"Buy Signal:  Price drops by {self.buy_drop_percent}% or more")
        print(f"Sell Signal: Price increases by {self.sell_increase_percent}% or more")
        if self.enable_rsi:
            print(f"RSI Enabled: period {self.rsi_period}, timeframe {self.rsi_timeframe}, oversold ≤ {self.rsi_oversold}, overbought ≥ {self.rsi_overbought}")
        print(f"{'='*60}\n")
    
    def set_reference_price(self, symbol: str, price: float):
        """
        Set or update the reference price for a symbol.
        This is used to track price changes and generate signals.
        
        Args:
            symbol: Trading pair symbol
            price: Reference price to set
        """
        self.reference_prices[symbol] = price

    # --- RSI utilities ---
    def _compute_rsi(self, closes: List[float], period: int) -> Optional[float]:
        """
        Compute RSI from a list of close prices.
        Returns the latest RSI value or None if insufficient data.
        """
        if len(closes) < period + 1:
            return None

        gains: List[float] = []
        losses: List[float] = []

        for i in range(1, len(closes)):
            delta = closes[i] - closes[i - 1]
            gains.append(max(delta, 0.0))
            losses.append(max(-delta, 0.0))

        # Wilder's smoothing
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        if avg_loss == 0 and avg_gain == 0:
            return 50.0
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def get_rsi(self, exchange, symbol: str) -> Optional[float]:
        """
        Fetch OHLCV and compute latest RSI for the configured timeframe/period.
        """
        if not self.enable_rsi:
            return None
        try:
            candles = exchange.get_ohlcv(symbol, timeframe=self.rsi_timeframe, limit=max(200, self.rsi_period * 3))
            if not candles:
                return None
            closes = [c[4] for c in candles]
            return self._compute_rsi(closes, self.rsi_period)
        except Exception:
            return None

    # --- Robust mean utilities ---
    def _select_price_series(self, candles: List[List[float]], mode: str) -> List[float]:
        prices: List[float] = []
        for c in candles:
            o, h, l, cl = c[1], c[2], c[3], c[4]
            if mode == 'hl2':
                prices.append((h + l) / 2.0)
            elif mode == 'hlc3':
                prices.append((h + l + cl) / 3.0)
            elif mode == 'ohlc4':
                prices.append((o + h + l + cl) / 4.0)
            else:
                prices.append(cl)
        return prices

    def _iqr_filter(self, values: List[float], k: float) -> List[float]:
        if len(values) < 8:
            return values
        sorted_vals = sorted(values)
        q1_idx = int(0.25 * (len(sorted_vals) - 1))
        q3_idx = int(0.75 * (len(sorted_vals) - 1))
        q1 = sorted_vals[q1_idx]
        q3 = sorted_vals[q3_idx]
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        return [v for v in values if lower <= v <= upper]

    def _zscore_filter(self, values: List[float], z: float) -> List[float]:
        if len(values) < 3:
            return values
        mean_v = sum(values) / len(values)
        var = sum((v - mean_v) ** 2 for v in values) / (len(values) - 1)
        std = var ** 0.5 if var > 0 else 0.0
        if std == 0:
            return values
        return [v for v in values if abs((v - mean_v) / std) <= z]

    def _mad_filter(self, values: List[float], k: float) -> List[float]:
        if not values:
            return values
        median = sorted(values)[len(values)//2]
        deviations = [abs(v - median) for v in values]
        mad = sorted(deviations)[len(deviations)//2] or 1e-9
        return [v for v in values if abs(v - median) / mad <= k]

    def get_robust_mean(self, exchange, symbol: str) -> Optional[float]:
        if not getattr(config, 'ENABLE_ROBUST_MEAN', True):
            return None
        try:
            timeframe = getattr(config, 'ROBUST_MEAN_TIMEFRAME', '1m')
            hours = getattr(config, 'ROBUST_MEAN_LOOKBACK_HOURS', 6)
            method = getattr(config, 'ROBUST_MEAN_METHOD', 'iqr')
            use_price = getattr(config, 'ROBUST_MEAN_USE_PRICE', 'close')
            iqr_k = getattr(config, 'ROBUST_MEAN_IQR_K', 1.5)
            z_k = getattr(config, 'ROBUST_MEAN_ZSCORE', 3.0)
            mad_k = getattr(config, 'ROBUST_MEAN_MAD_K', 3.5)

            # approximate bars needed
            bars_per_hour = {
                '1m': 60, '3m': 20, '5m': 12, '15m': 4, '30m': 2,
                '1h': 1
            }.get(timeframe, 60)
            limit = max(100, hours * bars_per_hour)
            candles = exchange.get_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not candles:
                return None
            prices = self._select_price_series(candles, use_price)

            if method == 'zscore':
                filtered = self._zscore_filter(prices, z_k)
            elif method == 'mad':
                filtered = self._mad_filter(prices, mad_k)
            else:
                filtered = self._iqr_filter(prices, iqr_k)

            if not filtered:
                return None
            return sum(filtered) / len(filtered)
        except Exception:
            return None
    
    def get_reference_price(self, symbol: str) -> Optional[float]:
        """
        Get the reference price for a symbol.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Reference price or None if not set
        """
        return self.reference_prices.get(symbol)
    
    def should_buy(self, symbol: str, current_price: float, exchange=None) -> bool:
        """
        Determine if we should buy based on price drop and volatility ceiling.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
        
        Returns:
            True if buy signal is triggered, False otherwise
        """
        # Check volatility ceiling first (prevent buying at inflated prices)
        if config.ENABLE_VOLATILITY_CEILING and symbol == 'SOL/USDT':
            if current_price > config.SOL_VOLATILITY_CEILING:
                print(f"\n🚫 VOLATILITY CEILING: Price ${current_price:,.2f} exceeds ceiling ${config.SOL_VOLATILITY_CEILING:,.2f}")
                print(f"   Waiting for price to drop below ceiling before buying else we can buy in potential loss...")
                print(f"   If you want to buy anyway then quickly get to the code and stop and rerun the bot immediately.")
                return False
        
        # If no reference price, set current price as reference and don't buy yet
        if symbol not in self.reference_prices:
            self.set_reference_price(symbol, current_price)
            return False
        
        reference_price = self.reference_prices[symbol]
        price_change_percent = ((current_price - reference_price) / reference_price) * 100
        
        # Buy if price dropped by the threshold percentage and RSI (if enabled) confirms
        if price_change_percent <= -self.buy_drop_percent:
            if self.enable_rsi and exchange is not None:
                rsi_value = self.get_rsi(exchange, symbol)
                if rsi_value is not None and rsi_value > self.rsi_oversold:
                    # RSI not oversold yet; skip buy
                    print(f"   RSI {rsi_value:.1f} > oversold {self.rsi_oversold} → skipping buy")
                    return False
            print(f"\n✅ BUY SIGNAL TRIGGERED for {symbol}")
            print(f"   Current Price: ${current_price:,.2f}")
            print(f"   Reference Price: ${reference_price:,.2f}")
            print(f"   Price Change: {price_change_percent:.2f}%")
            if self.enable_rsi and exchange is not None:
                if 'rsi_value' in locals() and rsi_value is not None:
                    print(f"   RSI: {rsi_value:.1f} (≤ {self.rsi_oversold} oversold ✓)")
                else:
                    print(f"   RSI: n/a")
            if config.ENABLE_VOLATILITY_CEILING and symbol == 'SOL/USDT':
                print(f"   Volatility Ceiling: ${config.SOL_VOLATILITY_CEILING:,.2f} ✓")
            return True
        
        return False
    
    def should_sell(self, symbol: str, current_price: float, avg_buy_price: float, exchange=None) -> bool:
        """
        Determine if we should sell based on price increase from buy price.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            avg_buy_price: Average price at which we bought
        
        Returns:
            True if sell signal is triggered, False otherwise
        """
        price_change_percent = ((current_price - avg_buy_price) / avg_buy_price) * 100
        
        # Sell if price increased by the threshold percentage and RSI (if enabled) confirms
        if price_change_percent >= self.sell_increase_percent:
            if self.enable_rsi and exchange is not None:
                rsi_value = self.get_rsi(exchange, symbol)
                if rsi_value is not None and rsi_value < self.rsi_overbought:
                    # RSI not overbought yet; skip sell
                    print(f"   RSI {rsi_value:.1f} < overbought {self.rsi_overbought} → holding")
                    return False
            print(f"\nSELL SIGNAL TRIGGERED for {symbol}")
            print(f"   Current Price: ${current_price:,.2f}")
            print(f"   Avg Buy Price: ${avg_buy_price:,.2f}")
            print(f"   Price Change: {price_change_percent:.2f}%")
            if self.enable_rsi and exchange is not None:
                if 'rsi_value' in locals() and rsi_value is not None:
                    print(f"   RSI: {rsi_value:.1f} (≥ {self.rsi_overbought} overbought ✓)")
                else:
                    print(f"   RSI: n/a")
            return True
        
        return False
    
    def calculate_buy_amount(self, available_balance: float, current_price: float, 
                           trade_percentage: float = config.TRADE_PERCENTAGE) -> float:
        """
        Calculate how much crypto to buy based on available balance.
        Accounts for platform fees to ensure we don't exceed available balance.
        
        Args:
            available_balance: Available cash balance
            current_price: Current price of the crypto
            trade_percentage: Percentage of balance to use (default from config)
        
        Returns:
            Amount of crypto to buy
        """
        # Use a percentage of available balance for this trade
        trade_amount_usd = available_balance * (trade_percentage / 100)
        
        # Account for platform fees if enabled
        if config.ENABLE_PLATFORM_FEES:
            # Calculate the amount that would result in the desired trade amount after fees
            # If we want to spend X dollars including fees, we need to solve:
            # X = crypto_amount * price + crypto_amount * price * fee_rate
            # X = crypto_amount * price * (1 + fee_rate)
            # crypto_amount = X / (price * (1 + fee_rate))
            fee_rate = config.BINANCE_BUY_FEE_PERCENT / 100
            crypto_amount = trade_amount_usd / (current_price * (1 + fee_rate))
        else:
            crypto_amount = trade_amount_usd / current_price
        
        return crypto_amount
    
    def update_reference_price_after_buy(self, symbol: str, buy_price: float):
        """
        Update reference price after a successful buy.
        Set it slightly higher than buy price to avoid immediate re-buying.
        
        Args:
            symbol: Trading pair symbol
            buy_price: Price at which we bought
        """
        # Set reference price slightly above buy price (e.g., 1% above)
        new_reference = buy_price * 1.01
        self.set_reference_price(symbol, new_reference)
        print(f"   Updated reference price to ${new_reference:,.2f}")
    
    def check_volatility_ceiling(self, symbol: str, current_price: float) -> bool:
        """
        Check if current price is within volatility ceiling limits.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
        
        Returns:
            True if price is within acceptable range, False if too high
        """
        if not config.ENABLE_VOLATILITY_CEILING:
            return True
            
        if symbol == 'SOL/USDT':
            ceiling = config.SOL_VOLATILITY_CEILING
            if current_price > ceiling:
                return False
                
        return True
    
    def get_volatility_status(self, symbol: str, current_price: float) -> str:
        """
        Get a status message about volatility ceiling.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
        
        Returns:
            Status message about volatility ceiling
        """
        if not config.ENABLE_VOLATILITY_CEILING:
            return ""
            
        if symbol == 'SOL/USDT':
            ceiling = config.SOL_VOLATILITY_CEILING
            if current_price > ceiling:
                return f"🚫 Above ceiling (${ceiling:,.2f})"
            else:
                return f"✅ Below ceiling (${ceiling:,.2f})"
                
        return ""

