"""
Trading Strategy Module
Implements the buy/sell logic based on price movements.
"""

from typing import Optional, Dict, List
import config
import time


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
        
        # Support/Resistance trend tracking
        self.support_history: Dict[str, List[float]] = {}
        self.resistance_history: Dict[str, List[float]] = {}
        self.market_trend: Dict[str, str] = {}  # 'bullish', 'bearish', 'neutral'
        self.last_support: Dict[str, float] = {}
        self.last_resistance: Dict[str, float] = {}
        self.support_trend: Dict[str, str] = {}  # 'increasing', 'decreasing', 'stable'
        self.resistance_trend: Dict[str, str] = {}  # 'increasing', 'decreasing', 'stable'
        
        # Mean-based strategy tracking
        self.last_buy_prices: Dict[str, float] = {}  # Track last buy price for sell decisions
        
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
    
    def should_buy(self, symbol: str, current_price: float, exchange=None, ml_predictor=None) -> bool:
        """
        Determine if we should buy based on price drop and volatility ceiling.
        Can use ML predictor for dynamic buy percentage calculation.
        Uses mean-based strategy if enabled.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            exchange: Exchange connector instance
            ml_predictor: ML predictor instance for dynamic buy percentage
        
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
        
        # Use mean-based strategy (primary strategy)
        return self._should_buy_mean_based(symbol, current_price, exchange, ml_predictor)
    
    def _should_buy_mean_based(self, symbol: str, current_price: float, exchange=None, ml_predictor=None) -> bool:
        """
        Mean-based buy strategy: Buy when price goes x% below the mean price.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            exchange: Exchange connector instance
            ml_predictor: ML predictor instance for dynamic buy percentage
            
        Returns:
            True if buy signal is triggered, False otherwise
        """
        if not exchange:
            print("⚠️  Exchange required for mean-based strategy")
            return False
        
        # Get current robust mean price
        mean_price = self.get_robust_mean(exchange, symbol)
        if mean_price is None:
            print(f"⚠️  Could not calculate robust mean price for {symbol}")
            return False
        
        # Calculate how much below mean the current price is
        price_below_mean_percent = ((mean_price - current_price) / mean_price) * 100
        
        # Determine buy threshold - use ML prediction if available
        buy_threshold = config.MEAN_BUY_THRESHOLD_PERCENT
        ml_confidence = 0.0
        
        if ml_predictor and config.ENABLE_ML_BUY_DECISION:
            try:
                # Get trend analysis and ML prediction
                trend_analysis = ml_predictor.get_trend_analysis(exchange, symbol)
                ml_trend = trend_analysis.get('trend', 'Unknown')
                
                predicted_percentage, confidence = ml_predictor.predict_buy_percentage(
                    exchange, symbol, current_price
                )
                
                # Always show what the ML would have suggested
                if ml_predictor.should_use_ml_prediction(confidence):
                    print(f"🤖 ML Prediction: Would buy at {predicted_percentage:.2f}% below mean (confidence: {confidence:.2f}) - Trend: {ml_trend}")
                else:
                    print(f"⚠️  ML confidence too low ({confidence:.2f}), using config threshold {config.MEAN_BUY_THRESHOLD_PERCENT:.2f}% - Trend: {ml_trend}")
                
                # If ML trend is bearish, skip buying regardless of confidence
                if config.ENABLE_TREND_PROTECTION and ml_trend == 'Bearish':
                    print(f"🚫 BLOCKED: ML Trend is {ml_trend} - Skipping buy to avoid falling market")
                    return False
                
                # Use ML prediction if confidence is high enough
                if ml_predictor.should_use_ml_prediction(confidence):
                    buy_threshold = predicted_percentage
                    ml_confidence = confidence
                else:
                    print(f"⚠️  Using config threshold due to low ML confidence")
                    
            except Exception as e:
                print(f"⚠️  ML prediction failed: {e}, using config threshold")
        
        # Check if price is enough below mean to trigger buy
        if price_below_mean_percent >= buy_threshold:
            if self.enable_rsi and exchange is not None:
                rsi_value = self.get_rsi(exchange, symbol)
                if rsi_value is not None:
                    # Additional safety: If RSI is extremely oversold (< 20), be more cautious
                    if config.ENABLE_EXTREME_OVERSOLD_PROTECTION and rsi_value < 20:
                        print(f"⚠️  RSI extremely oversold ({rsi_value:.1f} < 20) - Market may be in free fall, skipping buy")
                        return False
                    elif rsi_value > self.rsi_oversold:
                        # RSI not oversold yet; skip buy
                        print(f"   RSI {rsi_value:.1f} > oversold {self.rsi_oversold} → skipping buy")
                        return False
            
            print(f"\n✅ MEAN-BASED BUY SIGNAL TRIGGERED for {symbol}")
            print(f"   Current Price: ${current_price:,.2f}")
            print(f"   Mean Price: ${mean_price:,.2f}")
            print(f"   Price Below Mean: {price_below_mean_percent:.2f}%")
            print(f"   Buy Threshold: {buy_threshold:.2f}%")
            
            if ml_confidence > 0:
                print(f"   ML Confidence: {ml_confidence:.2f} ✓")
            
            if self.enable_rsi and exchange is not None:
                if 'rsi_value' in locals() and rsi_value is not None:
                    print(f"   RSI: {rsi_value:.1f} (≤ {self.rsi_oversold} oversold ✓)")
                else:
                    print(f"   RSI: n/a")
            
            # Store the buy price for sell decisions
            self.last_buy_prices[symbol] = current_price
            
            return True
        
        return False
    
    def should_sell(self, symbol: str, current_price: float, avg_buy_price: float, exchange=None) -> bool:
        """
        Determine if we should sell based on price increase from buy price.
        Uses mean-based strategy if enabled.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            avg_buy_price: Average price at which we bought
            exchange: Exchange connector instance
        
        Returns:
            True if sell signal is triggered, False otherwise
        """
        # Use mean-based strategy (primary strategy)
        return self._should_sell_mean_based(symbol, current_price, avg_buy_price, exchange)
    
    def _should_sell_mean_based(self, symbol: str, current_price: float, avg_buy_price: float, exchange=None) -> bool:
        """
        Mean-based sell strategy: Sell when price reaches target profit from buy price.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            avg_buy_price: Average price at which we bought
            exchange: Exchange connector instance
            
        Returns:
            True if sell signal is triggered, False otherwise
        """
        # Calculate profit percentage from buy price
        profit_percent = ((current_price - avg_buy_price) / avg_buy_price) * 100
        
        # Sell if we've reached the target profit percentage
        if profit_percent >= config.MEAN_SELL_PROFIT_PERCENT:
            if self.enable_rsi and exchange is not None:
                rsi_value = self.get_rsi(exchange, symbol)
                if rsi_value is not None and rsi_value < self.rsi_overbought:
                    # RSI not overbought yet; skip sell
                    print(f"   RSI {rsi_value:.1f} < overbought {self.rsi_overbought} → holding")
                    return False
            
            print(f"\n✅ MEAN-BASED SELL SIGNAL TRIGGERED for {symbol}")
            print(f"   Current Price: ${current_price:,.2f}")
            print(f"   Buy Price: ${avg_buy_price:,.2f}")
            print(f"   Profit: {profit_percent:,.2f}%")
            print(f"   Target Profit: {config.MEAN_SELL_PROFIT_PERCENT:.2f}%")
            
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
    
    def update_support_resistance_analysis(self, symbol: str, current_price: float, exchange=None):
        """
        Update support/resistance analysis and detect market trends.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            exchange: Exchange connector for getting historical data
        """
        if not exchange:
            return
            
        base_currency = symbol.split('/')[0]
        
        try:
            # Get recent OHLCV data for support/resistance calculation
            candles = exchange.get_ohlcv(symbol, timeframe='5m', limit=50)
            if not candles or len(candles) < 20:
                return
            
            # Calculate current support and resistance
            recent_lows = [candle[3] for candle in candles[-20:]]  # Low prices
            recent_highs = [candle[2] for candle in candles[-20:]]  # High prices
            
            current_support = min(recent_lows)
            current_resistance = max(recent_highs)
            
            # Initialize history if first time
            if base_currency not in self.support_history:
                self.support_history[base_currency] = []
                self.resistance_history[base_currency] = []
            
            # Add to history (keep last 10 values)
            self.support_history[base_currency].append(current_support)
            self.resistance_history[base_currency].append(current_resistance)
            
            # Keep only last 10 values
            if len(self.support_history[base_currency]) > 10:
                self.support_history[base_currency] = self.support_history[base_currency][-10:]
                self.resistance_history[base_currency] = self.resistance_history[base_currency][-10:]
            
            # Need at least 3 values to determine trend
            if len(self.support_history[base_currency]) >= 3:
                self._analyze_support_resistance_trends(base_currency)
            
            # Store current values
            self.last_support[base_currency] = current_support
            self.last_resistance[base_currency] = current_resistance
            
        except Exception as e:
            print(f"⚠️  Support/Resistance analysis failed: {e}")
    
    def _analyze_support_resistance_trends(self, base_currency: str):
        """
        Analyze support and resistance trends to determine market direction.
        
        Args:
            base_currency: Base currency symbol
        """
        support_values = self.support_history[base_currency]
        resistance_values = self.resistance_history[base_currency]
        
        # Analyze support trend (last 3 values)
        if len(support_values) >= 3:
            recent_support = support_values[-3:]
            if recent_support[-1] > recent_support[-2] > recent_support[-3]:
                self.support_trend[base_currency] = 'increasing'
            elif recent_support[-1] < recent_support[-2] < recent_support[-3]:
                self.support_trend[base_currency] = 'decreasing'
            else:
                self.support_trend[base_currency] = 'stable'
        
        # Analyze resistance trend (last 3 values)
        if len(resistance_values) >= 3:
            recent_resistance = resistance_values[-3:]
            if recent_resistance[-1] > recent_resistance[-2] > recent_resistance[-3]:
                self.resistance_trend[base_currency] = 'increasing'
            elif recent_resistance[-1] < recent_resistance[-2] < recent_resistance[-3]:
                self.resistance_trend[base_currency] = 'decreasing'
            else:
                self.resistance_trend[base_currency] = 'stable'
        
        # Determine overall market trend
        support_trend = self.support_trend.get(base_currency, 'stable')
        resistance_trend = self.resistance_trend.get(base_currency, 'stable')
        
        if support_trend == 'decreasing' and resistance_trend == 'increasing':
            self.market_trend[base_currency] = 'bearish'  # Falling market
        elif support_trend == 'increasing' and resistance_trend == 'decreasing':
            self.market_trend[base_currency] = 'bullish'  # Rising market
        elif support_trend == 'increasing' and resistance_trend == 'stable':
            self.market_trend[base_currency] = 'bullish'  # Support building
        elif support_trend == 'stable' and resistance_trend == 'decreasing':
            self.market_trend[base_currency] = 'bullish'  # Resistance breaking
        else:
            self.market_trend[base_currency] = 'neutral'
    
    def get_dynamic_buy_threshold(self, symbol: str, current_price: float, 
                                 base_threshold: float, exchange=None) -> float:
        """
        Calculate dynamic buy threshold based on support/resistance analysis.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            base_threshold: Base buy threshold percentage
            exchange: Exchange connector for getting historical data
            
        Returns:
            Adjusted buy threshold percentage
        """
        base_currency = symbol.split('/')[0]
        
        # Update support/resistance analysis
        self.update_support_resistance_analysis(symbol, current_price, exchange)
        
        market_trend = self.market_trend.get(base_currency, 'neutral')
        support_trend = self.support_trend.get(base_currency, 'stable')
        resistance_trend = self.resistance_trend.get(base_currency, 'stable')
        
        # Start with base threshold
        adjusted_threshold = base_threshold
        
        # Apply support/resistance-based adjustments
        if market_trend == 'bearish':
            # Falling market: Support decreasing + Resistance increasing
            # Be more conservative - require bigger drops
            adjusted_threshold *= 2.0  # Double the threshold (e.g., 0.5% -> 1.0%)
            print(f"📉 Bearish market detected: Support {support_trend}, Resistance {resistance_trend}")
            print(f"   Adjusted threshold: {adjusted_threshold:.2f}% (more conservative)")
            
        elif market_trend == 'bullish':
            # Rising market: Support increasing + Resistance decreasing
            # Be more aggressive - smaller drops needed
            adjusted_threshold *= 0.6  # Reduce threshold (e.g., 0.5% -> 0.3%)
            print(f"📈 Bullish market detected: Support {support_trend}, Resistance {resistance_trend}")
            print(f"   Adjusted threshold: {adjusted_threshold:.2f}% (more aggressive)")
            
        elif support_trend == 'increasing':
            # Support is building up - good sign for buying
            adjusted_threshold *= 0.7  # Slightly more aggressive
            print(f"🛡️  Support building: {support_trend}")
            print(f"   Adjusted threshold: {adjusted_threshold:.2f}% (support building)")
            
        elif resistance_trend == 'decreasing':
            # Resistance is weakening - good sign for buying
            adjusted_threshold *= 0.8  # Slightly more aggressive
            print(f"💪 Resistance weakening: {resistance_trend}")
            print(f"   Adjusted threshold: {adjusted_threshold:.2f}% (resistance breaking)")
        
        # Ensure threshold stays within reasonable bounds
        adjusted_threshold = max(0.2, min(3.0, adjusted_threshold))
        
        return adjusted_threshold
    
    def check_instant_buy_signal(self, symbol: str, current_price: float) -> bool:
        """
        Check for instant buy signal when support starts increasing (reversal signal).
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            
        Returns:
            True if instant buy signal is triggered
        """
        base_currency = symbol.split('/')[0]
        
        # Check if we have enough history
        if (base_currency not in self.support_history or 
            len(self.support_history[base_currency]) < 3):
            return False
        
        support_trend = self.support_trend.get(base_currency, 'stable')
        
        # Instant buy when support starts increasing (reversal signal)
        if support_trend == 'increasing':
            # Check if this is a new trend (wasn't increasing before)
            if len(self.support_history[base_currency]) >= 4:
                previous_trend = 'stable'
                prev_support = self.support_history[base_currency][-4:-1]
                if len(prev_support) >= 3:
                    if prev_support[-1] > prev_support[-2] > prev_support[-3]:
                        previous_trend = 'increasing'
                    elif prev_support[-1] < prev_support[-2] < prev_support[-3]:
                        previous_trend = 'decreasing'
                
                # If support was decreasing or stable before, and now increasing
                if previous_trend in ['decreasing', 'stable']:
                    print(f"🚀 INSTANT BUY SIGNAL: Support reversal detected!")
                    print(f"   Previous trend: {previous_trend}")
                    print(f"   Current trend: {support_trend}")
                    print(f"   Support values: {self.support_history[base_currency][-3:]}")
                    return True
        
        return False
    
    def get_support_resistance_status(self, symbol: str) -> str:
        """
        Get support/resistance status for display.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Status string for display
        """
        base_currency = symbol.split('/')[0]
        
        if base_currency not in self.market_trend:
            return ""
        
        market_trend = self.market_trend[base_currency]
        support_trend = self.support_trend.get(base_currency, 'stable')
        resistance_trend = self.resistance_trend.get(base_currency, 'stable')
        
        status_parts = []
        
        # Market trend
        trend_emoji = {'bullish': '📈', 'bearish': '📉', 'neutral': '➡️'}
        status_parts.append(f"Trend: {trend_emoji.get(market_trend, '➡️')} {market_trend}")
        
        # Support trend
        support_emoji = {'increasing': '🛡️', 'decreasing': '🛡️', 'stable': '🛡️'}
        status_parts.append(f"Support: {support_emoji.get(support_trend, '🛡️')} {support_trend}")
        
        # Resistance trend
        resistance_emoji = {'increasing': '💪', 'decreasing': '💪', 'stable': '💪'}
        status_parts.append(f"Resistance: {resistance_emoji.get(resistance_trend, '💪')} {resistance_trend}")
        
        return " | ".join(status_parts)

