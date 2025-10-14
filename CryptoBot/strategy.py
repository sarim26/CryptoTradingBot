"""
Trading Strategy Module
Implements the buy/sell logic based on price movements.
"""

from typing import Optional, Dict
import config


class TradingStrategy:
    """
    Implements a simple percentage-based trading strategy:
    - Buy when price drops by X% from last buy/reference price
    - Sell when price increases by Y% from buy price
    """
    
    def __init__(self, 
                 buy_drop_percent: float = config.BUY_DROP_PERCENTAGE,
                 sell_increase_percent: float = config.SELL_INCREASE_PERCENTAGE):
        """
        Initialize the trading strategy.
        
        Args:
            buy_drop_percent: Percentage drop to trigger buy signal
            sell_increase_percent: Percentage increase to trigger sell signal
        """
        self.buy_drop_percent = buy_drop_percent
        self.sell_increase_percent = sell_increase_percent
        
        # Track reference prices for each symbol
        self.reference_prices: Dict[str, float] = {}
        
        print(f"\n{'='*60}")
        print(f"Trading Strategy Initialized")
        print(f"{'='*60}")
        print(f"Buy Signal:  Price drops by {self.buy_drop_percent}% or more")
        print(f"Sell Signal: Price increases by {self.sell_increase_percent}% or more")
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
    
    def get_reference_price(self, symbol: str) -> Optional[float]:
        """
        Get the reference price for a symbol.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Reference price or None if not set
        """
        return self.reference_prices.get(symbol)
    
    def should_buy(self, symbol: str, current_price: float) -> bool:
        """
        Determine if we should buy based on price drop.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
        
        Returns:
            True if buy signal is triggered, False otherwise
        """
        # If no reference price, set current price as reference and don't buy yet
        if symbol not in self.reference_prices:
            self.set_reference_price(symbol, current_price)
            return False
        
        reference_price = self.reference_prices[symbol]
        price_change_percent = ((current_price - reference_price) / reference_price) * 100
        
        # Buy if price dropped by the threshold percentage
        if price_change_percent <= -self.buy_drop_percent:
            print(f"\nBUY SIGNAL TRIGGERED for {symbol}")
            print(f"   Current Price: ${current_price:,.2f}")
            print(f"   Reference Price: ${reference_price:,.2f}")
            print(f"   Price Change: {price_change_percent:.2f}%")
            return True
        
        return False
    
    def should_sell(self, symbol: str, current_price: float, avg_buy_price: float) -> bool:
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
        
        # Sell if price increased by the threshold percentage
        if price_change_percent >= self.sell_increase_percent:
            print(f"\nSELL SIGNAL TRIGGERED for {symbol}")
            print(f"   Current Price: ${current_price:,.2f}")
            print(f"   Avg Buy Price: ${avg_buy_price:,.2f}")
            print(f"   Price Change: {price_change_percent:.2f}%")
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

