"""
Main Trading Bot Module
Orchestrates all components and runs the main trading loop.
"""

import time
import sys
import os
from datetime import datetime
import config
from exchange_connector import ExchangeConnector
from portfolio import Portfolio
from strategy import TradingStrategy

sys.stdout.reconfigure(line_buffering=True)

# Fix Unicode encoding issues on Windows
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
elif sys.platform == "darwin":  # macOS
    # macOS handles Unicode well by default, but ensure UTF-8
    import locale
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
elif sys.platform.startswith("linux"):  # Linux
    # Linux handles Unicode well by default
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        # Fallback if en_US.UTF-8 not available
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')


class TradingBot:
    """
    Main trading bot that coordinates exchange connection, portfolio management,
    and trading strategy execution.
    """
    
    def __init__(self):
        """Initialize the trading bot with all necessary components."""
        print("\n" + "="*60)
        print("🤖 CRYPTO TRADING BOT INITIALIZING")
        print("="*60 + "\n")
        
        # Initialize components
        self.exchange = ExchangeConnector()
        self.portfolio = Portfolio(exchange_connector=self.exchange)
        self.strategy = TradingStrategy()
        
        # ML predictor (will be initialized if user chooses ML option)
        self.ml_predictor = None
        self.use_ml_buy_decision = False
        
        # Trading state
        self.is_running = False
        self.current_symbol = None
        
        print("✓ All components initialized successfully\n")
    
    def select_buy_decision_mode(self) -> bool:
        """
        Let user choose between ML-based or config-based buy percentage decisions.
        
        Returns:
            True if user chooses ML-based decisions, False for config-based
        """
        print("\n" + "="*60)
        print("🤖 BUY DECISION MODE SELECTION")
        print("="*60)
        print("\nChoose how the bot should decide buy percentages:")
        print("\n1. 🤖 ML-Based Decision (Recommended)")
        print("   - Bot analyzes historical trends and market conditions")
        print("   - Dynamically adjusts buy percentage based on ML predictions")
        print("   - Uses technical indicators and price patterns")
        print("   - More adaptive to market conditions")
        
        print("\n2. ⚙️  Config-Based Decision (Traditional)")
        print("   - Uses fixed buy percentage from config settings")
        print("   - Simple and predictable")
        print("   - Current setting: -{config.BUY_DROP_PERCENTAGE}%")
        
        print("\n" + "="*60)
        
        while True:
            try:
                choice = input("\nYour choice (1 for ML, 2 for Config): ").strip()
                
                if choice == '1':
                    print("\n✅ Selected ML-Based Buy Decision")
                    print("   The bot will analyze market trends to determine optimal buy percentages")
                    return True
                elif choice == '2':
                    print(f"\n✅ Selected Config-Based Buy Decision")
                    print(f"   Using fixed buy percentage: -{config.BUY_DROP_PERCENTAGE}%")
                    return False
                else:
                    print("❌ Invalid choice. Please enter 1 or 2.")
                    
            except (EOFError, KeyboardInterrupt):
                print("\n⚠️  Using default: Config-Based Decision")
                return False
    
    def initialize_ml_predictor(self, symbol: str) -> bool:
        """
        Initialize and train the ML predictor for the selected symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            True if ML predictor initialized successfully, False otherwise
        """
        try:
            from ml_predictor import MLPredictor
            
            print(f"\n🔄 Initializing ML Predictor for {symbol}...")
            self.ml_predictor = MLPredictor()
            
            # Train the model
            success = self.ml_predictor.train_model(self.exchange, symbol)
            
            if success:
                print("✅ ML Predictor ready!")
                return True
            else:
                print("❌ ML Predictor training failed, falling back to config-based decisions")
                self.ml_predictor = None
                self.use_ml_buy_decision = False
                return False
                
        except ImportError as e:
            print(f"❌ Failed to import ML dependencies: {e}")
            print("   Please install required packages: pip install scikit-learn pandas numpy")
            return False
        except Exception as e:
            print(f"❌ Error initializing ML predictor: {e}")
            return False
    
    def select_crypto(self) -> str:
        """
        Let user manually select which crypto to trade.
        
        Returns:
            Selected trading pair symbol
        """
        print("\n" + "="*60)
        print("📋 SELECT CRYPTOCURRENCY TO TRADE")
        print("="*60)
        
        # Show some popular options
        popular_cryptos = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 
                          'ADA/USDT', 'XRP/USDT', 'DOGE/USDT']
        
        print("\nPopular options:")
        for i, symbol in enumerate(popular_cryptos, 1):
            print(f"  {i}. {symbol}")
        
        print(f"\n  0. Enter custom symbol")
        print(f"  (Press Enter for default: {config.DEFAULT_SYMBOL})")
        
        while True:
            choice = input("\nYour choice: ").strip()
            
            # Default option
            if not choice:
                return config.DEFAULT_SYMBOL
            
            # Custom symbol
            if choice == '0':
                custom = input("Enter symbol (e.g., BTC/USDT): ").strip().upper()
                if '/' in custom:
                    return custom
                else:
                    print("✗ Invalid format. Use format like 'BTC/USDT'")
                    continue
            
            # Numbered option
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(popular_cryptos):
                    return popular_cryptos[choice_num - 1]
                else:
                    print(f"✗ Please enter a number between 0 and {len(popular_cryptos)}")
            except ValueError:
                print("✗ Invalid input. Please enter a number.")
    
    def display_current_status(self, symbol: str, current_price: float):
        """
        Display current trading status in a clean format.
        
        Args:
            symbol: Trading pair being monitored
            current_price: Current market price
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        base_currency = symbol.split('/')[0]
        
        print(f"\n{'─'*60}")
        print(f"⏰ {timestamp}")
        print(f"{'─'*60}")
        print(f"Symbol:        {symbol}")
        print(f"Current Price: ${current_price:,.2f}")
        
        # Show reference price and change
        ref_price = self.strategy.get_reference_price(symbol)
        if ref_price:
            change = ((current_price - ref_price) / ref_price) * 100
            print(f"Reference:     ${ref_price:,.2f} ({change:+.2f}%)")
        
        # Show volatility ceiling status
        volatility_status = self.strategy.get_volatility_status(symbol, current_price)
        if volatility_status:
            print(f"Volatility:    {volatility_status}")

        # Show RSI if enabled
        if getattr(self.strategy, 'enable_rsi', False):
            rsi_value = self.strategy.get_rsi(self.exchange, symbol)
            if rsi_value is not None:
                rsi_note = []
                if rsi_value <= self.strategy.rsi_oversold:
                    rsi_note.append("oversold")
                if rsi_value >= self.strategy.rsi_overbought:
                    rsi_note.append("overbought")
                note_str = f" ({', '.join(rsi_note)})" if rsi_note else ""
                print(f"RSI:           {rsi_value:.1f}{note_str}")
            else:
                print("RSI:           n/a")

        # Show Robust Mean (used for mean-based strategy)
        robust_mean = self.strategy.get_robust_mean(self.exchange, symbol)
        if robust_mean is not None:
            delta = ((current_price - robust_mean) / robust_mean) * 100
            print(f"Robust Mean:   ${robust_mean:,.2f} ({delta:+.2f}%)")
        else:
            print("Robust Mean:   n/a")
        
        # Show Support/Resistance analysis
        support_resistance_status = self.strategy.get_support_resistance_status(symbol)
        if support_resistance_status:
            print(f"S/R Analysis:  {support_resistance_status}")
        
        # Show ML trend analysis if ML predictor is available
        if self.ml_predictor and self.use_ml_buy_decision:
            try:
                trend_analysis = self.ml_predictor.get_trend_analysis(self.exchange, symbol)
                if "error" not in trend_analysis:
                    print(f"ML Trend:      {trend_analysis['trend']} (vol: {trend_analysis['volatility']:.2f}%)")
                    
                    # Show support/resistance levels
                    support_dist = trend_analysis['support_distance']
                    resistance_dist = trend_analysis['resistance_distance']
                    print(f"Support:       {support_dist:.2f}% away")
                    print(f"Resistance:    {resistance_dist:.2f}% away")
            except Exception as e:
                print(f"ML Analysis:   Error - {e}")
        
        # Show position if we have one
        if base_currency in self.portfolio.positions:
            position = self.portfolio.positions[base_currency]
            amount = position['amount']
            avg_buy_price = position['avg_buy_price']
            current_value = amount * current_price
            position_pl = ((current_price - avg_buy_price) / avg_buy_price) * 100
            
            print(f"\nPosition:      {amount:.8f} {base_currency}")
            print(f"Avg Buy Price: ${avg_buy_price:,.2f}")
            print(f"Current Value: ${current_value:,.2f}")
            print(f"Position P/L:  {position_pl:+.2f}%")
        
        print(f"\nCash Balance:  ${self.portfolio.balance:,.2f} USDT")
        
        # Show total portfolio value
        current_prices = {symbol: current_price}
        total_value = self.portfolio.get_portfolio_value(current_prices)
        total_pl, total_pl_percent = self.portfolio.get_total_profit_loss(current_prices)
        
        print(f"Total Value:   ${total_value:,.2f}")
        print(f"Total P/L:     ${total_pl:,.2f} ({total_pl_percent:+.2f}%)")
        print(f"{'─'*60}")
    
    def execute_trading_cycle(self, symbol: str):
        """
        Execute one iteration of the trading cycle:
        - Fetch current price
        - Check for buy/sell signals
        - Execute trades if signals are triggered
        
        Args:
            symbol: Trading pair to trade
        """
        # Fetch current price
        current_price = self.exchange.get_current_price(symbol)
        
        if current_price is None:
            print("✗ Failed to fetch price, skipping this cycle")
            return
        
        # Display current status
        self.display_current_status(symbol, current_price)
        
        base_currency = symbol.split('/')[0]
        
        # Check if we have an open position
        has_position = base_currency in self.portfolio.positions
        
        if has_position:
            # We have a position - check if we should sell
            position = self.portfolio.positions[base_currency]
            avg_buy_price = position['avg_buy_price']
            
            if self.strategy.should_sell(symbol, current_price, avg_buy_price, exchange=self.exchange):
                # Sell the entire position
                amount_to_sell = position['amount']
                success = self.portfolio.sell(symbol, amount_to_sell, current_price)
                
                if success:
                    # Update reference price after sell to prepare for next buy
                    self.strategy.set_reference_price(symbol, current_price)
        
        else:
            # No position - check if we should buy
            if self.strategy.should_buy(symbol, current_price, exchange=self.exchange, 
                                      ml_predictor=self.ml_predictor if self.use_ml_buy_decision else None):
                # Calculate buy amount
                amount_to_buy = self.strategy.calculate_buy_amount(
                    self.portfolio.balance, 
                    current_price
                )
                
                # Execute buy
                success = self.portfolio.buy(symbol, amount_to_buy, current_price)
                
                if success:
                    # Update reference price after buy
                    self.strategy.update_reference_price_after_buy(symbol, current_price)
    
    def run(self):
        """
        Main run loop for the trading bot.
        Continuously monitors price and executes trading strategy.
        """
        # Let user select crypto to trade
        self.current_symbol = self.select_crypto()
        
        # Let user choose buy decision mode
        self.use_ml_buy_decision = self.select_buy_decision_mode()
        
        # Initialize ML predictor if user chose ML mode
        if self.use_ml_buy_decision:
            success = self.initialize_ml_predictor(self.current_symbol)
            if not success:
                print("⚠️  Falling back to config-based buy decisions")
                self.use_ml_buy_decision = False
        
        print(f"\n{'='*60}")
        print(f"🚀 STARTING TRADING BOT")
        print(f"{'='*60}")
        print(f"Trading Pair:     {self.current_symbol}")
        print(f"Mode:             {config.TRADING_MODE.upper()}")
        print(f"Check Interval:   {config.PRICE_CHECK_INTERVAL} seconds")
        
        # Show mean-based strategy information
        print(f"Strategy:         📊 Robust Mean-Based Trading")
        print(f"Robust Mean:      {getattr(config, 'ROBUST_MEAN_LOOKBACK_HOURS', 6)} hours ({getattr(config, 'ROBUST_MEAN_TIMEFRAME', '1m')})")
        print(f"Update Interval:  Every {getattr(config, 'ROBUST_MEAN_REFRESH_SECONDS', 10)} seconds")
        
        if self.use_ml_buy_decision:
            print(f"Buy Decision:     🤖 ML-Based (Dynamic below mean)")
        else:
            print(f"Buy Threshold:    {config.MEAN_BUY_THRESHOLD_PERCENT}% below mean")
            
        print(f"Sell Target:      {config.MEAN_SELL_PROFIT_PERCENT}% profit from buy")
            
        print(f"Trade Size:       {config.TRADE_PERCENTAGE}% of balance")
        print(f"\nPress Ctrl+C to stop the bot\n")
        print(f"{'='*60}\n")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Execute one trading cycle
                self.execute_trading_cycle(self.current_symbol)
                
                # Wait before next check
                time.sleep(config.PRICE_CHECK_INTERVAL)
                
        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print("🛑 STOPPING TRADING BOT")
            print("="*60)
            self.is_running = False
            
            # Display final portfolio status
            current_price = self.exchange.get_current_price(self.current_symbol)
            if current_price:
                current_prices = {self.current_symbol: current_price}
                self.portfolio.display_portfolio(current_prices)
            
            # Show trade history
            self.portfolio.display_trade_history()
            
            print("\n✓ Bot stopped successfully\n")
    
    def display_menu(self):
        """Display an interactive menu for bot control."""
        while True:
            print("\n" + "="*60)
            print("🤖 CRYPTO TRADING BOT - MAIN MENU")
            print("="*60)
            print("1. Start Trading")
            print("2. View Portfolio")
            print("3. View Trade History")
            print("4. Export Trade History (CSV)")
            print("5. Export Trade History (TXT)")
            print("6. Change Trading Pair")
            print("7. Adjust Strategy Parameters")
            print("8. Exit")
            print("="*60)
            
            try:
                choice = input("\nSelect option: ").strip()
            except EOFError:
                # Running in Docker without interactive terminal
                print("\n🤖 Running in Docker mode - starting trading automatically...")
                choice = '1'  # Auto-start trading
            
            if choice == '1':
                self.run()
            elif choice == '2':
                current_price = None
                if self.current_symbol:
                    current_price = self.exchange.get_current_price(self.current_symbol)
                    current_prices = {self.current_symbol: current_price}
                    self.portfolio.display_portfolio(current_prices)
                else:
                    self.portfolio.display_portfolio()
            elif choice == '3':
                self.portfolio.display_trade_history()
            elif choice == '4':
                self.portfolio.export_trade_history_to_csv()
            elif choice == '5':
                self.portfolio.save_trade_history_to_file()
            elif choice == '6':
                self.current_symbol = self.select_crypto()
                print(f"\n✓ Trading pair changed to {self.current_symbol}")
            elif choice == '7':
                self.adjust_strategy()
            elif choice == '8':
                print("\n👋 Goodbye!\n")
                break
            else:
                print("\n✗ Invalid option. Please try again.")
    
    def adjust_strategy(self):
        """Allow user to adjust strategy parameters."""
        print("\n" + "="*60)
        print("⚙️  ADJUST STRATEGY PARAMETERS")
        print("="*60)
        print(f"Current Settings:")
        print(f"  Buy Threshold:  -{self.strategy.buy_drop_percent}%")
        print(f"  Sell Threshold: +{self.strategy.sell_increase_percent}%")
        
        try:
            buy_input = input(f"\nNew buy threshold (current: {self.strategy.buy_drop_percent}%, Enter to keep): ").strip()
            if buy_input:
                self.strategy.buy_drop_percent = float(buy_input)
            
            sell_input = input(f"New sell threshold (current: {self.strategy.sell_increase_percent}%, Enter to keep): ").strip()
            if sell_input:
                self.strategy.sell_increase_percent = float(sell_input)
            
            print(f"\n✓ Strategy updated:")
            print(f"  Buy Threshold:  -{self.strategy.buy_drop_percent}%")
            print(f"  Sell Threshold: +{self.strategy.sell_increase_percent}%")
            
        except ValueError:
            print("\n✗ Invalid input. Strategy unchanged.")


def main():
    """Entry point for the trading bot."""
    try:
        bot = TradingBot()
        bot.display_menu()
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

