"""
Quick start script - bypasses menu and goes straight to trading.
"""

from trading_bot_simple import TradingBot

def quick_start():
    print("QUICK START - Starting trading immediately...")
    print("="*60)
    
    try:
        bot = TradingBot()
        
        # Skip menu and go straight to trading
        print("\nStarting BTC/USDT trading with current settings...")
        bot.current_symbol = 'BTC/USDT'  # Set default symbol
        bot.run()  # Start trading loop
        
    except KeyboardInterrupt:
        print("\n\nTrading stopped by user.")
        print("Final portfolio summary:")
        if bot.current_symbol:
            current_price = bot.exchange.get_current_price(bot.current_symbol)
            if current_price:
                current_prices = {bot.current_symbol: current_price}
                bot.portfolio.display_portfolio(current_prices)
                bot.portfolio.display_annual_tax_summary()
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    quick_start()
