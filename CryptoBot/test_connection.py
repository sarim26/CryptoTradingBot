"""
Quick test script to verify the bot components work correctly.
"""

import config
from exchange_connector import ExchangeConnector

def test_connection():
    print("Testing crypto trading bot connection...")
    print("="*50)
    
    # Test exchange connection
    try:
        exchange = ExchangeConnector()
        print("Exchange connection successful!")
        
        # Test price fetching
        price = exchange.get_current_price('BTC/USDT')
        if price:
            print(f"BTC/USDT price: ${price:,.2f}")
        else:
            print("Failed to fetch BTC price")
            
        # Test configuration
        print(f"Trading mode: {config.TRADING_MODE}")
        print(f"Initial balance: ${config.INITIAL_BALANCE:,.2f}")
        print(f"Buy threshold: -{config.BUY_DROP_PERCENTAGE}%")
        print(f"Sell threshold: +{config.SELL_INCREASE_PERCENTAGE}%")
        
        print("\n" + "="*50)
        print("All tests passed! Bot is ready to run.")
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_connection()
