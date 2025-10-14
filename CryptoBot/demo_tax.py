"""
Demo script to show tax calculations working.
"""

from tax_calculator import TaxCalculator
from portfolio import Portfolio

def demo_tax_calculations():
    print("DEMO: UK Tax Calculations for Crypto Trading")
    print("="*60)
    
    # Initialize tax calculator for UK
    tax_calc = TaxCalculator('UK')
    
    # Demo different profit scenarios
    scenarios = [
        {"profit": 1000, "description": "Small profit"},
        {"profit": 5000, "description": "Medium profit"},
        {"profit": 10000, "description": "Large profit"},
    ]
    
    print("\nTAX IMPACT EXAMPLES:")
    print("-"*60)
    
    for scenario in scenarios:
        profit = scenario["profit"]
        description = scenario["description"]
        
        print(f"\n{description}: ${profit:,} profit")
        tax_calc.display_tax_breakdown(profit)
    
    # Demo portfolio with tax calculations
    print("\nPORTFOLIO DEMO:")
    print("-"*60)
    
    portfolio = Portfolio()
    
    # Simulate a profitable trade
    print("\nSimulating a profitable trade...")
    success = portfolio.sell('BTC/USDT', 0.01, 50000)  # Sell 0.01 BTC at $50,000
    
    if success:
        print("Trade executed successfully!")
        print("Check the tax breakdown above!")

if __name__ == "__main__":
    demo_tax_calculations()
