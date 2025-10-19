#!/usr/bin/env python3
"""
Simple script to view trade log file.
Usage: python view_trades.py [number_of_trades]
"""

import sys
import os
from portfolio import Portfolio

def main():
    """View trade log file."""
    
    # Get number of trades to show (default 20)
    try:
        last_n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    except ValueError:
        print("Usage: python view_trades.py [number_of_trades]")
        print("Example: python view_trades.py 10")
        return
    
    # Create portfolio instance to use the view method
    portfolio = Portfolio()
    
    # View the trade log
    portfolio.view_trade_log(last_n)
    
    # Show file info
    filename = "trades.log"
    if os.path.exists(filename):
        file_size = os.path.getsize(filename)
        print(f"📁 Trade log file: {filename}")
        print(f"📊 File size: {file_size:,} bytes")
    else:
        print(f"📁 No trade log file found: {filename}")

if __name__ == "__main__":
    main()
