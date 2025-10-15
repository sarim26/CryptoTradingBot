"""
Portfolio Management Module
Tracks virtual balance, positions, and calculates profit/loss.
"""

from typing import Dict, List
from datetime import datetime
import config
import csv
import os
from tax_calculator import TaxCalculator


class Portfolio:
    """
    Manages the trading portfolio including balance and positions.
    In paper trading mode, this simulates a real portfolio.
    """
    
    def __init__(self, initial_balance: float = config.INITIAL_BALANCE, exchange_connector=None):
        """
        Initialize portfolio with starting balance.
        
        Args:
            initial_balance: Starting balance in quote currency (USDT) - only used in paper mode
            exchange_connector: Exchange connector for fetching real balance in live mode
        """
        self.positions: Dict[str, Dict] = {}  # Current open positions
        self.trade_history: List[Dict] = []  # History of all trades
        
        # Initialize tax calculator if enabled
        if config.ENABLE_TAX_CALCULATIONS:
            self.tax_calculator = TaxCalculator(config.TAX_COUNTRY)
        else:
            self.tax_calculator = None
        
        # Set balance based on trading mode
        if config.TRADING_MODE == 'live' and exchange_connector:
            # Live mode: fetch real balance from exchange
            try:
                real_balance = self._fetch_real_balance(exchange_connector)
                self.balance = real_balance
                self.initial_balance = real_balance
                print(f"\n{'='*60}")
                print(f"Portfolio Initialized (LIVE MODE)")
                print(f"{'='*60}")
                print(f"Real Balance: ${self.balance:,.2f} USDT")
                print(f"{'='*60}\n")
            except Exception as e:
                print(f"Warning: Could not fetch real balance: {e}")
                print("Using fallback balance...")
                self.initial_balance = initial_balance
                self.balance = initial_balance
                print(f"\n{'='*60}")
                print(f"Portfolio Initialized (FALLBACK)")
                print(f"{'='*60}")
                print(f"Fallback Balance: ${self.balance:,.2f} USDT")
                print(f"{'='*60}\n")
        else:
            # Paper mode: use virtual balance
            self.initial_balance = initial_balance
            self.balance = initial_balance
            print(f"\n{'='*60}")
            print(f"Portfolio Initialized (PAPER MODE)")
            print(f"{'='*60}")
            print(f"Virtual Balance: ${self.balance:,.2f} USDT")
            print(f"{'='*60}\n")
    
    def _fetch_real_balance(self, exchange_connector):
        """
        Fetch real USDT balance from the exchange.
        
        Args:
            exchange_connector: Exchange connector instance
        
        Returns:
            Real USDT balance
        """
        try:
            # Fetch account balance
            balance = exchange_connector.exchange.fetch_balance()
            
            # Get USDT balance
            usdt_balance = balance.get('USDT', {}).get('free', 0.0)
            
            if usdt_balance is None:
                usdt_balance = 0.0
            
            print(f"Fetched real USDT balance: ${usdt_balance:,.2f}")
            return float(usdt_balance)
            
        except Exception as e:
            print(f"Error fetching real balance: {e}")
            raise
    
    def buy(self, symbol: str, amount: float, price: float) -> bool:
        """
        Execute a buy order (add to portfolio).
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            amount: Amount of crypto to buy
            price: Price per unit
        
        Returns:
            True if successful, False otherwise
        """
        cost = amount * price
        
        # Calculate platform fee if enabled
        platform_fee = 0.0
        if config.ENABLE_PLATFORM_FEES:
            platform_fee = cost * (config.BINANCE_BUY_FEE_PERCENT / 100)
        
        total_cost = cost + platform_fee
        
        # Check if we have enough balance (including fees)
        if total_cost > self.balance:
            print(f"X Insufficient balance to buy {amount} {symbol}")
            print(f"  Required: ${total_cost:,.2f} (including ${platform_fee:,.2f} fees) | Available: ${self.balance:,.2f}")
            return False
        
        # Extract base currency (e.g., 'BTC' from 'BTC/USDT')
        base_currency = symbol.split('/')[0]
        
        # Update balance (deduct total cost including fees)
        self.balance -= total_cost
        
        # Update or create position
        if base_currency in self.positions:
            # Average the buy price if we already have this position
            old_amount = self.positions[base_currency]['amount']
            old_avg_price = self.positions[base_currency]['avg_buy_price']
            new_amount = old_amount + amount
            new_avg_price = ((old_amount * old_avg_price) + (amount * price)) / new_amount
            
            self.positions[base_currency] = {
                'amount': new_amount,
                'avg_buy_price': new_avg_price,
                'symbol': symbol
            }
        else:
            # Create new position
            self.positions[base_currency] = {
                'amount': amount,
                'avg_buy_price': price,
                'symbol': symbol
            }
        
        # Record trade in history
        trade = {
            'timestamp': datetime.now(),
            'type': 'BUY',
            'symbol': symbol,
            'amount': amount,
            'price': price,
            'total': cost,
            'platform_fee': platform_fee,
            'total_cost': total_cost
        }
        self.trade_history.append(trade)
        
        # Print trade confirmation
        print(f"\n{'-'*60}")
        print(f"BUY ORDER EXECUTED")
        print(f"{'-'*60}")
        print(f"Symbol:         {symbol}")
        print(f"Amount:         {amount:.8f} {base_currency}")
        print(f"Price:          ${price:,.2f}")
        print(f"Subtotal:       ${cost:,.2f}")
        if platform_fee > 0:
            print(f"Platform Fee:   ${platform_fee:,.2f} ({config.BINANCE_BUY_FEE_PERCENT}%)")
        print(f"Total Cost:     ${total_cost:,.2f}")
        print(f"New Balance:    ${self.balance:,.2f} USDT")
        print(f"{'-'*60}\n")
        
        return True
    
    def sell(self, symbol: str, amount: float, price: float) -> bool:
        """
        Execute a sell order (remove from portfolio).
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            amount: Amount of crypto to sell
            price: Price per unit
        
        Returns:
            True if successful, False otherwise
        """
        base_currency = symbol.split('/')[0]
        
        # Check if we have this position
        if base_currency not in self.positions:
            print(f"✗ No position found for {base_currency}")
            return False
        
        # Check if we have enough amount to sell
        if amount > self.positions[base_currency]['amount']:
            print(f"✗ Insufficient {base_currency} to sell")
            print(f"  Requested: {amount} | Available: {self.positions[base_currency]['amount']}")
            return False
        
        revenue = amount * price
        avg_buy_price = self.positions[base_currency]['avg_buy_price']
        
        # Calculate platform fee if enabled
        platform_fee = 0.0
        if config.ENABLE_PLATFORM_FEES:
            platform_fee = revenue * (config.BINANCE_SELL_FEE_PERCENT / 100)
        
        net_revenue = revenue - platform_fee
        
        # Calculate profit/loss (gross and net)
        gross_profit_loss = (price - avg_buy_price) * amount
        net_profit_loss = gross_profit_loss - platform_fee
        profit_loss_percent = ((price - avg_buy_price) / avg_buy_price) * 100
        
        # Update balance (add net revenue after fees)
        self.balance += net_revenue
        
        # Update position
        self.positions[base_currency]['amount'] -= amount
        
        # Remove position if amount is zero or very close to zero
        if self.positions[base_currency]['amount'] < 0.00000001:
            del self.positions[base_currency]
        
        # Calculate tax if enabled and profitable
        tax_amount = 0.0
        net_profit_after_tax = net_profit_loss
        
        if net_profit_loss > 0 and self.tax_calculator:
            # Calculate annual profits for tax calculation (use net profits after fees)
            annual_profits = sum(trade.get('net_profit_loss', 0) for trade in self.trade_history 
                               if trade.get('net_profit_loss', 0) > 0)
            
            tax_amount, net_profit_after_tax, tax_percentage = self.tax_calculator.calculate_tax(
                net_profit_loss, annual_profits
            )
        
        # Record trade in history
        trade = {
            'timestamp': datetime.now(),
            'type': 'SELL',
            'symbol': symbol,
            'amount': amount,
            'price': price,
            'total': revenue,
            'platform_fee': platform_fee,
            'net_revenue': net_revenue,
            'gross_profit_loss': gross_profit_loss,
            'net_profit_loss': net_profit_loss,
            'profit_loss_percent': profit_loss_percent,
            'tax_amount': tax_amount,
            'net_profit_after_tax': net_profit_after_tax
        }
        self.trade_history.append(trade)
        
        # Print trade confirmation
        profit_symbol = "(PROFIT)" if net_profit_loss >= 0 else "(LOSS)"
        print(f"\n{'-'*60}")
        print(f"SELL ORDER EXECUTED {profit_symbol}")
        print(f"{'-'*60}")
        print(f"Symbol:         {symbol}")
        print(f"Amount:         {amount:.8f} {base_currency}")
        print(f"Sell Price:     ${price:,.2f}")
        print(f"Avg Buy Price:  ${avg_buy_price:,.2f}")
        print(f"Total Revenue:  ${revenue:,.2f}")
        if platform_fee > 0:
            print(f"Platform Fee:   ${platform_fee:,.2f} ({config.BINANCE_SELL_FEE_PERCENT}%)")
            print(f"Net Revenue:    ${net_revenue:,.2f}")
        print(f"Gross P/L:      ${gross_profit_loss:,.2f} ({profit_loss_percent:+.2f}%)")
        if platform_fee > 0:
            print(f"Net P/L:        ${net_profit_loss:,.2f}")
        
        # Show tax information if applicable
        if net_profit_loss > 0 and self.tax_calculator:
            print(f"Tax ({self.tax_calculator.country}):     ${tax_amount:,.2f}")
            print(f"Final P/L:     ${net_profit_after_tax:,.2f}")
            # Show detailed tax breakdown
            self.tax_calculator.display_tax_breakdown(net_profit_loss)
        
        print(f"New Balance:    ${self.balance:,.2f} USDT")
        print(f"{'-'*60}\n")
        
        return True
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value (balance + positions at current prices).
        
        Args:
            current_prices: Dictionary of current prices for each position
        
        Returns:
            Total portfolio value in USDT
        """
        total_value = self.balance
        
        for currency, position in self.positions.items():
            symbol = position['symbol']
            if symbol in current_prices:
                position_value = position['amount'] * current_prices[symbol]
                total_value += position_value
        
        return total_value
    
    def get_total_profit_loss(self, current_prices: Dict[str, float]) -> tuple:
        """
        Calculate total profit/loss compared to initial balance.
        
        Args:
            current_prices: Dictionary of current prices for each position
        
        Returns:
            Tuple of (profit_loss_amount, profit_loss_percent)
        """
        current_value = self.get_portfolio_value(current_prices)
        profit_loss = current_value - self.initial_balance
        profit_loss_percent = (profit_loss / self.initial_balance) * 100
        
        return profit_loss, profit_loss_percent
    
    def display_portfolio(self, current_prices: Dict[str, float] = None):
        """
        Display current portfolio status with all positions and balances.
        
        Args:
            current_prices: Dictionary of current prices for each position
        """
        print(f"\n{'='*60}")
        print(f"PORTFOLIO SUMMARY")
        print(f"{'='*60}")
        print(f"Cash Balance: ${self.balance:,.2f} USDT")
        
        if self.positions:
            print(f"\n{'-'*60}")
            print(f"OPEN POSITIONS:")
            print(f"{'-'*60}")
            
            total_positions_value = 0
            for currency, position in self.positions.items():
                amount = position['amount']
                avg_buy_price = position['avg_buy_price']
                symbol = position['symbol']
                
                print(f"\n{currency}:")
                print(f"  Amount:          {amount:.8f}")
                print(f"  Avg Buy Price:   ${avg_buy_price:,.2f}")
                print(f"  Total Cost:      ${amount * avg_buy_price:,.2f}")
                
                if current_prices and symbol in current_prices:
                    current_price = current_prices[symbol]
                    current_value = amount * current_price
                    unrealized_pl = current_value - (amount * avg_buy_price)
                    unrealized_pl_percent = ((current_price - avg_buy_price) / avg_buy_price) * 100
                    
                    print(f"  Current Price:   ${current_price:,.2f}")
                    print(f"  Current Value:   ${current_value:,.2f}")
                    print(f"  Unrealized P/L:  ${unrealized_pl:,.2f} ({unrealized_pl_percent:+.2f}%)")
                    
                    total_positions_value += current_value
        else:
            print("\nNo open positions")
        
        if current_prices:
            total_value = self.get_portfolio_value(current_prices)
            profit_loss, profit_loss_percent = self.get_total_profit_loss(current_prices)
            
            print(f"\n{'-'*60}")
            print(f"TOTAL PORTFOLIO VALUE: ${total_value:,.2f}")
            print(f"Initial Balance:       ${self.initial_balance:,.2f}")
            print(f"Total P/L:             ${profit_loss:,.2f} ({profit_loss_percent:+.2f}%)")
        
        print(f"{'='*60}\n")
    
    def display_annual_tax_summary(self):
        """
        Display annual tax summary for all trades.
        """
        if not self.tax_calculator or not self.trade_history:
            return
        
        # Calculate annual totals (use net profits after fees)
        total_profits = sum(trade.get('net_profit_loss', 0) for trade in self.trade_history 
                           if trade.get('net_profit_loss', 0) > 0)
        total_losses = sum(abs(trade.get('net_profit_loss', 0)) for trade in self.trade_history 
                          if trade.get('net_profit_loss', 0) < 0)
        
        summary = self.tax_calculator.get_annual_summary(
            len(self.trade_history), total_profits, total_losses
        )
        
        print(f"\n{'='*60}")
        print(f"ANNUAL TAX SUMMARY ({self.tax_calculator.country})")
        print(f"{'='*60}")
        print(f"Total Trades:     {summary['total_trades']}")
        print(f"Gross Profits:    ${total_profits:,.2f}")
        print(f"Gross Losses:     ${total_losses:,.2f}")
        print(f"Net Profit:       ${summary['net_profit']:,.2f}")
        print(f"Tax Due:          ${summary['tax_due']:,.2f}")
        print(f"Net After Tax:    ${summary['net_after_tax']:,.2f}")
        print(f"Effective Tax Rate: {summary['tax_percentage']:.1f}%")
        print(f"{'='*60}\n")
    
    def display_trade_history(self, last_n: int = 10):
        """
        Display recent trade history.
        
        Args:
            last_n: Number of recent trades to display
        """
        if not self.trade_history:
            print("\nNo trades executed yet")
            return
        
        print(f"\n{'='*60}")
        print(f"TRADE HISTORY (Last {min(last_n, len(self.trade_history))} trades)")
        print(f"{'='*60}")
        
        for trade in self.trade_history[-last_n:]:
            print(f"\n{trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | {trade['type']}")
            print(f"  {trade['symbol']}: {trade['amount']:.8f} @ ${trade['price']:,.2f}")
            
            if trade['type'] == 'BUY':
                print(f"  Subtotal: ${trade['total']:,.2f}")
                if trade.get('platform_fee', 0) > 0:
                    print(f"  Fee: ${trade['platform_fee']:,.2f}")
                print(f"  Total: ${trade.get('total_cost', trade['total']):,.2f}")
            else:  # SELL
                print(f"  Revenue: ${trade['total']:,.2f}")
                if trade.get('platform_fee', 0) > 0:
                    print(f"  Fee: ${trade['platform_fee']:,.2f}")
                    print(f"  Net: ${trade.get('net_revenue', trade['total']):,.2f}")
                if 'gross_profit_loss' in trade:
                    print(f"  Gross P/L: ${trade['gross_profit_loss']:,.2f} ({trade['profit_loss_percent']:+.2f}%)")
                if 'net_profit_loss' in trade and trade.get('platform_fee', 0) > 0:
                    print(f"  Net P/L: ${trade['net_profit_loss']:,.2f}")
                if 'net_profit_after_tax' in trade and trade.get('tax_amount', 0) > 0:
                    print(f"  Final P/L: ${trade['net_profit_after_tax']:,.2f}")
        
        print(f"{'='*60}\n")
    
    def export_trade_history_to_csv(self, filename: str = None):
        """
        Export trade history to CSV file.
        
        Args:
            filename: Optional custom filename. Defaults to 'trade_history_YYYY-MM-DD.csv'
        """
        if not self.trade_history:
            print("\nNo trades to export")
            return
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"trade_history_{timestamp}.csv"
        
        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'timestamp', 'type', 'symbol', 'amount', 'price', 'total',
                    'platform_fee', 'total_cost', 'net_revenue', 'gross_profit_loss',
                    'profit_loss_percent', 'net_profit_loss', 'tax_amount', 'net_profit_after_tax'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for trade in self.trade_history:
                    # Create a clean row with all possible fields
                    row = {
                        'timestamp': trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                        'type': trade['type'],
                        'symbol': trade['symbol'],
                        'amount': trade['amount'],
                        'price': trade['price'],
                        'total': trade['total'],
                        'platform_fee': trade.get('platform_fee', 0),
                        'total_cost': trade.get('total_cost', trade['total']),
                        'net_revenue': trade.get('net_revenue', trade['total']),
                        'gross_profit_loss': trade.get('gross_profit_loss', ''),
                        'profit_loss_percent': trade.get('profit_loss_percent', ''),
                        'net_profit_loss': trade.get('net_profit_loss', ''),
                        'tax_amount': trade.get('tax_amount', 0),
                        'net_profit_after_tax': trade.get('net_profit_after_tax', '')
                    }
                    writer.writerow(row)
            
            print(f"\n✅ Trade history exported to: {filename}")
            print(f"📊 Total trades exported: {len(self.trade_history)}")
            
        except Exception as e:
            print(f"\n❌ Error exporting trade history: {e}")
    
    def save_trade_history_to_file(self, filename: str = None):
        """
        Save trade history to a text file with formatted output.
        
        Args:
            filename: Optional custom filename. Defaults to 'trade_history_YYYY-MM-DD.txt'
        """
        if not self.trade_history:
            print("\nNo trades to save")
            return
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"trade_history_{timestamp}.txt"
        
        # Ensure filename has .txt extension
        if not filename.endswith('.txt'):
            filename += '.txt'
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("CRYPTO TRADING BOT - TRADE HISTORY\n")
                f.write("="*60 + "\n")
                f.write(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Trades: {len(self.trade_history)}\n")
                f.write("="*60 + "\n\n")
                
                for i, trade in enumerate(self.trade_history, 1):
                    f.write(f"Trade #{i}\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Date: {trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Type: {trade['type']}\n")
                    f.write(f"Symbol: {trade['symbol']}\n")
                    f.write(f"Amount: {trade['amount']:.8f}\n")
                    f.write(f"Price: ${trade['price']:,.2f}\n")
                    f.write(f"Total: ${trade['total']:,.2f}\n")
                    
                    if trade.get('platform_fee', 0) > 0:
                        f.write(f"Platform Fee: ${trade['platform_fee']:,.2f}\n")
                    
                    if trade['type'] == 'BUY':
                        f.write(f"Total Cost: ${trade.get('total_cost', trade['total']):,.2f}\n")
                    else:  # SELL
                        f.write(f"Net Revenue: ${trade.get('net_revenue', trade['total']):,.2f}\n")
                        if 'gross_profit_loss' in trade:
                            f.write(f"Gross P/L: ${trade['gross_profit_loss']:,.2f} ({trade['profit_loss_percent']:+.2f}%)\n")
                        if 'net_profit_loss' in trade and trade.get('platform_fee', 0) > 0:
                            f.write(f"Net P/L: ${trade['net_profit_loss']:,.2f}\n")
                        if 'net_profit_after_tax' in trade and trade.get('tax_amount', 0) > 0:
                            f.write(f"Final P/L: ${trade['net_profit_after_tax']:,.2f}\n")
                    
                    f.write("\n")
            
            print(f"\n✅ Trade history saved to: {filename}")
            print(f"📊 Total trades saved: {len(self.trade_history)}")
            
        except Exception as e:
            print(f"\n❌ Error saving trade history: {e}")

