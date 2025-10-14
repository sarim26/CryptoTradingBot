"""
Tax Calculator Module
Calculates taxes for different countries based on trading profits.
"""

from typing import Dict, Tuple
import config


class TaxCalculator:
    """
    Calculates taxes on trading profits based on country-specific rules.
    """
    
    def __init__(self, country: str = 'UK'):
        """
        Initialize tax calculator for specific country.
        
        Args:
            country: Country code for tax calculations
        """
        self.country = country
        self.tax_rules = self._get_tax_rules(country)
        
        print(f"\n{'='*60}")
        print(f"Tax Calculator Initialized ({country})")
        print(f"{'='*60}")
        print(f"Capital Gains Rate: {self.tax_rules['capital_gains_rate']}%")
        print(f"Annual Allowance: £{self.tax_rules['annual_allowance']:,.0f}")
        print(f"Currency: {self.tax_rules['currency']}")
        print(f"{'='*60}\n")
    
    def _get_tax_rules(self, country: str) -> Dict:
        """
        Get tax rules for specific country.
        
        Args:
            country: Country code
        
        Returns:
            Dictionary with tax rules
        """
        tax_rules = {
            'UK': {
                'capital_gains_rate': 20,  # Higher rate (adjust if you're basic rate)
                'annual_allowance': 3000,  # £3,000 for 2024-25
                'currency': 'GBP',
                'exchange_rate': 1.25,  # USD to GBP (approximate)
                'income_tax_threshold': 50000,  # If you trade frequently
            },
            'US': {
                'capital_gains_rate': 15,  # Long-term (1+ year) or 22% short-term
                'annual_allowance': 0,  # No allowance in US
                'currency': 'USD',
                'exchange_rate': 1.0,
            },
            'INDIA': {
                'capital_gains_rate': 30,  # As you mentioned
                'annual_allowance': 0,
                'currency': 'INR',
                'exchange_rate': 83.0,  # USD to INR (approximate)
            },
            'NONE': {
                'capital_gains_rate': 0,
                'annual_allowance': 0,
                'currency': 'USD',
                'exchange_rate': 1.0,
            }
        }
        
        return tax_rules.get(country.upper(), tax_rules['NONE'])
    
    def calculate_tax(self, gross_profit_usd: float, annual_profits_usd: float = 0) -> Tuple[float, float, float]:
        """
        Calculate tax on trading profit.
        
        Args:
            gross_profit_usd: Profit from this trade in USD
            annual_profits_usd: Total profits this year in USD
        
        Returns:
            Tuple of (tax_amount_usd, net_profit_usd, tax_percentage)
        """
        if self.country == 'NONE':
            return 0.0, gross_profit_usd, 0.0
        
        # Convert to local currency
        local_currency = self.tax_rules['currency']
        exchange_rate = self.tax_rules['exchange_rate']
        
        gross_profit_local = gross_profit_usd * exchange_rate
        annual_profits_local = annual_profits_usd * exchange_rate
        
        # Calculate taxable amount (after allowance)
        annual_allowance = self.tax_rules['annual_allowance']
        total_annual_local = annual_profits_local + gross_profit_local
        
        if total_annual_local <= annual_allowance:
            # Within allowance, no tax
            tax_amount_local = 0.0
            tax_percentage = 0.0
        else:
            # Calculate tax on amount above allowance
            if annual_profits_local < annual_allowance:
                # This trade pushes us over the allowance
                taxable_amount = total_annual_local - annual_allowance
            else:
                # Already over allowance, tax full amount
                taxable_amount = gross_profit_local
            
            tax_rate = self.tax_rules['capital_gains_rate'] / 100
            tax_amount_local = taxable_amount * tax_rate
            tax_percentage = (tax_amount_local / gross_profit_local) * 100 if gross_profit_local > 0 else 0
        
        # Convert tax back to USD
        tax_amount_usd = tax_amount_local / exchange_rate
        net_profit_usd = gross_profit_usd - tax_amount_usd
        
        return tax_amount_usd, net_profit_usd, tax_percentage
    
    def display_tax_breakdown(self, gross_profit_usd: float, annual_profits_usd: float = 0):
        """
        Display detailed tax breakdown.
        
        Args:
            gross_profit_usd: Profit from this trade in USD
            annual_profits_usd: Total profits this year in USD
        """
        if gross_profit_usd <= 0:
            print(f"\n{'-'*60}")
            print(f"TAX BREAKDOWN (No profit to tax)")
            print(f"{'-'*60}\n")
            return
        
        tax_amount_usd, net_profit_usd, tax_percentage = self.calculate_tax(gross_profit_usd, annual_profits_usd)
        
        # Convert to local currency for display
        local_currency = self.tax_rules['currency']
        exchange_rate = self.tax_rules['exchange_rate']
        
        gross_local = gross_profit_usd * exchange_rate
        tax_local = tax_amount_usd * exchange_rate
        net_local = net_profit_usd * exchange_rate
        
        print(f"\n{'-'*60}")
        print(f"TAX BREAKDOWN ({self.country})")
        print(f"{'-'*60}")
        print(f"Gross Profit:     ${gross_profit_usd:,.2f} USD ({local_currency}{gross_local:,.2f})")
        print(f"Tax Rate:         {tax_percentage:.1f}%")
        print(f"Tax Amount:       ${tax_amount_usd:,.2f} USD ({local_currency}{tax_local:,.2f})")
        print(f"Net Profit:       ${net_profit_usd:,.2f} USD ({local_currency}{net_local:,.2f})")
        print(f"Annual Allowance: {local_currency}{self.tax_rules['annual_allowance']:,.0f}")
        print(f"{'-'*60}\n")
    
    def get_annual_summary(self, total_trades: int, total_profits_usd: float, total_losses_usd: float):
        """
        Get annual tax summary.
        
        Args:
            total_trades: Number of trades this year
            total_profits_usd: Total profits in USD
            total_losses_usd: Total losses in USD
        
        Returns:
            Tax summary information
        """
        net_profit_usd = total_profits_usd - total_losses_usd
        
        if net_profit_usd <= 0:
            return {
                'net_profit': net_profit_usd,
                'tax_due': 0.0,
                'net_after_tax': net_profit_usd,
                'tax_percentage': 0.0
            }
        
        tax_amount_usd, net_after_tax_usd, tax_percentage = self.calculate_tax(net_profit_usd, 0)
        
        return {
            'net_profit': net_profit_usd,
            'tax_due': tax_amount_usd,
            'net_after_tax': net_after_tax_usd,
            'tax_percentage': tax_percentage,
            'total_trades': total_trades
        }
