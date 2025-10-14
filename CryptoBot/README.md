# 🤖 Crypto Trading Bot

A simple yet powerful cryptocurrency trading bot built in Python. This bot connects to Binance (or other exchanges) and executes automated trades based on percentage-based price movements.

## ✨ Features

- **Paper Trading Mode**: Test strategies with virtual money (no real trades)
- **Live Trading Mode**: Execute real trades on supported exchanges
- **Percentage-Based Strategy**: Buy on X% price drops, sell on Y% price increases
- **Real-Time Market Data**: Fetches live prices from exchange APIs
- **Portfolio Management**: Tracks balance, positions, and profit/loss
- **Trade History**: Complete log of all executed trades
- **Interactive Menu**: Easy-to-use interface for bot control
- **Configurable**: Adjust all parameters without touching code

## 📋 Requirements

- Python 3.8 or higher
- Internet connection
- (Optional) Binance API keys for live trading

## 🚀 Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the bot**:
   - The bot comes pre-configured for paper trading
   - Edit `.env` file to change settings:
     ```
     TRADING_MODE=paper  # Change to 'live' for real trading
     INITIAL_BALANCE=10000  # Starting balance for paper trading
     ```

4. **(Optional) For live trading**:
   - Get API keys from Binance: https://www.binance.com/en/my/settings/api-management
   - Add them to `.env`:
     ```
     BINANCE_API_KEY=your_key_here
     BINANCE_API_SECRET=your_secret_here
     TRADING_MODE=live
     ```
   - ⚠️ **WARNING**: Start with small amounts in live mode!

## 🎮 Usage

### Quick Start

Run the bot:
```bash
python trading_bot.py
```

The bot will show an interactive menu where you can:
1. Start trading
2. View your portfolio
3. Check trade history
4. Change trading pair
5. Adjust strategy parameters

### Paper Trading (Recommended First)

Paper trading lets you test the bot with virtual money:
- No risk
- No API keys needed
- Same real-time market data
- Perfect for learning and testing strategies

### Live Trading

⚠️ **Only use live trading after thorough testing in paper mode!**

To switch to live trading:
1. Get Binance API keys
2. Add them to `.env`
3. Set `TRADING_MODE=live`
4. Start with small amounts

## ⚙️ Configuration

Edit `config.py` to customize the bot:

```python
# Trading Strategy
BUY_DROP_PERCENTAGE = 2.0      # Buy when price drops 2%
SELL_INCREASE_PERCENTAGE = 3.0  # Sell when price increases 3%
TRADE_PERCENTAGE = 10          # Use 10% of balance per trade

# Data Fetching
PRICE_CHECK_INTERVAL = 10      # Check price every 10 seconds

# Default Trading Pair
DEFAULT_SYMBOL = 'BTC/USDT'    # Change to trade other cryptos
```

## 📊 How It Works

### Strategy Logic

1. **Monitoring**: Bot continuously fetches real-time prices
2. **Buy Signal**: When price drops by X% from reference price
   - Calculate buy amount (% of available balance)
   - Execute buy order
   - Update reference price
3. **Sell Signal**: When price increases by Y% from average buy price
   - Sell entire position
   - Calculate profit/loss
   - Reset for next opportunity

### Example

With default settings (2% drop to buy, 3% rise to sell):

1. BTC price: $50,000 (reference price set)
2. Price drops to $49,000 (-2%) → **BUY SIGNAL** 
   - Bot buys BTC with 10% of balance
3. Price rises to $50,470 (+3% from buy) → **SELL SIGNAL**
   - Bot sells all BTC
   - Records profit

## 📁 Project Structure

```
CryptoBot/
│
├── trading_bot.py          # Main bot orchestration
├── exchange_connector.py   # Exchange API communication
├── portfolio.py            # Portfolio and balance management
├── strategy.py             # Trading strategy logic
├── config.py               # Configuration parameters
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (your settings)
└── README.md              # This file
```

## 🔧 Customization Ideas

Once you're comfortable with the basics, try:

1. **Different Strategies**:
   - Add stop-loss protection
   - Implement trailing stops
   - Use time-based rules (e.g., don't trade on weekends)

2. **Technical Indicators**:
   - Add RSI (Relative Strength Index)
   - Use Moving Averages (MA, EMA)
   - Implement MACD

3. **Machine Learning**:
   - Train models on historical data
   - Predict price movements
   - Optimize strategy parameters

4. **Risk Management**:
   - Position sizing based on volatility
   - Maximum drawdown limits
   - Portfolio diversification

## ⚠️ Important Warnings

1. **Cryptocurrency trading is risky** - Only invest what you can afford to lose
2. **Start with paper trading** - Test thoroughly before using real money
3. **Markets are unpredictable** - Past performance doesn't guarantee future results
4. **Use small amounts** - When starting live trading, use minimal capital
5. **Monitor regularly** - Don't leave the bot running unattended for long periods
6. **Keep API keys secure** - Never share your API keys or commit them to git

## 🐛 Troubleshooting

### "Error initializing exchange"
- Check internet connection
- Verify exchange name is correct (default: binance)
- For live mode, check API keys are valid

### "Insufficient balance"
- Increase `INITIAL_BALANCE` in `.env` for paper trading
- For live trading, ensure you have USDT in your account

### "No position found"
- Normal message - means you don't own any of that crypto yet
- Bot will buy when price drops enough

### Price not updating
- Check `PRICE_CHECK_INTERVAL` in config.py
- Verify internet connection
- Exchange might be rate-limiting (bot has protection for this)

## 📚 Next Steps

1. **Learn the basics**: Run in paper trading mode and watch how it works
2. **Experiment**: Adjust strategy parameters and see how they affect performance
3. **Study markets**: Understand why prices move (news, trends, volume)
4. **Add features**: Implement your own ideas (see Customization Ideas)
5. **Backtest**: Test strategies on historical data
6. **Go live carefully**: Start with small amounts when ready

## 📝 License

This project is provided for educational purposes. Use at your own risk.

## 🤝 Contributing

Feel free to fork this project and add your own features! Some ideas:
- Support for more exchanges (Coinbase, Kraken, etc.)
- Web dashboard for monitoring
- Telegram/Discord notifications
- More advanced strategies
- Backtesting framework

## 💡 Tips for Success

1. **Start simple**: Master the basics before adding complexity
2. **Keep learning**: Study technical analysis and market dynamics
3. **Stay disciplined**: Don't override the bot with emotional decisions
4. **Track everything**: Review your trade history regularly
5. **Be patient**: Good opportunities take time to develop

Happy trading! 🚀📈

