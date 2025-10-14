# 🚀 Quick Start Guide

Get your crypto trading bot running in 3 minutes!

## Step 1: Install Dependencies

Open your terminal and run:

```bash
pip install -r requirements.txt
```

This installs:
- `ccxt` - Connects to cryptocurrency exchanges
- `python-dotenv` - Manages configuration
- `colorama` - Makes terminal output prettier

## Step 2: Run the Bot

```bash
python trading_bot.py
```

That's it! The bot will start in **paper trading mode** (no real money).

## Step 3: Try It Out

When the bot starts, you'll see a menu:

```
1. Start Trading    ← Pick this!
2. View Portfolio
3. View Trade History
4. Change Trading Pair
5. Adjust Strategy Parameters
6. Exit
```

### What to do:

1. **Press `1`** to start trading
2. **Select a crypto** - Try BTC/USDT first (option 1)
3. **Watch it work!** The bot will:
   - Monitor prices every 10 seconds
   - Buy when price drops 2%
   - Sell when price rises 3%
   - Show you all trades and profits

### Stop the bot:

Press **`Ctrl + C`** to stop safely. The bot will show you:
- Final portfolio value
- All trades executed
- Total profit/loss

## Understanding the Output

When running, you'll see:

```
⏰ 2025-10-12 14:30:45
──────────────────────────────────────────────
Symbol:        BTC/USDT
Current Price: $49,500.00
Reference:     $50,000.00 (-1.00%)

Cash Balance:  $10,000.00 USDT
Total Value:   $10,000.00
Total P/L:     $0.00 (+0.00%)
──────────────────────────────────────────────
```

**What this means:**
- **Current Price**: What BTC costs right now
- **Reference**: Price we're watching for buy signals
- **Cash Balance**: Virtual money available
- **Total P/L**: Your profit or loss

## Tips for First-Time Users

### 1. Be Patient
The bot won't trade immediately. It needs to wait for:
- Price to drop 2% (to trigger a BUY)
- Then rise 3% from buy price (to trigger a SELL)

### 2. Test Different Settings
Press `5` in the menu to adjust:
- Buy threshold (try 1% for faster action)
- Sell threshold (try 2% for quicker profits)

### 3. Watch Real Market Data
The bot uses **real live prices** from Binance, so you're seeing actual market conditions!

### 4. Check Your Portfolio
While bot is running, press `Ctrl+C` to stop, then:
- Press `2` to see your portfolio
- Press `3` to see trade history

## Example Trading Cycle

Here's what happens in a typical trade:

1. **Bot starts monitoring BTC at $50,000**
   ```
   Current Price: $50,000.00
   Reference:     $50,000.00
   ```

2. **Price drops to $49,000 (-2%)**
   ```
   🔔 BUY SIGNAL TRIGGERED for BTC/USDT
   ✓ BUY ORDER EXECUTED
   Amount:      0.02040816 BTC
   Price:       $49,000.00
   Total Cost:  $1,000.00
   ```

3. **Price rises to $50,470 (+3% from buy price)**
   ```
   🔔 SELL SIGNAL TRIGGERED for BTC/USDT
   ✓ SELL ORDER EXECUTED 📈
   Sell Price:     $50,470.00
   Avg Buy Price:  $49,000.00
   Profit/Loss:    $30.00 (+3.00%)
   ```

4. **Profit made: $30!** 🎉

## Common Questions

### Q: Is this using real money?
**A:** No! By default, it's paper trading with virtual money ($10,000 USDT).

### Q: Can I trade other cryptocurrencies?
**A:** Yes! Choose option `4` in the menu to switch to ETH, SOL, DOGE, etc.

### Q: How do I make it trade faster?
**A:** Choose option `5` and lower the percentages:
- Buy threshold: 1% (instead of 2%)
- Sell threshold: 1.5% (instead of 3%)

### Q: Can I use real money?
**A:** Yes, but **not recommended yet**! First:
1. Test in paper trading for at least a week
2. Understand how the strategy works
3. Get Binance API keys
4. Start with small amounts ($50-100)

### Q: The bot isn't trading. Is it broken?
**A:** Probably not! The bot waits for the right conditions. If the price doesn't drop 2%, it won't buy. Try lowering the threshold to 1% for more action.

## Next Steps

Once you're comfortable:

1. **Try different cryptos** - Some are more volatile (more trades)
2. **Adjust strategy** - Find what works best
3. **Learn more** - Read about technical indicators (RSI, MACD)
4. **Expand the bot** - Add your own features!

## Need Help?

Check the main **README.md** for:
- Detailed explanations
- Troubleshooting guide
- Customization ideas
- Live trading setup

---

**Ready?** Run this command and start trading:
```bash
python trading_bot.py
```

Good luck! 📈🚀

