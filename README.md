Stock Analyzer
A terminal-based stock analysis tool that evaluates any ticker across 5 structured steps using data from Yahoo Finance.

Features
Step	Analysis
1. Fundamentals	P/E, EPS, Revenue, Debt/Equity, Dividend Yield, P/B, Market Cap
2. Price & Volume	52-week range, Moving Averages (50/200-day), Volume, Support/Resistance
3. Momentum	RSI, MACD, Bollinger Bands
4. Market Context	Sector ETF comparison, S&P 500 correlation
5. Risk Metrics	Beta, Std Dev, Max Drawdown, Sharpe Ratio, Short Interest
Each sub-metric is scored 0–100 and rolled up into an overall rating (Poor → Weak → Neutral → Good → Strong).

Installation
git clone https://github.com/pabloresgo/stock_analyzer.git
cd stock_analyzer
pip install -r requirements.txt
pandas-ta is optional but recommended for more accurate RSI calculations. If it's not installed, a built-in fallback is used automatically.

Usage
# Basic analysis (1-year history)
python stock_analyzer.py NVDA

# Custom history period
python stock_analyzer.py TD --period 2y

# Output as JSON (for scripting/piping)
python stock_analyzer.py AAPL --json
Arguments
Argument	Default	Description
ticker	(required)	Stock ticker symbol (e.g. AAPL, NVDA, TD)
--period	1y	History window: 6mo, 1y, 2y, 5y
--json	off	Print output as JSON instead of formatted text
Example Output
============================================================
  STOCK ANALYSIS: NVDA
  Date: 2026-05-13 10:00
  Period: 1y
============================================================

  Fetching data from Yahoo Finance... ✓ NVIDIA Corporation

  ────────────────────────────────────────────────────────
  OVERALL SCORE: ████████████████░░░░ 80/100 (Good)
  Current Price: $105.42
  ────────────────────────────────────────────────────────

  🔵 STEP 1: FUNDAMENTAL ANALYSIS
     Score: ████████████████░░░░ 78/100 (Good)
  ...
Requirements
Python 3.8+
yfinance >= 0.2.0
pandas >= 2.0.0
pandas-ta >= 0.3.14b (optional)
Disclaimer
This tool is for informational and educational purposes only. It is not financial advice. Always do your own research before making investment decisions.

License
MIT — see LICENSE.
