#!/usr/bin/env python3
"""
Stock Analysis Tool — 5-Step Framework
=======================================
Analyzes any stock ticker across 5 steps:
  1. Fundamental Analysis (P/E, EPS, Revenue, D/E, Dividend, P/B, Market Cap)
  2. Price & Volume (52-week range, Moving Averages, Volume, Support/Resistance)
  3. Momentum Indicators (RSI, MACD, Bollinger Bands)
  4. Market Context (Sector ETF comparison, S&P 500 correlation)
  5. Risk Metrics (Beta, Std Dev, Max Drawdown, Sharpe Ratio, Short Interest)

Usage:
  python stock_analyzer.py NVDA
  python stock_analyzer.py TD --period 2y
  python stock_analyzer.py AAPL --json

Requirements:
  pip install yfinance pandas-ta pandas
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

try:
    import pandas_ta as ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

warnings.filterwarnings("ignore")

# ── Sector ETF mapping ─────────────────────────────────────────────
SECTOR_ETFS = {
    "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Energy": "XLE",
    "Industrials": "XLI", "Utilities": "XLU", "Real Estate": "XLRE",
    "Communication Services": "XLC", "Basic Materials": "XLB",
}


def score_label(score):
    """Convert a 0-100 score to a human label."""
    if score >= 80: return "Strong"
    if score >= 60: return "Good"
    if score >= 40: return "Neutral"
    if score >= 20: return "Weak"
    return "Poor"


def safe_get(info, key, default=None):
    """Safely get a value from the yfinance info dict."""
    val = info.get(key, default)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return val


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — FUNDAMENTAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════
def analyze_fundamentals(info, financials):
    results = {}
    scores = []

    # P/E Ratio
    pe = safe_get(info, "trailingPE")
    forward_pe = safe_get(info, "forwardPE")
    if pe is not None:
        if pe < 0:
            s = 10; verdict = "Negative earnings — company losing money"
        elif pe < 10:
            s = 75; verdict = "Very cheap — may be undervalued or slow-growth"
        elif pe < 16:
            s = 85; verdict = "Undervalued — attractive price for earnings"
        elif pe <= 25:
            s = 70; verdict = "Reasonable — fairly priced"
        elif pe <= 35:
            s = 45; verdict = "Pricey — high expectations baked in"
        else:
            s = 30; verdict = "Expensive — needs strong growth to justify"
        results["P/E Ratio"] = {"value": round(pe, 2), "forward_pe": round(forward_pe, 2) if forward_pe else None, "score": s, "verdict": verdict}
        scores.append(s)
    else:
        results["P/E Ratio"] = {"value": None, "score": None, "verdict": "Not available"}

    # EPS
    eps = safe_get(info, "trailingEps")
    eps_growth = None
    if financials is not None and not financials.empty:
        try:
            eps_row = None
            for label in ["Basic EPS", "Diluted EPS"]:
                if label in financials.index:
                    eps_row = financials.loc[label].dropna().sort_index()
                    break
            if eps_row is not None and len(eps_row) >= 2:
                old, new = eps_row.iloc[0], eps_row.iloc[-1]
                if old != 0:
                    eps_growth = ((new - old) / abs(old)) * 100
        except Exception:
            pass
    if eps is not None:
        if eps_growth is not None and eps_growth > 15:
            s = 85; verdict = f"Strong growth ({eps_growth:+.1f}% over reporting period)"
        elif eps_growth is not None and eps_growth > 0:
            s = 65; verdict = f"Positive growth ({eps_growth:+.1f}%)"
        elif eps_growth is not None and eps_growth > -10:
            s = 40; verdict = f"Flat or slight decline ({eps_growth:+.1f}%)"
        elif eps_growth is not None:
            s = 20; verdict = f"Declining ({eps_growth:+.1f}%) — investigate cause"
        else:
            s = 50; verdict = "EPS available but trend data insufficient"
        results["EPS"] = {"value": round(eps, 2), "growth_pct": round(eps_growth, 2) if eps_growth else None, "score": s, "verdict": verdict}
        scores.append(s)
    else:
        results["EPS"] = {"value": None, "score": None, "verdict": "Not available"}

    # Revenue
    revenue = safe_get(info, "totalRevenue")
    rev_growth = safe_get(info, "revenueGrowth")
    if revenue is not None:
        rev_b = revenue / 1e9
        if rev_growth is not None:
            rg = rev_growth * 100
            if rg > 20: s = 90; verdict = f"Exceptional revenue growth ({rg:+.1f}%)"
            elif rg > 5: s = 70; verdict = f"Healthy growth ({rg:+.1f}%)"
            elif rg > 0: s = 55; verdict = f"Modest growth ({rg:+.1f}%)"
            else: s = 30; verdict = f"Revenue declining ({rg:+.1f}%)"
        else:
            s = 50; verdict = "Revenue data available but growth rate unknown"
        results["Revenue"] = {"value_billions": round(rev_b, 2), "growth_pct": round(rev_growth * 100, 2) if rev_growth else None, "score": s, "verdict": verdict}
        scores.append(s)
    else:
        results["Revenue"] = {"value_billions": None, "score": None, "verdict": "Not available"}

    # Debt/Equity
    de = safe_get(info, "debtToEquity")
    sector = safe_get(info, "sector", "")
    if de is not None:
        de_ratio = de / 100 if de > 10 else de  # yfinance sometimes returns as percentage
        is_bank = "Financial" in sector or "Bank" in safe_get(info, "industry", "")
        if is_bank:
            s = 60; verdict = f"D/E = {de_ratio:.2f} — banks carry high debt by design, check Tier 1 capital instead"
        elif de_ratio < 0.5:
            s = 90; verdict = f"Very low debt ({de_ratio:.2f}) — conservative balance sheet"
        elif de_ratio < 1:
            s = 75; verdict = f"Low debt ({de_ratio:.2f}) — healthy"
        elif de_ratio < 2:
            s = 55; verdict = f"Moderate debt ({de_ratio:.2f}) — acceptable"
        else:
            s = 25; verdict = f"High debt ({de_ratio:.2f}) — risky if rates rise"
        results["Debt/Equity"] = {"value": round(de_ratio, 2), "is_bank": is_bank, "score": s, "verdict": verdict}
        scores.append(s)
    else:
        results["Debt/Equity"] = {"value": None, "score": None, "verdict": "Not available"}

    # Dividend Yield
    div_yield = safe_get(info, "dividendYield")
    if div_yield is not None:
        dy = div_yield * 100
        if dy == 0:
            s = 50; verdict = "No dividend — growth company reinvesting profits"
        elif dy < 2:
            s = 55; verdict = f"Low yield ({dy:.2f}%) — token dividend"
        elif dy <= 5:
            s = 80; verdict = f"Solid income ({dy:.2f}%) — good for TFSA/RRSP"
        elif dy <= 7:
            s = 65; verdict = f"High yield ({dy:.2f}%) — verify sustainability"
        else:
            s = 30; verdict = f"Suspiciously high ({dy:.2f}%) — may be cut"
        results["Dividend Yield"] = {"value_pct": round(dy, 2), "score": s, "verdict": verdict}
        scores.append(s)
    else:
        results["Dividend Yield"] = {"value_pct": 0, "score": 50, "verdict": "No dividend — growth company"}
        scores.append(50)

    # P/B Ratio
    pb = safe_get(info, "priceToBook")
    if pb is not None:
        if pb < 1:
            s = 80; verdict = f"Trading below book value ({pb:.2f}) — potential bargain"
        elif pb < 3:
            s = 65; verdict = f"Normal range ({pb:.2f})"
        elif pb < 5:
            s = 45; verdict = f"Above book ({pb:.2f}) — growth expectations priced in"
        else:
            s = 30; verdict = f"Very high ({pb:.2f}) — only justified by exceptional growth"
        results["P/B Ratio"] = {"value": round(pb, 2), "score": s, "verdict": verdict}
        scores.append(s)
    else:
        results["P/B Ratio"] = {"value": None, "score": None, "verdict": "Not available"}

    # Market Cap
    mcap = safe_get(info, "marketCap")
    if mcap is not None:
        mcap_b = mcap / 1e9
        if mcap_b > 200: s = 85; label = "Mega-cap"
        elif mcap_b > 10: s = 75; label = "Large-cap"
        elif mcap_b > 2: s = 55; label = "Mid-cap"
        else: s = 35; label = "Small-cap"
        results["Market Cap"] = {"value_billions": round(mcap_b, 2), "category": label, "score": s, "verdict": f"{label} (${mcap_b:.1f}B) — {'lower risk, more stable' if mcap_b > 10 else 'higher growth potential but more volatile'}"}
        scores.append(s)
    else:
        results["Market Cap"] = {"value_billions": None, "score": None, "verdict": "Not available"}

    step_score = round(sum(scores) / len(scores)) if scores else 0
    return {"step": "Fundamental Analysis", "step_number": 1, "score": step_score, "label": score_label(step_score), "sub_steps": results}


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — PRICE & VOLUME
# ═══════════════════════════════════════════════════════════════════
def analyze_price_volume(info, hist):
    results = {}
    scores = []

    # 52-Week Range
    low52 = safe_get(info, "fiftyTwoWeekLow")
    high52 = safe_get(info, "fiftyTwoWeekHigh")
    price = safe_get(info, "currentPrice") or safe_get(info, "regularMarketPrice")
    if low52 and high52 and price:
        position = (price - low52) / (high52 - low52) * 100
        if position > 90:
            s = 55; verdict = f"Near 52-week high ({position:.0f}th percentile) — strong momentum but limited upside"
        elif position > 70:
            s = 70; verdict = f"Upper range ({position:.0f}th percentile) — healthy uptrend"
        elif position > 30:
            s = 60; verdict = f"Mid-range ({position:.0f}th percentile) — neutral position"
        elif position > 10:
            s = 50; verdict = f"Near lows ({position:.0f}th percentile) — potential bargain or falling knife"
        else:
            s = 35; verdict = f"At 52-week low ({position:.0f}th percentile) — investigate cause"
        results["52-Week Range"] = {"low": round(low52, 2), "high": round(high52, 2), "current": round(price, 2), "position_pct": round(position, 1), "score": s, "verdict": verdict}
        scores.append(s)
    else:
        results["52-Week Range"] = {"score": None, "verdict": "Not available"}

    # Moving Averages
    if hist is not None and len(hist) >= 200:
        ma50 = hist["Close"].rolling(50).mean().iloc[-1]
        ma200 = hist["Close"].rolling(200).mean().iloc[-1]
        current = hist["Close"].iloc[-1]
        above_50 = current > ma50
        above_200 = current > ma200
        golden_cross = ma50 > ma200

        if above_50 and above_200 and golden_cross:
            s = 85; verdict = "Strong uptrend — price above both MAs, golden cross active"
        elif above_200 and not above_50:
            s = 55; verdict = "Long-term uptrend but short-term weakness — price below 50-day MA"
        elif above_50 and not above_200:
            s = 50; verdict = "Short-term bounce but long-term downtrend"
        else:
            s = 25; verdict = "Downtrend — price below both moving averages"
        results["Moving Averages"] = {"ma_50": round(float(ma50), 2), "ma_200": round(float(ma200), 2), "price": round(float(current), 2), "above_50": above_50, "above_200": above_200, "golden_cross": golden_cross, "score": s, "verdict": verdict}
        scores.append(s)
    elif hist is not None and len(hist) >= 50:
        ma50 = hist["Close"].rolling(50).mean().iloc[-1]
        current = hist["Close"].iloc[-1]
        above_50 = current > ma50
        s = 60 if above_50 else 40
        results["Moving Averages"] = {"ma_50": round(float(ma50), 2), "ma_200": None, "price": round(float(current), 2), "above_50": above_50, "score": s, "verdict": f"Price {'above' if above_50 else 'below'} 50-day MA (insufficient data for 200-day)"}
        scores.append(s)
    else:
        results["Moving Averages"] = {"score": None, "verdict": "Insufficient price history"}

    # Volume
    avg_vol = safe_get(info, "averageVolume")
    vol_today = safe_get(info, "volume") or (int(hist["Volume"].iloc[-1]) if hist is not None and len(hist) > 0 else None)
    if avg_vol and vol_today:
        vol_ratio = vol_today / avg_vol
        if vol_ratio > 2:
            s = 70; verdict = f"Very high volume ({vol_ratio:.1f}x average) — strong conviction in today's move"
        elif vol_ratio > 1.2:
            s = 65; verdict = f"Above average volume ({vol_ratio:.1f}x) — move is supported"
        elif vol_ratio > 0.8:
            s = 55; verdict = f"Normal volume ({vol_ratio:.1f}x average)"
        else:
            s = 40; verdict = f"Low volume ({vol_ratio:.1f}x average) — move may not hold"
        results["Volume"] = {"today": vol_today, "average": avg_vol, "ratio": round(vol_ratio, 2), "score": s, "verdict": verdict}
        scores.append(s)
    else:
        results["Volume"] = {"score": None, "verdict": "Not available"}

    # Support & Resistance (simplified — recent swing highs/lows)
    if hist is not None and len(hist) >= 60:
        recent = hist["Close"].tail(60)
        support = float(recent.rolling(20).min().dropna().iloc[-1])
        resistance = float(recent.rolling(20).max().dropna().iloc[-1])
        current = float(hist["Close"].iloc[-1])
        dist_to_support = (current - support) / current * 100
        dist_to_resist = (resistance - current) / current * 100

        if dist_to_support < 2:
            s = 65; verdict = f"Near support (${support:.2f}) — potential buy zone if it holds"
        elif dist_to_resist < 2:
            s = 45; verdict = f"Near resistance (${resistance:.2f}) — may stall here"
        else:
            s = 55; verdict = f"Between support (${support:.2f}) and resistance (${resistance:.2f})"
        results["Support & Resistance"] = {"support": round(support, 2), "resistance": round(resistance, 2), "current": round(current, 2), "score": s, "verdict": verdict}
        scores.append(s)
    else:
        results["Support & Resistance"] = {"score": None, "verdict": "Insufficient data"}

    step_score = round(sum(scores) / len(scores)) if scores else 0
    return {"step": "Price & Volume", "step_number": 2, "score": step_score, "label": score_label(step_score), "sub_steps": results}


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — MOMENTUM INDICATORS
# ═══════════════════════════════════════════════════════════════════
def analyze_momentum(hist):
    results = {}
    scores = []

    if hist is None or len(hist) < 30:
        return {"step": "Momentum Indicators", "step_number": 3, "score": 0, "label": "No Data", "sub_steps": {"RSI": {"score": None, "verdict": "Insufficient data"}, "MACD": {"score": None, "verdict": "Insufficient data"}, "Bollinger Bands": {"score": None, "verdict": "Insufficient data"}}}

    close = hist["Close"].squeeze()

    # RSI
    if HAS_TA:
        rsi_series = ta.rsi(close, length=14)
        if rsi_series is not None and len(rsi_series.dropna()) > 0:
            rsi = float(rsi_series.dropna().iloc[-1])
        else:
            rsi = None
    else:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi = float(rsi_series.dropna().iloc[-1]) if len(rsi_series.dropna()) > 0 else None

    if rsi is not None:
        if rsi > 70:
            s = 35; verdict = f"Overbought ({rsi:.1f}) — stock may be due for a pullback"
        elif rsi > 60:
            s = 65; verdict = f"Bullish momentum ({rsi:.1f}) — trending up but not extreme"
        elif rsi > 40:
            s = 55; verdict = f"Neutral ({rsi:.1f}) — no strong momentum signal"
        elif rsi > 30:
            s = 60; verdict = f"Mild weakness ({rsi:.1f}) — approaching oversold"
        else:
            s = 70; verdict = f"Oversold ({rsi:.1f}) — potential bounce opportunity"
        results["RSI"] = {"value": round(rsi, 2), "score": s, "verdict": verdict}
        scores.append(s)
    else:
        results["RSI"] = {"value": None, "score": None, "verdict": "Could not calculate"}

    # MACD
    if len(close) >= 35:
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        macd_val = float(macd_line.iloc[-1])
        signal_val = float(signal_line.iloc[-1])
        histogram = macd_val - signal_val

        # Check crossover direction
        prev_hist = float(macd_line.iloc[-2] - signal_line.iloc[-2])
        bullish_cross = prev_hist < 0 and histogram > 0
        bearish_cross = prev_hist > 0 and histogram < 0

        if bullish_cross:
            s = 80; verdict = f"Bullish crossover just occurred — strong buy signal"
        elif bearish_cross:
            s = 25; verdict = f"Bearish crossover just occurred — sell signal"
        elif histogram > 0 and histogram > prev_hist:
            s = 70; verdict = f"Bullish and accelerating — MACD above signal and widening"
        elif histogram > 0:
            s = 60; verdict = f"Bullish but decelerating — MACD above signal but narrowing"
        elif histogram < 0 and histogram < prev_hist:
            s = 25; verdict = f"Bearish and accelerating — MACD below signal and widening"
        else:
            s = 40; verdict = f"Bearish but improving — MACD below signal but narrowing"
        results["MACD"] = {"macd": round(macd_val, 4), "signal": round(signal_val, 4), "histogram": round(histogram, 4), "score": s, "verdict": verdict}
        scores.append(s)
    else:
        results["MACD"] = {"score": None, "verdict": "Insufficient data for MACD"}

    # Bollinger Bands
    if len(close) >= 20:
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        current = float(close.iloc[-1])
        upper_val = float(upper.iloc[-1])
        lower_val = float(lower.iloc[-1])
        ma20_val = float(ma20.iloc[-1])
        band_width = (upper_val - lower_val) / ma20_val * 100

        # Percent B: where price sits within the bands (0=lower, 1=upper)
        pct_b = (current - lower_val) / (upper_val - lower_val)

        if pct_b > 1.0:
            s = 30; verdict = f"Price ABOVE upper band — overbought, likely to pull back"
        elif pct_b > 0.8:
            s = 45; verdict = f"Near upper band ({pct_b:.0%}) — stretched high"
        elif pct_b > 0.5:
            s = 60; verdict = f"Upper half of bands ({pct_b:.0%}) — mild bullish"
        elif pct_b > 0.2:
            s = 55; verdict = f"Lower half of bands ({pct_b:.0%}) — mild bearish"
        elif pct_b > 0:
            s = 60; verdict = f"Near lower band ({pct_b:.0%}) — potentially oversold"
        else:
            s = 65; verdict = f"Price BELOW lower band — oversold, potential bounce"

        squeeze = band_width < 10
        if squeeze:
            verdict += " | Bands squeezing — big move may be coming"

        results["Bollinger Bands"] = {"upper": round(upper_val, 2), "lower": round(lower_val, 2), "middle": round(ma20_val, 2), "pct_b": round(pct_b, 3), "bandwidth_pct": round(band_width, 2), "squeeze": squeeze, "score": s, "verdict": verdict}
        scores.append(s)
    else:
        results["Bollinger Bands"] = {"score": None, "verdict": "Insufficient data"}

    step_score = round(sum(scores) / len(scores)) if scores else 0
    return {"step": "Momentum Indicators", "step_number": 3, "score": step_score, "label": score_label(step_score), "sub_steps": results}


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — MARKET CONTEXT
# ═══════════════════════════════════════════════════════════════════
def analyze_context(ticker, info, hist):
    results = {}
    scores = []

    sector = safe_get(info, "sector", "")
    etf_ticker = SECTOR_ETFS.get(sector)

    # Sector ETF comparison
    if etf_ticker and hist is not None and len(hist) >= 60:
        try:
            etf = yf.download(etf_ticker, period="1y", progress=False, auto_adjust=True)
            if etf is not None and len(etf) >= 60:
                stock_ret = (float(hist["Close"].iloc[-1]) / float(hist["Close"].iloc[0]) - 1) * 100
                etf_ret = (float(etf["Close"].iloc[-1]) / float(etf["Close"].iloc[0]) - 1) * 100
                outperformance = stock_ret - etf_ret

                if outperformance > 20:
                    s = 90; verdict = f"Massively outperforming sector ({ticker} {stock_ret:+.1f}% vs {etf_ticker} {etf_ret:+.1f}%)"
                elif outperformance > 5:
                    s = 75; verdict = f"Outperforming sector ({ticker} {stock_ret:+.1f}% vs {etf_ticker} {etf_ret:+.1f}%)"
                elif outperformance > -5:
                    s = 55; verdict = f"In line with sector ({ticker} {stock_ret:+.1f}% vs {etf_ticker} {etf_ret:+.1f}%)"
                else:
                    s = 30; verdict = f"Underperforming sector ({ticker} {stock_ret:+.1f}% vs {etf_ticker} {etf_ret:+.1f}%)"
                results["Sector ETF"] = {"sector": sector, "etf": etf_ticker, "stock_return_pct": round(stock_ret, 2), "etf_return_pct": round(etf_ret, 2), "outperformance_pct": round(outperformance, 2), "score": s, "verdict": verdict}
                scores.append(s)
            else:
                results["Sector ETF"] = {"sector": sector, "etf": etf_ticker, "score": None, "verdict": "Could not download ETF data"}
        except Exception:
            results["Sector ETF"] = {"sector": sector, "score": None, "verdict": "Error fetching sector ETF data"}
    else:
        results["Sector ETF"] = {"sector": sector or "Unknown", "score": None, "verdict": f"No sector ETF mapping found for '{sector}'"}

    # S&P 500 Correlation
    if hist is not None and len(hist) >= 60:
        try:
            spy = yf.download("SPY", period="1y", progress=False, auto_adjust=True)
            if spy is not None and len(spy) >= 60:
                stock_returns = hist["Close"].pct_change().dropna().squeeze()
                spy_returns = spy["Close"].pct_change().dropna().squeeze()
                # Align indices
                common = stock_returns.index.intersection(spy_returns.index)
                if len(common) > 30:
                    corr = float(stock_returns.loc[common].corr(spy_returns.loc[common]))
                    if corr > 0.8:
                        s = 50; verdict = f"High correlation ({corr:.2f}) — moves closely with S&P 500, less diversification benefit"
                    elif corr > 0.5:
                        s = 60; verdict = f"Moderate correlation ({corr:.2f}) — partially follows market"
                    elif corr > 0.2:
                        s = 70; verdict = f"Low correlation ({corr:.2f}) — somewhat independent, good for diversification"
                    else:
                        s = 75; verdict = f"Very low correlation ({corr:.2f}) — moves independently from market"
                    results["S&P 500 Correlation"] = {"value": round(corr, 3), "score": s, "verdict": verdict}
                    scores.append(s)
                else:
                    results["S&P 500 Correlation"] = {"score": None, "verdict": "Insufficient overlapping data"}
            else:
                results["S&P 500 Correlation"] = {"score": None, "verdict": "Could not download S&P 500 data"}
        except Exception:
            results["S&P 500 Correlation"] = {"score": None, "verdict": "Error computing correlation"}
    else:
        results["S&P 500 Correlation"] = {"score": None, "verdict": "Insufficient price history"}

    step_score = round(sum(scores) / len(scores)) if scores else 0
    return {"step": "Market Context", "step_number": 4, "score": step_score, "label": score_label(step_score), "sub_steps": results}


# ═══════════════════════════════════════════════════════════════════
# STEP 5 — RISK METRICS
# ═══════════════════════════════════════════════════════════════════
def analyze_risk(info, hist):
    results = {}
    scores = []

    # Beta
    beta = safe_get(info, "beta")
    if beta is not None:
        if beta < 0.5:
            s = 85; verdict = f"Very low volatility ({beta:.2f}) — defensive stock"
        elif beta < 0.8:
            s = 75; verdict = f"Low volatility ({beta:.2f}) — calmer than market"
        elif beta <= 1.2:
            s = 60; verdict = f"Market-like volatility ({beta:.2f})"
        elif beta <= 1.8:
            s = 40; verdict = f"Above-market volatility ({beta:.2f}) — expect bigger swings"
        else:
            s = 25; verdict = f"High volatility ({beta:.2f}) — moves {beta:.1f}x the market in both directions"
        results["Beta"] = {"value": round(beta, 2), "score": s, "verdict": verdict}
        scores.append(s)
    else:
        results["Beta"] = {"value": None, "score": None, "verdict": "Not available"}

    # Standard Deviation (annualized)
    if hist is not None and len(hist) >= 60:
        daily_returns = hist["Close"].pct_change().dropna()
        std_daily = float(daily_returns.std())
        std_annual = std_daily * (252 ** 0.5) * 100

        if std_annual < 15:
            s = 80; verdict = f"Low volatility ({std_annual:.1f}% annualized) — steady and predictable"
        elif std_annual < 25:
            s = 65; verdict = f"Moderate volatility ({std_annual:.1f}% annualized) — normal range"
        elif std_annual < 40:
            s = 40; verdict = f"High volatility ({std_annual:.1f}% annualized) — significant daily swings"
        else:
            s = 20; verdict = f"Very high volatility ({std_annual:.1f}% annualized) — aggressive risk"
        results["Standard Deviation"] = {"daily_pct": round(std_daily * 100, 3), "annual_pct": round(std_annual, 2), "score": s, "verdict": verdict}
        scores.append(s)
    else:
        results["Standard Deviation"] = {"score": None, "verdict": "Insufficient data"}

    # Max Drawdown
    if hist is not None and len(hist) >= 60:
        close = hist["Close"].squeeze()
        cummax = close.cummax()
        drawdown = (close - cummax) / cummax * 100
        max_dd = float(drawdown.min())
        dd_date = drawdown.idxmin()

        if max_dd > -10:
            s = 85; verdict = f"Shallow max drawdown ({max_dd:.1f}%) — limited downside experienced"
        elif max_dd > -20:
            s = 65; verdict = f"Moderate drawdown ({max_dd:.1f}%) — normal correction range"
        elif max_dd > -40:
            s = 40; verdict = f"Significant drawdown ({max_dd:.1f}%) — serious decline occurred"
        else:
            s = 20; verdict = f"Severe drawdown ({max_dd:.1f}%) — catastrophic decline in this period"
        results["Max Drawdown"] = {"value_pct": round(max_dd, 2), "date": str(dd_date.date()) if hasattr(dd_date, 'date') else str(dd_date), "score": s, "verdict": verdict}
        scores.append(s)
    else:
        results["Max Drawdown"] = {"score": None, "verdict": "Insufficient data"}

    # Sharpe Ratio (using 5% risk-free rate approximation)
    if hist is not None and len(hist) >= 120:
        daily_returns = hist["Close"].pct_change().dropna()
        annual_return = float(daily_returns.mean()) * 252 * 100
        annual_std = float(daily_returns.std()) * (252 ** 0.5) * 100
        risk_free = 5.0  # approximate 2026 rate
        sharpe = (annual_return - risk_free) / annual_std if annual_std > 0 else 0

        if sharpe > 2:
            s = 90; verdict = f"Excellent risk-adjusted return (Sharpe = {sharpe:.2f})"
        elif sharpe > 1:
            s = 70; verdict = f"Good risk-adjusted return (Sharpe = {sharpe:.2f})"
        elif sharpe > 0.5:
            s = 50; verdict = f"Mediocre risk-adjusted return (Sharpe = {sharpe:.2f})"
        elif sharpe > 0:
            s = 35; verdict = f"Poor risk-adjusted return (Sharpe = {sharpe:.2f}) — barely compensating for risk"
        else:
            s = 15; verdict = f"Negative Sharpe ({sharpe:.2f}) — risk-free rate beat this stock"
        results["Sharpe Ratio"] = {"value": round(sharpe, 3), "annual_return_pct": round(annual_return, 2), "annual_std_pct": round(annual_std, 2), "risk_free_pct": risk_free, "score": s, "verdict": verdict}
        scores.append(s)
    else:
        results["Sharpe Ratio"] = {"score": None, "verdict": "Insufficient data (need 120+ days)"}

    # Short Interest
    short_pct = safe_get(info, "shortPercentOfFloat")
    if short_pct is not None:
        sp = short_pct * 100
        if sp < 2:
            s = 80; verdict = f"Very low short interest ({sp:.2f}%) — market is confident"
        elif sp < 5:
            s = 60; verdict = f"Moderate short interest ({sp:.2f}%) — some bearish bets"
        elif sp < 10:
            s = 40; verdict = f"Elevated short interest ({sp:.2f}%) — notable bearish sentiment"
        else:
            s = 20; verdict = f"High short interest ({sp:.2f}%) — significant bets against this stock"
        results["Short Interest"] = {"value_pct": round(sp, 2), "score": s, "verdict": verdict}
        scores.append(s)
    else:
        results["Short Interest"] = {"score": None, "verdict": "Not available"}

    step_score = round(sum(scores) / len(scores)) if scores else 0
    return {"step": "Risk Metrics", "step_number": 5, "score": step_score, "label": score_label(step_score), "sub_steps": results}


# ═══════════════════════════════════════════════════════════════════
# MAIN ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════
def run_analysis(ticker, period="1y"):
    """Run all 5 steps of analysis for a given ticker."""
    print(f"\n{'='*60}")
    print(f"  STOCK ANALYSIS: {ticker.upper()}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Period: {period}")
    print(f"{'='*60}\n")

    # Fetch data
    print("  Fetching data from Yahoo Finance...", end=" ", flush=True)
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    name = safe_get(info, "longName", ticker.upper())
    print(f"✓ {name}\n")

    hist = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if hist is not None and len(hist) > 0 and isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.droplevel(1)

    try:
        financials = stock.quarterly_income_stmt
    except Exception:
        financials = None

    # Run all steps
    steps = []
    steps.append(analyze_fundamentals(info, financials))
    steps.append(analyze_price_volume(info, hist))
    steps.append(analyze_momentum(hist))
    steps.append(analyze_context(ticker, info, hist))
    steps.append(analyze_risk(info, hist))

    # Overall score
    valid_scores = [s["score"] for s in steps if s["score"] > 0]
    overall = round(sum(valid_scores) / len(valid_scores)) if valid_scores else 0

    report = {
        "ticker": ticker.upper(),
        "name": name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "period": period,
        "overall_score": overall,
        "overall_label": score_label(overall),
        "price": safe_get(info, "currentPrice") or safe_get(info, "regularMarketPrice"),
        "steps": steps,
    }

    return report


def print_report(report):
    """Print a formatted text report."""
    BAR = "█"
    EMPTY = "░"

    def bar(score, width=20):
        if score is None: return "N/A"
        filled = round(score / 100 * width)
        return f"{BAR * filled}{EMPTY * (width - filled)} {score}/100"

    print(f"  {'─'*56}")
    print(f"  OVERALL SCORE: {bar(report['overall_score'])} ({report['overall_label']})")
    print(f"  Current Price: ${report['price']:.2f}" if report['price'] else "")
    print(f"  {'─'*56}\n")

    for step in report["steps"]:
        cat_colors = {1: "🔵", 2: "🟡", 3: "🟡", 4: "🟢", 5: "🔴"}
        icon = cat_colors.get(step["step_number"], "⚪")
        print(f"  {icon} STEP {step['step_number']}: {step['step'].upper()}")
        print(f"     Score: {bar(step['score'])} ({step['label']})")
        print()

        for name, data in step["sub_steps"].items():
            score = data.get("score")
            verdict = data.get("verdict", "N/A")
            if score is not None:
                mini_bar = bar(score, 10)
                print(f"     ├─ {name}")
                print(f"     │  {mini_bar}")
                # Print key values
                for k, v in data.items():
                    if k not in ("score", "verdict") and v is not None:
                        print(f"     │  {k}: {v}")
                print(f"     │  → {verdict}")
            else:
                print(f"     ├─ {name}: {verdict}")
            print(f"     │")
        print()

    # Summary
    print(f"  {'═'*56}")
    print(f"  SUMMARY FOR {report['name']}")
    print(f"  {'═'*56}")
    print()

    # Find strongest and weakest
    valid_steps = [s for s in report["steps"] if s["score"] > 0]
    if valid_steps:
        strongest = max(valid_steps, key=lambda x: x["score"])
        weakest = min(valid_steps, key=lambda x: x["score"])
        print(f"  ✅ Strongest area: {strongest['step']} ({strongest['score']}/100)")
        print(f"  ⚠️  Weakest area:  {weakest['step']} ({weakest['score']}/100)")
        print()

    # Top concerns
    concerns = []
    for step in report["steps"]:
        for name, data in step["sub_steps"].items():
            score = data.get("score")
            if score is not None and score < 35:
                concerns.append(f"  🔻 {name}: {data.get('verdict', '')}")
    if concerns:
        print("  KEY CONCERNS:")
        for c in concerns:
            print(c)
        print()

    # Top strengths
    strengths = []
    for step in report["steps"]:
        for name, data in step["sub_steps"].items():
            score = data.get("score")
            if score is not None and score >= 75:
                strengths.append(f"  🔹 {name}: {data.get('verdict', '')}")
    if strengths:
        print("  KEY STRENGTHS:")
        for s in strengths:
            print(s)
        print()

    print(f"  {'─'*56}")
    print("  ⚠️  This is analysis, not advice. Always do your own research.")
    print(f"  {'─'*56}\n")


# ═══════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Stock Analysis Tool — 5-Step Framework")
    parser.add_argument("ticker", help="Stock ticker (e.g. NVDA, TD, AAPL)")
    parser.add_argument("--period", default="1y", help="History period: 6mo, 1y, 2y, 5y (default: 1y)")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of text")
    args = parser.parse_args()

    report = run_analysis(args.ticker, args.period)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)

    return report


if __name__ == "__main__":
    main()
