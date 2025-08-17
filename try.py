import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import requests
import plotly.express as px
import plotly.graph_objects as go
import uuid
import concurrent.futures
from itertools import islice
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Telegram configuration
bot_token = "7756863824:AAGm11kD_R8MErGUxbUwzF7w60WYcY"
chat_id = "8227677"

# Function to send Telegram message
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            pass  # Silently skip Telegram errors
    except Exception:
        pass  # Silently skip exceptions

# Function to generate TradingView-style chart image
def generate_chart_image(ticker, period="1d", interval="5m"):
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        if data.empty:
            return None
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                            open=data['Open'],
                                            high=data['High'],
                                            low=data['Low'],
                                            close=data['Close'])])
        fig.add_hline(y=data['Close'].iloc[-1], line_dash="dash", line_color="green", annotation_text="Current Price")
        fig.update_layout(title=f"{ticker} Chart", xaxis_title="Time", yaxis_title="Price")
        buf = BytesIO()
        fig.write_image(buf, format="png")
        data = base64.b64encode(buf.getvalue()).decode()
        return data
    except Exception:
        return None

# Function to calculate support and resistance levels
def calculate_support_resistance(data):
    if data.empty or 'Close' not in data.columns:
        return 0, 0
    prices = data['Close'].values
    pivot_high = np.max(prices[-10:])
    pivot_low = np.min(prices[-10:])
    return pivot_low, pivot_high

# Updated sector details with new stock list
sector_details = {
    "METAL": {"index": "^CNXMETAL", "stocks": ["HINDCOPPER.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "JSL.NS", "NATIONALUM.NS", "ADANIENT.NS", "JINDALSTEL.NS", "HINDALCO.NS", "SAIL.NS", "HINDZINC.NS", "APLAPOLLO.NS", "NMDC.NS", "VEDL.NS"]},
    "PSU BANK": {"index": "^CNXPSUBANK", "stocks": ["PNB.NS", "BANKINDIA.NS", "UNIONBANK.NS", "BANKBARODA.NS", "CANBK.NS", "INDIANB.NS", "SBIN.NS"]},
    "REALTY": {"index": "^CNXREALTY", "stocks": ["NCC.NS", "OBEROIRLTY.NS", "DLF.NS", "GODREJPROP.NS", "PHOENIXLTD.NS", "LODHA.NS", "PRESTIGE.NS", "NBCC.NS"]},
    "ENERGY": {"index": "^CNXENERGY", "stocks": ["PETRONET.NS", "NTPC.NS", "RELIANCE.NS", "ADANIENSOL.NS", "OIL.NS", "ADANIGREEN.NS", "COALINDIA.NS", "IGL.NS", "POWERGRID.NS", "INOXWIND.NS", "ATGL.NS", "TORNTPOWER.NS", "JSWENERGY.NS", "ONGC.NS", "CESC.NS", "BDL.NS", "TATAPOWER.NS", "MAZDOCK.NS", "IOC.NS", "IREDA.NS", "CGPOWER.NS", "BPCL.NS", "SOLARINDS.NS", "BLUESTARCO.NS", "GMRAIRPORT.NS", "NHPC.NS", "SJVN.NS"]},
    "AUTO": {"index": "^CNXAUTO", "stocks": ["BOSCHLTD.NS", "BALKRISIND.NS", "EXIDEIND.NS", "EICHERMOT.NS", "BHARATFORG.NS", "TITAGARH.NS", "TIINDIA.NS", "TATAMOTORS.NS", "UNOMINDA.NS", "TVSMOTOR.NS", "SONACOMS.NS", "M&M.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "MARUTI.NS", "MOTHERSON.NS"]},
    "IT": {"index": "^CNXIT", "stocks": ["KPITTECH.NS", "WIPRO.NS", "CYIENT.NS", "KAYNES.NS", "TATAELXSI.NS", "TATATECH.NS", "MPHASIS.NS", "OFSS.NS", "INFY.NS", "TCS.NS", "CAMS.NS", "PERSISTENT.NS", "HCLTECH.NS", "COFORGE.NS", "HFCL.NS", "LTIM.NS", "TECHM.NS"]},
    "PHARMA": {"index": "^CNXPHARMA", "stocks": ["PPLPHARMA.NS", "GRANULES.NS", "DIVISLAB.NS", "LAURUSLABS.NS", "TORNTPHARM.NS", "AUROPHARMA.NS", "BIOCON.NS", "LUPIN.NS", "ZYDUSLIFE.NS", "DRREDDY.NS", "GLENMARK.NS", "FORTIS.NS", "CIPLA.NS", "MANKIND.NS", "SUNPHARMA.NS", "PEL.NS", "ALKEM.NS"]},
    "NIFTY 50": {"index": "^NIFTY50", "stocks": ["ASIANPAINT.NS", "LT.NS", "JSWSTEEL.NS", "INDUSINDBK.NS", "NTPC.NS", "EICHERMOT.NS", "AXISBANK.NS", "RELIANCE.NS", "BEL.NS", "TATASTEEL.NS", "HDFCBANK.NS", "WIPRO.NS", "BRITANNIA.NS", "BAJAJFINSV.NS", "BHARTIARTL.NS", "COALINDIA.NS", "POWERGRID.NS", "APOLLOHOSP.NS", "TATAMOTORS.NS", "BAJFINANCE.NS", "NESTLEIND.NS", "INFY.NS", "TATACONSUM.NS", "ITC.NS", "DRREDDY.NS", "ULTRACEMCO.NS", "SBILIFE.NS", "ONGC.NS", "ADANIENT.NS", "HINDUNILVR.NS", "GRASIM.NS", "TCS.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SHRIRAMFIN.NS", "M&M.NS", "HINDALCO.NS", "CIPLA.NS", "TITAN.NS", "HEROMOTOCO.NS", "SUNPHARMA.NS", "BAJAJ-AUTO.NS", "MARUTI.NS", "BPCL.NS", "HCLTECH.NS", "SBIN.NS", "ADANIPORTS.NS", "HDFCLIFE.NS", "TECHM.NS", "TRENT.NS"]},
    "PVT BANK": {"index": "^CNXPSUBANK", "stocks": ["INDUSINDBK.NS", "AXISBANK.NS", "HDFCBANK.NS", "FEDERALBNK.NS", "BANDHANBNK.NS", "RBLBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "IDFCFIRSTB.NS"]},
    "BANK": {"index": "^CNXPSUBANK", "stocks": ["INDUSINDBK.NS", "AXISBANK.NS", "HDFCBANK.NS", "PNB.NS", "BANKINDIA.NS", "FEDERALBNK.NS", "BANKBARODA.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "CANBK.NS", "AUBANK.NS", "INDIANB.NS", "SBIN.NS", "IDFCFIRSTB.NS"]},
    "FIN SERVICE": {"index": "^CNXFIN", "stocks": ["MUTHOOTFIN.NS", "JIOFIN.NS", "IIFL.NS", "HDFCAMC.NS", "RECLTD.NS", "AXISBANK.NS", "HDFCBANK.NS", "PFC.NS", "BAJAJFINSV.NS", "PAYTM.NS", "POONAWALLA.NS", "PNBHOUSING.NS", "SBICARD.NS", "POLICYBZR.NS", "BAJFINANCE.NS", "HUDCO.NS", "IRFC.NS", "LICI.NS", "LICHSGFIN.NS", "CHOLAFIN.NS", "SBILIFE.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "CDSL.NS", "SHRIRAMFIN.NS", "MAXHEALTH.NS", "ICICIGI.NS", "ICICIPRULI.NS", "SBIN.NS", "ANGELONE.NS", "BSE.NS", "HDFCLIFE.NS"]},
    "FMCG": {"index": "^CNXFMCG", "stocks": ["VBL.NS", "PATANJALI.NS", "BRITANNIA.NS", "DABUR.NS", "DMART.NS", "MARICO.NS", "NESTLEIND.NS", "TATACONSUM.NS", "ITC.NS", "HINDUNILVR.NS", "SUPREMEIND.NS", "ETERNAL.NS", "KALYANKJIL.NS", "UNITDSPR.NS", "NYKAA.NS", "COLPAL.NS", "GODREJCP.NS"]},
    "CEMENT": {"index": "^NIFTY50", "stocks": ["AMBUJACEM.NS", "ULTRACEMCO.NS", "ACC.NS", "DALBHARAT.NS", "SHREECEM.NS"]},
    "NIFTY MID SELECT": {"index": "^NIFTY50", "stocks": ["PIIND.NS", "HDFCAMC.NS", "BHARATFORG.NS", "AUROPHARMA.NS", "LUPIN.NS", "POLYCAB.NS", "GODREJPROP.NS", "UPL.NS", "FEDERALBNK.NS", "ASHOKLEY.NS", "VOLTAS.NS", "PAGEIND.NS", "MPHASIS.NS", "JUBLFOOD.NS", "INDHOTEL.NS", "CUMMINSIND.NS", "PERSISTENT.NS", "ASTRAL.NS", "RVNL.NS", "CONCOR.NS", "AUBANK.NS", "HINDPETRO.NS", "COFORGE.NS", "IDFCFIRSTB.NS"]},
    "SENSEX": {"index": "^BSESN", "stocks": ["ASIANPAINT.NS", "LT.NS", "JSWSTEEL.NS", "INDUSINDBK.NS", "NTPC.NS", "AXISBANK.NS", "RELIANCE.NS", "TATASTEEL.NS", "HDFCBANK.NS", "BAJAJFINSV.NS", "BHARTIARTL.NS", "POWERGRID.NS", "TATAMOTORS.NS", "BAJFINANCE.NS", "NESTLEIND.NS", "INFY.NS", "ITC.NS", "ULTRACEMCO.NS", "HINDUNILVR.NS", "TCS.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "M&M.NS", "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "HCLTECH.NS", "SBIN.NS", "ADANIPORTS.NS", "TECHM.NS"]}
}

# Function to check if the market is open
def is_market_open():
    now = datetime.now()
    market_open_time = now.replace(hour=9, minute=15, second=0)
    market_close_time = now.replace(hour=15, minute=30, second=0)
    return market_open_time <= now <= market_close_time and now.weekday() < 5

# Enhanced calculate_technicals function with better handling for missing data
def calculate_technicals(data):
    if data.empty or 'Close' not in data.columns or 'Volume' not in data.columns or len(data) < 2:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # Return default values if data is insufficient
    try:
        # RSI (14)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.replace([np.inf, -np.inf], np.nan).fillna(0)))
        rsi_val = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 0
        
        # EMA (20, 50)
        ema_20 = data['Close'].ewm(span=20, adjust=False).mean()
        ema_50 = data['Close'].ewm(span=50, adjust=False).mean()
        ema_20_val = ema_20.iloc[-1] if not np.isnan(ema_20.iloc[-1]) else 0
        ema_50_val = ema_50.iloc[-1] if not np.isnan(ema_50.iloc[-1]) else 0
        
        # MACD (12, 26, 9)
        ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_val = macd.iloc[-1] if not np.isnan(macd.iloc[-1]) else 0
        signal_val = signal.iloc[-1] if not np.isnan(signal.iloc[-1]) else 0
        
        # On-Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        obv = pd.Series(obv, index=data.index)
        obv_val = obv.iloc[-1] if not np.isnan(obv.iloc[-1]) else 0
        
        # Bollinger Bands (20, 2)
        sma_20 = data['Close'].rolling(window=20).mean()
        std_20 = data['Close'].rolling(window=20).std()
        bb_upper = sma_20 + (std_20 * 2)
        bb_lower = sma_20 - (std_20 * 2)
        bb_upper_val = bb_upper.iloc[-1] if not np.isnan(bb_upper.iloc[-1]) else 0
        bb_lower_val = bb_lower.iloc[-1] if not np.isnan(bb_lower.iloc[-1]) else 0
        
        # Average True Range (ATR, 14)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        atr_val = atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else 0
        
        # VWAP (Volume Weighted Average Price)
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        
        # Enhanced R Factor Calculation
        open_price = data['Open'].iloc[0]
        latest_price = data['Close'].iloc[-1]
        change_pct = ((latest_price - open_price) / open_price) * 100 if not np.isnan([open_price, latest_price]).any() else 0
        volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
        volume_factor = min(2.0, volume / (avg_volume + 1e-10))
        volatility = atr_val / latest_price if latest_price != 0 else 0
        momentum = (abs(change_pct) * volume_factor * (1 + volatility))
        r_factor = np.clip((momentum * 1.5), 0, 10)
        r_factor = r_factor if not np.isnan(r_factor) else 0
        
        # Institutional Activity Score (Simulated)
        volume_spike = volume / (avg_volume + 1e-10) if avg_volume != 0 else 0
        price_breakout = (latest_price > bb_upper_val or latest_price < bb_lower_val)
        institutional_score = (volume_spike * 0.6 + (2 if price_breakout else 0) * 0.4) * 10
        institutional_score = institutional_score if not np.isnan(institutional_score) else 0
        
        return (rsi_val, ema_20_val, ema_50_val, macd_val, signal_val, obv_val, 
                r_factor, institutional_score, bb_upper_val, bb_lower_val)
    except Exception:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

# Function to fetch data for a single stock or index
def fetch_ticker_data(ticker, period="1d", interval="5m"):
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        if data.empty:
            return ticker, pd.DataFrame()
        return ticker, data
    except Exception:
        return ticker, pd.DataFrame()

# Function to fetch previous day's data for top gainers/losers
def fetch_previous_day_data(ticker):
    try:
        end_date = datetime.now().date() - timedelta(days=1)
        start_date = end_date - timedelta(days=1)
        data = yf.Ticker(ticker).history(start=start_date, end=end_date, interval="1d")
        if data.empty:
            return ticker, pd.DataFrame()
        return ticker, data
    except Exception:
        return ticker, pd.DataFrame()

# Function to fetch live sector and stock data in batches with breakout list
def fetch_sector_stock_data():
    fetch_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')
    sector_data = []
    stock_data = []
    top_movers = []
    high_activity_stocks = []  # List for high activity stocks during market hours
    breakout_stocks = []  # New list for breakout stocks
    best_stock = None
    best_stock_score = -float('inf')
    sector_flows = []
    period = "1d"
    interval = "5m"
    
    # Check if market is open for filtering high activity stocks
    is_market = is_market_open()
    
    # Fetch all index data in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        index_futures = {
            executor.submit(fetch_ticker_data, details["index"], period, interval): sector
            for sector, details in sector_details.items()
        }
        for future in concurrent.futures.as_completed(index_futures):
            sector = index_futures[future]
            try:
                ticker, index_data = future.result()
                if not index_data.empty:
                    open_price = index_data['Open'].iloc[0]
                    latest_price = index_data['Close'].iloc[-1]
                    change = ((latest_price - open_price) / open_price) * 100
                    sector_data.append([sector, round(change, 2), round(latest_price, 2)])
            except Exception:
                pass
    
    # Fetch all stock data in batches
    batch_size = 5
    for sector, details in sector_details.items():
        max_change = 0
        top_stock = None
        sector_volume = 0
        sector_investment_flow = 0
        stock_list = details["stocks"]
        
        # Process stocks in batches
        for i in range(0, len(stock_list), batch_size):
            batch = list(islice(stock_list, i, i + batch_size))
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                stock_futures = {
                    executor.submit(fetch_ticker_data, stock, period, interval): stock
                    for stock in batch
                }
                for future in concurrent.futures.as_completed(stock_futures):
                    stock = stock_futures[future]
                    try:
                        ticker, stock_info = future.result()
                        if not stock_info.empty:
                            open_price = stock_info['Open'].iloc[0]
                            latest_price = stock_info['Close'].iloc[-1]
                            volume = stock_info['Volume'].sum()
                            change = ((latest_price - open_price) / open_price) * 100
                            volume_spike = volume / stock_info['Volume'].rolling(window=20).mean().iloc[-1] if stock_info['Volume'].rolling(window=20).mean().iloc[-1] != 0 else 0
                            price_movement = abs(change) > 2  # Significant movement threshold
                            
                            rsi, ema_20, ema_50, macd, signal, obv, r_factor, institutional_score, bb_upper, bb_lower = calculate_technicals(stock_info)
                            trend = "Bullish" if latest_price > ema_50 else "Bearish" if latest_price < ema_50 else "Neutral"
                            
                            stock_data.append([
                                sector, stock.replace('.NS', ''), round(latest_price, 2),
                                round(change, 2), volume, round(institutional_score, 2), round(r_factor, 2)
                            ])
                            
                            # Add to high activity stocks list if market is open and criteria met
                            if is_market and r_factor > 7 and volume_spike > 1.5 and price_movement:
                                high_activity_stocks.append([
                                    sector, stock.replace('.NS', ''), round(latest_price, 2),
                                    round(change, 2), volume, round(r_factor, 2),
                                    round(institutional_score, 2), datetime.now().strftime('%H:%M:%S')
                                ])
                            
                            # Detect breakout stocks with institutional activity
                            support, resistance = calculate_support_resistance(stock_info)
                            if (latest_price > resistance or latest_price < support) and institutional_score > 5 and volume_spike > 1.5:
                                chart_image = generate_chart_image(stock)
                                entry_candle = "Current Candle" if latest_price > resistance else "Previous Breakout Candle"
                                stop_loss = support if latest_price > resistance else resistance
                                target = latest_price + (resistance - support) * 1.5 if latest_price > resistance else latest_price - (support - resistance) * 1.5
                                breakout_stocks.append([
                                    sector, stock.replace('.NS', ''), round(latest_price, 2),
                                    round(change, 2), volume, round(r_factor, 2),
                                    round(institutional_score, 2), datetime.now().strftime('%H:%M:%S'),
                                    support, resistance, entry_candle, stop_loss, target, chart_image
                                ])
                            
                            sector_volume += volume
                            sector_investment_flow += (change * volume) / 1e6
                            
                            if abs(change) > max_change:
                                max_change = abs(change)
                                top_stock = {
                                    "sector": sector,
                                    "stock": stock,
                                    "stock_name": stock.replace('.NS', ''),
                                    "change": round(change, 2),
                                    "latest_price": round(latest_price, 2),
                                    "volume": volume,
                                    "trend": trend,
                                    "institutional_score": round(institutional_score, 2)
                                }
                                top_stock["trade_suggestion"] = suggest_option_trade(
                                    stock, latest_price, rsi, ema_20, ema_50, macd, signal, obv, r_factor
                                )
                            
                            trade_suggestion = suggest_option_trade(stock, latest_price, rsi, ema_20, ema_50, macd, signal, obv, r_factor)
                            if trade_suggestion["signal"] in ["Buy Call", "Buy Put"]:
                                score = abs(change) + institutional_score / 10
                                if volume_spike:
                                    score += 2
                                if 30 < rsi < 70:
                                    score += 1
                                if r_factor > 7:
                                    score += 2
                                if score > best_stock_score:
                                    best_stock_score = score
                                    best_stock = {
                                        "sector": sector,
                                        "stock_name": stock.replace('.NS', ''),
                                        "change": round(change, 2),
                                        "latest_price": round(latest_price, 2),
                                        "trend": trend,
                                        "volume": volume,
                                        "institutional_score": round(institutional_score, 2),
                                        "trade_suggestion": trade_suggestion
                                    }
                    except Exception:
                        pass
        
        if top_stock:
            top_movers.append(top_stock)
        sector_flows.append([sector, sector_volume, sector_investment_flow])
    
    # Fetch previous day's top gainers and losers in batches
    previous_day_data = []
    all_stocks = [stock for sector, details in sector_details.items() for stock in details["stocks"]]
    for i in range(0, len(all_stocks), batch_size):
        batch = list(islice(all_stocks, i, i + batch_size))
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            stock_futures = {
                executor.submit(fetch_previous_day_data, stock): stock
                for stock in batch
            }
            for future in concurrent.futures.as_completed(stock_futures):
                stock = stock_futures[future]
                try:
                    ticker, stock_info = future.result()
                    if not stock_info.empty:
                        open_price = stock_info['Open'].iloc[0]
                        close_price = stock_info['Close'].iloc[-1]
                        change = ((close_price - open_price) / open_price) * 100
                        previous_day_data.append([
                            stock.replace('.NS', ''), round(change, 2), round(close_price, 2)
                        ])
                except Exception:
                    pass
    
    sector_df = pd.DataFrame(sector_data, columns=["Sector", "% Change", "Current Price"])
    stock_df = pd.DataFrame(stock_data, columns=[
        "Sector", "Stock", "Current Price", "% Change", "Volume", 
        "Institutional Activity", "R Factor"
    ])
    top_movers_df = pd.DataFrame(top_movers)
    sector_flows_df = pd.DataFrame(sector_flows, columns=["Sector", "Total Volume", "Investment Flow"])
    previous_day_df = pd.DataFrame(previous_day_data, columns=["Stock", "% Change", "Close Price"])
    high_activity_df = pd.DataFrame(high_activity_stocks, columns=[
        "Sector", "Stock", "Current Price", "% Change", "Volume", "R Factor", "Institutional Score", "Detection Time"
    ])
    breakout_df = pd.DataFrame(breakout_stocks, columns=[
        "Sector", "Stock", "Current Price", "% Change", "Volume", "R Factor",
        "Institutional Score", "Detection Time", "Support", "Resistance",
        "Entry Candle", "Stop Loss", "Target", "Chart Image"
    ])
    
    if best_stock:
        trade = best_stock["trade_suggestion"]
        r_factor_analysis = (
            f"Strong momentum" if trade['r_factor'] > 7 else 
            f"Moderate momentum" if trade['r_factor'] > 4 else "Weak momentum"
        )
        message = (
            f"📈 *Best Intraday Opportunity (Analysis) {datetime.now().strftime('%Y-%m-%d')}*\n\n"
            f"**Stock**: {best_stock['stock_name']} ({best_stock['sector']})\n"
            f"**% Change**: {best_stock['change']:.2f}%\n"
            f"**Price**: ₹{best_stock['latest_price']:.2f}\n"
            f"**Trend**: {best_stock['trend']}\n"
            f"**Volume**: {best_stock['volume']:,.0f}\n"
            f"**Institutional Score**: {best_stock['institutional_score']:.2f}\n"
            f"**R Factor**: {trade['r_factor']:.2f} ({r_factor_analysis})\n\n"
            f"**Technical Analysis**:\n"
            f"- RSI: {trade['rsi']:.2f}\n"
            f"- EMA20: {trade['ema_20']:.2f}, EMA50: {trade['ema_50']:.2f}\n"
            f"- MACD: {trade['macd']:.2f}, Signal: {trade['signal_line']:.2f}\n"
            f"- OBV: {trade['obv']:,.0f}\n\n"
            f"**Trade Recommendation**:\n"
            f"- Signal: {trade['signal']}\n"
            f"- Option: {trade['signal']} at Strike ₹{trade['strike_price']}\n"
            f"- Entry: ₹{trade['entry_price']:.2f}\n"
            f"- Expiry: {trade['expiry']}\n"
            f"**Data Fetched At**: {fetch_timestamp}\n"
        )
        send_telegram_message(message)
    
    # Send breakout list to Telegram after market close
    if not is_market_open() and not breakout_df.empty:
        message = f"📊 *Final Breakout Stocks for {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        for _, row in breakout_df.iterrows():
            message += (
                f"**Stock**: {row['Stock']} ({row['Sector']})\n"
                f"**Time**: {row['Detection Time']}\n"
                f"**Price**: ₹{row['Current Price']:.2f}\n"
                f"**% Change**: {row['% Change']:.2f}%\n"
                f"**Support**: ₹{row['Support']:.2f}\n"
                f"**Resistance**: ₹{row['Resistance']:.2f}\n"
                f"**Entry Candle**: {row['Entry Candle']}\n"
                f"**Stop Loss**: ₹{row['Stop Loss']:.2f}\n"
                f"**Target**: ₹{row['Target']:.2f}\n\n"
            )
        send_telegram_message(message)
    
    return sector_df, stock_df, top_movers_df, sector_flows_df, previous_day_df, high_activity_df, breakout_df, fetch_timestamp

# Function to suggest option trade
def suggest_option_trade(stock, latest_price, rsi, ema_20, ema_50, macd, signal, obv, r_factor):
    if np.isnan([latest_price, rsi, ema_20, ema_50, macd, signal, obv, r_factor]).any():
        return {
            "signal": "No Trade",
            "trend": "Neutral",
            "strike_price": np.nan,
            "entry_price": np.nan,
            "rsi": 0,
            "ema_20": 0,
            "ema_50": 0,
            "macd": 0,
            "signal_line": 0,
            "obv": 0,
            "r_factor": 0,
            "expiry": "N/A"
        }
    
    trend = "Neutral"
    signal_type = "No Trade"
    
    if (rsi > 50 and rsi < 70 and ema_20 > ema_50 and macd > signal and 
        r_factor > 4.0 and obv > 0):
        signal_type = "Buy Call"
        trend = "Bullish"
    elif (rsi < 50 and rsi > 30 and ema_20 < ema_50 and macd < signal and 
          r_factor > 4.0 and obv < 0):
        signal_type = "Buy Put"
        trend = "Bearish"
    
    strike_price = round(latest_price / 100) * 100
    expiry = "Weekly"
    
    return {
        "signal": signal_type,
        "trend": trend,
        "strike_price": strike_price,
        "entry_price": latest_price,
        "rsi": round(rsi, 2),
        "ema_20": round(ema_20, 2),
        "ema_50": round(ema_50, 2),
        "macd": round(macd, 2),
        "signal_line": round(signal, 2),
        "obv": round(obv, 2),
        "r_factor": round(r_factor, 2),
        "expiry": expiry
    }

# Streamlit App
st.set_page_config(page_title="AI-Based Market Analysis Dashboard", layout="wide")
st.title("📈 AI-Based Intraday Market Analysis Dashboard")

# Show Market Status
market_status = "🟢 Market is Open" if is_market_open() else "🔴 Market is Closed (Using Previous Day's Data)"
st.sidebar.subheader(market_status)

# Fetch and display data
sector_df, stock_df, top_movers_df, sector_flows_df, previous_day_df, high_activity_df, breakout_df, fetch_timestamp = fetch_sector_stock_data()

# Display Timestamp
st.sidebar.subheader(f"Data Last Updated: {fetch_timestamp}")

# Overall Sector View (Treemap)
st.subheader("🌐 Overall Sector Performance (Treemap)")
st.write(f"Updated at: {fetch_timestamp}")
if not sector_df.empty:
    fig = px.treemap(
        sector_df,
        path=['Sector'],
        values='Current Price',
        color='% Change',
        color_continuous_scale=['red', 'white', 'green'],
        title="Sector-wise Performance Overview"
    )
    fig.update_traces(
        text=sector_df['Sector'] + "<br>" + sector_df['% Change'].astype(str) + "%",
        textposition='middle center',
        textfont=dict(size=14),
        hovertemplate='<b>%{label}</b><br>% Change: %{color:.2f}%<br>Current Price: ₹%{value:.2f}'
    )
    fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

# Display Sector Performance Graph
st.subheader("📊 Sector Performance")
st.write(f"Updated at: {fetch_timestamp}")
if not sector_df.empty:
    fig = px.bar(sector_df, x="Sector", y="% Change", 
                 color="% Change", 
                 color_continuous_scale=["red", "white", "green"],
                 title="Sector-wise Performance (%)")
    fig.update_layout(xaxis_title="Sector", yaxis_title="% Change")
    st.plotly_chart(fig, use_container_width=True)

# Display Institutional Activity by Sector
st.subheader("💼 Institutional Activity by Sector")
st.write(f"Updated at: {fetch_timestamp}")
if not sector_flows_df.empty:
    fig = px.bar(sector_flows_df, x="Sector", y="Investment Flow",
                 color="Investment Flow",
                 color_continuous_scale=["red", "white", "green"],
                 title="Sector-wise Investment Flow (Positive: Inflow, Negative: Outflow)")
    fig.update_layout(xaxis_title="Sector", yaxis_title="Investment Flow (₹ Cr)")
    st.plotly_chart(fig, use_container_width=True)

# Display Previous Day's Top Gainers and Losers
st.subheader("📅 Previous Day's Top Gainers and Losers")
st.write(f"Updated at: {fetch_timestamp}")
if not previous_day_df.empty:
    top_gainers = previous_day_df[previous_day_df['% Change'] > 0].sort_values(by="% Change", ascending=False).head(5)
    top_losers = previous_day_df[previous_day_df['% Change'] < 0].sort_values(by="% Change", ascending=True).head(5)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Top Gainers**")
        st.dataframe(
            top_gainers.style.format({
                "% Change": "{:.2f}%",
                "Close Price": "{:.2f}"
            }).highlight_max(subset=["% Change"], color='lightgreen')
        )
    with col2:
        st.write("**Top Losers**")
        st.dataframe(
            top_losers.style.format({
                "% Change": "{:.2f}%",
                "Close Price": "{:.2f}"
            }).highlight_min(subset=["% Change"], color='lightcoral')
        )

# Display Top Movers
st.subheader("🔥 Top Movers with Trading Signals")
st.write(f"Updated at: {fetch_timestamp}")
if not top_movers_df.empty:
    for _, row in top_movers_df.iterrows():
        trade = row['trade_suggestion']
        r_factor_analysis = (
            f"Strong momentum" if trade['r_factor'] > 7 else 
            f"Moderate momentum" if trade['r_factor'] > 4 else "Weak momentum"
        )
        with st.expander(f"{row['sector']} - {row['stock_name']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Price**: ₹{row['latest_price']:.2f}")
                st.write(f"**% Change**: {row['change']:.2f}%")
                st.write(f"**Trend**: {row['trend']}")
                st.write(f"**Volume**: {row['volume']:,.0f}")
                st.write(f"**Institutional Score**: {row['institutional_score']:.2f}")
            with col2:
                st.write(f"**R Factor**: {trade['r_factor']:.2f} ({r_factor_analysis})")
                st.write(f"**RSI**: {trade['rsi']:.2f}")
                st.write(f"**MACD**: {trade['macd']:.2f} (Signal: {trade['signal_line']:.2f})")
                st.write(f"**OBV**: {trade['obv']:,.0f}")
            if trade['signal'] != "No Trade":
                st.success(f"**Trade**: {trade['signal']} @ ₹{trade['strike_price']} (Entry: ₹{trade['entry_price']:.2f}, Expiry: {trade['expiry']})")

# Display Stock-wise Analysis
st.subheader("🚀 Stock-wise Analysis")
st.write(f"Updated at: {fetch_timestamp}")
if not stock_df.empty:
    tabs = st.tabs(list(sector_details.keys()))
    for tab, sector in zip(tabs, sector_details.keys()):
        with tab:
            sector_stocks = stock_df[stock_df['Sector'] == sector]
            sector_stocks = sector_stocks.sort_values(by="% Change", key=abs, ascending=False)
            st.dataframe(
                sector_stocks.style.format({
                    "% Change": "{:.2f}%",
                    "Current Price": "{:.2f}",
                    "Volume": "{:,.0f}",
                    "Institutional Activity": "{:.2f}",
                    "R Factor": "{:.2f}"
                }).highlight_max(subset=["% Change"], color='lightgreen')
                .highlight_min(subset=["% Change"], color='lightcoral')
            )

# Display High Activity Stocks During Market Hours
st.subheader("🌟 High Activity Stocks During Market Hours")
st.write(f"Updated at: {fetch_timestamp}")
if not high_activity_df.empty:
    high_activity_df = high_activity_df.sort_values(by="R Factor", ascending=False)
    st.dataframe(
        high_activity_df.style.format({
            "% Change": "{:.2f}%",
            "Current Price": "{:.2f}",
            "Volume": "{:,.0f}",
            "R Factor": "{:.2f}",
            "Institutional Score": "{:.2f}",
            "Detection Time": "{}"
        }).highlight_max(subset=["R Factor"], color='lightgreen')
    )
else:
    st.write("No high activity stocks identified during market hours yet.")

# Display Breakout Stocks
st.subheader("🚀 Breakout Stocks with Trade Plans")
st.write(f"Updated at: {fetch_timestamp}")
if not breakout_df.empty:
    breakout_df = breakout_df.sort_values(by="R Factor", ascending=False)
    for _, row in breakout_df.iterrows():
        with st.expander(f"{row['Sector']} - {row['Stock']} (Detected: {row['Detection Time']})"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Price**: ₹{row['Current Price']:.2f}")
                st.write(f"**% Change**: {row['% Change']:.2f}%")
                st.write(f"**Volume**: {row['Volume']:,.0f}")
                st.write(f"**R Factor**: {row['R Factor']:.2f}")
                st.write(f"**Institutional Score**: {row['Institutional Score']:.2f}")
                st.write(f"**Support**: ₹{row['Support']:.2f}")
                st.write(f"**Resistance**: ₹{row['Resistance']:.2f}")
            with col2:
                st.write(f"**Entry Candle**: {row['Entry Candle']}")
                st.write(f"**Stop Loss**: ₹{row['Stop Loss']:.2f}")
                st.write(f"**Target**: ₹{row['Target']:.2f}")
                if row['Chart Image']:
                    st.image(f"data:image/png;base64,{row['Chart Image']}", use_column_width=True)

# Auto-refresh
st.sidebar.header("🔄 Auto Refresh")
refresh_rate = st.sidebar.slider("Refresh Every (Seconds)", 30, 300, 60)
time.sleep(refresh_rate)
st.rerun()

