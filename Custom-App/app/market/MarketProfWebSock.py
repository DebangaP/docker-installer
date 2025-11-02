import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import websocket
from datetime import datetime, timedelta
import pytz
from threading import Lock, Thread
import struct
import redis
import psycopg2
from psycopg2.extras import execute_values

# Configure logging
logging.basicConfig(level=logging.INFO)

# Zerodha Configuration
API_KEY = "YOUR_API_KEY"  # Replace
ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"  # Replace
INSTRUMENT_TOKENS = [256265]  # Nifty 50
MODE = "quote"

# Time Zone
IST = pytz.timezone('Asia/Kolkata')

# Redis Configuration
REDIS_HOST = "redis"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_TTL = 86400  # 24 hours

# PostgreSQL Configuration
PG_HOST = "postgres"
PG_PORT = 5432
PG_DATABASE = "mydb"
PG_USER = "postgres"
PG_PASSWORD = "postgres"

# Global data storage
data_lock = Lock()
raw_ticks = []
ohlcv_data = {}
market_structure_data = {}
current_session_start = datetime(2025, 10, 2, 9, 15, 0, tzinfo=IST)
current_session_date = current_session_start.date()

# Initialize Redis
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

# Initialize PostgreSQL
conn = psycopg2.connect(
    host=PG_HOST, port=PG_PORT, database=PG_DATABASE, user=PG_USER, password=PG_PASSWORD
)
cursor = conn.cursor()

# Create tables
cursor.execute("""
    CREATE TABLE IF NOT EXISTS raw_ticks (
        timestamp TIMESTAMP,
        instrument_token INTEGER,
        ltp FLOAT,
        high FLOAT,
        low FLOAT,
        open FLOAT,
        close FLOAT,
        volume INTEGER
    );
    CREATE TABLE IF NOT EXISTS bars (
        timestamp TIMESTAMP,
        instrument_token INTEGER,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        volume INTEGER,
        PRIMARY KEY (timestamp, instrument_token)
    );
    CREATE TABLE IF NOT EXISTS market_structure (
        session_date DATE,
        instrument_token INTEGER,
        poc FLOAT,
        vah FLOAT,
        val FLOAT,
        day_type VARCHAR(50),
        opening_type VARCHAR(50),
        ib_high FLOAT,
        ib_low FLOAT,
        single_prints JSONB,
        poor_high BOOLEAN,
        poor_low BOOLEAN,
        overnight_high FLOAT,
        overnight_low FLOAT,
        PRIMARY KEY (session_date, instrument_token)
    );
    CREATE TABLE IF NOT EXISTS sessions (
        session_date DATE,
        instrument_token INTEGER,
        overnight_high FLOAT,
        overnight_low FLOAT,
        PRIMARY KEY (session_date, instrument_token)
    );
""")
conn.commit()

# Parse binary quote packet
def parse_quote_packet(packet_bytes):
    if len(packet_bytes) < 28:
        return None
    unpacked = struct.unpack('<I i i i i i i I', packet_bytes[:28])
    token = unpacked[0]
    ltp = unpacked[1] / 100.0
    high = unpacked[2] / 100.0
    low = unpacked[3] / 100.0
    open_price = unpacked[4] / 100.0
    close = unpacked[5] / 100.0
    timestamp_ms = unpacked[7]
    timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=IST)
    return {
        'token': token,
        'ltp': ltp,
        'high': high,
        'low': low,
        'open': open_price,
        'close': close,
        'volume': 0,
        'timestamp': timestamp
    }

# WebSocket callbacks
def on_open(ws):
    logging.info("WebSocket connected")
    ws.send(json.dumps({"a": "subscribe", "v": INSTRUMENT_TOKENS}))
    ws.send(json.dumps({"a": "mode", "v": [MODE] + INSTRUMENT_TOKENS}))

def on_message(ws, message):
    if isinstance(message, bytes):
        try:
            num_packets = struct.unpack('<H', message[:2])[0]
            offset = 2
            for _ in range(num_packets):
                packet_len = struct.unpack('<H', message[offset:offset+2])[0]
                offset += 2
                packet = message[offset:offset+packet_len]
                tick = parse_quote_packet(packet)
                if tick and tick['token'] == 256265:
                    # Store in Redis
                    redis_key = f"tick:{tick['token']}:{tick['timestamp'].strftime('%Y%m%d%H%M%S%f')}"
                    redis_client.setex(redis_key, REDIS_TTL, json.dumps(tick))
                    
                    # Store raw tick in memory and Postgres
                    with data_lock:
                        raw_ticks.append(tick)
                        cursor.execute("""
                            INSERT INTO my_schema.raw_ticks (timestamp, instrument_token, ltp, high, low, open, close, volume)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            tick['timestamp'], tick['token'], tick['ltp'], tick['high'], tick['low'],
                            tick['open'], tick['close'], tick['volume']
                        ))
                        conn.commit()
                        
                        # Aggregate to 30-min OHLCV
                        ts = tick['timestamp']
                        bar_start = ts.replace(minute=(ts.minute // 30) * 30, second=0, microsecond=0)
                        if bar_start.date() != current_session_date:
                            return
                        if bar_start not in ohlcv_data:
                            ohlcv_data[bar_start] = tick.copy()
                        else:
                            data = ohlcv_data[bar_start]
                            data['high'] = max(data['high'], tick['high'])
                            data['low'] = min(data['low'], tick['low'])
                            data['close'] = tick['ltp']
                            data['volume'] += tick['volume']
                            data['timestamp'] = ts
                        
                        # Store bar in Postgres
                        cursor.execute("""
                            INSERT INTO my_schema.bars (timestamp, instrument_token, open, high, low, close, volume)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (timestamp, instrument_token) DO UPDATE
                            SET open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                                close = EXCLUDED.close, volume = EXCLUDED.volume
                        """, (
                            bar_start, tick['token'], ohlcv_data[bar_start]['open'],
                            ohlcv_data[bar_start]['high'], ohlcv_data[bar_start]['low'],
                            ohlcv_data[bar_start]['close'], ohlcv_data[bar_start]['volume']
                        ))
                        conn.commit()
                offset += packet_len
        except Exception as e:
            logging.error(f"Error parsing tick: {e}")
    else:
        logging.info(f"Text message: {message}")

def on_error(ws, error):
    logging.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    logging.info("WebSocket closed")
    conn.close()

def start_websocket():
    ws_url = f"wss://ws.kite.trade?api_key={API_KEY}&access_token={ACCESS_TOKEN}"
    ws = websocket.WebSocketApp(ws_url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()


# Market Structure Calculations
def calculate_market_structure(df, prev_day_data=None):
    if df.empty:
        return None
    
    tick_size = 0.05
    periods = [f"{i//2+1:02d}:{('00' if i%2==0 else '30')}" for i in range(len(df)*2)][:len(df)]
    
    # TPO Calculation
    min_price = df['low'].min()
    max_price = df['high'].max()
    price_levels = np.arange(np.floor(min_price / tick_size) * tick_size,
                             np.ceil(max_price / tick_size) * tick_size + tick_size,
                             tick_size)
    price_levels = np.round(price_levels, 2)
    tpo_matrix = pd.DataFrame(index=price_levels, columns=periods)
    tpo_matrix[:] = ''
    
    for i, (_, row) in enumerate(df.iterrows()):
        period = periods[i]
        period_prices = np.arange(np.floor(row['low'] / tick_size) * tick_size,
                                  np.ceil(row['high'] / tick_size) * tick_size + tick_size,
                                  tick_size)
        period_prices = np.round(period_prices, 2)
        for price in period_prices:
            if price in tpo_matrix.index:
                tpo_matrix.loc[price, period] = period[0]
    
    tpo_counts = tpo_matrix.apply(lambda x: sum(x != ''), axis=1)
    if tpo_counts.empty:
        return None
    
    # POC and Value Area
    poc_price = tpo_counts.idxmax()
    poc_count = tpo_counts.max()
    total_tpos = tpo_counts.sum()
    value_area_tpos = total_tpos * 0.7
    current_tpos = poc_count
    vah = poc_price
    val = poc_price
    sorted_levels = tpo_counts.sort_index()
    poc_idx = sorted_levels.index.get_loc(poc_price)
    upper_idx, lower_idx = poc_idx + 1, poc_idx - 1
    while current_tpos < value_area_tpos and (upper_idx < len(sorted_levels) or lower_idx >= 0):
        upper_count = sorted_levels.iloc[upper_idx] if upper_idx < len(sorted_levels) else 0
        lower_count = sorted_levels.iloc[lower_idx] if lower_idx >= 0 else 0
        if upper_count >= lower_count and upper_idx < len(sorted_levels):
            vah = sorted_levels.index[upper_idx]
            current_tpos += upper_count
            upper_idx += 1
        elif lower_idx >= 0:
            val = sorted_levels.index[lower_idx]
            current_tpos += lower_count
            lower_idx -= 1
        else:
            break
    
    # Initial Balance (first hour, 9:15–10:15)
    ib_end = current_session_start + timedelta(hours=1)
    ib_df = df[df['timestamp'] <= ib_end]
    ib_high = ib_df['high'].max() if not ib_df.empty else None
    ib_low = ib_df['low'].min() if not ib_df.empty else None
    
    # Opening Type
    opening_type = "Unknown"
    first_bar = df.iloc[0] if not df.empty else None
    if prev_day_data and first_bar is not None:
        prev_vah = prev_day_data.get('vah')
        prev_val = prev_day_data.get('val')
        prev_overnight_high = prev_day_data.get('overnight_high')
        prev_overnight_low = prev_day_data.get('overnight_low')
        open_price = first_bar['open']
        first_hour_close = df[df['timestamp'] <= ib_end]['close'].iloc[-1] if not ib_df.empty else open_price
        
        # Open Drive: Opens outside VA and moves further
        if open_price > prev_vah and first_hour_close > open_price:
            opening_type = "Open Drive Up"
        elif open_price < prev_val and first_hour_close < open_price:
            opening_type = "Open Drive Down"
        # Open Test Drive: Tests overnight high/low or VA edge, then drives
        elif (open_price >= prev_vah and open_price <= prev_overnight_high and first_hour_close > prev_vah) or \
             (open_price <= prev_val and open_price >= prev_overnight_low and first_hour_close < prev_val):
            opening_type = "Open Test Drive"
        # Open Rejection Reverse: Tests one direction, reverses
        elif (open_price > prev_vah and first_hour_close < prev_vah) or \
             (open_price < prev_val and first_hour_close > prev_val):
            opening_type = "Open Rejection Reverse"
        # Open Auction: Rotational within/near VA
        else:
            opening_type = "Open Auction"
    
    # Day Type
    day_type = "Normal"
    if not ib_df.empty:
        session_range = max_price - min_price
        ib_range = ib_high - ib_low
        if session_range > 2 * ib_range:
            day_type = "Trend" if df['close'].iloc[-1] > ib_high or df['close'].iloc[-1] < ib_low else "Normal Variation"
        # Double Distribution: Check for two distinct TPO clusters
        tpo_peaks = (tpo_counts.shift(1) < tpo_counts) & (tpo_counts.shift(-1) < tpo_counts)
        if tpo_peaks.sum() > 1:
            day_type = "Double Distribution"
        elif abs(df['close'].iloc[-1] - (ib_high + ib_low)/2) < ib_range/2:
            day_type = "Neutral"
    
    # Single Prints and Poor Highs/Lows
    single_prints = [price for price, count in tpo_counts.items() if count == 1 and val < price < vah]
    session_high = max_price
    session_low = min_price
    poor_high = tpo_counts.get(session_high, 0) > 1
    poor_low = tpo_counts.get(session_low, 0) > 1
    
    # Overnight High/Low (approximate from pre-market ticks if available)
    pre_market_ticks = [t for t in raw_ticks if t['timestamp'] < current_session_start]
    overnight_high = max([t['high'] for t in pre_market_ticks], default=None)
    overnight_low = min([t['low'] for t in pre_market_ticks], default=None)
    
    # Store in market_structure_data
    structure = {
        'poc': poc_price,
        'vah': vah,
        'val': val,
        'day_type': day_type,
        'opening_type': opening_type,
        'ib_high': ib_high,
        'ib_low': ib_low,
        'single_prints': single_prints,
        'poor_high': poor_high,
        'poor_low': poor_low,
        'overnight_high': overnight_high,
        'overnight_low': overnight_low
    }
    
    # Store in PostgreSQL
    cursor.execute("""
        INSERT INTO my_schema.market_structure (session_date, instrument_token, poc, vah, val, day_type, opening_type,
                                     ib_high, ib_low, single_prints, poor_high, poor_low, overnight_high, overnight_low)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (session_date, instrument_token) DO UPDATE
        SET poc = EXCLUDED.poc, vah = EXCLUDED.vah, val = EXCLUDED.val, day_type = EXCLUDED.day_type,
            opening_type = EXCLUDED.opening_type, ib_high = EXCLUDED.ib_high, ib_low = EXCLUDED.ib_low,
            single_prints = EXCLUDED.single_prints, poor_high = EXCLUDED.poor_high, poor_low = EXCLUDED.poor_low,
            overnight_high = EXCLUDED.overnight_high, overnight_low = EXCLUDED.overnight_low
    """, (
        current_session_date, 256265, poc_price, vah, val, day_type, opening_type,
        ib_high, ib_low, json.dumps(single_prints), poor_high, poor_low, overnight_high, overnight_low
    ))
    if overnight_high and overnight_low:
        cursor.execute("""
            INSERT INTO my_schema.sessions (session_date, instrument_token, overnight_high, overnight_low)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (session_date, instrument_token) DO UPDATE
            SET overnight_high = EXCLUDED.overnight_high, overnight_low = EXCLUDED.overnight_low
        """, (current_session_date, 256265, overnight_high, overnight_low))
    conn.commit()
    
    return {
        'tpo_matrix': tpo_matrix,
        'tpo_counts': tpo_counts,
        'structure': structure
    }

# Plot Update
fig, ax = plt.subplots(figsize=(12, 8))
def update_plot(frame):
    ax.clear()
    with data_lock:
        df = pd.DataFrame.from_dict(ohlcv_data, orient='index').sort_index() if ohlcv_data else pd.DataFrame()
    
    # Fetch previous day's data
    cursor.execute("""
        SELECT poc, vah, val, overnight_high, overnight_low
        FROM my_schema.market_structure
        WHERE session_date = %s AND instrument_token = %s
    """, (current_session_date - timedelta(days=1), 256265))
    prev_day_data = cursor.fetchone()
    prev_day_data = {
        'poc': prev_day_data[0],
        'vah': prev_day_data[1],
        'val': prev_day_data[2],
        'overnight_high': prev_day_data[3],
        'overnight_low': prev_day_data[4]
    } if prev_day_data else {}
    
    if df.empty:
        ax.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center', transform=ax.transAxes)
        return ax
    
    result = calculate_market_structure(df, prev_day_data)
    if not result:
        return ax
    
    tpo_matrix = result['tpo_matrix']
    tpo_counts = result['tpo_counts']
    structure = result['structure']
    
    # Plot TPOs
    for price in tpo_matrix.index:
        tpos = ''.join(tpo_matrix.loc[price].values)
        ax.text(0, price, tpos, fontsize=10, verticalalignment='center', fontfamily='monospace')
        if price in structure['single_prints']:
            ax.axhline(price, color='purple', linestyle=':', alpha=0.5, label='Single Print' if 'Single Print' not in ax.get_legend_handles_labels()[1] else '')
    
    # Plot POC, VAH, VAL
    if structure['poc']:
        ax.axhline(structure['poc'], color='r', linestyle='--', label=f"POC: {structure['poc']:.2f}")
    if structure['vah'] and structure['val']:
        ax.axhline(structure['vah'], color='g', linestyle='--', label=f"VAH: {structure['vah']:.2f}")
        ax.axhline(structure['val'], color='b', linestyle='--', label=f"VAL: {structure['val']:.2f}")
    
    # Plot IB
    if structure['ib_high'] and structure['ib_low']:
        ax.axhline(structure['ib_high'], color='orange', linestyle='-.', label=f"IB High: {structure['ib_high']:.2f}")
        ax.axhline(structure['ib_low'], color='orange', linestyle='-.', label=f"IB Low: {structure['ib_low']:.2f}")
    
    # Plot previous day's VA (if available)
    if prev_day_data.get('vah') and prev_day_data.get('val'):
        ax.axhline(prev_day_data['vah'], color='gray', linestyle=':', alpha=0.5, label=f"Prev VAH: {prev_day_data['vah']:.2f}")
        ax.axhline(prev_day_data['val'], color='gray', linestyle=':', alpha=0.5, label=f"Prev VAL: {prev_day_data['val']:.2f}")
    
    # Histogram
    ax_twin = ax.twinx()
    ax_twin.barh(tpo_counts.index, tpo_counts.values, height=0.05, alpha=0.3, color='gray')
    
    # Annotations
    ax.text(0.05, 0.95, f"Day Type: {structure['day_type']}\nOpening Type: {structure['opening_type']}",
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    if structure['poor_high']:
        ax.text(0.05, 0.90, "Poor High", color='red', transform=ax.transAxes)
    if structure['poor_low']:
        ax.text(0.05, 0.85, "Poor Low", color='red', transform=ax.transAxes)
    
    ax.set_xlabel('TPO Letters (Time Periods)')
    ax.set_ylabel('Price')
    ax_twin.set_ylabel('TPO Count')
    ax.set_title(f'Live TPO Chart - Nifty 50 ({datetime.now(IST).strftime("%Y-%m-%d %H:%M")})')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    logging.info(f"Updated: Bars: {len(df)}, POC: {structure['poc']:.2f if structure['poc'] else 'N/A'}, Day: {structure['day_type']}")
    return ax

# Start WebSocket
ws_thread = Thread(target=start_websocket, daemon=True)
ws_thread.start()

# Animate plot
ani = FuncAnimation(fig, update_plot, interval=30000, cache_frame_data=False)
plt.show()