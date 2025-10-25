"""
Market Charts Generator using Matplotlib
Replicates Grafana dashboard functionality using matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
from sqlalchemy import create_engine, text
import io
import base64

class MarketChartsGenerator:
    def __init__(self, db_config):
        """Initialize with database configuration"""
        self.db_config = db_config
        self.engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        self.ist = pytz.timezone('Asia/Kolkata')
        
    def get_pre_market_ticks(self, analysis_date=None):
        """Get pre-market tick data (9:05-9:15)"""
        if not analysis_date:
            analysis_date = datetime.now().strftime("%Y-%m-%d")
            
        query = """
        SELECT timestamp + interval '5 hours 30 minutes' as timestamp, 
               open, high, low, last_price as close
        FROM my_schema.ticks
        WHERE instrument_token = 256265
        AND run_date = :analysis_date
        AND timestamp + interval '5 hours 30 minutes' < CAST(:analysis_date AS DATE) + INTERVAL '9 hours 15 minutes'
        ORDER BY timestamp
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {
                'analysis_date': analysis_date
            })
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
        return df
    
    def get_market_ticks(self, analysis_date=None):
        """Get market hours tick data (9:15-15:30)"""
        if not analysis_date:
            analysis_date = datetime.now().strftime("%Y-%m-%d")
            
        query = """
        SELECT timestamp, open, high, low, last_price as close
        FROM my_schema.ticks
        WHERE instrument_token = 256265
        AND timestamp + interval '5 hours 30 minutes' BETWEEN CAST(:analysis_date AS DATE) + INTERVAL '9 hours 15 minutes' 
        AND CAST(:analysis_date AS DATE) + INTERVAL '15 hours 30 minutes'
        ORDER BY timestamp
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {
                'analysis_date': analysis_date
            })
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
        return df
    
    def get_latest_trades(self):
        """Get latest trades data"""
        query = """
        SELECT trade_id, order_id, exchange, trading_symbol, transaction_type, quantity, average_price 
        FROM my_schema.trades
        WHERE run_date = (SELECT MAX(run_date) FROM my_schema.trades)
        ORDER BY trade_id DESC
        LIMIT 20
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
        return df
    
    def get_margin_data(self):
        """Get margin/cash available data"""
        query = """
        SELECT DISTINCT margin_type, net, available_cash, available_live_balance
        FROM my_schema.margins
        WHERE run_date = (SELECT MAX(run_date) FROM my_schema.margins)
        AND enabled IS TRUE
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
        return df
    
    def get_holdings_data(self):
        """Get holdings data"""
        query = """
        SELECT trading_symbol, quantity, average_price, close_price, pnl
        FROM my_schema.holdings 
        WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
        ORDER BY trading_symbol DESC
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
        return df
    
    def get_positions_data(self):
        """Get positions data"""
        query = """
        SELECT fetch_timestamp, position_type, trading_symbol, product, exchange, average_price, pnl 
        FROM my_schema.positions 
        WHERE run_date = (SELECT MAX(run_date) FROM my_schema.positions)
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
        return df
    
    def get_gainers_losers(self):
        """Get today's biggest gainers and losers"""
        query = """
        SELECT "Curr".scrip_id, 100*("Curr".price_close - "Prev".price_close)/"Prev".price_close "Gain"
        FROM my_schema.master_scrips ms,
        (
        SELECT SCRIP_ID, PRICE_CLOSE FROM my_schema.rt_intraday_price rip 
        WHERE PRICE_DATE = (SELECT MAX(PRICE_DATE) FROM my_schema.rt_intraday_price rip2 WHERE country = 'IN' 
                        AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH'))
        AND country = 'IN'
        AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
        ) "Curr",
        (
        SELECT SCRIP_ID, PRICE_CLOSE FROM my_schema.rt_intraday_price rip 
        WHERE PRICE_DATE = (SELECT DATE(MAX(PRICE_DATE))::text FROM my_schema.rt_intraday_price rip2 
                        WHERE price_date < (SELECT max(price_date) FROM my_schema.rt_intraday_price rip5 
                                        WHERE country = 'IN' 
                                        AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')) )
        AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
        AND country = 'IN'
        ) "Prev"
        WHERE "Curr".scrip_id = "Prev".scrip_id
        ORDER BY "Gain" DESC
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
        return df
    
    def plot_candlestick_chart(self, df, title, ax=None):
        """Plot candlestick chart"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        if df.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return ax
        
        # Create candlestick data
        dates = df.index
        opens = df['open']
        highs = df['high']
        lows = df['low']
        closes = df['close']
        
        # Plot candlesticks
        for i, (date, open_price, high, low, close) in enumerate(zip(dates, opens, highs, lows, closes)):
            color = 'green' if close >= open_price else 'red'
            
            # Draw the high-low line
            ax.plot([i, i], [low, high], color='black', linewidth=1)
            
            # Draw the open-close rectangle
            height = abs(close - open_price)
            bottom = min(open_price, close)
            
            rect = Rectangle((i-0.3, bottom), 0.6, height, 
                           facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        
        # Format x-axis
        if len(dates) > 0:
            step = max(1, len(dates) // 10)
            ax.set_xticks(range(0, len(dates), step))
            ax.set_xticklabels([dates[i].strftime('%H:%M') for i in range(0, len(dates), step)], 
                              rotation=45)
        
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_data_table(self, df, title, ax=None, page=1, rows_per_page=10, font_size=12):
        """Plot data as a table with pagination"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 8))
        
        if df.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=font_size)
            ax.set_title(title, fontsize=font_size+2, fontweight='bold')
            return ax
        
        # Calculate pagination
        total_rows = len(df)
        total_pages = (total_rows + rows_per_page - 1) // rows_per_page
        start_idx = (page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)
        
        # Get page data
        page_df = df.iloc[start_idx:end_idx]
        
        # Create table
        table_data = []
        for _, row in page_df.iterrows():
            table_data.append([str(val) for val in row.values])
        
        table = ax.table(cellText=table_data,
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 0.9])
        
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Add pagination info
        page_info = f"Page {page} of {total_pages} | Showing {start_idx+1}-{end_idx} of {total_rows} records"
        ax.text(0.5, 0.95, page_info, ha='center', va='top', transform=ax.transAxes, 
                fontsize=font_size-2, style='italic')
        
        ax.set_title(title, fontsize=font_size+2, fontweight='bold', pad=20)
        ax.axis('off')
        
        return ax
    
    def generate_market_dashboard(self, analysis_date=None):
        """Generate complete market dashboard with improved layout"""
        if not analysis_date:
            analysis_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create figure with subplots - wider TPO charts
        fig = plt.figure(figsize=(24, 20))
        
        # Define grid layout - TPO charts take full width, tables below
        gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 1], width_ratios=[1, 1])
        
        # Get data
        pre_market_df = self.get_pre_market_ticks(analysis_date)
        market_df = self.get_market_ticks(analysis_date)
        trades_df = self.get_latest_trades()
        holdings_df = self.get_holdings_data()
        positions_df = self.get_positions_data()
        gainers_df = self.get_gainers_losers()
        
        # Plot 1: Pre-market candlesticks (full width)
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_candlestick_chart(pre_market_df, f"Nifty50 Pre-Market Ticks ({analysis_date})", ax1)
        
        # Plot 2: Market hours candlesticks (full width)
        ax2 = fig.add_subplot(gs[1, :])
        self.plot_candlestick_chart(market_df, f"Nifty50 Market Ticks ({analysis_date})", ax2)
        
        # Plot 3: Latest trades table (full width)
        ax3 = fig.add_subplot(gs[2, :])
        self.plot_data_table(trades_df, "Latest Trades", ax3, font_size=14)
        
        # Plot 4: Positions table (full width)
        ax4 = fig.add_subplot(gs[3, :])
        self.plot_data_table(positions_df, "Positions", ax4, font_size=14)
        
        # Plot 5: Holdings table (full width)
        ax5 = fig.add_subplot(gs[4, :])
        self.plot_data_table(holdings_df, "Holdings", ax5, font_size=14)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def generate_gainers_losers_chart(self):
        """Generate gainers and losers chart"""
        df = self.get_gainers_losers()
        
        if df.empty:
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.text(0.5, 0.5, 'No gainers/losers data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Today's Biggest Gainers & Losers", fontsize=16, fontweight='bold')
            ax.axis('off')
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Top 10 gainers
            top_gainers = df.head(10)
            colors_gainers = ['green' if x >= 0 else 'red' for x in top_gainers['Gain']]
            
            bars1 = ax1.barh(range(len(top_gainers)), top_gainers['Gain'], color=colors_gainers, alpha=0.7)
            ax1.set_yticks(range(len(top_gainers)))
            ax1.set_yticklabels(top_gainers['scrip_id'])
            ax1.set_xlabel('Gain %')
            ax1.set_title('Top 10 Gainers', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars1, top_gainers['Gain'])):
                ax1.text(value + (0.1 if value >= 0 else -0.1), bar.get_y() + bar.get_height()/2, 
                        f'{value:.2f}%', ha='left' if value >= 0 else 'right', va='center', fontweight='bold')
            
            # Top 10 losers
            top_losers = df.tail(10)
            colors_losers = ['green' if x >= 0 else 'red' for x in top_losers['Gain']]
            
            bars2 = ax2.barh(range(len(top_losers)), top_losers['Gain'], color=colors_losers, alpha=0.7)
            ax2.set_yticks(range(len(top_losers)))
            ax2.set_yticklabels(top_losers['scrip_id'])
            ax2.set_xlabel('Gain %')
            ax2.set_title('Top 10 Losers', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars2, top_losers['Gain'])):
                ax2.text(value + (0.1 if value >= 0 else -0.1), bar.get_y() + bar.get_height()/2, 
                        f'{value:.2f}%', ha='left' if value >= 0 else 'right', va='center', fontweight='bold')
            
            plt.suptitle("Today's Biggest Gainers & Losers", fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def generate_candlestick_chart(self, analysis_date=None, chart_type="market"):
        """Generate individual candlestick chart"""
        if chart_type == "pre_market":
            df = self.get_pre_market_ticks(analysis_date)
            title = f"Nifty50 Pre-Market Candlesticks ({analysis_date})"
        else:
            df = self.get_market_ticks(analysis_date)
            title = f"Nifty50 Market Candlesticks ({analysis_date})"
        
        fig, ax = plt.subplots(figsize=(15, 8))
        self.plot_candlestick_chart(df, title, ax)
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def generate_gainers_losers_chart(self):
        """Generate gainers and losers chart"""
        df = self.get_gainers_losers()
        
        if df.empty:
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.text(0.5, 0.5, 'No gainers/losers data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Today's Biggest Gainers & Losers", fontsize=16, fontweight='bold')
            ax.axis('off')
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Top 10 gainers
            top_gainers = df.head(10)
            colors_gainers = ['green' if x >= 0 else 'red' for x in top_gainers['Gain']]
            
            bars1 = ax1.barh(range(len(top_gainers)), top_gainers['Gain'], color=colors_gainers, alpha=0.7)
            ax1.set_yticks(range(len(top_gainers)))
            ax1.set_yticklabels(top_gainers['scrip_id'])
            ax1.set_xlabel('Gain %')
            ax1.set_title('Top 10 Gainers', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars1, top_gainers['Gain'])):
                ax1.text(value + (0.1 if value >= 0 else -0.1), bar.get_y() + bar.get_height()/2, 
                        f'{value:.2f}%', ha='left' if value >= 0 else 'right', va='center', fontweight='bold')
            
            # Top 10 losers
            top_losers = df.tail(10)
            colors_losers = ['green' if x >= 0 else 'red' for x in top_losers['Gain']]
            
            bars2 = ax2.barh(range(len(top_losers)), top_losers['Gain'], color=colors_losers, alpha=0.7)
            ax2.set_yticks(range(len(top_losers)))
            ax2.set_yticklabels(top_losers['scrip_id'])
            ax2.set_xlabel('Gain %')
            ax2.set_title('Top 10 Losers', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars2, top_losers['Gain'])):
                ax2.text(value + (0.1 if value >= 0 else -0.1), bar.get_y() + bar.get_height()/2, 
                        f'{value:.2f}%', ha='left' if value >= 0 else 'right', va='center', fontweight='bold')
            
            plt.suptitle("Today's Biggest Gainers & Losers", fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def generate_trades_table(self):
        """Generate trades table chart"""
        df = self.get_latest_trades()
        
        fig, ax = plt.subplots(figsize=(15, 8))
        self.plot_data_table(df, "Latest Trades", ax)
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def generate_gainers_losers_chart(self):
        """Generate gainers and losers chart"""
        df = self.get_gainers_losers()
        
        if df.empty:
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.text(0.5, 0.5, 'No gainers/losers data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Today's Biggest Gainers & Losers", fontsize=16, fontweight='bold')
            ax.axis('off')
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Top 10 gainers
            top_gainers = df.head(10)
            colors_gainers = ['green' if x >= 0 else 'red' for x in top_gainers['Gain']]
            
            bars1 = ax1.barh(range(len(top_gainers)), top_gainers['Gain'], color=colors_gainers, alpha=0.7)
            ax1.set_yticks(range(len(top_gainers)))
            ax1.set_yticklabels(top_gainers['scrip_id'])
            ax1.set_xlabel('Gain %')
            ax1.set_title('Top 10 Gainers', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars1, top_gainers['Gain'])):
                ax1.text(value + (0.1 if value >= 0 else -0.1), bar.get_y() + bar.get_height()/2, 
                        f'{value:.2f}%', ha='left' if value >= 0 else 'right', va='center', fontweight='bold')
            
            # Top 10 losers
            top_losers = df.tail(10)
            colors_losers = ['green' if x >= 0 else 'red' for x in top_losers['Gain']]
            
            bars2 = ax2.barh(range(len(top_losers)), top_losers['Gain'], color=colors_losers, alpha=0.7)
            ax2.set_yticks(range(len(top_losers)))
            ax2.set_yticklabels(top_losers['scrip_id'])
            ax2.set_xlabel('Gain %')
            ax2.set_title('Top 10 Losers', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars2, top_losers['Gain'])):
                ax2.text(value + (0.1 if value >= 0 else -0.1), bar.get_y() + bar.get_height()/2, 
                        f'{value:.2f}%', ha='left' if value >= 0 else 'right', va='center', fontweight='bold')
            
            plt.suptitle("Today's Biggest Gainers & Losers", fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"

# Example usage
if __name__ == "__main__":
    DB_CONFIG = {
        'host': 'postgres',
        'database': 'mydb',
        'user': 'postgres',
        'password': 'postgres',
        'port': 5432
    }
    
    chart_generator = MarketChartsGenerator(DB_CONFIG)
    
    # Generate complete dashboard
    dashboard_image = chart_generator.generate_market_dashboard("2025-10-20")
    print("Dashboard generated successfully")
    
    # Generate individual charts
    candlestick_image = chart_generator.generate_candlestick_chart("2025-10-20", "market")
    trades_image = chart_generator.generate_trades_table()
    print("Individual charts generated successfully")
