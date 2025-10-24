import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sqlalchemy import create_engine, text, select, and_, Table, MetaData
from datetime import datetime, timedelta
from collections import defaultdict
import logging

class TPOProfile:
    def __init__(self, tick_size=0.05):
        """
        Initialize TPO Profile calculator
        
        Args:
            tick_size: Price increment for TPO calculation
        """
        self.tick_size = tick_size
        self.tpo_data = None
        self.poc = None
        self.value_area_high = None
        self.value_area_low = None
        self.initial_balance_high = None
        self.initial_balance_low = None
        
    def calculate_tpo(self, df, time_column='timestamp', price_column='last_price'):
        """
        Calculate TPO profile from tick data with 1-minute bins
        
        Args:
            df: DataFrame with tick data
            time_column: Name of timestamp column
            price_column: Name of price column
        """
        # Ensure timestamp is datetime
        df[time_column] = pd.to_datetime(df[time_column])
        
        # Create 1-minute bins
        df['time_period'] = df[time_column].dt.floor('1min')
        
        # Convert price_column to float to handle decimal.Decimal
        df[price_column] = df[price_column].astype(float)
        
        print(df)

        # Round prices to tick size
        df['price_level'] = (df[price_column] / self.tick_size).round() * self.tick_size
        
        # Build TPO profile
        tpo_dict = defaultdict(list)
        for period in sorted(df['time_period'].unique()):
            period_data = df[df['time_period'] == period]
            prices_in_period = period_data['price_level'].unique()
            letter = chr(65 + len(tpo_dict) % 26)  # A, B, C, ...
            
            for price in prices_in_period:
                tpo_dict[price].append(letter)
        
        # Convert to DataFrame
        self.tpo_data = pd.DataFrame([
            {'price': price, 'tpo_count': len(letters), 'tpo_letters': ''.join(letters)}
            for price, letters in sorted(tpo_dict.items(), reverse=True)
        ])
        
        # Calculate Initial Balance (first 2 time periods or 1 hour)
        time_periods = sorted(df['time_period'].unique())
        ib_periods = time_periods[:min(2, len(time_periods))]
        ib_data = df[df['time_period'].isin(ib_periods)]
        if not ib_data.empty:
            self.initial_balance_high = ib_data['price_level'].max()
            self.initial_balance_low = ib_data['price_level'].min()
        
        # Calculate Point of Control (POC) - price with most TPO count
        if not self.tpo_data.empty:
            self.poc = self.tpo_data.loc[self.tpo_data['tpo_count'].idxmax(), 'price']
        
        # Calculate Value Area (70% of TPO volume)
        self._calculate_value_area()
        
        return self.tpo_data

    
    def _calculate_value_area(self):
        """Calculate Value Area High and Low (70% of volume)"""
        if self.tpo_data is None or self.tpo_data.empty:
            return
        
        total_tpo = self.tpo_data['tpo_count'].sum()
        target_tpo = total_tpo * 0.70
        
        # Start from POC and expand up/down
        poc_idx = self.tpo_data[self.tpo_data['price'] == self.poc].index[0]
        
        accumulated_tpo = self.tpo_data.loc[poc_idx, 'tpo_count']
        upper_idx = poc_idx
        lower_idx = poc_idx
        
        while accumulated_tpo < target_tpo:
            # Check if we can expand up or down
            can_go_up = upper_idx > 0
            can_go_down = lower_idx < len(self.tpo_data) - 1
            
            if not can_go_up and not can_go_down:
                break
            
            # Expand in direction with more TPO
            upper_tpo = self.tpo_data.loc[upper_idx - 1, 'tpo_count'] if can_go_up else 0
            lower_tpo = self.tpo_data.loc[lower_idx + 1, 'tpo_count'] if can_go_down else 0
            
            if upper_tpo >= lower_tpo and can_go_up:
                upper_idx -= 1
                accumulated_tpo += upper_tpo
            elif can_go_down:
                lower_idx += 1
                accumulated_tpo += lower_tpo
        
        self.value_area_high = self.tpo_data.loc[upper_idx, 'price']
        self.value_area_low = self.tpo_data.loc[lower_idx, 'price']
    
    def plot_profile(self, ax=None, show_metrics=True):
        """
        Plot TPO profile with key metrics
        
        Args:
            ax: Matplotlib axis object (creates new if None)
            show_metrics: Whether to show POC, VA, and IB lines
        """
        if self.tpo_data is None or self.tpo_data.empty:
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            ax.clear()
        
        # Plot horizontal bars for TPO count
        prices = self.tpo_data['price'].values
        counts = self.tpo_data['tpo_count'].values
        
        ax.barh(prices, counts, height=self.tick_size*0.8, 
                color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        if show_metrics:
            # Point of Control
            ax.axhline(y=self.poc, color='red', linewidth=2, 
                      linestyle='-', label=f'POC: {self.poc:.2f}', zorder=10)
            
            # Value Area
            if self.value_area_high and self.value_area_low:
                ax.axhline(y=self.value_area_high, color='green', linewidth=1.5,
                          linestyle='--', label=f'VAH: {self.value_area_high:.2f}', zorder=9)
                ax.axhline(y=self.value_area_low, color='green', linewidth=1.5,
                          linestyle='--', label=f'VAL: {self.value_area_low:.2f}', zorder=9)
                
                # Shade value area
                ax.axhspan(self.value_area_low, self.value_area_high, 
                          alpha=0.1, color='green', zorder=1)
            
            # Initial Balance
            if self.initial_balance_high and self.initial_balance_low:
                ax.axhline(y=self.initial_balance_high, color='orange', linewidth=1.5,
                          linestyle=':', label=f'IBH: {self.initial_balance_high:.2f}', zorder=8)
                ax.axhline(y=self.initial_balance_low, color='orange', linewidth=1.5,
                          linestyle=':', label=f'IBL: {self.initial_balance_low:.2f}', zorder=8)
        
        ax.set_xlabel('TPO Count', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_title('Market Profile (TPO)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        return ax


class PostgresDataFetcher:
    def __init__(self, host, database, user, password, port=5432):
        """Initialize PostgreSQL connection using SQLAlchemy"""
        self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        self.engine = create_engine(self.connection_string, pool_pre_ping=True)
    
    def fetch_tick_data(self, table_name, instrument_token, start_time=None, end_time=None):
        """
        Fetch tick data from PostgreSQL using a simple SQL query.

        Args:
            table_name: Name of the table containing tick data
            instrument_token: Instrument token to filter
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
        """
        # Validate table name to prevent SQL injection
        if not table_name.isidentifier():
            raise ValueError("Invalid table name")
        
        # Build query
        query = "SELECT last_price, open, high, low, close, timestamp FROM my_schema.\"{}\"".format(table_name)
        params = {}
        
        conditions = []
        if instrument_token:
            conditions.append("instrument_token = :instrument_token")
            params['instrument_token'] = instrument_token
        if start_time:  # Tick timestamp seems to be UTC, adjust accordingly
            conditions.append("timestamp + interval '5 hours 30 minutes'>= :start_time")
            params['start_time'] = start_time
        if end_time:
            conditions.append("timestamp <= :end_time")   
            params['end_time'] = end_time
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp"
        
        try:
            logging.info(f"Executing query: {query} with params: {params}")
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def close(self):
        """Close the database connection"""
        self.engine.dispose()



class LiveTPOChart:
    def __init__(self, db_fetcher, table_name, symbol=None, 
                 tick_size=0.05, refresh_interval=5000):
        """
        Create live-updating TPO chart
        
        Args:
            db_fetcher: PostgresDataFetcher instance
            table_name: Database table name
            symbol: Trading symbol
            tick_size: Price tick size
            refresh_interval: Update interval in milliseconds
        """
        self.db_fetcher = db_fetcher
        self.table_name = table_name
        self.symbol = symbol
        self.tpo_profile = TPOProfile(tick_size=tick_size)
        self.refresh_interval = refresh_interval
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.suptitle(f'Live TPO Profile - {symbol or "All Symbols"}', 
                         fontsize=16, fontweight='bold')
        
    def update(self, frame):
        """Update function for animation"""
        # Fetch latest data (e.g., today's data)
        end_time = datetime.now()
        start_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
        
        try:
            df = self.db_fetcher.fetch_tick_data(
                self.table_name, 
                symbol=self.symbol,
                start_time=start_time,
                end_time=end_time
            )
            
            if not df.empty:
                self.tpo_profile.calculate_tpo(df)
                self.tpo_profile.plot_profile(ax=self.ax, show_metrics=True)
                
                # Add timestamp
                self.ax.text(0.02, 0.98, f'Last Updated: {datetime.now().strftime("%H:%M:%S")}',
                           transform=self.ax.transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='wheat', alpha=0.5))
        except Exception as e:
            print(f"Error updating chart: {e}")
    
    def start(self):
        """Start the live chart"""
        ani = FuncAnimation(self.fig, self.update, interval=self.refresh_interval, 
                          cache_frame_data=False)
        plt.tight_layout()
        plt.show()
        return ani


# Example Usage
if __name__ == "__main__":
    # Configuration
    DB_CONFIG = {
        'host': 'localhost',
        'database': 'mydb',
        'user': 'postgres',
        'password': 'postgres',
        'port': 5434
    }
    
    # Initialize components
    db_fetcher = PostgresDataFetcher(**DB_CONFIG)
    
    # Option 1: Create static TPO profile
    print("Fetching data...")
    df = db_fetcher.fetch_tick_data(
        table_name='ticks',
        instrument_token=256265,
        start_time=datetime.now().replace(hour=9, minute=5),
        end_time=datetime.now()
    )

    logging.info("Calculating TPO profile...")
    print(df)
    
    if not df.empty:
        tpo = TPOProfile(tick_size=0.05)
        tpo.calculate_tpo(df)
        
        print(f"\nTPO Metrics:")
        print(f"Point of Control: {tpo.poc}")
        print(f"Value Area High: {tpo.value_area_high}")
        print(f"Value Area Low: {tpo.value_area_low}")
        print(f"Initial Balance High: {tpo.initial_balance_high}")
        print(f"Initial Balance Low: {tpo.initial_balance_low}")
        
        # Plot
        tpo.plot_profile()
        plt.show()
    
    # Option 2: Create live-updating chart (uncomment to use)
    # live_chart = LiveTPOChart(
    #     db_fetcher=db_fetcher,
    #     table_name='tick_data',
    #     symbol='NIFTY',
    #     tick_size=0.05,
    #     refresh_interval=5000  # 5 seconds
    # )
    # ani = live_chart.start()