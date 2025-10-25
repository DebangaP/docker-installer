import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sqlalchemy import create_engine, text, select, and_, Table, MetaData
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import threading
import time
from pytz import timezone

class TPOProfile:
    def __init__(self, tick_size=5):
        """
        Initialize TPO Profile calculator
        
        Args:
            tick_size: Price increment for TPO calculation
        """
        self.tick_size = tick_size
        self.tpo_data = None
        self.poc = None
        self.poc_high = None
        self.poc_low = None
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
        
        # Calculate Point of Control (POC) - price(s) with most TPO count
        if not self.tpo_data.empty:
            max_tpo_count = self.tpo_data['tpo_count'].max()
            poc_levels = self.tpo_data[self.tpo_data['tpo_count'] == max_tpo_count]
            
            if len(poc_levels) == 1:
                # Single POC level
                self.poc = poc_levels['price'].iloc[0]
                self.poc_high = self.poc
                self.poc_low = self.poc
            else:
                # POC range - multiple levels with same max TPO count
                self.poc_low = poc_levels['price'].min()
                self.poc_high = poc_levels['price'].max()
                self.poc = (self.poc_low + self.poc_high) / 2  # Mid-point for single value representation
        
        # Calculate Value Area (70% of TPO volume)
        self._calculate_value_area()
        
        return self.tpo_data

    
    def _calculate_value_area(self):
        """Calculate Value Area High and Low (70% of volume)"""
        if self.tpo_data is None or self.tpo_data.empty:
            return
        
        total_tpo = self.tpo_data['tpo_count'].sum()
        target_tpo = total_tpo * 0.70
        
        # Handle POC range - start from the middle of POC range
        if self.poc_low == self.poc_high:
            # Single POC level
            poc_idx = self.tpo_data[self.tpo_data['price'] == self.poc].index[0]
        else:
            # POC range - find the middle price level or closest to middle
            poc_range_data = self.tpo_data[
                (self.tpo_data['price'] >= self.poc_low) & 
                (self.tpo_data['price'] <= self.poc_high)
            ]
            if not poc_range_data.empty:
                # Use the price level closest to the middle of POC range
                poc_idx = poc_range_data.iloc[len(poc_range_data)//2].name
            else:
                # Fallback to any POC level
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
    
    def plot_profile(self, ax=None, show_metrics=True, show_letters=True):
        """
        Plot TPO profile with key metrics and TPO letters
        
        Args:
            ax: Matplotlib axis object (creates new if None)
            show_metrics: Whether to show POC, VA, and IB lines
            show_letters: Whether to show TPO letters on the bars
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
        
        bars = ax.barh(prices, counts, height=self.tick_size*0.8, 
                      color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add TPO letters on the bars
        if show_letters and 'tpo_letters' in self.tpo_data.columns:
            for i, (price, count, letters) in enumerate(zip(prices, counts, self.tpo_data['tpo_letters'])):
                if letters and count > 0:
                    # Position text in the middle of the bar
                    ax.text(count/2, price, letters, 
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           color='white', zorder=15)
        
        if show_metrics:
            # Point of Control
            if self.poc_low == self.poc_high:
                # Single POC level
                ax.axhline(y=self.poc, color='red', linewidth=2, 
                          linestyle='-', label=f'POC: {self.poc:.2f}', zorder=10)
            else:
                # POC range
                ax.axhline(y=self.poc_low, color='red', linewidth=2, 
                          linestyle='-', label=f'POC Range: {self.poc_low:.2f} - {self.poc_high:.2f}', zorder=10)
                ax.axhline(y=self.poc_high, color='red', linewidth=2, 
                          linestyle='-', zorder=10)
                # Shade POC range
                ax.axhspan(self.poc_low, self.poc_high, 
                          alpha=0.1, color='red', zorder=1)
            
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
        ax.legend(loc='upper right', fontsize=14)
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
                 tick_size=5, refresh_interval=5000):
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
                self.tpo_profile.plot_profile(ax=self.ax, show_metrics=True, show_letters=True)
                
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


class RealTimeTPOProfile:
    def __init__(self, db_fetcher, table_name, instrument_token, tick_size=5, 
                 update_interval=5, market_start_time="09:15", market_end_time="15:30",
                 analysis_date=None):
        """
        Real-time TPO Profile generator for continuous market data processing
        
        Args:
            db_fetcher: PostgresDataFetcher instance
            table_name: Database table name
            instrument_token: Instrument token to filter
            tick_size: Price tick size for TPO calculation
            update_interval: Update interval in seconds
            market_start_time: Market start time in HH:MM format (IST)
            market_end_time: Market end time in HH:MM format (IST)
            analysis_date: Date for analysis in 'YYYY-MM-DD' format, None for current date
        """
        self.db_fetcher = db_fetcher
        self.table_name = table_name
        self.instrument_token = instrument_token
        self.tick_size = tick_size
        self.update_interval = update_interval
        self.market_start_time = market_start_time
        self.market_end_time = market_end_time
        self.analysis_date = analysis_date
        
        # Initialize TPO profile
        self.tpo_profile = TPOProfile(tick_size=tick_size)
        
        # Market hours in IST
        self.ist = timezone('Asia/Kolkata')
        
        # Threading control
        self.is_running = False
        self.update_thread = None
        
        # Data storage for incremental updates
        self.last_update_time = None
        self.cumulative_data = pd.DataFrame()
        
        # Callbacks for external updates
        self.on_update_callbacks = []
        
    def add_update_callback(self, callback):
        """Add callback function to be called when TPO profile is updated"""
        self.on_update_callbacks.append(callback)
    
    def is_market_hours(self):
        """Check if current time is within market hours (IST)"""
        if self.analysis_date is None:
            # Use current date - check actual market hours
            now_ist = datetime.now(self.ist)
            current_time = now_ist.strftime("%H:%M")
            
            # Check if it's a weekday (Monday=0, Sunday=6)
            if now_ist.weekday() >= 5:  # Weekend
                return False
                
            return self.market_start_time <= current_time <= self.market_end_time
        else:
            # Use specified date - always return True for historical analysis
            return True
    
    def get_market_start_datetime(self):
        """Get today's market start datetime in IST"""
        now_ist = datetime.now(self.ist)
        start_hour, start_minute = map(int, self.market_start_time.split(':'))
        return now_ist.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
    
    def fetch_new_ticks(self):
        """Fetch new ticks since last update"""
        try:
            # Determine the analysis date
            if self.analysis_date is None:
                # Use current date
                analysis_date = datetime.now(self.ist).date()
            else:
                # Use specified date
                analysis_date = datetime.strptime(self.analysis_date, '%Y-%m-%d').date()
            
            # Set start time for fetching data
            if self.last_update_time is None:
                # First run - fetch from market start on analysis date
                start_hour, start_minute = map(int, self.market_start_time.split(':'))
                start_time = datetime.combine(analysis_date, datetime.min.time().replace(
                    hour=start_hour, minute=start_minute, second=0, microsecond=0
                )).replace(tzinfo=self.ist)
            else:
                # Subsequent runs - fetch from last update
                start_time = self.last_update_time
            
            if self.analysis_date is None:
                # Real-time mode - use current time as end time
                end_time = datetime.now(self.ist)
            else:
                # Historical mode - simulate progressive time
                if self.last_update_time is None:
                    # Start with first 5 minutes of market data
                    end_time = start_time + timedelta(minutes=5)
                else:
                    # Increment by 2 minutes for each update (simulating real-time)
                    end_time = self.last_update_time + timedelta(minutes=2)
                
                # Don't go beyond market end time
                end_hour, end_minute = map(int, self.market_end_time.split(':'))
                market_end = datetime.combine(analysis_date, datetime.min.time().replace(
                    hour=end_hour, minute=end_minute, second=0, microsecond=0
                )).replace(tzinfo=self.ist)
                if end_time > market_end:
                    end_time = market_end
            
            # Fetch data from database
            df = self.db_fetcher.fetch_tick_data(
                table_name=self.table_name,
                instrument_token=self.instrument_token,
                start_time=start_time,
                end_time=end_time
            )
            
            if not df.empty:
                # Convert timestamp to IST if needed
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Update last update time
                self.last_update_time = df['timestamp'].max()
                
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logging.error(f"Error fetching new ticks: {e}")
            return pd.DataFrame()
    
    def update_tpo_profile(self):
        """Update TPO profile with new data"""
        try:
            # Fetch new ticks
            new_data = self.fetch_new_ticks()
            
            if not new_data.empty:
                # Append to cumulative data
                self.cumulative_data = pd.concat([self.cumulative_data, new_data], 
                                               ignore_index=True)
                
                # Remove duplicates based on timestamp
                self.cumulative_data = self.cumulative_data.drop_duplicates(
                    subset=['timestamp'], keep='last'
                ).sort_values('timestamp')
                
                # Calculate TPO profile with all data
                self.tpo_profile.calculate_tpo(self.cumulative_data)
                
                # Call update callbacks
                for callback in self.on_update_callbacks:
                    try:
                        callback(self.tpo_profile)
                    except Exception as e:
                        logging.error(f"Error in update callback: {e}")
                
                logging.info(f"TPO Profile updated. POC: {self.tpo_profile.poc}, "
                           f"VAH: {self.tpo_profile.value_area_high}, "
                           f"VAL: {self.tpo_profile.value_area_low}")
                
        except Exception as e:
            logging.error(f"Error updating TPO profile: {e}")
    
    def start_real_time_updates(self):
        """Start real-time updates in a separate thread"""
        if self.is_running:
            logging.warning("Real-time updates already running")
            return
        
        self.is_running = True
        
        def update_loop():
            while self.is_running:
                try:
                    if self.is_market_hours():
                        self.update_tpo_profile()
                    else:
                        logging.info("Outside market hours, skipping update")
                    
                    # Wait for next update
                    time.sleep(self.update_interval)
                    
                except Exception as e:
                    logging.error(f"Error in update loop: {e}")
                    time.sleep(self.update_interval)
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
        logging.info("Real-time TPO profile updates started")
    
    def stop_real_time_updates(self):
        """Stop real-time updates"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        logging.info("Real-time TPO profile updates stopped")
    
    def get_current_profile(self):
        """Get current TPO profile data"""
        return {
            'tpo_data': self.tpo_profile.tpo_data,
            'poc': self.tpo_profile.poc,
            'poc_low': self.tpo_profile.poc_low,
            'poc_high': self.tpo_profile.poc_high,
            'value_area_high': self.tpo_profile.value_area_high,
            'value_area_low': self.tpo_profile.value_area_low,
            'initial_balance_high': self.tpo_profile.initial_balance_high,
            'initial_balance_low': self.tpo_profile.initial_balance_low,
            'last_update': self.last_update_time,
            'total_ticks': len(self.cumulative_data)
        }
    
    def plot_current_profile(self, ax=None, show_metrics=True, show_letters=True):
        """Plot current TPO profile"""
        return self.tpo_profile.plot_profile(ax=ax, show_metrics=show_metrics, show_letters=show_letters)
    
    def reset_profile(self):
        """Reset the TPO profile and start fresh"""
        self.tpo_profile = TPOProfile(tick_size=self.tick_size)
        self.cumulative_data = pd.DataFrame()
        self.last_update_time = None
        logging.info("TPO profile reset")


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
    
    # Date Configuration - Set to None for current date, or specify a date like '2025-10-20'
    ANALYSIS_DATE = None  # Change to '2025-10-20' or any other date in 'YYYY-MM-DD' format
    
    # Initialize components
    db_fetcher = PostgresDataFetcher(**DB_CONFIG)
    
    # Determine analysis date
    if ANALYSIS_DATE is None:
        analysis_date = datetime.now().strftime('%Y-%m-%d')
        print(f"Using current date: {analysis_date}")
    else:
        analysis_date = ANALYSIS_DATE
        print(f"Using specified date: {analysis_date}")
    
    # Option 1: Create static TPO profile
    print("Generating Pre-market TPO Profile...")
    df = db_fetcher.fetch_tick_data(
        table_name='ticks',
        instrument_token=256265,
        start_time=f'{analysis_date} 09:05:00.000 +0530',
        end_time=f'{analysis_date} 09:15:00.000 +0530'
    )

    # Pre-market TPO Profile (9:05am to 9:15am IST)
    logging.info("Calculating Pre-market TPO profile...")
    print(df)
    
    # Initialize pre-market profile variables
    pre_market_tpo = None
    pre_market_profile_data = None
    
    if not df.empty:
        pre_market_tpo = TPOProfile(tick_size=5)
        pre_market_tpo.calculate_tpo(df)
        
        print(f"\nPre-market TPO Metrics:")
        # Display POC (single level or range)
        if pre_market_tpo.poc_low == pre_market_tpo.poc_high:
            print(f"Point of Control: {pre_market_tpo.poc}")
        else:
            print(f"Point of Control Range: {pre_market_tpo.poc_low} - {pre_market_tpo.poc_high}")
        print(f"Value Area High: {pre_market_tpo.value_area_high}")
        print(f"Value Area Low: {pre_market_tpo.value_area_low}")
        print(f"Initial Balance High: {pre_market_tpo.initial_balance_high}")
        print(f"Initial Balance Low: {pre_market_tpo.initial_balance_low}")
        
        # Store pre-market profile for side-by-side comparison
        pre_market_profile_data = {
            'tpo_data': pre_market_tpo.tpo_data,
            'poc': pre_market_tpo.poc,
            'value_area_high': pre_market_tpo.value_area_high,
            'value_area_low': pre_market_tpo.value_area_low,
            'initial_balance_high': pre_market_tpo.initial_balance_high,
            'initial_balance_low': pre_market_tpo.initial_balance_low
        }
    
    # Real-time TPO Profile Generator for Market Hours (9:15am to 3:30pm IST)
    print("\n" + "="*60)
    print("REAL-TIME TPO PROFILE GENERATOR")
    print("="*60)
    
    # Initialize real-time TPO profile generator
    real_time_tpo = RealTimeTPOProfile(
        db_fetcher=db_fetcher,
        table_name='ticks',
        instrument_token=256265,
        tick_size=5,
        update_interval=10,  # Update every 10 seconds
        market_start_time="09:15",
        market_end_time="15:30",
        analysis_date=ANALYSIS_DATE
    )
    
    # Create side-by-side comparison plot
    def plot_side_by_side_comparison():
        """Plot pre-market and real-time profiles side by side"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot pre-market profile on the left
        if pre_market_tpo is not None and pre_market_profile_data is not None:
            pre_market_tpo.plot_profile(ax=ax1, show_metrics=True, show_letters=True)
            ax1.set_title("Pre-market TPO Profile\n(9:05am - 9:15am IST)", fontsize=14, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No Pre-market Data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title("Pre-market TPO Profile\n(9:05am - 9:15am IST)", fontsize=14, fontweight='bold')
        
        # Plot real-time profile on the right
        real_time_tpo.plot_current_profile(ax=ax2, show_metrics=True, show_letters=True)
        ax2.set_title("Real-time TPO Profile\n(9:15am - 3:30pm IST)", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    # Add callback function to handle updates
    def on_tpo_update(tpo_profile):
        """Callback function called when TPO profile is updated"""
        print(f"\n[REAL-TIME UPDATE] {datetime.now().strftime('%H:%M:%S')}")
        
        # Display POC (single level or range)
        if tpo_profile.poc_low == tpo_profile.poc_high:
            print(f"POC: {tpo_profile.poc}")
        else:
            print(f"POC Range: {tpo_profile.poc_low} - {tpo_profile.poc_high}")
            
        print(f"VAH: {tpo_profile.value_area_high}")
        print(f"VAL: {tpo_profile.value_area_low}")
        print(f"IBH: {tpo_profile.initial_balance_high}")
        print(f"IBL: {tpo_profile.initial_balance_low}")
        
        # Update side-by-side comparison
        plot_side_by_side_comparison()
    
    real_time_tpo.add_update_callback(on_tpo_update)
    
    # Start real-time updates based on configuration
    if ANALYSIS_DATE is None:
        print("Starting real-time TPO updates for current date...")
        print("Updates will fetch live data during market hours")
    else:
        print(f"Starting simulated real-time TPO updates for {ANALYSIS_DATE}...")
        print("Updates will simulate progressive market data")
    
    # Start real-time updates
    real_time_tpo.start_real_time_updates()
    
    # Keep the program running to see updates
    try:
        if ANALYSIS_DATE is None:
            print("Real-time TPO profile is running with live data. Press Ctrl+C to stop.")
        else:
            print(f"Real-time TPO profile is running with {ANALYSIS_DATE} data. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping real-time updates...")
        real_time_tpo.stop_real_time_updates()
        
        # Get final profile
        final_profile = real_time_tpo.get_current_profile()
        print(f"\nFinal TPO Profile Summary:")
        print(f"Total ticks processed: {final_profile['total_ticks']}")
        print(f"Last update: {final_profile['last_update']}")
        
        # Plot final side-by-side comparison
        print("Generating final side-by-side comparison...")
        plot_side_by_side_comparison()
    
    # Option 2: Create live-updating chart (uncomment to use)
    # live_chart = LiveTPOChart(
    #     db_fetcher=db_fetcher,
    #     table_name='tick_data',
    #     symbol='NIFTY',
    #     tick_size=0.05,
    #     refresh_interval=5000  # 5 seconds
    # )
    # ani = live_chart.start()