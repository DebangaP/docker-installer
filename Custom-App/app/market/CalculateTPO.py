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

        # Round prices to tick size
        df['price_level'] = (df[price_column] / self.tick_size).round() * self.tick_size
        
        # Build TPO profile
        tpo_dict = defaultdict(list)
        period_count = 0
        for period in sorted(df['time_period'].unique()):
            period_data = df[df['time_period'] == period]
            prices_in_period = period_data['price_level'].unique()
            
            # Generate TPO letters: A-Z, then AA, AB, AC, etc. (all uppercase)
            if period_count < 26:
                letter = chr(65 + period_count)  # A-Z
            else:
                # Double letters: AA, AB, AC, etc. (all uppercase)
                first_letter_idx = (period_count - 26) // 26
                second_letter_idx = (period_count - 26) % 26
                letter = chr(65 + first_letter_idx) + chr(65 + second_letter_idx)
            
            for price in prices_in_period:
                tpo_dict[price].append(letter)
            
            period_count += 1
        
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
    
    def plot_profile(self, ax=None, show_metrics=True, show_letters=True, dark_mode=False):
        """
        Plot TPO profile with key metrics and TPO letters
        
        Args:
            ax: Matplotlib axis object (creates new if None)
            show_metrics: Whether to show POC, VA, and IB lines
            show_letters: Whether to show TPO letters on the bars
            dark_mode: Whether to use dark background styling
        """
        if self.tpo_data is None or self.tpo_data.empty:
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
            if dark_mode:
                fig.patch.set_facecolor('black')
        else:
            ax.clear()
        
        # Apply dark mode styling if requested
        if dark_mode:
            ax.set_facecolor('black')
            ax.tick_params(colors='white', labelsize=10)
            ax.spines['top'].set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            title_color = 'white'
            label_color = 'white'
            grid_color = 'gray'
        else:
            title_color = 'black'
            label_color = 'black'
            grid_color = 'gray'
        
        # Plot horizontal bars for TPO count
        prices = self.tpo_data['price'].values
        counts = self.tpo_data['tpo_count'].values
        
        # Color coding: green for regular TPOs, purple for Value Area, orange for Initial Balance (matching 5-Day TPO)
        if dark_mode:
            colors = []
            for price in prices:
                color = '#32CD32'  # Default: green for regular TPOs
                # Check Value Area first (purple)
                if self.value_area_high and self.value_area_low:
                    if self.value_area_low <= price <= self.value_area_high:
                        color = '#9370DB'  # Purple for Value Area
                # Check Initial Balance (orange/yellow) - only if not in Value Area
                elif self.initial_balance_high and self.initial_balance_low:
                    if self.initial_balance_low <= price <= self.initial_balance_high:
                        color = '#FFA500'  # Orange for Initial Balance
                colors.append(color)
            bars = ax.barh(prices, counts, height=self.tick_size*0.8, 
                          color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        else:
            bars = ax.barh(prices, counts, height=self.tick_size*0.8, 
                          color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add TPO letters on the bars
        if show_letters and 'tpo_letters' in self.tpo_data.columns:
            for i, (price, count, letters) in enumerate(zip(prices, counts, self.tpo_data['tpo_letters'])):
                if letters and count > 0:
                    # Position text in the middle of the bar
                    ax.text(count/2, price, letters, 
                           ha='center', va='center', fontsize=10, fontweight='bold',
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
        
        ax.set_xlabel('TPO Count', fontsize=12, color=label_color)
        ax.set_ylabel('Price', fontsize=12, color=label_color)
        ax.set_title('Market Profile (TPO)', fontsize=14, fontweight='bold', color=title_color)
        
        # Set legend with appropriate colors for dark mode
        if dark_mode:
            legend = ax.legend(loc='upper right', fontsize=14, facecolor='black', edgecolor='white', labelcolor='white')
        else:
            legend = ax.legend(loc='upper right', fontsize=14)
        
        ax.grid(True, alpha=0.3, axis='y', color=grid_color)
        
        return ax

    def predict_support_resistance_levels(self, current_price: float = None, lookback_days: int = 5) -> dict:
        """
        Predict upcoming support and resistance levels based on TPO patterns
        
        Uses three methods:
        1. TPO Cluster Analysis - identifies significant price clusters
        2. Value Area Extensions - projects levels from Value Area boundaries
        3. Fibonacci-like Extensions - applies Fibonacci ratios to Value Area width
        
        Args:
            current_price: Current market price (if None, uses POC)
            lookback_days: Number of days for analysis (for future use)
            
        Returns:
            Dictionary with support_levels, resistance_levels, and key TPO metrics
        """
        if self.tpo_data is None or self.tpo_data.empty:
            return {
                'support_levels': [],
                'resistance_levels': [],
                'current_price': current_price if current_price else None,
                'poc': float(self.poc) if self.poc else None,
                'vah': float(self.value_area_high) if self.value_area_high else None,
                'val': float(self.value_area_low) if self.value_area_low else None
            }
        
        # Determine current price
        if current_price is None:
            if self.poc:
                current_price = float(self.poc)
            else:
                # Use middle of price range
                prices = self.tpo_data['price'].values
                current_price = float((prices.min() + prices.max()) / 2)
        else:
            current_price = float(current_price)
        
        predicted_support = []
        predicted_resistance = []
        
        # Method 1: TPO Cluster Analysis
        # Identify price levels with significant TPO count (clusters)
        for idx, row in self.tpo_data.iterrows():
            price = float(row['price'])
            tpo_count = int(row['tpo_count'])
            
            # Significant cluster if TPO count >= 3
            if tpo_count >= 3:
                distance_from_poc = abs(price - self.poc) if self.poc else 0
                
                if price < current_price:
                    # Potential support level
                    strength = 'Strong' if tpo_count >= 5 else 'Medium'
                    confidence = min(100, tpo_count * 15)  # Higher TPO count = higher confidence
                    predicted_support.append({
                        'price': price,
                        'strength': strength,
                        'tpo_count': tpo_count,
                        'distance_from_current': float(current_price - price),
                        'type': 'TPO Cluster',
                        'confidence': confidence
                    })
                elif price > current_price:
                    # Potential resistance level
                    strength = 'Strong' if tpo_count >= 5 else 'Medium'
                    confidence = min(100, tpo_count * 15)
                    predicted_resistance.append({
                        'price': price,
                        'strength': strength,
                        'tpo_count': tpo_count,
                        'distance_from_current': float(price - current_price),
                        'type': 'TPO Cluster',
                        'confidence': confidence
                    })
        
        # Method 2: Value Area Extensions
        if self.value_area_high and self.value_area_low:
            va_width = float(self.value_area_high - self.value_area_low)
            
            if va_width > 0:
                # Extension multipliers
                extension_multipliers = [0.5, 1.0, 1.5]
                
                # Resistance levels above VAH
                for multiplier in extension_multipliers:
                    extension_above = float(self.value_area_high) + (va_width * multiplier)
                    if extension_above > current_price:
                        predicted_resistance.append({
                            'price': extension_above,
                            'strength': 'Medium',
                            'tpo_count': 0,
                            'distance_from_current': float(extension_above - current_price),
                            'type': f'VAH Extension ({multiplier}x)',
                            'confidence': 60
                        })
                
                # Support levels below VAL
                for multiplier in extension_multipliers:
                    extension_below = float(self.value_area_low) - (va_width * multiplier)
                    if extension_below < current_price:
                        predicted_support.append({
                            'price': extension_below,
                            'strength': 'Medium',
                            'tpo_count': 0,
                            'distance_from_current': float(current_price - extension_below),
                            'type': f'VAL Extension ({multiplier}x)',
                            'confidence': 60
                        })
        
        # Method 3: Fibonacci-like Extensions
        if self.value_area_high and self.value_area_low:
            va_width = float(self.value_area_high - self.value_area_low)
            
            if va_width > 0:
                # Fibonacci ratios
                fib_ratios = [0.382, 0.5, 0.618, 1.0, 1.382, 1.618]
                
                # Confidence mapping for Fibonacci ratios
                confidence_map = {
                    0.382: 50,
                    0.5: 55,
                    0.618: 70,
                    1.0: 65,
                    1.382: 60,
                    1.618: 70
                }
                
                # Resistance levels above VAH
                for ratio in fib_ratios:
                    resistance_level = float(self.value_area_high) + (va_width * ratio)
                    if resistance_level > current_price:
                        predicted_resistance.append({
                            'price': resistance_level,
                            'strength': 'Medium',
                            'tpo_count': 0,
                            'distance_from_current': float(resistance_level - current_price),
                            'type': f'VA Fib Extension ({ratio})',
                            'confidence': confidence_map.get(ratio, 55)
                        })
                
                # Support levels below VAL
                for ratio in fib_ratios:
                    support_level = float(self.value_area_low) - (va_width * ratio)
                    if support_level < current_price:
                        predicted_support.append({
                            'price': support_level,
                            'strength': 'Medium',
                            'tpo_count': 0,
                            'distance_from_current': float(current_price - support_level),
                            'type': f'VA Fib Extension ({ratio})',
                            'confidence': confidence_map.get(ratio, 55)
                        })
        
        # Sort and filter - keep top 3 most relevant levels per category
        # Sort support descending (highest first), resistance ascending (lowest first)
        predicted_support = sorted(predicted_support, key=lambda x: x['price'], reverse=True)[:3]
        predicted_resistance = sorted(predicted_resistance, key=lambda x: x['price'])[:3]
        
        return {
            'support_levels': predicted_support,
            'resistance_levels': predicted_resistance,
            'current_price': current_price,
            'poc': float(self.poc) if self.poc else None,
            'vah': float(self.value_area_high) if self.value_area_high else None,
            'val': float(self.value_area_low) if self.value_area_low else None
        }


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
        
        # Build query: select timestamp and last_price for TPO profile
        query = "SELECT timestamp, last_price FROM my_schema.\"{}\"".format(table_name)
        params = {}
        
        conditions = []
        if instrument_token:
            conditions.append("instrument_token = :instrument_token")
            params['instrument_token'] = instrument_token
        # Compare directly against stored timestamps; no offset applied
        if start_time:
            conditions.append("timestamp >= :start_time")
            params['start_time'] = start_time
        if end_time:
            conditions.append("timestamp <= :end_time")
            params['end_time'] = end_time
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp"
        
        try:
            #logging.info(f"Executing query: {query} with params: {params}")
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                #logging.info(f"Fetched {len(df)} tick rows for instrument_token={params.get('instrument_token')} from {table_name}")
                if not df.empty:
                    try:
                        ts = pd.to_datetime(df['timestamp'])
                        logging.info(f"Tick time window (DB UTC): min={ts.min()}, max={ts.max()}")
                    except Exception:
                        pass
                return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def close(self):
        """Close the database connection"""
        self.engine.dispose()
    
    def fetch_trading_days_data(self, table_name, instrument_token, num_days=5, 
                               market_start_time="09:15", market_end_time="15:30"):
        """
        Fetch tick data for the last N trading days from current week + previous week
        
        Args:
            table_name: Name of the table containing tick data
            instrument_token: Instrument token to filter
            num_days: Number of trading days to fetch (default: 5)
            market_start_time: Market start time in HH:MM format (default: "09:15")
            market_end_time: Market end time in HH:MM format (default: "15:30")
            
        Returns:
            List of tuples: (date, DataFrame) for each trading day, ordered from oldest to newest.
            If data is missing for a day, returns (date, empty DataFrame)
        """
        from pytz import timezone
        
        IST = timezone('Asia/Kolkata')
        today = datetime.now(IST).date()
        
        # Calculate the start of current week (Monday)
        days_since_monday = today.weekday()  # 0=Monday, 6=Sunday
        current_week_start = today - timedelta(days=days_since_monday)
        
        # Calculate the start of previous week (Monday of previous week)
        previous_week_start = current_week_start - timedelta(days=7)
        
        # Get all trading days (Mon-Fri) from previous week and current week
        # We want up to 5 trading days, prioritizing current week first, then previous week
        all_trading_dates = []
        
        # First, add current week trading days (from Monday to today)
        for day_offset in range(days_since_monday + 1):  # +1 to include today
            check_date = current_week_start + timedelta(days=day_offset)
            if check_date.weekday() < 5:  # Monday (0) to Friday (4)
                all_trading_dates.append(check_date)
        
        # Then, add previous week trading days (all 5 days)
        for day_offset in range(5):  # Monday to Friday
            check_date = previous_week_start + timedelta(days=day_offset)
            if check_date.weekday() < 5:  # Monday (0) to Friday (4)
                all_trading_dates.append(check_date)
        
        # Sort all dates (oldest first)
        all_trading_dates.sort()
        
        # Take the last num_days trading days (most recent ones)
        target_dates = all_trading_dates[-num_days:] if len(all_trading_dates) >= num_days else all_trading_dates
        
        # Ensure we have exactly num_days slots (from current week + previous week only)
        # If we don't have enough days from these two weeks, we'll still show what we have
        # but pad with placeholder dates to maintain the 5-day structure
        if len(target_dates) < num_days:
            # We don't have enough days from current week + previous week
            # This can happen if it's early in the week or if there were holidays
            # Pad with placeholder dates (these will show as blank charts)
            while len(target_dates) < num_days:
                # Add a placeholder date going backwards from the earliest date we have
                earliest_date = target_dates[0] if target_dates else today
                # Go back to find the previous trading day
                placeholder_date = earliest_date - timedelta(days=1)
                # Skip weekends
                while placeholder_date.weekday() >= 5:  # Saturday (5) or Sunday (6)
                    placeholder_date = placeholder_date - timedelta(days=1)
                if placeholder_date not in target_dates:
                    target_dates.insert(0, placeholder_date)
                else:
                    # If we've already added this date, go back further
                    placeholder_date = placeholder_date - timedelta(days=1)
                    while placeholder_date.weekday() >= 5:
                        placeholder_date = placeholder_date - timedelta(days=1)
                    target_dates.insert(0, placeholder_date)
        
        # Fetch data for each target date
        trading_days_data = []
        for check_date in target_dates[-num_days:]:  # Take last num_days
            # Create start and end timestamps for this trading day
            start_time_str = f"{check_date} {market_start_time}"
            end_time_str = f"{check_date} {market_end_time}"
            
            try:
                start_time = IST.localize(datetime.strptime(start_time_str, "%Y-%m-%d %H:%M"))
                end_time = IST.localize(datetime.strptime(end_time_str, "%Y-%m-%d %H:%M"))
            except:
                # If date parsing fails, add empty DataFrame
                trading_days_data.append((check_date, pd.DataFrame()))
                continue
            
            # Fetch data for this day
            day_data = self.fetch_tick_data(
                table_name=table_name,
                instrument_token=instrument_token,
                start_time=start_time,
                end_time=end_time
            )
            
            # Always add the date, even if data is empty (for blank chart display)
            if not day_data.empty:
                day_data['trading_date'] = check_date
            trading_days_data.append((check_date, day_data))
        
        # Sort by date (oldest first)
        trading_days_data.sort(key=lambda x: x[0])
        
        return trading_days_data



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
        """Plot current TPO profile with latest tick line"""
        ax_returned = self.tpo_profile.plot_profile(ax=ax, show_metrics=show_metrics, show_letters=show_letters)
        
        # Add latest tick line if we have data
        if ax_returned is not None and not self.cumulative_data.empty:
            # Get the latest tick
            latest_tick = self.cumulative_data.iloc[-1]
            latest_price = latest_tick['last_price']
            
            # Add a horizontal line for the latest tick price
            ax_returned.axhline(y=latest_price, color='yellow', linewidth=4, 
                              linestyle='-.', label=f'Nifty50: {latest_price:.2f}', 
                              alpha=0.8, zorder=20)
            ax_returned.legend(loc='upper right', fontsize=14)
        
        return ax_returned
    
    def reset_profile(self):
        """Reset the TPO profile and start fresh"""
        self.tpo_profile = TPOProfile(tick_size=self.tick_size)
        self.cumulative_data = pd.DataFrame()
        self.last_update_time = None
        logging.info("TPO profile reset")


def plot_5day_tpo_chart(db_fetcher, table_name, instrument_token=256265, 
                        tick_size=5, market_start_time="09:15", market_end_time="15:30"):
    """
    Generate a 5-day TPO chart with TPO profiles
    
    Args:
        db_fetcher: PostgresDataFetcher instance
        table_name: Table name containing tick data
        instrument_token: Instrument token (default: 256265 for Nifty 50)
        tick_size: Price tick size for TPO calculation
        market_start_time: Market start time
        market_end_time: Market end time
        
    Returns:
        Base64 encoded image string
    """
    import base64
    import io
    
    # Fetch data for last 5 trading days from current week + previous week
    trading_days = db_fetcher.fetch_trading_days_data(
        table_name=table_name,
        instrument_token=instrument_token,
        num_days=5,
        market_start_time=market_start_time,
        market_end_time=market_end_time
    )
    
    # Ensure we always have exactly 5 days (even if some have no data)
    # fetch_trading_days_data should return exactly 5 days, but ensure it here
    num_days = 5
    if len(trading_days) < num_days:
        # Pad with empty dataframes for missing days
        from pytz import timezone
        IST = timezone('Asia/Kolkata')
        today = datetime.now(IST).date()
        while len(trading_days) < num_days:
            # Add placeholder dates with empty data
            placeholder_date = today - timedelta(days=len(trading_days))
            trading_days.append((placeholder_date, pd.DataFrame()))
    
    # Ensure days are ordered chronologically from left to right: oldest (left) to newest (right)
    trading_days.sort(key=lambda x: x[0])
    
    # Take only the last 5 days
    trading_days = trading_days[-num_days:]
    
    fig = plt.figure(figsize=(20, 12), facecolor='black')
    
    # Create subplots: 1 row x num_days columns
    # Each day has TPO only (price scale removed)
    # Layout: [TPO1][TPO2][TPO3][TPO4][TPO5]
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, num_days, figure=fig, hspace=0.2, wspace=0.1,
                 width_ratios=[1] * num_days)
    
    # Calculate global price range for all days (only from days with data)
    all_prices = []
    for _, day_data in trading_days:
        if not day_data.empty and 'last_price' in day_data.columns:
            # Convert to float to handle Decimal types
            prices = day_data['last_price'].dropna().astype(float).tolist()
            all_prices.extend(prices)
    
    # If no prices found, use a default range (this shouldn't happen in practice)
    if not all_prices:
        # Try to get a recent price from database as fallback
        try:
            IST = timezone('Asia/Kolkata')
            today = datetime.now(IST).date()
            fallback_data = db_fetcher.fetch_tick_data(
                table_name=table_name,
                instrument_token=instrument_token,
                start_time=IST.localize(datetime.strptime(f"{today} {market_start_time}", "%Y-%m-%d %H:%M")),
                end_time=IST.localize(datetime.strptime(f"{today} {market_end_time}", "%Y-%m-%d %H:%M"))
            )
            if not fallback_data.empty and 'last_price' in fallback_data.columns:
                fallback_price = float(fallback_data['last_price'].iloc[-1])
                price_min = fallback_price - 1000
                price_max = fallback_price + 1000
            else:
                price_min = 0
                price_max = 100
        except:
            price_min = 0
            price_max = 100
    else:
        price_min = float(min(all_prices))
        price_max = float(max(all_prices))
    
    price_range = price_max - price_min
    price_padding = price_range * 0.05 if price_range > 0 else 50  # 5% padding or default
    
    y_min = price_min - price_padding
    y_max = price_max + price_padding
    
    # Process each day
    for day_idx, (trading_date, day_data) in enumerate(trading_days):
        # Check if data is empty (missing day)
        is_empty = day_data.empty or (not day_data.empty and 'last_price' not in day_data.columns)
        
        # Calculate TPO profile for this day (only if data exists)
        if not is_empty:
            tpo_profile = TPOProfile(tick_size=tick_size)
            tpo_profile.calculate_tpo(day_data)
        else:
            tpo_profile = TPOProfile(tick_size=tick_size)
            tpo_profile.tpo_data = pd.DataFrame()
        
        # TPO Profile
        ax_tpo = fig.add_subplot(gs[0, day_idx])
        
        # Show blank chart if data is missing
        if is_empty:
            ax_tpo.text(0.5, 0.5, 'No Data', 
                       ha='center', va='center', fontsize=12, color='gray',
                       transform=ax_tpo.transAxes)
        elif tpo_profile.tpo_data is not None and not tpo_profile.tpo_data.empty:
            prices = tpo_profile.tpo_data['price'].values
            counts = tpo_profile.tpo_data['tpo_count'].values
            letters_list = tpo_profile.tpo_data['tpo_letters'].values if 'tpo_letters' in tpo_profile.tpo_data.columns else [''] * len(prices)
            
            # Color coding: green for regular TPOs, purple for Value Area, orange for Initial Balance
            colors = []
            for price in prices:
                color = '#32CD32'  # Default: green for regular TPOs
                # Check Value Area first (purple)
                if tpo_profile.value_area_high and tpo_profile.value_area_low:
                    if tpo_profile.value_area_low <= price <= tpo_profile.value_area_high:
                        color = '#9370DB'  # Purple for Value Area
                # Check Initial Balance (orange/yellow) - only if not in Value Area
                elif tpo_profile.initial_balance_high and tpo_profile.initial_balance_low:
                    if tpo_profile.initial_balance_low <= price <= tpo_profile.initial_balance_high:
                        color = '#FFA500'  # Orange for Initial Balance
                colors.append(color)
            
            bars = ax_tpo.barh(prices, counts, height=tick_size*0.8, 
                              color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add TPO letters - positioned from left to right (earliest to latest)
            for price, count, letters in zip(prices, counts, letters_list):
                if letters and count > 0:
                    # Ensure letters are a string and not duplicated
                    if isinstance(letters, str):
                        # Only show letters if we have valid data
                        if len(letters) > 0 and count > 0:
                            # Position text centered in the bar (bars grow left to right)
                            ax_tpo.text(count/2, price, letters, 
                                       ha='center', va='center', fontsize=8, fontweight='bold',
                                       color='white', zorder=15, clip_on=True)
            
            # Add POC line (red)
            if tpo_profile.poc:
                ax_tpo.axhline(y=tpo_profile.poc, color='red', linewidth=2, 
                              linestyle='-', zorder=10)
            
            # Add Value Area lines (green)
            if tpo_profile.value_area_high:
                ax_tpo.axhline(y=tpo_profile.value_area_high, color='green', 
                              linewidth=1.5, linestyle='--', zorder=9)
            if tpo_profile.value_area_low:
                ax_tpo.axhline(y=tpo_profile.value_area_low, color='green', 
                              linewidth=1.5, linestyle='--', zorder=9)
            
            # Add Initial Balance lines (orange/yellow)
            if tpo_profile.initial_balance_high:
                ax_tpo.axhline(y=tpo_profile.initial_balance_high, color='orange', 
                              linewidth=1.5, linestyle=':', zorder=8)
            if tpo_profile.initial_balance_low:
                ax_tpo.axhline(y=tpo_profile.initial_balance_low, color='orange', 
                              linewidth=1.5, linestyle=':', zorder=8)
            
            # Add red triangle for last price
            if not day_data.empty and 'last_price' in day_data.columns:
                last_price = float(day_data['last_price'].iloc[-1])
                max_count = float(counts.max()) if len(counts) > 0 else 1.0
                # Position triangle at the right edge of the chart
                ax_tpo.plot([max_count * 1.1], [last_price], 'r^', markersize=10, 
                          zorder=20, clip_on=False, markeredgecolor='red', markeredgewidth=1)
        
        ax_tpo.set_xlabel('TPO Count', fontsize=9, color='white')
        ax_tpo.set_title(f'{trading_date.strftime("%m/%d")}', fontsize=11, fontweight='bold', color='white', pad=5)
        ax_tpo.set_ylim(y_min, y_max)
        ax_tpo.grid(True, alpha=0.3, axis='y', color='gray')
        ax_tpo.set_facecolor('black')
        ax_tpo.tick_params(colors='white', labelsize=8)
        # Hide y-axis labels for inner days (only show on first day)
        if day_idx > 0:
            ax_tpo.set_yticklabels([])
        else:
            ax_tpo.set_ylabel('Price', fontsize=10, color='white')
    
    # Add overall title
    fig.suptitle('5-Day TPO Market Profile', 
                fontsize=16, fontweight='bold', y=0.98, color='white')
    
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='black')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"


def plot_volume_profile(scrip_id: str, num_days: int = 5, tick_size: float = 5.0):
    """
    Generate a volume profile chart using data from rt_intraday_price table
    
    Args:
        scrip_id: Stock symbol (e.g., 'NIFTY', 'RELIANCE')
        num_days: Number of trading days to include (default: 5)
        tick_size: Price tick size for volume profile calculation (default: 5.0)
        
    Returns:
        Base64 encoded image string or None if no data available
    """
    import base64
    import io
    import psycopg2.extras
    from common.Boilerplate import get_db_connection
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Fetch volume data from rt_intraday_price for the last N trading days
        cursor.execute("""
            SELECT 
                price_date::date as trading_date,
                price_high,
                price_low,
                price_close,
                volume
            FROM my_schema.rt_intraday_price
            WHERE scrip_id = %s
            AND country = 'IN'
            AND volume IS NOT NULL
            AND volume > 0
            AND price_high IS NOT NULL
            AND price_low IS NOT NULL
            AND price_close IS NOT NULL
            AND price_date::date >= CURRENT_DATE - make_interval(days => %s)
            ORDER BY price_date::date DESC
            LIMIT %s
        """, (scrip_id, num_days * 2, num_days))
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not rows or len(rows) == 0:
            logging.warning(f"No volume data found for {scrip_id}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        df['trading_date'] = pd.to_datetime(df['trading_date'])
        df = df.sort_values('trading_date')
        
        # Take only the last num_days
        df = df.tail(num_days)
        
        if df.empty:
            return None
        
        # Calculate price range
        all_prices = []
        for _, row in df.iterrows():
            # Create price levels from low to high
            low = float(row['price_low'])
            high = float(row['price_high'])
            price_levels = np.arange(
                (low / tick_size).round() * tick_size,
                (high / tick_size).round() * tick_size + tick_size,
                tick_size
            )
            all_prices.extend(price_levels)
        
        if not all_prices:
            return None
        
        y_min = min(all_prices)
        y_max = max(all_prices)
        
        # Create figure
        fig = plt.figure(figsize=(20, 12), facecolor='black')
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, num_days + 1, figure=fig, hspace=0.2, wspace=0.1,
                     width_ratios=[1] * num_days + [0.4])
        
        # Process each day
        for day_idx, (_, day_row) in enumerate(df.iterrows()):
            trading_date = day_row['trading_date']
            high = float(day_row['price_high'])
            low = float(day_row['price_low'])
            close = float(day_row['price_close'])
            volume = float(day_row['volume'])
            
            # Create price levels for this day
            price_levels = np.arange(
                (low / tick_size).round() * tick_size,
                (high / tick_size).round() * tick_size + tick_size,
                tick_size
            )
            
            # Distribute volume across price levels (simplified - assumes uniform distribution)
            # In reality, you'd need tick-by-tick data for accurate volume profile
            num_levels = len(price_levels)
            if num_levels > 0:
                volume_per_level = volume / num_levels
                
                # Create volume profile DataFrame
                volume_profile = pd.DataFrame({
                    'price': price_levels,
                    'volume': [volume_per_level] * num_levels
                })
                volume_profile = volume_profile.sort_values('price', ascending=False)
                
                # Find POC (price with highest volume)
                poc_price = volume_profile.loc[volume_profile['volume'].idxmax(), 'price']
                
                # Plot volume profile
                ax_vol = fig.add_subplot(gs[0, day_idx])
                
                vol_prices = volume_profile['price'].values
                volumes = volume_profile['volume'].values
                
                # Color coding: blue for volume, magenta for POC
                vol_colors = []
                for price in vol_prices:
                    if abs(price - poc_price) < tick_size:
                        vol_colors.append('#FF00FF')  # Magenta for POC
                    else:
                        vol_colors.append('#4169E1')  # Blue for regular volume
                
                ax_vol.barh(vol_prices, volumes, height=tick_size*0.8,
                           color=vol_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Add POC line
                ax_vol.axhline(y=poc_price, color='magenta', linewidth=2,
                              linestyle='-', zorder=10)
                
                # Add close price indicator
                ax_vol.axhline(y=close, color='yellow', linewidth=1.5,
                              linestyle='--', zorder=9, alpha=0.7)
                
                ax_vol.set_xlabel('Volume', fontsize=9, color='white')
                ax_vol.set_title(f'{trading_date.strftime("%m/%d")}\nVol: {volume:,.0f}', 
                               fontsize=11, fontweight='bold', color='white', pad=5)
                ax_vol.set_ylim(y_min, y_max)
                ax_vol.grid(True, alpha=0.3, axis='y', color='gray')
                ax_vol.set_facecolor('black')
                ax_vol.tick_params(colors='white', labelsize=8)
                
                # Hide y-axis labels for inner days
                if day_idx > 0:
                    ax_vol.set_yticklabels([])
                else:
                    ax_vol.set_ylabel('Price', fontsize=10, color='white')
        
        # Add price scale on the right
        ax_price_scale = fig.add_subplot(gs[0, num_days])
        price_ticks = np.arange(y_min, y_max + tick_size, tick_size * 5)
        price_labels = [f'{p:.2f}' for p in price_ticks]
        
        # Add current price indicator
        if not df.empty:
            current_price = float(df.iloc[-1]['price_close'])
            price_labels_with_current = []
            for p in price_ticks:
                if abs(p - current_price) < tick_size * 2:
                    price_labels_with_current.append(f'{p:.2f} (C)')
                else:
                    price_labels_with_current.append(f'{p:.2f}')
            price_labels = price_labels_with_current
        
        ax_price_scale.set_ylim(y_min, y_max)
        ax_price_scale.set_yticks(price_ticks)
        ax_price_scale.set_yticklabels(price_labels, fontsize=9, color='white')
        ax_price_scale.set_ylabel('Price', fontsize=12, fontweight='bold', color='white')
        ax_price_scale.set_facecolor('black')
        ax_price_scale.spines['top'].set_color('white')
        ax_price_scale.spines['bottom'].set_color('white')
        ax_price_scale.spines['left'].set_color('white')
        ax_price_scale.spines['right'].set_color('white')
        ax_price_scale.tick_params(colors='white', labelsize=9)
        ax_price_scale.grid(True, alpha=0.2, axis='y', color='gray')
        
        # Add overall title
        fig.suptitle(f'{scrip_id} - {num_days}-Day Volume Profile', 
                    fontsize=16, fontweight='bold', y=0.98, color='white')
        
        plt.tight_layout(rect=[0, 0, 0.95, 0.96])
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='black')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        logging.error(f"Error generating volume profile: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None


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