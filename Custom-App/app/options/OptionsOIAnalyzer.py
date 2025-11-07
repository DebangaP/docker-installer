"""
Options Open Interest (OI) Analyzer
Joins options_tick, options_tick_ohlc, and options_tick_depth tables
to analyze and visualize OI distribution by strike price
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from common.Boilerplate import get_db_connection
import base64
import io


class OptionsOIAnalyzer:
    """
    Analyze Open Interest distribution across option strikes
    Joins options_tick, options_tick_ohlc, and options_tick_depth tables
    """
    
    def __init__(self):
        """Initialize Options OI Analyzer"""
        pass
    
    def get_oi_by_strike(self,
                        expiry: Optional[date] = None,
                        option_type: Optional[str] = None,
                        analysis_date: Optional[date] = None,
                        include_ohlc: bool = True,
                        include_depth: bool = False) -> pd.DataFrame:
        """
        Get aggregated OI data by strike price with optional OHLC and depth data
        
        Args:
            expiry: Filter by expiry date (None = latest expiry or all)
            option_type: 'CE' or 'PE' or None for both
            analysis_date: Analysis date (None = latest data across all dates)
            include_ohlc: Whether to include OHLC data from options_tick_ohlc
            include_depth: Whether to include depth data from options_tick_depth
            
        Returns:
            DataFrame with strike_price, option_type, total_oi, timestamp, and other metrics
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Build WHERE clause - get latest data across all dates unless analysis_date is specified
            where_conditions = []
            params = []
            
            if analysis_date:
                where_conditions.append("ot.run_date = %s")
                params.append(analysis_date)
            
            if expiry:
                where_conditions.append("ot.expiry = %s")
                params.append(expiry)
            
            if option_type:
                where_conditions.append("ot.option_type = %s")
                params.append(option_type)
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # Base query - get latest tick for each (strike_price, option_type, expiry) combination
            # Join with OHLC and optionally with depth data
            if include_ohlc and include_depth:
                # Full join with OHLC and depth
                query = f"""
                    WITH latest_ticks AS (
                        SELECT DISTINCT ON (ot.strike_price, ot.option_type, ot.expiry)
                            ot.id as tick_id,
                            ot.strike_price,
                            ot.option_type,
                            ot.expiry,
                            ot.tradingsymbol,
                            ot.oi,
                            ot.volume,
                            ot.last_price,
                            ot.average_price,
                            ot.buy_quantity,
                            ot.sell_quantity,
                            ot.oi_day_high,
                            ot.oi_day_low,
                            ot.timestamp
                        FROM my_schema.options_ticks ot
                        WHERE {where_clause}
                        ORDER BY ot.strike_price, ot.option_type, ot.expiry, ot.timestamp DESC
                    )
                    SELECT 
                        lt.strike_price,
                        lt.option_type,
                        lt.expiry,
                        lt.tradingsymbol,
                        lt.oi as total_oi,
                        lt.volume as total_volume,
                        lt.last_price as avg_last_price,
                        lt.average_price as avg_price,
                        lt.buy_quantity as total_buy_quantity,
                        lt.sell_quantity as total_sell_quantity,
                        lt.oi_day_high as max_oi_day_high,
                        lt.oi_day_low as min_oi_day_low,
                        lt.timestamp,
                        ohlc.open,
                        ohlc.high,
                        ohlc.low,
                        ohlc.close,
                        COALESCE(SUM(depth.quantity), 0) as total_depth_quantity,
                        COUNT(depth.id) as depth_entries
                    FROM latest_ticks lt
                    LEFT JOIN my_schema.options_tick_ohlc ohlc ON lt.tick_id = ohlc.tick_id
                    LEFT JOIN my_schema.options_tick_depth depth ON lt.tick_id = depth.tick_id
                    GROUP BY 
                        lt.strike_price, lt.option_type, lt.expiry, lt.tradingsymbol,
                        lt.oi, lt.volume, lt.last_price, lt.average_price,
                        lt.buy_quantity, lt.sell_quantity, lt.oi_day_high, lt.oi_day_low,
                        lt.timestamp, ohlc.open, ohlc.high, ohlc.low, ohlc.close
                    ORDER BY lt.timestamp DESC, lt.oi DESC
                """
            elif include_ohlc:
                # Join with OHLC only
                query = f"""
                    WITH latest_ticks AS (
                        SELECT DISTINCT ON (ot.strike_price, ot.option_type, ot.expiry)
                            ot.id as tick_id,
                            ot.strike_price,
                            ot.option_type,
                            ot.expiry,
                            ot.tradingsymbol,
                            ot.oi,
                            ot.volume,
                            ot.last_price,
                            ot.average_price,
                            ot.buy_quantity,
                            ot.sell_quantity,
                            ot.oi_day_high,
                            ot.oi_day_low,
                            ot.timestamp
                        FROM my_schema.options_ticks ot
                        WHERE {where_clause}
                        ORDER BY ot.strike_price, ot.option_type, ot.expiry, ot.timestamp DESC
                    )
                    SELECT 
                        lt.strike_price,
                        lt.option_type,
                        lt.expiry,
                        lt.tradingsymbol,
                        lt.oi as total_oi,
                        lt.volume as total_volume,
                        lt.last_price as avg_last_price,
                        lt.average_price as avg_price,
                        lt.buy_quantity as total_buy_quantity,
                        lt.sell_quantity as total_sell_quantity,
                        lt.oi_day_high as max_oi_day_high,
                        lt.oi_day_low as min_oi_day_low,
                        lt.timestamp,
                        ohlc.open,
                        ohlc.high,
                        ohlc.low,
                        ohlc.close
                    FROM latest_ticks lt
                    LEFT JOIN my_schema.options_tick_ohlc ohlc ON lt.tick_id = ohlc.tick_id
                    ORDER BY lt.timestamp DESC, lt.oi DESC
                """
            else:
                # Simple aggregation without OHLC
                query = f"""
                    SELECT DISTINCT ON (ot.strike_price, ot.option_type, ot.expiry)
                        ot.strike_price,
                        ot.option_type,
                        ot.expiry,
                        ot.tradingsymbol,
                        ot.oi as total_oi,
                        ot.volume as total_volume,
                        ot.last_price as avg_last_price,
                        ot.average_price as avg_price,
                        ot.buy_quantity as total_buy_quantity,
                        ot.sell_quantity as total_sell_quantity,
                        ot.oi_day_high as max_oi_day_high,
                        ot.oi_day_low as min_oi_day_low,
                        ot.timestamp,
                        ot.run_date
                    FROM my_schema.options_ticks ot
                    WHERE {where_clause}
                    ORDER BY ot.strike_price, ot.option_type, ot.expiry, ot.timestamp DESC
                """
            
            logging.info(f"Executing OI analysis query with params: {params}")
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                logging.warning("No OI data found for the specified criteria")
                return pd.DataFrame()
            
            # Determine columns based on query
            if include_ohlc and include_depth:
                columns = [
                    'strike_price', 'option_type', 'expiry', 'tradingsymbol',
                    'total_oi', 'total_volume', 'avg_last_price', 'avg_price',
                    'total_buy_quantity', 'total_sell_quantity',
                    'max_oi_day_high', 'min_oi_day_low',
                    'timestamp', 'open', 'high', 'low', 'close',
                    'total_depth_quantity', 'depth_entries'
                ]
            elif include_ohlc:
                columns = [
                    'strike_price', 'option_type', 'expiry', 'tradingsymbol',
                    'total_oi', 'total_volume', 'avg_last_price', 'avg_price',
                    'total_buy_quantity', 'total_sell_quantity',
                    'max_oi_day_high', 'min_oi_day_low',
                    'timestamp', 'open', 'high', 'low', 'close'
                ]
            else:
                columns = [
                    'strike_price', 'option_type', 'expiry', 'tradingsymbol',
                    'total_oi', 'total_volume', 'avg_last_price', 'avg_price',
                    'total_buy_quantity', 'total_sell_quantity',
                    'max_oi_day_high', 'min_oi_day_low',
                    'timestamp', 'run_date'
                ]
            
            df = pd.DataFrame(rows, columns=columns)
            
            # Fill NaN values
            df['total_oi'] = df['total_oi'].fillna(0)
            df['total_volume'] = df['total_volume'].fillna(0)
            
            # Sort by timestamp (latest first) then by OI descending
            if 'timestamp' in df.columns:
                df = df.sort_values(['timestamp', 'total_oi'], ascending=[False, False]).reset_index(drop=True)
            else:
                df = df.sort_values('total_oi', ascending=False).reset_index(drop=True)
            
            logging.info(f"Retrieved {len(df)} strike-option combinations sorted by timestamp and OI")
            return df
            
        except Exception as e:
            logging.error(f"Error fetching OI by strike: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return pd.DataFrame()
    
    def get_oi_summary(self,
                       expiry: Optional[date] = None,
                       analysis_date: Optional[date] = None) -> Dict:
        """
        Get summary statistics of OI distribution
        
        Args:
            expiry: Filter by expiry date
            analysis_date: Analysis date
            
        Returns:
            Dictionary with summary statistics
        """
        df = self.get_oi_by_strike(expiry=expiry, analysis_date=analysis_date, include_ohlc=False)
        
        if df.empty:
            return {
                'total_strikes': 0,
                'total_oi': 0,
                'avg_oi_per_strike': 0,
                'max_oi_strike': None,
                'min_oi_strike': None,
                'ce_total_oi': 0,
                'pe_total_oi': 0
            }
        
        # Separate CE and PE
        ce_df = df[df['option_type'] == 'CE'] if 'option_type' in df.columns else pd.DataFrame()
        pe_df = df[df['option_type'] == 'PE'] if 'option_type' in df.columns else pd.DataFrame()
        
        max_oi_row = df.loc[df['total_oi'].idxmax()]
        min_oi_row = df.loc[df['total_oi'].idxmin()]
        
        return {
            'total_strikes': len(df),
            'total_oi': int(df['total_oi'].sum()),
            'avg_oi_per_strike': float(df['total_oi'].mean()),
            'max_oi_strike': {
                'strike': float(max_oi_row['strike_price']),
                'option_type': str(max_oi_row.get('option_type', 'N/A')),
                'oi': int(max_oi_row['total_oi'])
            },
            'min_oi_strike': {
                'strike': float(min_oi_row['strike_price']),
                'option_type': str(min_oi_row.get('option_type', 'N/A')),
                'oi': int(min_oi_row['total_oi'])
            },
            'ce_total_oi': int(ce_df['total_oi'].sum()) if not ce_df.empty else 0,
            'pe_total_oi': int(pe_df['total_oi'].sum()) if not pe_df.empty else 0,
            'top_10_strikes': df.head(10)[['strike_price', 'option_type', 'total_oi']].to_dict('records')
        }
    
    def plot_oi_distribution(self,
                             expiry: Optional[date] = None,
                             option_type: Optional[str] = None,
                             analysis_date: Optional[date] = None,
                             top_n: int = 50,
                             save_path: Optional[str] = None) -> str:
        """
        Generate OI distribution chart by strike price
        
        Args:
            expiry: Filter by expiry date
            option_type: 'CE' or 'PE' or None for both
            analysis_date: Analysis date
            top_n: Number of top strikes to display
            save_path: Path to save the chart (optional)
            
        Returns:
            Base64 encoded image string
        """
        df = self.get_oi_by_strike(
            expiry=expiry,
            option_type=option_type,
            analysis_date=analysis_date,
            include_ohlc=False
        )
        
        if df.empty:
            # Create empty chart
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.text(0.5, 0.5, 'No OI data available', 
                   ha='center', va='center', fontsize=16,
                   transform=ax.transAxes)
            ax.set_title('Options Open Interest Distribution', fontsize=18, fontweight='bold')
        else:
            # Take top N strikes
            top_df = df.head(top_n).copy()
            
            # Create strike labels
            top_df['strike_label'] = top_df.apply(
                lambda x: f"{x['strike_price']:.0f} {x['option_type']}", 
                axis=1
            )
            
            # Create figure with subplots - make top 50 strikes chart bigger, OI Distribution smaller
            # height_ratios: 3:1 means top chart gets 75% of height, bottom chart gets 25%
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), height_ratios=[3, 1])
            
            # Plot 1: Bar chart of OI by strike (bigger chart)
            colors = top_df['option_type'].map({'CE': 'green', 'PE': 'red'})
            ax1.barh(range(len(top_df)), top_df['total_oi'], color=colors, alpha=0.7)
            ax1.set_yticks(range(len(top_df)))
            ax1.set_yticklabels(top_df['strike_label'], fontsize=11)
            ax1.set_xlabel('Open Interest', fontsize=13, fontweight='bold')
            ax1.set_title(f'Top {top_n} Strikes by Open Interest', fontsize=16, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
            ax1.invert_yaxis()  # Highest OI at top
            
            # Add value labels on bars
            for i, (idx, row) in enumerate(top_df.iterrows()):
                ax1.text(row['total_oi'], i, f"{int(row['total_oi']):,}", 
                        va='center', fontsize=10, fontweight='bold')
            
            # Plot 2: Line chart of OI distribution (smaller chart)
            if len(top_df) > 1:
                # Sort by strike price for line chart
                top_df_sorted = top_df.sort_values('strike_price')
                
                # Separate CE and PE
                ce_data = top_df_sorted[top_df_sorted['option_type'] == 'CE']
                pe_data = top_df_sorted[top_df_sorted['option_type'] == 'PE']
                
                if not ce_data.empty:
                    ax2.plot(ce_data['strike_price'], ce_data['total_oi'], 
                            marker='o', label='CE', color='green', linewidth=2, markersize=5)
                
                if not pe_data.empty:
                    ax2.plot(pe_data['strike_price'], pe_data['total_oi'], 
                            marker='s', label='PE', color='red', linewidth=2, markersize=5)
                
                ax2.set_xlabel('Strike Price', fontsize=11, fontweight='bold')
                ax2.set_ylabel('Open Interest', fontsize=11, fontweight='bold')
                ax2.set_title('OI Distribution Across Strikes', fontsize=12, fontweight='bold')
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
                # Reduce tick label sizes for smaller chart
                ax2.tick_params(labelsize=9)
            
            # Add summary text
            summary = self.get_oi_summary(expiry=expiry, analysis_date=analysis_date)
            summary_text = (
                f"Total OI: {summary['total_oi']:,} | "
                f"CE OI: {summary['ce_total_oi']:,} | "
                f"PE OI: {summary['pe_total_oi']:,} | "
                f"Max OI Strike: {summary['max_oi_strike']['strike']:.0f} {summary['max_oi_strike']['option_type']}"
            )
            fig.suptitle(f'Options Open Interest Analysis - {analysis_date or date.today()}', 
                        fontsize=16, fontweight='bold', y=0.98)
            fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def generate_oi_analysis_report(self,
                                   expiry: Optional[date] = None,
                                   analysis_date: Optional[date] = None) -> Dict:
        """
        Generate comprehensive OI analysis report
        
        Args:
            expiry: Filter by expiry date
            analysis_date: Analysis date
            
        Returns:
            Dictionary with analysis data and chart
        """
        oi_data = self.get_oi_by_strike(
            expiry=expiry,
            analysis_date=analysis_date,
            include_ohlc=True
        )
        
        summary = self.get_oi_summary(expiry=expiry, analysis_date=analysis_date)
        chart_image = self.plot_oi_distribution(
            expiry=expiry,
            analysis_date=analysis_date
        )
        
        # Convert DataFrame to dict for JSON serialization
        oi_data_dict = oi_data.to_dict('records') if not oi_data.empty else []
        
        # Convert numpy types to native Python types
        for record in oi_data_dict:
            for key, value in record.items():
                if isinstance(value, (np.integer, np.int64)):
                    record[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    record[key] = float(value) if not pd.isna(value) else None
                elif isinstance(value, pd.Timestamp):
                    record[key] = value.isoformat()
                elif pd.isna(value):
                    record[key] = None
        
        # Determine analysis date for response
        analysis_date_str = None
        if analysis_date:
            analysis_date_str = analysis_date.isoformat()
        elif not oi_data.empty and 'run_date' in oi_data.columns:
            # Get the latest run_date from the data
            latest_run_date = oi_data['run_date'].max()
            if pd.notna(latest_run_date):
                analysis_date_str = latest_run_date.isoformat() if hasattr(latest_run_date, 'isoformat') else str(latest_run_date)
        elif not oi_data.empty and 'timestamp' in oi_data.columns:
            # Extract date from latest timestamp
            latest_timestamp = oi_data['timestamp'].max()
            if pd.notna(latest_timestamp):
                if isinstance(latest_timestamp, pd.Timestamp):
                    analysis_date_str = latest_timestamp.date().isoformat()
                else:
                    analysis_date_str = str(latest_timestamp)
        
        return {
            'success': True,
            'analysis_date': analysis_date_str or 'latest',
            'expiry': expiry.isoformat() if expiry else None,
            'summary': summary,
            'oi_data': oi_data_dict,
            'total_records': len(oi_data),
            'chart_image': chart_image
        }

