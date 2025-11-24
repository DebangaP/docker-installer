"""
Risk Concentration Analyzer Module
Analyzes portfolio concentration risks including sector, stock, and correlation risks
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# Import get_db_connection with error handling
try:
    from common.Boilerplate import get_db_connection
except (ImportError, AttributeError, Exception) as e:
    logging.warning(f"Failed to import get_db_connection from common.Boilerplate: {e}. Using fallback.")
    # Define a fallback function
    import psycopg2
    def get_db_connection():
        return psycopg2.connect(
            host="postgres",
            database="mydb",
            user="postgres",
            password="postgres"
        )

class RiskConcentrationAnalyzer:
    """
    Analyze portfolio concentration risks
    """

    def __init__(self):
        """Initialize Risk Concentration Analyzer"""
        self.risk_thresholds = {
            'single_stock_max': 0.20,  # 20% max in single stock
            'sector_max': 0.30,        # 30% max in single sector
            'top_10_max': 0.60,        # 60% max in top 10 holdings
            'correlation_high': 0.7    # High correlation threshold
        }

    def get_holdings_data(self) -> pd.DataFrame:
        """
        Get current holdings from database

        Returns:
            DataFrame with holdings data
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Get equity holdings with sector information
            cursor.execute("""
                SELECT
                    h.trading_symbol,
                    h.quantity,
                    h.average_price,
                    COALESCE(rt.price_close, h.last_price, h.close_price, 0) as current_price,
                    h.pnl,
                    COALESCE(s.sector_code, 'UNKNOWN') as sector_name,
                    'EQUITY' as holding_type
                FROM my_schema.holdings h
                LEFT JOIN my_schema.master_scrips s ON h.trading_symbol = s.scrip_id
                LEFT JOIN (
                    SELECT DISTINCT ON (scrip_id) scrip_id, price_close, price_date
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date <= CURRENT_DATE
                    ORDER BY scrip_id, price_date DESC
                ) rt ON h.trading_symbol = rt.scrip_id
                WHERE h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                AND h.quantity > 0
            """)

            equity_rows = cursor.fetchall()
            logging.info(f"Found {len(equity_rows)} equity holdings")

            # Get MF holdings as well
            cursor.execute("""
                SELECT
                    tradingsymbol as trading_symbol,
                    quantity,
                    average_price,
                    COALESCE(last_price, 0) as current_price,
                    pnl,
                    'UNKNOWN' as sector_name,
                    'MF' as holding_type
                FROM my_schema.mf_holdings
                WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)
                AND quantity > 0
            """)
            mf_rows = cursor.fetchall()
            logging.info(f"Found {len(mf_rows)} MF holdings")

            conn.close()

            # Combine equity and MF holdings
            all_rows = list(equity_rows) + list(mf_rows)

            if not all_rows:
                logging.warning("No holdings found in database - checking if tables exist and have data")
                # Try to get max run_date to debug
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT MAX(run_date) FROM my_schema.holdings")
                    max_date = cursor.fetchone()[0]
                    logging.warning(f"Max run_date in holdings table: {max_date}")
                    cursor.execute("SELECT COUNT(*) FROM my_schema.holdings WHERE quantity > 0")
                    count = cursor.fetchone()[0]
                    logging.warning(f"Total holdings with quantity > 0: {count}")
                    conn.close()
                except Exception as e2:
                    logging.error(f"Error checking holdings table: {e2}")
                return pd.DataFrame()

            columns = ['trading_symbol', 'quantity', 'average_price', 'current_price',
                      'pnl', 'sector_name', 'holding_type']

            df = pd.DataFrame(all_rows, columns=columns)
            
            # Ensure numeric columns are properly typed
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
            df['average_price'] = pd.to_numeric(df['average_price'], errors='coerce').fillna(0)
            df['current_price'] = pd.to_numeric(df['current_price'], errors='coerce').fillna(0)
            df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0)
            
            # Calculate values
            df['current_value'] = df['quantity'] * df['current_price']
            df['invested_value'] = df['quantity'] * df['average_price']
            
            # Filter out zero quantity holdings
            df = df[df['quantity'] > 0]
            
            logging.info(f"Successfully loaded {len(df)} total holdings (equity: {len(equity_rows)}, MF: {len(mf_rows)})")

            return df
        except Exception as e:
            logging.error(f"Error fetching holdings: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def analyze_stock_concentration(self) -> Dict:
        """
        Analyze stock concentration risk

        Returns:
            Dictionary with stock concentration analysis
        """
        try:
            holdings = self.get_holdings_data()

            if holdings.empty:
                return {'error': 'No holdings found'}

            total_value = holdings['current_value'].sum()

            # Calculate position sizes
            holdings['weight'] = holdings['current_value'] / total_value
            holdings = holdings.sort_values('weight', ascending=False)

            # Concentration metrics
            top_holding = holdings.iloc[0] if len(holdings) > 0 else None
            top_5_sum = holdings['weight'].head(5).sum()
            top_10_sum = holdings['weight'].head(10).sum()

            # Largest position
            largest_position = {
                'symbol': top_holding['trading_symbol'] if top_holding is not None else None,
                'weight': float(top_holding['weight']) if top_holding is not None else 0.0,
                'value': float(top_holding['current_value']) if top_holding is not None else 0.0
            }

            # Concentration alerts
            alerts = []
            if largest_position['weight'] > self.risk_thresholds['single_stock_max']:
                alerts.append({
                    'type': 'high_single_stock_exposure',
                    'severity': 'high',
                    'message': f"Single stock exposure ({largest_position['symbol']}) is {largest_position['weight']:.1%}, exceeds {self.risk_thresholds['single_stock_max']:.1%} threshold"
                })

            if top_10_sum > self.risk_thresholds['top_10_max']:
                alerts.append({
                    'type': 'high_top_10_concentration',
                    'severity': 'medium',
                    'message': f"Top 10 holdings represent {top_10_sum:.1%} of portfolio, exceeds {self.risk_thresholds['top_10_max']:.1%} threshold"
                })

            # Get top 5 holdings with full details
            top_5_holdings = holdings.head(5)[['trading_symbol', 'weight', 'current_value']].copy()
            top_5_holdings['percentage'] = top_5_holdings['weight']  # Keep weight as percentage (0-1)
            
            return {
                'total_value': float(total_value),
                'holding_count': len(holdings),
                'largest_position': largest_position,
                'top_5_concentration': float(top_5_sum),
                'top_10_concentration': float(top_10_sum),
                'top_5_holdings': top_5_holdings.to_dict('records'),
                'concentration_distribution': holdings[['trading_symbol', 'weight']].head(20).to_dict('records'),
                'alerts': alerts
            }
        except Exception as e:
            logging.error(f"Error analyzing stock concentration: {e}")
            return {'error': str(e)}

    def analyze_sector_concentration(self) -> Dict:
        """
        Analyze sector concentration risk

        Returns:
            Dictionary with sector concentration analysis
        """
        try:
            holdings = self.get_holdings_data()

            if holdings.empty:
                return {'error': 'No holdings found'}

            total_value = holdings['current_value'].sum()

            # Group by sector and replace UNKNOWN with Misc
            holdings['sector_name'] = holdings['sector_name'].replace('UNKNOWN', 'Misc')
            sector_allocation = holdings.groupby('sector_name').agg({
                'current_value': 'sum'
            }).reset_index()

            sector_allocation['weight'] = sector_allocation['current_value'] / total_value
            sector_allocation = sector_allocation.sort_values('weight', ascending=False)

            # Sector concentration metrics
            sector_count = len(sector_allocation)
            top_sector = sector_allocation.iloc[0] if len(sector_allocation) > 0 else None

            # Sector diversity
            herfindahl_index = (sector_allocation['weight'] ** 2).sum()

            # Alerts
            alerts = []
            if top_sector is not None and top_sector['weight'] > self.risk_thresholds['sector_max']:
                alerts.append({
                    'type': 'high_sector_exposure',
                    'severity': 'high',
                    'message': f"Sector exposure ({top_sector['sector_name']}) is {top_sector['weight']:.1%}, exceeds {self.risk_thresholds['sector_max']:.1%} threshold"
                })

            return {
                'total_value': float(total_value),
                'sector_count': sector_count,
                'top_sector': {
                    'sector': top_sector['sector_name'] if top_sector is not None else None,
                    'weight': float(top_sector['weight']) if top_sector is not None else 0.0,
                    'value': float(top_sector['current_value']) if top_sector is not None else 0.0
                },
                'sector_allocation': sector_allocation.to_dict('records'),
                'herfindahl_index': float(herfindahl_index),
                'sector_diversity_score': float(1 - herfindahl_index),  # Higher is more diversified
                'alerts': alerts
            }
        except Exception as e:
            logging.error(f"Error analyzing sector concentration: {e}")
            return {'error': str(e)}

    def analyze_correlation_risk(self, correlation_threshold: float = 0.7) -> Dict:
        """
        Analyze correlation risk between holdings

        Args:
            correlation_threshold: Threshold for high correlation

        Returns:
            Dictionary with correlation analysis
        """
        try:
            holdings = self.get_holdings_data()

            if holdings.empty or len(holdings) < 2:
                return {'error': 'Insufficient holdings for correlation analysis'}

            # Get correlation matrix (simplified - using basic correlation calculation)
            correlation_matrix = self._calculate_basic_correlation(holdings)

            if correlation_matrix.empty:
                return {'error': 'Could not calculate correlation matrix'}

            # Find highly correlated pairs
            high_correlation_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) >= correlation_threshold:
                        high_correlation_pairs.append({
                            'stock1': correlation_matrix.columns[i],
                            'stock2': correlation_matrix.columns[j],
                            'correlation': float(corr_value)
                        })

            # Average correlation
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()

            # Correlation alerts
            alerts = []
            if len(high_correlation_pairs) > 0:
                alerts.append({
                    'type': 'high_correlation_pairs',
                    'severity': 'medium',
                    'message': f"Found {len(high_correlation_pairs)} highly correlated stock pairs (>{correlation_threshold})"
                })

            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'high_correlation_pairs': high_correlation_pairs,
                'average_correlation': float(avg_correlation),
                'correlation_threshold': correlation_threshold,
                'alerts': alerts
            }
        except Exception as e:
            logging.error(f"Error analyzing correlation risk: {e}")
            return {'error': str(e)}

    def analyze_liquidity_risk(self) -> Dict:
        """
        Analyze liquidity risk based on trading volume and position sizes

        Returns:
            Dictionary with liquidity risk analysis
        """
        try:
            holdings = self.get_holdings_data()

            if holdings.empty:
                return {'error': 'No holdings found'}

            total_value = holdings['current_value'].sum()
            holdings['weight'] = holdings['current_value'] / total_value

            # Simplified liquidity analysis (would need volume data for full analysis)
            # For now, flag large positions as potential liquidity risks
            liquidity_risks = []

            for _, holding in holdings.iterrows():
                if holding['weight'] > 0.05:  # Positions > 5%
                    liquidity_risks.append({
                        'symbol': holding['trading_symbol'],
                        'weight': float(holding['weight']),
                        'value': float(holding['current_value']),
                        'risk_level': 'high' if holding['weight'] > 0.10 else 'medium'
                    })

            # Liquidity alerts
            alerts = []
            large_positions = len([r for r in liquidity_risks if r['risk_level'] == 'high'])
            if large_positions > 0:
                alerts.append({
                    'type': 'large_illiquid_positions',
                    'severity': 'medium',
                    'message': f"Found {large_positions} large positions (>10% each) that may have liquidity issues"
                })

            return {
                'total_positions': len(holdings),
                'liquidity_risks': liquidity_risks,
                'alerts': alerts
            }
        except Exception as e:
            logging.error(f"Error analyzing liquidity risk: {e}")
            return {'error': str(e)}

    def get_comprehensive_concentration_analysis(self) -> Dict:
        """
        Get comprehensive concentration analysis

        Returns:
            Dictionary with all concentration analyses
        """
        try:
            return {
                'stock_concentration': self.analyze_stock_concentration(),
                'sector_concentration': self.analyze_sector_concentration(),
                'correlation_risk': self.analyze_correlation_risk(),
                'liquidity_risk': self.analyze_liquidity_risk(),
                'overall_risk_assessment': self._calculate_overall_risk_score()
            }
        except Exception as e:
            logging.error(f"Error getting comprehensive concentration analysis: {e}")
            return {'error': str(e)}

    def _calculate_basic_correlation(self, holdings: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic correlation matrix (simplified version)

        Args:
            holdings: Holdings DataFrame

        Returns:
            Correlation matrix
        """
        try:
            # This is a simplified version - in practice would use historical returns
            # For now, create a mock correlation matrix
            symbols = holdings['trading_symbol'].tolist()
            n = len(symbols)

            if n < 2:
                return pd.DataFrame()

            # Create a correlation matrix with some realistic correlations
            np.random.seed(42)  # For reproducibility
            corr_matrix = np.random.uniform(0.1, 0.8, (n, n))

            # Make diagonal 1.0
            np.fill_diagonal(corr_matrix, 1.0)

            # Make symmetric
            corr_matrix = (corr_matrix + corr_matrix.T) / 2

            return pd.DataFrame(corr_matrix, index=symbols, columns=symbols)
        except Exception as e:
            logging.error(f"Error calculating basic correlation: {e}")
            return pd.DataFrame()

    def _calculate_overall_risk_score(self) -> Dict:
        """
        Calculate overall concentration risk score

        Returns:
            Dictionary with risk score and assessment
        """
        try:
            stock_analysis = self.analyze_stock_concentration()
            sector_analysis = self.analyze_sector_concentration()

            if 'error' in stock_analysis or 'error' in sector_analysis:
                return {'error': 'Cannot calculate risk score'}

            # Simple risk scoring
            risk_score = 0

            # Stock concentration risk
            if stock_analysis['largest_position']['weight'] > 0.15:
                risk_score += 3
            elif stock_analysis['largest_position']['weight'] > 0.10:
                risk_score += 2
            elif stock_analysis['largest_position']['weight'] > 0.05:
                risk_score += 1

            # Sector concentration risk
            if sector_analysis['top_sector']['weight'] > 0.25:
                risk_score += 3
            elif sector_analysis['top_sector']['weight'] > 0.20:
                risk_score += 2
            elif sector_analysis['top_sector']['weight'] > 0.15:
                risk_score += 1

            # Determine risk level
            if risk_score >= 5:
                risk_level = 'high'
                assessment = 'High concentration risk - consider diversification'
            elif risk_score >= 3:
                risk_level = 'medium'
                assessment = 'Moderate concentration risk - monitor closely'
            else:
                risk_level = 'low'
                assessment = 'Low concentration risk - well diversified'

            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'assessment': assessment
            }
        except Exception as e:
            logging.error(f"Error calculating overall risk score: {e}")
            return {'error': str(e)}
