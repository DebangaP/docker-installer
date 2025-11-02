"""
Mutual Fund Benchmark Mapper
Maps Mutual Funds to appropriate benchmark indices
"""
import logging
from typing import Optional, Dict


class MFBenchmarkMapper:
    """Maps Mutual Funds to appropriate benchmarks"""
    
    def __init__(self):
        """Initialize benchmark mapper with default mappings"""
        self.benchmark_mappings = self._init_mappings()
    
    def _init_mappings(self) -> Dict[str, str]:
        """
        Initialize benchmark mappings
        
        Returns:
            Dictionary mapping fund keywords to benchmark symbols
        """
        return {
            # Sector-specific funds
            'IT': 'NIFTY_IT',  # Information Technology
            'FIN': 'NIFTY_FIN',  # Financial Services
            'BANK': 'NIFTY_BNK',  # Banking
            'PHARMA': 'NIFTY_PHA',  # Pharmaceuticals
            'FMCG': 'NIFTY_FMC',  # FMCG
            'AUTO': 'NIFTY_AUT',  # Automobile
            'REALTY': 'NIFTY_REA',  # Realty
            'ENERGY': 'NIFTY_ENE',  # Energy
            'METAL': 'NIFTY_MET',  # Metal/Materials
            
            # Default benchmark
            'default': 'NIFTY50'  # NIFTY 50
        }
    
    def _get_benchmark_yahoo_code(self, benchmark_symbol: str) -> str:
        """
        Get Yahoo Finance code for benchmark symbol
        
        Args:
            benchmark_symbol: Benchmark symbol (e.g., 'NIFTY50')
            
        Returns:
            Yahoo Finance code (e.g., '^NSEI')
        """
        yahoo_codes = {
            'NIFTY50': '^NSEI',
            'NIFTY_IT': '^CNXIT',
            'NIFTY_FIN': '^CNXFIN',
            'NIFTY_BNK': '^NSEBANK',
            'NIFTY_PHA': '^CNXPHARMA',
            'NIFTY_FMC': '^CNXFMCG',
            'NIFTY_AUT': '^CNXAUTO',
            'NIFTY_REA': '^CNXREALTY',
            'NIFTY_ENE': '^CNXENERGY',
            'NIFTY_MET': '^CNXMETAL'
        }
        
        return yahoo_codes.get(benchmark_symbol, '^NSEI')
    
    def map_to_benchmark(self, mf_symbol: str, fund_name: str = None) -> Dict[str, str]:
        """
        Map MF to appropriate benchmark
        
        Args:
            mf_symbol: Mutual Fund trading symbol
            fund_name: Mutual Fund name (optional, used for better mapping)
            
        Returns:
            Dictionary with benchmark_symbol and yahoo_code
        """
        # Normalize for case-insensitive matching
        search_text = (fund_name or mf_symbol).upper()
        
        # Check for sector keywords
        for keyword, benchmark in self.benchmark_mappings.items():
            if keyword != 'default' and keyword in search_text:
                benchmark_symbol = benchmark
                yahoo_code = self._get_benchmark_yahoo_code(benchmark_symbol)
                
                logging.debug(f"Mapped {mf_symbol} to {benchmark_symbol} (keyword: {keyword})")
                return {
                    'benchmark_symbol': benchmark_symbol,
                    'yahoo_code': yahoo_code,
                    'match_reason': f"Matched keyword: {keyword}"
                }
        
        # Default to NIFTY50
        benchmark_symbol = self.benchmark_mappings['default']
        yahoo_code = self._get_benchmark_yahoo_code(benchmark_symbol)
        
        logging.debug(f"Mapped {mf_symbol} to {benchmark_symbol} (default)")
        return {
            'benchmark_symbol': benchmark_symbol,
            'yahoo_code': yahoo_code,
            'match_reason': 'Default benchmark'
        }
    
    def get_benchmark_for_mf(self, mf_symbol: str, fund_name: str = None) -> Optional[str]:
        """
        Get benchmark symbol for a mutual fund (simplified method)
        
        Args:
            mf_symbol: Mutual Fund trading symbol
            fund_name: Mutual Fund name (optional)
            
        Returns:
            Benchmark symbol or None
        """
        mapping = self.map_to_benchmark(mf_symbol, fund_name)
        return mapping.get('benchmark_symbol')
    
    def set_custom_mapping(self, mf_symbol: str, benchmark_symbol: str):
        """
        Set custom benchmark mapping for a specific MF
        
        Args:
            mf_symbol: Mutual Fund trading symbol
            benchmark_symbol: Benchmark symbol to map to
        """
        # This could be extended to store custom mappings in database
        logging.info(f"Custom mapping set: {mf_symbol} -> {benchmark_symbol}")
        # For now, this is a placeholder for future enhancement
        # Could be implemented with a database table for custom mappings

