#!/usr/bin/env python3
"""
Test runner for Risk Management tests
Run this script to execute all test cases
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_tests(include_integration=False):
    """Run all test cases
    
    Args:
        include_integration: If True, include integration tests (requires database)
    """
    loader = unittest.TestLoader()
    
    # Always run unit tests
    suite = loader.discover(os.path.dirname(__file__), pattern='test_risk_management.py')
    
    # Optionally include integration tests
    if include_integration:
        integration_suite = loader.discover(os.path.dirname(__file__), pattern='test_risk_integration.py')
        suite.addTest(integration_suite)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    print("="*70)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Risk Management tests')
    parser.add_argument('--integration', action='store_true', 
                       help='Include integration tests (requires database)')
    args = parser.parse_args()
    
    exit_code = run_tests(include_integration=args.integration)
    sys.exit(exit_code)

