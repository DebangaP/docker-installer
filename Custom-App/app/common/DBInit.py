"""
Database Initialization and Update Script
Safely adds/updates database tables without deleting existing data
Also keeps Schema.sql in sync with database changes
"""

import logging
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from datetime import datetime
from typing import List, Dict, Optional


# add usage to this file -- how to add new table DDLs?  

class DBInit:
    """
    Database initialization and schema update utility
    """
    
    def __init__(self, db_config: Optional[Dict] = None):
        """
        Initialize DBInit with database configuration
        
        Args:
            db_config: Database configuration dict with keys:
                - host: Database host (default: 'postgres')
                - database: Database name (default: 'mydb')
                - user: Database user (default: 'postgres')
                - password: Database password (default: 'postgres')
                - port: Database port (default: 5432)
        """
        if db_config is None:
            db_config = {
                'host': os.getenv('DB_HOST', 'postgres'),
                'database': os.getenv('DB_NAME', 'mydb'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'postgres'),
                'port': int(os.getenv('DB_PORT', 5432))
            }
        
        self.db_config = db_config
        self.schema_file = os.path.join(os.path.dirname(__file__), '..', '..', 'Postgres', 'Schema.sql')
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)
    
    def execute_safe_ddl(self, ddl_statements: List[str], dry_run: bool = False) -> Dict:
        """
        Execute DDL statements safely (with IF NOT EXISTS, etc.)
        
        Args:
            ddl_statements: List of DDL statements to execute
            dry_run: If True, only validate without executing
            
        Returns:
            Dictionary with execution results
        """
        results = {
            'success': True,
            'executed': [],
            'skipped': [],
            'errors': [],
            'dry_run': dry_run
        }
        
        conn = None
        cursor = None
        
        try:
            conn = self.get_connection()
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            for statement in ddl_statements:
                if not statement or not statement.strip():
                    continue
                
                # Clean statement
                statement = statement.strip()
                
                # Skip comments and empty lines
                if statement.startswith('--') or statement.startswith('/*') or not statement:
                    continue
                
                # Skip CREATE SCHEMA if it already exists
                if statement.upper().startswith('CREATE SCHEMA'):
                    schema_name = self._extract_schema_name(statement)
                    if schema_name and self._schema_exists(cursor, schema_name):
                        results['skipped'].append({
                            'statement': statement[:100] + '...' if len(statement) > 100 else statement,
                            'reason': f'Schema {schema_name} already exists'
                        })
                        continue
                
                # Skip CREATE TABLE if it already exists (for safety)
                if statement.upper().startswith('CREATE TABLE'):
                    table_name = self._extract_table_name(statement)
                    if table_name and self._table_exists(cursor, table_name):
                        # Check if statement has IF NOT EXISTS
                        if 'IF NOT EXISTS' not in statement.upper():
                            results['skipped'].append({
                                'statement': statement[:100] + '...' if len(statement) > 100 else statement,
                                'reason': f'Table {table_name} already exists (use IF NOT EXISTS or ALTER TABLE)'
                            })
                            continue
                
                # Skip CREATE INDEX if it already exists
                if statement.upper().startswith('CREATE INDEX'):
                    index_name = self._extract_index_name(statement)
                    if index_name and self._index_exists(cursor, index_name):
                        if 'IF NOT EXISTS' not in statement.upper():
                            results['skipped'].append({
                                'statement': statement[:100] + '...' if len(statement) > 100 else statement,
                                'reason': f'Index {index_name} already exists'
                            })
                            continue
                
                try:
                    if not dry_run:
                        cursor.execute(statement)
                        results['executed'].append({
                            'statement': statement[:100] + '...' if len(statement) > 100 else statement,
                            'success': True
                        })
                        logging.info(f"Executed: {statement[:100]}...")
                    else:
                        results['executed'].append({
                            'statement': statement[:100] + '...' if len(statement) > 100 else statement,
                            'success': True,
                            'dry_run': True
                        })
                        logging.info(f"DRY RUN - Would execute: {statement[:100]}...")
                        
                except psycopg2.Error as e:
                    error_msg = str(e)
                    # Some errors are expected (like table already exists with IF NOT EXISTS)
                    if 'already exists' in error_msg.lower() or 'duplicate' in error_msg.lower():
                        results['skipped'].append({
                            'statement': statement[:100] + '...' if len(statement) > 100 else statement,
                            'reason': error_msg
                        })
                    else:
                        results['errors'].append({
                            'statement': statement[:100] + '...' if len(statement) > 100 else statement,
                            'error': error_msg
                        })
                        logging.error(f"Error executing statement: {error_msg}")
                        results['success'] = False
            
            if not dry_run:
                conn.commit()
            
        except Exception as e:
            logging.error(f"Error in execute_safe_ddl: {e}")
            results['success'] = False
            results['errors'].append({
                'statement': 'Connection error',
                'error': str(e)
            })
        
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        
        return results
    
    def _schema_exists(self, cursor, schema_name: str) -> bool:
        """Check if schema exists"""
        try:
            cursor.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM information_schema.schemata 
                    WHERE schema_name = %s
                )
            """, (schema_name,))
            return cursor.fetchone()[0]
        except:
            return False
    
    def _table_exists(self, cursor, table_name: str) -> bool:
        """Check if table exists in my_schema"""
        try:
            # Handle schema.table format
            if '.' in table_name:
                schema, table = table_name.split('.', 1)
            else:
                schema = 'my_schema'
                table = table_name
            
            cursor.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = %s AND table_name = %s
                )
            """, (schema, table))
            return cursor.fetchone()[0]
        except:
            return False
    
    def _index_exists(self, cursor, index_name: str) -> bool:
        """Check if index exists"""
        try:
            cursor.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_indexes 
                    WHERE indexname = %s
                )
            """, (index_name,))
            return cursor.fetchone()[0]
        except:
            return False
    
    def _extract_schema_name(self, statement: str) -> Optional[str]:
        """Extract schema name from CREATE SCHEMA statement"""
        try:
            # CREATE SCHEMA my_schema
            parts = statement.upper().split()
            if 'SCHEMA' in parts:
                idx = parts.index('SCHEMA')
                if idx + 1 < len(parts):
                    return parts[idx + 1].lower()
        except:
            pass
        return None
    
    def _extract_table_name(self, statement: str) -> Optional[str]:
        """Extract table name from CREATE TABLE statement"""
        try:
            # CREATE TABLE my_schema.table_name
            parts = statement.upper().split()
            if 'TABLE' in parts:
                idx = parts.index('TABLE')
                # Skip IF NOT EXISTS
                if idx + 1 < len(parts) and parts[idx + 1] == 'IF':
                    if idx + 3 < len(parts):
                        table_part = parts[idx + 3]
                    else:
                        return None
                else:
                    table_part = parts[idx + 1] if idx + 1 < len(parts) else None
                
                if table_part:
                    # Remove parentheses and handle schema.table format
                    table_name = table_part.split('(')[0].strip()
                    return table_name.lower()
        except:
            pass
        return None
    
    def _extract_index_name(self, statement: str) -> Optional[str]:
        """Extract index name from CREATE INDEX statement"""
        try:
            # CREATE INDEX idx_name ON ...
            parts = statement.upper().split()
            if 'INDEX' in parts:
                idx = parts.index('INDEX')
                if idx + 1 < len(parts):
                    index_name = parts[idx + 1].lower()
                    # Remove IF NOT EXISTS
                    if index_name == 'if':
                        if idx + 3 < len(parts):
                            return parts[idx + 3].lower()
                    else:
                        return index_name
        except:
            pass
        return None
    
    def load_ddl_from_file(self, file_path: Optional[str] = None) -> List[str]:
        """
        Load DDL statements from Schema.sql file
        
        Args:
            file_path: Path to Schema.sql file (default: relative path)
            
        Returns:
            List of DDL statements
        """
        if file_path is None:
            file_path = self.schema_file
        
        if not os.path.exists(file_path):
            logging.warning(f"Schema file not found: {file_path}")
            return []
        
        ddl_statements = []
        current_statement = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Skip comments
                    if line.startswith('--'):
                        continue
                    
                    # Handle multi-line statements
                    current_statement.append(line)
                    
                    # Check if statement ends with semicolon
                    if line.endswith(';'):
                        statement = ' '.join(current_statement)
                        if statement:
                            ddl_statements.append(statement)
                        current_statement = []
            
            # Handle last statement if no semicolon
            if current_statement:
                statement = ' '.join(current_statement)
                if statement:
                    ddl_statements.append(statement)
            
            logging.info(f"Loaded {len(ddl_statements)} DDL statements from {file_path}")
            return ddl_statements
            
        except Exception as e:
            logging.error(f"Error loading DDL from file: {e}")
            return []
    
    def add_ddl_to_schema(self, new_ddl: str, description: str = None):
        """
        Add new DDL statement to Schema.sql file
        
        Args:
            new_ddl: New DDL statement to add
            description: Optional description comment
        """
        if not os.path.exists(self.schema_file):
            logging.error(f"Schema file not found: {self.schema_file}")
            return False
        
        try:
            with open(self.schema_file, 'a', encoding='utf-8') as f:
                f.write('\n\n')
                if description:
                    f.write(f'-- {description}\n')
                f.write(f'-- Added on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                f.write(new_ddl)
                if not new_ddl.strip().endswith(';'):
                    f.write(';')
                f.write('\n')
            
            logging.info(f"Added DDL to Schema.sql: {new_ddl[:100]}...")
            return True
            
        except Exception as e:
            logging.error(f"Error adding DDL to Schema.sql: {e}")
            return False
    
    def initialize_database(self, dry_run: bool = False) -> Dict:
        """
        Initialize database by executing all DDL statements from Schema.sql
        
        Args:
            dry_run: If True, only validate without executing
            
        Returns:
            Dictionary with execution results
        """
        logging.info("Starting database initialization...")
        
        ddl_statements = self.load_ddl_from_file()
        
        if not ddl_statements:
            return {
                'success': False,
                'error': 'No DDL statements found in Schema.sql',
                'executed': [],
                'skipped': [],
                'errors': []
            }
        
        results = self.execute_safe_ddl(ddl_statements, dry_run=dry_run)
        
        # Also create swing_trade_suggestions table if it doesn't exist
        swing_trade_table_ddl = """
        CREATE TABLE IF NOT EXISTS my_schema.swing_trade_suggestions (
            id SERIAL PRIMARY KEY,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            analysis_date DATE,
            run_date DATE DEFAULT CURRENT_DATE,
            scrip_id VARCHAR(10),
            instrument_token BIGINT,
            pattern_type VARCHAR(50),
            direction VARCHAR(10) DEFAULT 'BUY',
            entry_price DOUBLE PRECISION,
            target_price DOUBLE PRECISION,
            stop_loss DOUBLE PRECISION,
            potential_gain_pct DOUBLE PRECISION,
            risk_reward_ratio DOUBLE PRECISION,
            confidence_score DOUBLE PRECISION,
            holding_period_days INT,
            current_price DOUBLE PRECISION,
            sma_20 DOUBLE PRECISION,
            sma_50 DOUBLE PRECISION,
            sma_200 DOUBLE PRECISION,
            rsi_14 DOUBLE PRECISION,
            macd DOUBLE PRECISION,
            macd_signal DOUBLE PRECISION,
            atr_14 DOUBLE PRECISION,
            volume_trend VARCHAR(20),
            support_level DOUBLE PRECISION,
            resistance_level DOUBLE PRECISION,
            rationale TEXT,
            technical_context JSONB,
            diagnostics JSONB,
            filtering_criteria JSONB,
            status VARCHAR(20) DEFAULT 'ACTIVE'
        )
        """
        
        # Execute swing trade table creation
        swing_results = self.execute_safe_ddl([swing_trade_table_ddl], dry_run=dry_run)
        
        # Create indexes if table was created
        if swing_results['success'] or any('already exists' in str(e.get('reason', '')).lower() for e in swing_results.get('skipped', [])):
            index_statements = [
                "CREATE INDEX IF NOT EXISTS idx_swing_trade_analysis_date ON my_schema.swing_trade_suggestions(analysis_date)",
                "CREATE INDEX IF NOT EXISTS idx_swing_trade_scrip_id ON my_schema.swing_trade_suggestions(scrip_id)",
                "CREATE INDEX IF NOT EXISTS idx_swing_trade_pattern ON my_schema.swing_trade_suggestions(pattern_type)",
                "CREATE INDEX IF NOT EXISTS idx_swing_trade_confidence ON my_schema.swing_trade_suggestions(confidence_score DESC)",
                "CREATE INDEX IF NOT EXISTS idx_swing_trade_run_date ON my_schema.swing_trade_suggestions(run_date)"
            ]
            index_results = self.execute_safe_ddl(index_statements, dry_run=dry_run)
            swing_results['executed'].extend(index_results['executed'])
            swing_results['skipped'].extend(index_results['skipped'])
            swing_results['errors'].extend(index_results['errors'])
        
        # Merge results
        results['executed'].extend(swing_results['executed'])
        results['skipped'].extend(swing_results['skipped'])
        results['errors'].extend(swing_results['errors'])
        if not swing_results['success']:
            results['success'] = False
        
        # Also create prophet_predictions table if it doesn't exist
        prophet_table_ddl = """
        CREATE TABLE IF NOT EXISTS my_schema.prophet_predictions (
            id SERIAL PRIMARY KEY,
            scrip_id VARCHAR(10) NOT NULL,
            run_date DATE DEFAULT CURRENT_DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            prediction_days INTEGER DEFAULT 30,
            current_price DOUBLE PRECISION,
            predicted_price_30d DOUBLE PRECISION,
            predicted_price_change_pct DOUBLE PRECISION,
            prediction_confidence DOUBLE PRECISION,
            prediction_details JSONB,
            status VARCHAR(20) DEFAULT 'ACTIVE',
            UNIQUE(scrip_id, run_date, prediction_days)
        )
        """
        
        # Execute prophet predictions table creation
        prophet_results = self.execute_safe_ddl([prophet_table_ddl], dry_run=dry_run)
        
        # Create indexes if table was created
        if prophet_results['success'] or any('already exists' in str(e.get('reason', '')).lower() for e in prophet_results.get('skipped', [])):
            prophet_index_statements = [
                "CREATE INDEX IF NOT EXISTS idx_prophet_run_date ON my_schema.prophet_predictions(run_date)",
                "CREATE INDEX IF NOT EXISTS idx_prophet_scrip_id ON my_schema.prophet_predictions(scrip_id)",
                "CREATE INDEX IF NOT EXISTS idx_prophet_price_change ON my_schema.prophet_predictions(predicted_price_change_pct DESC)",
                "CREATE INDEX IF NOT EXISTS idx_prophet_status ON my_schema.prophet_predictions(status)"
            ]
            prophet_index_results = self.execute_safe_ddl(prophet_index_statements, dry_run=dry_run)
            prophet_results['executed'].extend(prophet_index_results['executed'])
            prophet_results['skipped'].extend(prophet_index_results['skipped'])
            prophet_results['errors'].extend(prophet_index_results['errors'])
        
        # Merge prophet results
        results['executed'].extend(prophet_results['executed'])
        results['skipped'].extend(prophet_results['skipped'])
        results['errors'].extend(prophet_results['errors'])
        if not prophet_results['success']:
            results['success'] = False
        
        logging.info(f"Database initialization completed. "
                     f"Executed: {len(results['executed'])}, "
                     f"Skipped: {len(results['skipped'])}, "
                     f"Errors: {len(results['errors'])}")
        
        return results
    
    def add_table(self, table_ddl: str, update_schema: bool = True) -> Dict:
        """
        Add a new table to database and optionally update Schema.sql
        
        Args:
            table_ddl: CREATE TABLE statement (with IF NOT EXISTS recommended)
            update_schema: If True, also add to Schema.sql
            
        Returns:
            Dictionary with execution results
        """
        results = self.execute_safe_ddl([table_ddl], dry_run=False)
        
        if results['success'] and update_schema:
            self.add_ddl_to_schema(table_ddl, description='New table DDL')
        
        return results
    
    def add_index(self, index_ddl: str, update_schema: bool = True) -> Dict:
        """
        Add a new index to database and optionally update Schema.sql
        
        Args:
            index_ddl: CREATE INDEX statement (with IF NOT EXISTS recommended)
            update_schema: If True, also add to Schema.sql
            
        Returns:
            Dictionary with execution results
        """
        results = self.execute_safe_ddl([index_ddl], dry_run=False)
        
        if results['success'] and update_schema:
            self.add_ddl_to_schema(index_ddl, description='New index DDL')
        
        return results
    
    def update_table_schema(self, alter_ddl: str, update_schema: bool = True) -> Dict:
        """
        Update existing table schema using ALTER TABLE
        
        Args:
            alter_ddl: ALTER TABLE statement
            update_schema: If True, also add to Schema.sql
            
        Returns:
            Dictionary with execution results
        """
        results = self.execute_safe_ddl([alter_ddl], dry_run=False)
        
        if results['success'] and update_schema:
            self.add_ddl_to_schema(alter_ddl, description='Table schema update (ALTER TABLE)')
        
        return results


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Database Initialization and Update Tool')
    parser.add_argument('--dry-run', action='store_true', help='Validate without executing')
    parser.add_argument('--init', action='store_true', help='Initialize database from Schema.sql')
    parser.add_argument('--ddl', type=str, help='Execute specific DDL statement')
    parser.add_argument('--file', type=str, help='Execute DDL statements from file')
    parser.add_argument('--update-schema', action='store_true', 
                       help='Update Schema.sql with new DDL statements')
    
    args = parser.parse_args()
    
    db_init = DBInit()
    
    if args.init:
        # Initialize database from Schema.sql
        results = db_init.initialize_database(dry_run=args.dry_run)
        
        print(f"\n{'='*60}")
        print("Database Initialization Results")
        print(f"{'='*60}")
        print(f"Success: {results['success']}")
        print(f"Executed: {len(results['executed'])}")
        print(f"Skipped: {len(results['skipped'])}")
        print(f"Errors: {len(results['errors'])}")
        
        if results['errors']:
            print("\nErrors:")
            for error in results['errors']:
                print(f"  - {error['error']}")
        
    elif args.ddl:
        # Execute specific DDL
        results = db_init.execute_safe_ddl([args.ddl], dry_run=args.dry_run)
        if args.update_schema:
            db_init.add_ddl_to_schema(args.ddl)
        
        print(f"\nDDL Execution Results:")
        print(f"Success: {results['success']}")
        if results['errors']:
            print(f"Errors: {results['errors']}")
    
    elif args.file:
        # Execute DDL from file
        ddl_statements = db_init.load_ddl_from_file(args.file)
        results = db_init.execute_safe_ddl(ddl_statements, dry_run=args.dry_run)
        
        print(f"\nDDL File Execution Results:")
        print(f"Success: {results['success']}")
        print(f"Executed: {len(results['executed'])}")
        if results['errors']:
            print(f"Errors: {results['errors']}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

