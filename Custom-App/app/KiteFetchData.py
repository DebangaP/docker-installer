from Boilerplate import *

def init_postgres_conn():
    try:
        print('init postgres')
        cursor = conn.cursor()
        # Create tables not created already, and not part of Schema.sql so that the Postgres container data does not need to be deleted using "docker rm -f $(docker ps -a -q)"
        cursor.execute("""
                CREATE TABLE IF NOT EXISTS my_schema.holdings (
                    fetch_timestamp TIMESTAMP DEFAULT current_timestamp,
                    run_date DATE DEFAULT CURRENT_DATE,
                    trading_symbol VARCHAR(50),
                    instrument_token INTEGER,
                    isin VARCHAR(12),
                    quantity INTEGER,
                    t1_quantity INTEGER,
                    authorised_quantity INTEGER,
                    average_price FLOAT,
                    close_price FLOAT,
                    last_price FLOAT,
                    pnl FLOAT,
                    collateral_quantity INTEGER,
                    collateral_type VARCHAR(20),
                    CONSTRAINT holdings_unique_key UNIQUE (instrument_token, run_date)
                );
            """)
        conn.commit()
        logging.info("Connected to PostgreSQL and initialized tables")
    except Exception as e:
        conn.rollback()
        logging.error(f"Postgres connection failed: {e}")
    

def fetch_and_save_profile():
    print('fetching profile')
    profile = kite.profile()  # attempt to fetch user profile
    logging.info(f"User profile: {profile}")
    print(type(profile))

    try:
        cursor.execute("""
                    INSERT INTO my_schema.profile (
                        user_id, user_name, email, user_type, broker,
                        products, order_types, exchanges
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE
                    SET user_name = EXCLUDED.user_name,
                        email = EXCLUDED.email,
                        user_type = EXCLUDED.user_type,
                        broker = EXCLUDED.broker,
                        products = EXCLUDED.products,
                        order_types = EXCLUDED.order_types,
                        exchanges = EXCLUDED.exchanges
                """, (
                    profile.get('user_id'), profile.get('user_name'),
                    profile.get('email'), profile.get('user_type'), profile.get('broker'),
                    profile.get('products', []), profile.get('order_types', []),
                    profile.get('exchanges', [])
                ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"Error saving profile: {e}")


def fetch_and_save_orders(kite: KiteConnect, cursor: psycopg2.extensions.cursor):
    kite = kite
    cursor = cursor
    fetch_and_save_orders1()


def fetch_and_save_orders1():
    try:
        print('Fetching orders')
        orders = kite.orders()  # Fetch orders from Kite API
        logging.info(f"User orders: {orders}")

        if not orders:
            logging.warning("No orders found")
            #return []

        # Prepare data for insertion
        order_data = [
            (
                order.get('order_id'),
                order.get('parent_order_id'),
                order.get('exchange_order_id'),
                order.get('status'),
                order.get('status_message'),
                order.get('order_type'),
                order.get('transaction_type'),
                order.get('exchange'),
                order.get('tradingsymbol'),
                order.get('instrument_token'),
                order.get('quantity'),
                order.get('price'),
                order.get('trigger_price'),
                order.get('average_price'),
                order.get('order_timestamp'),  # Already a datetime object
                order.get('exchange_timestamp')  # Already a datetime object
            ) for order in orders
        ]

        # Execute the INSERT statement
        cursor.executemany("""
            INSERT INTO my_schema.orders (
                order_id, parent_order_id, exchange_order_id, status,
                status_message, order_type, transaction_type, exchange, trading_symbol,
                instrument_token, quantity, price, trigger_price, average_price,
                order_timestamp, exchange_timestamp
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (order_id) DO UPDATE
            SET parent_order_id = EXCLUDED.parent_order_id,
                exchange_order_id = EXCLUDED.exchange_order_id,
                status = EXCLUDED.status,
                status_message = EXCLUDED.status_message,
                order_type = EXCLUDED.order_type,
                transaction_type = EXCLUDED.transaction_type,
                exchange = EXCLUDED.exchange,
                trading_symbol = EXCLUDED.trading_symbol,
                instrument_token = EXCLUDED.instrument_token,
                quantity = EXCLUDED.quantity,
                price = EXCLUDED.price,
                trigger_price = EXCLUDED.trigger_price,
                average_price = EXCLUDED.average_price,
                order_timestamp = EXCLUDED.order_timestamp,
                exchange_timestamp = EXCLUDED.exchange_timestamp
        """, order_data)

        conn.commit()
        logging.info(f"Inserted/Updated {cursor.rowcount} orders")
        #return orders

    except Exception as e:
        conn.rollback()
        logging.error(f"Error saving orders: {str(e)}")


def fetch_and_save_positions(kite: KiteConnect, cursor: psycopg2.extensions.cursor):
    kite = kite
    cursor = cursor
    fetch_and_save_positions1()


def fetch_and_save_positions1():
    print('fetching positions')
    positions = kite.positions()    # save into postgres table with current date
    logging.info(f"User positions: {positions}")

    if not positions:
        logging.warning("No positions found")
        #return []
            
    # Process both 'net' and 'day' positions
    all_positions = []
    for position_type in ['net', 'day']:
        for pos in positions.get(position_type, []):
            all_positions.append({
                        'position_type': position_type,
                        'tradingsymbol': pos.get('tradingsymbol'),
                        'instrument_token': pos.get('instrument_token'),
                        'exchange': pos.get('exchange'),
                        'product': pos.get('product'),
                        'quantity': pos.get('quantity'),
                        'buy_qty': pos.get('buy_quantity'),
                        'sell_qty': pos.get('sell_quantity'),
                        'buy_price': pos.get('buy_price'),
                        'sell_price': pos.get('sell_price'),
                        'average_price': pos.get('average_price'),
                        'last_price': pos.get('last_price'),
                        'pnl': pos.get('pnl'),
                        'm2m': pos.get('m2m'),
                        'realised': pos.get('realised'),
                        'unrealised': pos.get('unrealised'),
                        'value': pos.get('value')
                    })
            
        if all_positions:
            try:
                cursor.executemany("""
                        INSERT INTO my_schema.positions (
                            position_type, trading_symbol, instrument_token,
                            exchange, product, quantity, buy_qty, sell_qty, buy_price,
                            sell_price, average_price, last_price, pnl, m2m, realised,
                            unrealised, value
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (position_type, instrument_token) DO UPDATE
                        SET trading_symbol = EXCLUDED.trading_symbol,
                            exchange = EXCLUDED.exchange,
                            product = EXCLUDED.product,
                            quantity = EXCLUDED.quantity,
                            buy_qty = EXCLUDED.buy_qty,
                            sell_qty = EXCLUDED.sell_qty,
                            buy_price = EXCLUDED.buy_price,
                            sell_price = EXCLUDED.sell_price,
                            average_price = EXCLUDED.average_price,
                            last_price = EXCLUDED.last_price,
                            pnl = EXCLUDED.pnl,
                            m2m = EXCLUDED.m2m,
                            realised = EXCLUDED.realised,
                            unrealised = EXCLUDED.unrealised,
                            value = EXCLUDED.value
                    """, [
                        (
                            pos['position_type'], pos['tradingsymbol'],
                            pos['instrument_token'], pos['exchange'], pos['product'],
                            pos['quantity'], pos['buy_qty'], pos['sell_qty'],
                            pos['buy_price'], pos['sell_price'], pos['average_price'],
                            pos['last_price'], pos['pnl'], pos['m2m'],
                            pos['realised'], pos['unrealised'], pos['value']
                        ) for pos in all_positions
                    ])
                conn.commit()
                logging.info(f"Inserted/Updated {cursor.rowcount} positions")
            except Exception as e:
                conn.rollback()
                logging.error(f"Error in positions: {e}")
   

def fetch_and_save_holdings(kite: KiteConnect, cursor: psycopg2.extensions.cursor):
    kite = kite
    cursor = cursor
    fetch_and_save_holdings1()


def fetch_and_save_holdings1():
    print('fetching holdings')
    holdings = kite.holdings()  # save into postgres table with current date
    logging.info(f"User holdings: {holdings}")

    try:
        if not holdings:
            logging.warning("No holdings found")
            #return []
    except Exception as e:
        logging.error(f"Error XXXX: {e}")

    valid_holdings = []
    try:
        for h in holdings:
            # Validate critical fields
            if not h.get('instrument_token'):
                logging.warning(f"Skipping holding with missing instrument_token: {h}")
                continue
            holding = {
                'tradingsymbol': h.get('tradingsymbol', ''),
                'instrument_token': h.get('instrument_token', 0),
                'isin': h.get('isin', ''),
                'quantity': h.get('quantity', 0),
                't1_quantity': h.get('t1_quantity', 0),
                'authorised_quantity': h.get('authorised_quantity', 0),
                'average_price': h.get('average_price', 0.0),
                'close_price': h.get('close_price', None),
                'last_price': h.get('last_price', None),
                'pnl': h.get('pnl', None),
                'collateral_quantity': h.get('collateral_quantity', 0),
                'collateral_type': h.get('collateral_type', None)
            }
            valid_holdings.append(holding)
            logging.debug(f"Processed holding: {holding}")
    except Exception as e:
        logging.error(f"Error in validating: {e}")
       
    try:
        print('saving holdings')
        if valid_holdings:
            cursor.executemany("""
                INSERT INTO my_schema.holdings (
                    trading_symbol, instrument_token,
                    isin, quantity, t1_quantity, authorised_quantity, average_price,
                    close_price, last_price, pnl, collateral_quantity, collateral_type
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (instrument_token, run_date) DO UPDATE
                SET trading_symbol = EXCLUDED.trading_symbol,
                    isin = EXCLUDED.isin,
                    quantity = EXCLUDED.quantity,
                    t1_quantity = EXCLUDED.t1_quantity,
                    authorised_quantity = EXCLUDED.authorised_quantity,
                    average_price = EXCLUDED.average_price,
                    close_price = EXCLUDED.close_price,
                    last_price = EXCLUDED.last_price,
                    pnl = EXCLUDED.pnl,
                    collateral_quantity = EXCLUDED.collateral_quantity,
                    collateral_type = EXCLUDED.collateral_type
            """, [
                (
                    h['tradingsymbol'], h['instrument_token'],
                    h['isin'], h['quantity'],
                    h['t1_quantity'], h['authorised_quantity'],
                    h['average_price'], h['close_price'], h['last_price'],
                    h['pnl'], h['collateral_quantity'], h['collateral_type']
                ) for h in valid_holdings
            ])
            conn.commit()
            logging.info(f"Stored {len(valid_holdings)} holdings")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error saving holdings: {e}")  


def fetch_and_save_mf_holdings(kite: KiteConnect, cursor: psycopg2.extensions.cursor):
    kite = kite
    cursor = cursor
    fetch_and_save_mf_holdings1()


def fetch_and_save_mf_holdings1():
    print('fetching MF holdings')
    try:
        mf_holdings = kite.mf_holdings()
        logging.info(f"User MF holdings: {mf_holdings}")
    except Exception as e:
        logging.error(f"Error fetching MF holdings: {e}")
        return

    if not mf_holdings:
        logging.warning("No MF holdings found")
        return

    valid_mf_holdings = []
    try:
        for mf in mf_holdings:
            # Calculate values
            quantity = float(mf.get('quantity', 0))
            average_price = float(mf.get('average_price', 0))
            last_price = float(mf.get('last_price', 0))
            
            invested_amount = quantity * average_price
            current_value = quantity * last_price
            pnl = current_value - invested_amount
            
            # Calculate percentages
            net_change_percentage = 0.0
            if invested_amount > 0:
                net_change_percentage = ((current_value - invested_amount) / invested_amount) * 100
            
            # Day change percentage (if available in the API response)
            day_change_percentage = float(mf.get('day_change', 0)) if mf.get('day_change') else 0.0
            
            mf_holding = {
                'folio': mf.get('folio', ''),
                'fund': mf.get('fund', ''),
                'tradingsymbol': mf.get('tradingsymbol', ''),
                'isin': mf.get('isin', ''),
                'quantity': quantity,
                'average_price': average_price,
                'last_price': last_price,
                'invested_amount': invested_amount,
                'current_value': current_value,
                'pnl': pnl,
                'net_change_percentage': net_change_percentage,
                'day_change_percentage': day_change_percentage
            }
            valid_mf_holdings.append(mf_holding)
            logging.debug(f"Processed MF holding: {mf_holding}")
    except Exception as e:
        logging.error(f"Error in validating MF holdings: {e}")
        import traceback
        logging.error(traceback.format_exc())
       
    try:
        print('saving MF holdings')
        if valid_mf_holdings:
            cursor.executemany("""
                INSERT INTO my_schema.mf_holdings (
                    folio, fund, tradingsymbol, isin, quantity, average_price, last_price,
                    invested_amount, current_value, pnl, net_change_percentage, day_change_percentage
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (folio, tradingsymbol, run_date) DO UPDATE
                SET fund = EXCLUDED.fund,
                    isin = EXCLUDED.isin,
                    quantity = EXCLUDED.quantity,
                    average_price = EXCLUDED.average_price,
                    last_price = EXCLUDED.last_price,
                    invested_amount = EXCLUDED.invested_amount,
                    current_value = EXCLUDED.current_value,
                    pnl = EXCLUDED.pnl,
                    net_change_percentage = EXCLUDED.net_change_percentage,
                    day_change_percentage = EXCLUDED.day_change_percentage,
                    fetch_timestamp = CURRENT_TIMESTAMP
            """, [
                (
                    mf['folio'], mf['fund'], mf['tradingsymbol'], mf['isin'],
                    mf['quantity'], mf['average_price'], mf['last_price'],
                    mf['invested_amount'], mf['current_value'], mf['pnl'],
                    mf['net_change_percentage'], mf['day_change_percentage']
                ) for mf in valid_mf_holdings
            ])
            conn.commit()
            logging.info(f"Stored {len(valid_mf_holdings)} MF holdings")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error saving MF holdings: {e}")
        import traceback
        logging.error(traceback.format_exc())


def fetch_and_save_trades(kite: KiteConnect, cursor: psycopg2.extensions.cursor):
    kite = kite
    cursor = cursor
    fetch_and_save_trades1()

def fetch_and_save_trades1():
    print('fetching trades')
    trades = kite.trades()  # save into postgres table with current date
    logging.info(f"User trades: {trades}")

    if not trades:
        logging.warning("No trades found")
        #return []

    try:
        cursor.executemany("""
                    INSERT INTO my_schema.trades (
                        trade_id, order_id, exchange_order_id, exchange,
                        trading_symbol, instrument_token, transaction_type, quantity,
                        average_price, trade_timestamp, exchange_timestamp
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (trade_id) DO UPDATE
                    SET order_id = EXCLUDED.order_id,
                        exchange_order_id = EXCLUDED.exchange_order_id,
                        exchange = EXCLUDED.exchange,
                        trading_symbol = EXCLUDED.trading_symbol,
                        instrument_token = EXCLUDED.instrument_token,
                        transaction_type = EXCLUDED.transaction_type,
                        quantity = EXCLUDED.quantity,
                        average_price = EXCLUDED.average_price,
                        trade_timestamp = EXCLUDED.trade_timestamp,
                        exchange_timestamp = EXCLUDED.exchange_timestamp
                """, [
                    (
                        trade.get('trade_id'), trade.get('order_id'),
                        trade.get('exchange_order_id'), trade.get('exchange'),
                        trade.get('tradingsymbol'), trade.get('instrument_token'),
                        trade.get('transaction_type'), trade.get('quantity'),
                        trade.get('average_price'),
                        pd.to_datetime(trade.get('trade_timestamp')) if trade.get('trade_timestamp') else None,
                        pd.to_datetime(trade.get('exchange_timestamp')) if trade.get('exchange_timestamp') else None
                    ) for trade in trades
                ])
        conn.commit()
        logging.info(f"Inserted/Updated {cursor.rowcount} trades")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error saving trades: {e}")       


def fetch_and_save_margins(kite: KiteConnect, cursor: psycopg2.extensions.cursor):
    kite = kite
    cursor = cursor
    fetch_and_save_margins1()


def fetch_and_save_margins1():
    print('fetching margins')
    margins = kite.margins()    # save into postgres table with current date
    logging.info(f"User margins: {margins}")

    if not margins:
        logging.warning("No margins data found")
        #return []
    
    logging.info("Margin 1")
    try:
        valid_margins = []
        for margin_type in ['equity', 'commodity']:
            logging.info("Margin 2")
            margin = margins.get(margin_type, {})
            if not margin:
                logging.warning(f"No data for margin_type: {margin_type}")
                continue
            
            valid_margins.append({
                'margin_type': margin_type,
                'enabled': margin.get('enabled', False),
                'net': margin.get('net', 0.0),
                'available_adhoc_margin': margin.get('available', {}).get('adhoc_margin', 0.0),
                'available_cash': margin.get('available', {}).get('cash', 0.0),
                'available_opening_balance': margin.get('available', {}).get('opening_balance', 0.0),
                'available_live_balance': margin.get('available', {}).get('live_balance', 0.0),
                'available_collateral': margin.get('available', {}).get('collateral', 0.0),
                'available_intraday_payin': margin.get('available', {}).get('intraday_payin', 0.0),
                'utilised_debits': margin.get('utilised', {}).get('debits', 0.0),
                'utilised_exposure': margin.get('utilised', {}).get('exposure', 0.0),
                'utilised_m2m_realised': margin.get('utilised', {}).get('m2m_realised', 0.0),
                'utilised_m2m_unrealised': margin.get('utilised', {}).get('m2m_unrealised', 0.0),
                'utilised_option_premium': margin.get('utilised', {}).get('option_premium', 0.0),
                'utilised_payout': margin.get('utilised', {}).get('payout', 0.0),
                'utilised_span': margin.get('utilised', {}).get('span', 0.0),
                'utilised_holding_sales': margin.get('utilised', {}).get('holding_sales', 0.0),
                'utilised_turnover': margin.get('utilised', {}).get('turnover', 0.0),
                'utilised_liquid_collateral': margin.get('utilised', {}).get('liquid_collateral', 0.0),
                'utilised_stock_collateral': margin.get('utilised', {}).get('stock_collateral', 0.0),
                'utilised_equity': margin.get('utilised', {}).get('equity', 0.0),
                'utilised_delivery': margin.get('utilised', {}).get('delivery', 0.0)
            })
        logging.debug(f"Processed margin: {valid_margins[-1]}")

        logging.info("Margin 3")
  
        if valid_margins:
            logging.info("Margin 4")
            #print('saving margins')
            
            cursor.execute("SELECT NOW();")
            logging.info(cursor.fetchone())

            cursor.executemany("""
                INSERT INTO my_schema.margins (
                    margin_type, enabled, net,
                    available_adhoc_margin, available_cash, available_opening_balance,
                    available_live_balance, available_collateral, available_intraday_payin,
                    utilised_debits, utilised_exposure, utilised_m2m_realised,
                    utilised_m2m_unrealised, utilised_option_premium, utilised_payout,
                    utilised_span, utilised_holding_sales, utilised_turnover,
                    utilised_liquid_collateral, utilised_stock_collateral,
                    utilised_equity, utilised_delivery
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (margin_type) DO UPDATE
                SET enabled = EXCLUDED.enabled,
                    net = EXCLUDED.net,
                    available_adhoc_margin = EXCLUDED.available_adhoc_margin,
                    available_cash = EXCLUDED.available_cash,
                    available_opening_balance = EXCLUDED.available_opening_balance,
                    available_live_balance = EXCLUDED.available_live_balance,
                    available_collateral = EXCLUDED.available_collateral,
                    available_intraday_payin = EXCLUDED.available_intraday_payin,
                    utilised_debits = EXCLUDED.utilised_debits,
                    utilised_exposure = EXCLUDED.utilised_exposure,
                    utilised_m2m_realised = EXCLUDED.utilised_m2m_realised,
                    utilised_m2m_unrealised = EXCLUDED.utilised_m2m_unrealised,
                    utilised_option_premium = EXCLUDED.utilised_option_premium,
                    utilised_payout = EXCLUDED.utilised_payout,
                    utilised_span = EXCLUDED.utilised_span,
                    utilised_holding_sales = EXCLUDED.utilised_holding_sales,
                    utilised_turnover = EXCLUDED.utilised_turnover,
                    utilised_liquid_collateral = EXCLUDED.utilised_liquid_collateral,
                    utilised_stock_collateral = EXCLUDED.utilised_stock_collateral,
                    utilised_equity = EXCLUDED.utilised_equity,
                    utilised_delivery = EXCLUDED.utilised_delivery
            """, [
                (
                    m['margin_type'], m['enabled'], m['net'],
                    m['available_adhoc_margin'], m['available_cash'], m['available_opening_balance'],
                    m['available_live_balance'], m['available_collateral'], m['available_intraday_payin'],
                    m['utilised_debits'], m['utilised_exposure'], m['utilised_m2m_realised'],
                    m['utilised_m2m_unrealised'], m['utilised_option_premium'], m['utilised_payout'],
                    m['utilised_span'], m['utilised_holding_sales'], m['utilised_turnover'],
                    m['utilised_liquid_collateral'], m['utilised_stock_collateral'],
                    m['utilised_equity'], m['utilised_delivery']
                ) for m in valid_margins
            ])
            conn.commit()
            logging.info(f"Stored {len(valid_margins)} margins")
        #return valid_margins
    except Exception as e:
        conn.rollback()
        logging.error(f"Error fetching/storing margins: {e}")
        #return []


try:
    init_postgres_conn()
    print('1')
    fetch_and_save_holdings1()
    print('1a')
    fetch_and_save_mf_holdings1()
    print('2')

    fetch_and_save_orders1()
    print('3')
    fetch_and_save_positions1()
    print('4')
    fetch_and_save_trades1()
    print('5')
    fetch_and_save_margins1()
    print('6')
    fetch_and_save_profile()
except Exception as e:
    logging.error(f"Error saving data to database: {e}")
