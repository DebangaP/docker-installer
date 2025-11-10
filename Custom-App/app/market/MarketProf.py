import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample OHLCV data (prices in dollars, volume in shares)
data = {
    'Timestamp': [
        '2025-10-02 09:30:00', '2025-10-02 10:00:00', '2025-10-02 10:30:00',
        '2025-10-02 11:00:00', '2025-10-02 11:30:00', '2025-10-02 12:00:00',
        '2025-10-02 12:30:00', '2025-10-02 13:00:00', '2025-10-02 13:30:00',
        '2025-10-02 14:00:00', '2025-10-02 14:30:00', '2025-10-02 15:00:00',
        '2025-10-02 15:30:00'
    ],
    'Open': [100.0, 100.5, 101.0, 100.8, 101.5, 102.0, 102.5, 103.0, 102.8, 102.0, 101.8, 102.2, 102.5],
    'High': [100.8, 101.2, 101.5, 101.3, 102.0, 102.8, 103.0, 103.5, 103.2, 102.5, 102.3, 102.8, 103.0],
    'Low': [99.8, 100.2, 100.5, 100.3, 101.0, 101.5, 102.0, 102.5, 102.3, 101.5, 101.3, 101.8, 102.0],
    'Close': [100.5, 101.0, 100.8, 101.5, 102.0, 102.5, 103.0, 102.8, 102.0, 101.8, 102.2, 102.5, 102.8],
    'Volume': [10000, 12000, 8000, 9000, 11000, 13000, 15000, 14000, 10000, 9500, 10500, 11500, 12000]
}

df = pd.DataFrame(data)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Parameters
tick_size = 0.1  # Price increment for TPO chart
periods = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']  # Labels for 13 periods

# Step 1: Discretize price levels
min_price = df['Low'].min()
max_price = df['High'].max()
price_levels = np.arange(np.floor(min_price / tick_size) * tick_size,
                         np.ceil(max_price / tick_size) * tick_size + tick_size,
                         tick_size)
price_levels = np.round(price_levels, 2)

# Step 2: Create TPO matrix
tpo_matrix = pd.DataFrame(index=price_levels, columns=periods)
tpo_matrix[:] = ''

# Step 3: Assign TPOs to price levels
for i, period in enumerate(periods):
    row = df.iloc[i]
    period_prices = np.arange(np.floor(row['Low'] / tick_size) * tick_size,
                              np.ceil(row['High'] / tick_size) * tick_size + tick_size,
                              tick_size)
    period_prices = np.round(period_prices, 2)
    for price in period_prices:
        if price in tpo_matrix.index:
            tpo_matrix.loc[price, period] = period

# Step 4: Calculate TPO counts
tpo_counts = tpo_matrix.apply(lambda x: sum(x != ''), axis=1)

# Step 5: Identify POC and Value Area
poc_price = tpo_counts.idxmax()
poc_count = tpo_counts.max()
total_tpos = tpo_counts.sum()
value_area_tpos = total_tpos * 0.7  # 70% of TPOs for Value Area

# Calculate Value Area (VAH, VAL)
current_tpos = tpo_counts[poc_price]
vah = poc_price
val = poc_price
current_range = [poc_price]

while current_tpos < value_area_tpos:
    # Find the next highest and lowest prices with TPOs
    above = tpo_counts[tpo_counts.index > vah].index
    below = tpo_counts[tpo_counts.index < val].index
    next_above = above.min() if len(above) > 0 else None
    next_below = below.max() if len(below) > 0 else None
    
    # Compare TPO counts above and below
    above_count = tpo_counts.get(next_above, 0) if next_above is not None else 0
    below_count = tpo_counts.get(next_below, 0) if next_below is not None else 0
    
    if above_count >= below_count and next_above is not None:
        vah = next_above
        current_tpos += above_count
        current_range.append(next_above)
    elif next_below is not None:
        val = next_below
        current_tpos += below_count
        current_range.append(next_below)
    else:
        break

# Step 6: Visualize TPO Chart
plt.figure(figsize=(12, 8))

# Plot TPOs
for price in tpo_matrix.index:
    tpos = ''.join(tpo_matrix.loc[price].values)
    plt.text(0, price, tpos, fontsize=10, verticalalignment='center', fontfamily='monospace')

# Highlight POC, VAH, VAL
plt.axhline(poc_price, color='r', linestyle='--', label=f'POC: {poc_price}')
plt.axhline(vah, color='g', linestyle='--', label=f'VAH: {vah}')
plt.axhline(val, color='b', linestyle='--', label=f'VAL: {val}')

# Plot TPO count histogram
ax2 = plt.gca().twiny()
ax2.barh(tpo_counts.index, tpo_counts.values, height=tick_size, alpha=0.3, color='gray')

# Customize plot
plt.xlabel('TPO Letters (Time Periods)')
plt.ylabel('Price ($)')
ax2.set_xlabel('TPO Count')
plt.title('TPO Chart for Sample Stock Data (2025-10-02)')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Print key metrics
print(f"Point of Control (POC): ${poc_price}")
print(f"Value Area High (VAH): ${vah}")
print(f"Value Area Low (VAL): ${val}")