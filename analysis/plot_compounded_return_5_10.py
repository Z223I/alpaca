#!/usr/bin/env python3
"""
Plot compounded return for stocks in the $5-10 price range that were selected (predicted_outcome == 1)
from predictions_5pct.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Read the predictions CSV
csv_path = os.path.join(os.path.dirname(__file__), 'predictions_5pct.csv')
df = pd.read_csv(csv_path)

# Filter for $5-10 price range
df_filtered = df[(df['squeeze_entry_price'] >= 5) & (df['squeeze_entry_price'] <= 10)].copy()

# Filter for selected stocks (predicted_outcome == 1)
df_selected = df_filtered[df_filtered['predicted_outcome'] == 1].copy()

# Sort by timestamp
df_selected['timestamp'] = pd.to_datetime(df_selected['timestamp'])
df_selected = df_selected.sort_values('timestamp')

# Calculate compounded return
# Starting with $1.00, multiply by (1 + realistic_profit/100) for each trade
df_selected['compounded_value'] = 1.0
for i, row in enumerate(df_selected.itertuples()):
    if i == 0:
        df_selected.loc[row.Index, 'compounded_value'] = 1.0 * (1 + row.realistic_profit / 100)
    else:
        prev_value = df_selected.iloc[i-1]['compounded_value']
        df_selected.loc[row.Index, 'compounded_value'] = prev_value * (1 + row.realistic_profit / 100)

# Calculate cumulative profit percentage
df_selected['cumulative_profit_pct'] = (df_selected['compounded_value'] - 1.0) * 100

# Print summary statistics
print(f"\n{'='*60}")
print(f"Compounded Return Analysis: $5-10 Price Range (Selected)")
print(f"{'='*60}")
print(f"Total trades: {len(df_selected)}")
print(f"Starting value: $1.00")
print(f"Ending value: ${df_selected['compounded_value'].iloc[-1]:.4f}")
print(f"Total return: {df_selected['cumulative_profit_pct'].iloc[-1]:.2f}%")
print(f"Average profit per trade: {df_selected['realistic_profit'].mean():.2f}%")
print(f"Win rate: {(df_selected['actual_outcome'].sum() / len(df_selected) * 100):.2f}%")
print(f"{'='*60}\n")

# Create the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Compounded value over time
ax1.plot(range(1, len(df_selected) + 1), df_selected['compounded_value'].values,
         linewidth=2, color='#2E86AB', marker='o', markersize=4)
ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Starting Value')
ax1.set_xlabel('Trade Number', fontsize=12, fontweight='bold')
ax1.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
ax1.set_title('Compounded Portfolio Value - $5-10 Price Range (Selected Stocks)',
              fontsize=14, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Add final value annotation
final_value = df_selected['compounded_value'].iloc[-1]
ax1.annotate(f'Final: ${final_value:.4f}',
             xy=(len(df_selected), final_value),
             xytext=(10, 10), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
             fontsize=10, fontweight='bold')

# Plot 2: Cumulative profit percentage
colors = ['green' if val >= 0 else 'red' for val in df_selected['cumulative_profit_pct']]
ax2.plot(range(1, len(df_selected) + 1), df_selected['cumulative_profit_pct'].values,
         linewidth=2, color='#A23B72', marker='o', markersize=4)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
ax2.fill_between(range(1, len(df_selected) + 1),
                  df_selected['cumulative_profit_pct'].values, 0,
                  where=(df_selected['cumulative_profit_pct'].values >= 0),
                  alpha=0.3, color='green', label='Profit')
ax2.fill_between(range(1, len(df_selected) + 1),
                  df_selected['cumulative_profit_pct'].values, 0,
                  where=(df_selected['cumulative_profit_pct'].values < 0),
                  alpha=0.3, color='red', label='Loss')
ax2.set_xlabel('Trade Number', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
ax2.set_title('Cumulative Return Percentage - $5-10 Price Range (Selected Stocks)',
              fontsize=14, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Add final return annotation
final_return = df_selected['cumulative_profit_pct'].iloc[-1]
ax2.annotate(f'Final: {final_return:.2f}%',
             xy=(len(df_selected), final_return),
             xytext=(10, 10), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
             fontsize=10, fontweight='bold')

plt.tight_layout()

# Save the plot
output_path = os.path.join(os.path.dirname(__file__), 'plots', 'compounded_return_5_10_selected.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

plt.show()
