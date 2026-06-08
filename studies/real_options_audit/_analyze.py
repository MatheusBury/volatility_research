"""After-audit analysis: breakdown VRP by moneyness bands"""
import pandas as pd
import numpy as np

vrp = pd.read_csv('studies/real_options_audit/results/vrp_data.csv')
vrp = vrp.dropna(subset=['iv', 'rv'])

print(f'Total observations: {len(vrp)}')

# Banded VRP by moneyness
bands = [(0, 0.95, 'OTM'), (0.95, 1.05, 'ATM'), (1.05, 1.5, 'ITM'), (1.5, 5.0, 'Deep ITM')]
for lo, hi, label in bands:
    sub = vrp[(vrp['moneyness'] >= lo) & (vrp['moneyness'] < hi)]
    if len(sub) > 0:
        print(f'\n{label} (moneyness {lo}-{hi}): {len(sub)} obs')
        print(f'  Mean IV: {sub["iv"].mean():.2%}')
        print(f'  Mean RV: {sub["rv"].mean():.2%}')
        print(f'  Mean VRP: {sub["vrp"].mean():.2%}')
        print(f'  VRP > 0: {sub["vrp"].gt(0).mean():.1%}')
    else:
        print(f'\n{label} (moneyness {lo}-{hi}): no data')

# For ATM only: breakdown by stock
print('\n\nATM (0.95-1.05) by stock:')
atm = vrp[(vrp['moneyness'] >= 0.95) & (vrp['moneyness'] < 1.05)]
for prefix, stock in [('PETR', 'PETR4'), ('VALE', 'VALE3'), ('ITUB', 'ITUB4')]:
    sub = atm[atm['option'].str.startswith(prefix)]
    if len(sub) > 0:
        print(f'\n{stock}: {len(sub)} ATM obs')
        print(f'  Mean IV: {sub["iv"].mean():.2%}')
        print(f'  Mean VRP: {sub["vrp"].mean():.2%}')
        print(f'  Sample options: {sub["option"].unique()[:5]}')

# Check trades
trades = pd.read_csv('studies/real_options_audit/results/trades.csv')
print(f'\n\nTotal trades: {len(trades)}')
if len(trades) > 0:
    print(f'Avg PnL: {trades["pnl"].mean():.4f}')
    print(f'Avg Cost: {trades["cost"].mean():.4f}')
    print(f'Avg Hold (days): {(pd.to_datetime(trades["exit"]) - pd.to_datetime(trades["entry"])).dt.days.mean():.1f}')
    print(f'Total PnL: {trades["pnl"].sum():.2f}')
