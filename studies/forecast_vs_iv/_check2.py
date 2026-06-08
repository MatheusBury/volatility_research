"""Check filtered VRP records"""
import pandas as pd
import os

for f in ['PETR4_vrp_records.csv', 'VALE3_vrp_records.csv', 'ITUB4_vrp_records.csv']:
    path = os.path.join('studies/forecast_vs_iv/results', f)
    df = pd.read_csv(path)
    print(f'=== {f} ({len(df)} obs) ===')
    print(f'  Forecast RV: {df["forecast_rv"].mean()*100:.2f}%')
    print(f'  Implied IV:   {df["implied_iv"].mean()*100:.2f}%')
    print(f'  Future RV:    {df["future_rv"].mean()*100:.2f}%')
    print(f'  Spread:        {df["spread"].mean()*100:.2f}%')
    print(f'  DTE:           {df["dte"].mean():.0f}')
    print(f'  Moneyness:     {df["moneyness"].mean():.3f}')
    print(df[['forecast_rv','implied_iv','future_rv','spread','dte']].head(10).to_string())
    print()
