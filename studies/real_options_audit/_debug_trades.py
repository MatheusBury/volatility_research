"""Debug zero trades"""
import MetaTrader5 as mt5
import pandas as pd

mt5.initialize()

s = mt5.symbol_info('PETRG419')
print(f'Option: {s.name}, strike={s.option_strike}')
exp = pd.Timestamp(s.expiration_time, unit='s')

rates = mt5.copy_rates_from_pos(s.name, mt5.TIMEFRAME_D1, 0, 500)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df['dte'] = (exp - df['time']).dt.days
in_range = df[(df['dte'] >= 15) & (df['dte'] <= 45)]
print(f'Bars in 15-45 DTE: {len(in_range)}')
for _, r in in_range.iterrows():
    print(f'  {r["time"].date()} close={r["close"]:.2f} dte={r["dte"]}')

if len(in_range) >= 2:
    entry_date = in_range.iloc[0]['time']
    exit_date = in_range.iloc[-1]['time']
    diff = (exit_date - entry_date).days
    print(f'\nEntry: {entry_date.date()}, Exit: {exit_date.date()}, diff={diff} days')
    
    m15 = pd.read_parquet(r'C:\Users\mathe\Documents\GitHub\mt5\dataset\export_mt5\intraday\avista\M15\PETR4.parquet').reset_index()
    m15.columns = [c.lower() for c in m15.columns]
    m15['time'] = pd.to_datetime(m15['time'])
    
    for d, label in [(entry_date, 'entry'), (exit_date, 'exit')]:
        mask = (m15['time'] >= d) & (m15['time'] < d + pd.Timedelta(days=1))
        sub = m15[mask]
        print(f'{label} date {d.date()}: {len(sub)} M15 bars, last close={sub.iloc[-1]["close"]:.2f}' if len(sub) > 0 else f'{label} date {d.date()}: NO M15 data')
    
    # Check if timezone might be the issue
    print(f'\nEntry date tz: {entry_date.tz}')
    print(f'M15 time tz: {m15.time.iloc[0].tz}')

mt5.shutdown()
