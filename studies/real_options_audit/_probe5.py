"""Probe 5: Check ALL series letters K-Z and understand B3 naming convention"""
import MetaTrader5 as mt5
import pandas as pd

mt5.initialize()
symbols = mt5.symbols_get()

# Check ALL letters for PETR options
print('=== PETR4 ALL series letters ===')
for sletter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    opts = [s for s in symbols if s.name.startswith('PETR' + sletter) and s.option_mode > 0]
    if opts:
        # Determine type from series letter mapping
        # Standard B3: A-E=Jan-May CALL, F-J=Jun-Oct PUT, but what about K-Z?
        exps = sorted(set(pd.Timestamp(s.expiration_time, unit='s').date() for s in opts if s.expiration_time))
        total_vol = sum(s.session_volume for s in opts)
        types = set()
        for s in opts:
            typ = 'CALL' if s.name[4] in 'ABCDE' else 'PUT'
            types.add(typ)
        print(f'  {sletter}: {len(opts):3d} opts, {list(types)}, vol={total_vol:8.0f}, exps={exps[:3]}')

# Check if there are options where option_right != 0
opt_w_right = [s for s in symbols if s.option_mode > 0 and s.option_right != 0]
if opt_w_right:
    print(f'\nOptions with option_right != 0: {len(opt_w_right)}')
    for s in opt_w_right[:5]:
        print(f'  {s.name}: right={s.option_right}')
else:
    print('\nAll options have option_right = 0 (MT5 limitation)')

# What is the actual type field in MT5?
sample = [s for s in symbols if s.option_mode > 0][:1]
if sample:
    s = sample[0]
    print(f'\nMT5 option fields for {s.name}:')
    for attr in dir(s):
        if not attr.startswith('_') and 'option' in attr.lower() or 'type' in attr.lower():
            print(f'  {attr}: {getattr(s, attr)}')

# Check the actual PETR series that have both CALL/PUT volume
# Maybe the naming is: same letter for CALL and PUT, distinguished internally
print('\n=== PETR4: Check if ANY series has both CALL and PUT with volume ===')
all_petr = [s for s in symbols if s.name.startswith('PETR') and s.option_mode > 0]
from collections import defaultdict
by_series = defaultdict(lambda: {'symbols': [], 'vol': 0})
for s in all_petr:
    # Extract the series letter (first letter after PETR)
    # Names like PETRF419, PETRA370, PETRG414W1
    series_letter = s.name[4]
    by_series[series_letter]['symbols'].append(s.name)
    by_series[series_letter]['vol'] += s.session_volume

for sl in sorted(by_series.keys()):
    info = by_series[sl]
    print(f'  Series {sl}: {len(info["symbols"]):4d} symbols, vol={info["vol"]:10.0f}')

# Get the underlying symbols for these options
print('\n=== Underlying symbols (PETR4, VALE3, ITUB4) D1 data range ===')
for sym in ['PETR4', 'VALE3', 'ITUB4']:
    rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_D1, 0, 2000)
    if rates is not None and len(rates) > 0:
        first = pd.Timestamp(rates[0][0], unit='s')
        last = pd.Timestamp(rates[-1][0], unit='s')
        print(f'  {sym}: {len(rates)} bars from {first.date()} to {last.date()}, last close={rates[-1][4]:.2f}')

mt5.shutdown()
