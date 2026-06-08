"""Probe 3: Check CALL options, weekly series, and historical data depth"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

mt5.initialize()
symbols = mt5.symbols_get()

petr4 = mt5.symbol_info('PETR4')
spot = petr4.last if petr4 else 0
print(f'PETR4 spot: {spot:.2f}')

# All PETR options in 15-45 DTE range
print('\n=== ALL PETR options DTE=15-45, sorted by DTE ===')
petr_opts = [s for s in symbols if s.name.startswith('PETR') and s.option_mode > 0]
petr_1545 = []
for s in petr_opts:
    if s.expiration_time:
        exp = pd.Timestamp(s.expiration_time, unit='s')
        dte = (exp - pd.Timestamp.now()).days
        if 15 <= dte <= 45:
            petr_1545.append(s)

petr_1545.sort(key=lambda s: (s.expiration_time, s.option_strike))

print(f'Total options: {len(petr_1545)}')
for s in petr_1545:
    typ = 'C' if s.name[4] in 'ABCDE' else 'P'
    exp = pd.Timestamp(s.expiration_time, unit='s')
    dte = (exp - pd.Timestamp.now()).days
    # Check D1 historical data
    rates = mt5.copy_rates_from_pos(s.name, mt5.TIMEFRAME_D1, 0, 100)
    hist_bars = len(rates) if rates is not None else 0
    print(f'{s.name:15s} {typ} K={s.option_strike:6.2f} DTE={dte:2d} last={s.last:7.2f} bid={s.bid:7.2f} ask={s.ask:7.2f} vol={s.session_volume:10.0f} D1bars={hist_bars:4d}')

# Check VALE3 ALL options DTE 15-45
vale3 = mt5.symbol_info('VALE3')
vspot = vale3.last if vale3 else 0
print(f'\n=== VALE3 DTE=15-45 (spot={vspot:.2f}) ===')
vale_opts = [s for s in symbols if s.name.startswith('VALE') and s.option_mode > 0]
vale_1545 = []
for s in vale_opts:
    if s.expiration_time:
        exp = pd.Timestamp(s.expiration_time, unit='s')
        dte = (exp - pd.Timestamp.now()).days
        if 15 <= dte <= 45:
            vale_1545.append(s)
vale_1545.sort(key=lambda s: (s.expiration_time, s.option_strike))
print(f'Total options: {len(vale_1545)}')
for s in vale_1545:
    typ = 'C' if s.name[4] in 'ABCDE' else 'P'
    exp = pd.Timestamp(s.expiration_time, unit='s')
    dte = (exp - pd.Timestamp.now()).days
    rates = mt5.copy_rates_from_pos(s.name, mt5.TIMEFRAME_D1, 0, 100)
    hist_bars = len(rates) if rates is not None else 0
    print(f'{s.name:15s} {typ} K={s.option_strike:7.2f} DTE={dte:2d} last={s.last:7.2f} bid={s.bid:7.2f} ask={s.ask:7.2f} vol={s.session_volume:10.0f} D1bars={hist_bars:4d}')

# Check ITUB4
itub4 = mt5.symbol_info('ITUB4')
ispot = itub4.last if itub4 else 0
print(f'\n=== ITUB4 DTE=15-45 (spot={ispot:.2f}) ===')
itub_opts = [s for s in symbols if s.name.startswith('ITUB') and s.option_mode > 0]
itub_1545 = []
for s in itub_opts:
    if s.expiration_time:
        exp = pd.Timestamp(s.expiration_time, unit='s')
        dte = (exp - pd.Timestamp.now()).days
        if 15 <= dte <= 45:
            itub_1545.append(s)
itub_1545.sort(key=lambda s: (s.expiration_time, s.option_strike))
print(f'Total options: {len(itub_1545)}')
for s in itub_1545:
    typ = 'C' if s.name[4] in 'ABCDE' else 'P'
    exp = pd.Timestamp(s.expiration_time, unit='s')
    dte = (exp - pd.Timestamp.now()).days
    rates = mt5.copy_rates_from_pos(s.name, mt5.TIMEFRAME_D1, 0, 100)
    hist_bars = len(rates) if rates is not None else 0
    print(f'{s.name:15s} {typ} K={s.option_strike:7.2f} DTE={dte:2d} last={s.last:7.2f} bid={s.bid:7.2f} ask={s.ask:7.2f} vol={s.session_volume:10.0f} D1bars={hist_bars:4d}')

# Check weekly vs monthly expiration: what expiry dates have CALL+PUT pairs
print('\n=== PETR4 CALL+PUT pairs by expiry (DTE 15-45) ===')
from collections import defaultdict
by_expiry = defaultdict(lambda: {'calls': [], 'puts': []})
for s in petr_1545:
    typ = 'calls' if s.name[4] in 'ABCDE' else 'puts'
    by_expiry[s.expiration_time][typ].append((s.option_strike, s.name, s.session_volume, s.last, s.bid, s.ask))

for exp_ts, opts in sorted(by_expiry.items()):
    exp = pd.Timestamp(exp_ts, unit='s')
    dte = (exp - pd.Timestamp.now()).days
    calls = len(opts['calls'])
    puts = len(opts['puts'])
    # Find nearest ATM for each
    c_atm = min(opts['calls'], key=lambda x: abs(x[0] - spot)) if calls > 0 else None
    p_atm = min(opts['puts'], key=lambda x: abs(x[0] - spot)) if puts > 0 else None
    print(f'  {exp.date()} DTE={dte:2d}: {calls}C/{puts}P', end='')
    if c_atm:
        print(f' call={c_atm[0]:.2f}({c_atm[1]}) last={c_atm[3]:.2f} vol={c_atm[2]:.0f}', end='')
    if p_atm:
        print(f' put={p_atm[0]:.2f}({p_atm[1]}) last={p_atm[3]:.2f} vol={p_atm[2]:.0f}', end='')
    print()

mt5.shutdown()
