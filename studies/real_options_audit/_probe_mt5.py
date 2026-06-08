"""Probe MT5 data structure for B3 options - one-time investigation"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

mt5.initialize()

symbols = mt5.symbols_get()
petr_opts = [s for s in symbols if s.name.startswith('PETR') and s.option_mode > 0 and s.session_volume > 0]
petr_opts.sort(key=lambda s: s.session_volume, reverse=True)

print('=== D1 data structure for top PETR options ===')
for s in petr_opts[:5]:
    rates = mt5.copy_rates_from_pos(s.name, mt5.TIMEFRAME_D1, 0, 3)
    if rates is not None and len(rates) > 0:
        r = rates[-1]
        exp = pd.Timestamp(s.expiration_time, unit='s') if s.expiration_time else None
        dte = (exp - pd.Timestamp.now()).days if exp else None
        print()
        print(f'Symbol: {s.name}')
        print(f'  Strike: {s.option_strike}')
        print(f'  Expiration: {exp} (DTE: {dte})')
        print(f'  Session volume: {s.session_volume}')
        print(f'  Last: {s.last}, Bid: {s.bid}, Ask: {s.ask}')
        print(f'  Spread (pips): {s.spread}')
        print(f'  D1: time={pd.Timestamp(r[0], unit="s")}, O={r[1]:.2f}, H={r[2]:.2f}, L={r[3]:.2f}, C={r[4]:.2f}, tick_vol={r[5]}, spread={r[6]}, real_vol={r[7]}')

print('\n=== Point value (trade_contract_size) ===')
for s in petr_opts[:3]:
    info = mt5.symbol_info(s.name)
    print(f'{s.name}: contract_size={info.trade_contract_size}, tick_size={info.trade_tick_size}, tick_value={info.trade_tick_value}')

print('\n=== DTE distribution for top 50 PETR options ===')
dtes = []
for s in petr_opts[:50]:
    if s.expiration_time:
        exp = pd.Timestamp(s.expiration_time, unit='s')
        dte = (exp - pd.Timestamp.now()).days
        dtes.append(dte)
print(f'  Min DTE: {min(dtes)}, Max DTE: {max(dtes)}, Mean: {np.mean(dtes):.0f}')

print('\n=== All columns in D1 rates ===')
rates = mt5.copy_rates_from_pos(petr_opts[0].name, mt5.TIMEFRAME_D1, 0, 10)
print(f'  Columns: {rates.dtype.names}')
print(f'  Sample spread values: {[r[6] for r in rates]}')

# Also check VALE3 DTE
vale_opts = [s for s in symbols if s.name.startswith('VALE') and s.option_mode > 0]
vale_dtes = []
for s in vale_opts[:50]:
    if s.expiration_time:
        exp = pd.Timestamp(s.expiration_time, unit='s')
        dte = (exp - pd.Timestamp.now()).days
        vale_dtes.append(dte)
print(f'\n  VALE3 DTE - Min: {min(vale_dtes)}, Max: {max(vale_dtes)}, Mean: {np.mean(vale_dtes):.0f}')

itub_opts = [s for s in symbols if s.name.startswith('ITUB') and s.option_mode > 0]
itub_dtes = []
for s in itub_opts[:50]:
    if s.expiration_time:
        exp = pd.Timestamp(s.expiration_time, unit='s')
        dte = (exp - pd.Timestamp.now()).days
        itub_dtes.append(dte)
print(f'  ITUB4 DTE - Min: {min(itub_dtes)}, Max: {max(itub_dtes)}, Mean: {np.mean(itub_dtes):.0f}')

mt5.shutdown()
