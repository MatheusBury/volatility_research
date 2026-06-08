"""Second probe: spot prices, ATM options, DTE distribution details"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

mt5.initialize()

# 1. Spot prices
for sym_name in ['PETR4', 'VALE3', 'ITUB4']:
    sym = mt5.symbol_info(sym_name)
    if sym:
        print(f'{sym_name}: Bid={sym.bid:.2f}, Ask={sym.ask:.2f}, Last={sym.last:.2f}')

symbols = mt5.symbols_get()

# 2. Top PETR options FULL details
petr_opts = [s for s in symbols if s.name.startswith('PETR') and s.option_mode > 0 and s.session_volume > 0]
petr_opts.sort(key=lambda s: s.session_volume, reverse=True)
petr4 = mt5.symbol_info('PETR4')
spot = petr4.last if petr4 else 0

print('\n=== Top 20 PETR options with spreads ===')
for s in petr_opts[:20]:
    typ = 'CALL' if s.name[4] in 'ABCDE' else 'PUT'
    exp = pd.Timestamp(s.expiration_time, unit='s') if s.expiration_time else None
    dte = (exp - pd.Timestamp.now()).days if exp else 0
    spr = (s.ask - s.bid) if s.ask > 0 and s.bid > 0 else 0
    spr_pct = spr / s.last * 100 if s.last > 0 and spr > 0 else 0
    print(f'{s.name:12s} {typ:4s} K={s.option_strike:6.2f} last={s.last:8.2f} bid={s.bid:8.2f} ask={s.ask:8.2f} spread={spr:8.4f} ({spr_pct:6.1f}%) DTE={dte:3d} vol={s.session_volume:10.0f}')

# 3. All PETR expirations
print('\n=== PETR4 expirations ===')
expirations = {}
for s in petr_opts:
    if s.expiration_time:
        exp = pd.Timestamp(s.expiration_time, unit='s')
        ekey = exp.date()
        if ekey not in expirations:
            expirations[ekey] = {'count': 0, 'total_vol': 0}
        expirations[ekey]['count'] += 1
        expirations[ekey]['total_vol'] += s.session_volume

for e in sorted(expirations.keys()):
    dte = (pd.Timestamp(e) - pd.Timestamp.now()).days
    info = expirations[e]
    print(f'  {e} (DTE={dte:3d}): {info["count"]:4d} options, vol={info["total_vol"]:10.0f}')

# 4. Find ATM CALL + PUT pairs in 15-45 DTE
print(f'\n=== ATM straddle candidates for PETR4 (spot={spot:.2f}) ===')
for s in petr_opts:
    if s.expiration_time:
        exp = pd.Timestamp(s.expiration_time, unit='s')
        dte = (exp - pd.Timestamp.now()).days
        if 15 <= dte <= 45 and abs(s.option_strike - spot) < 1.0:
            typ = 'CALL' if s.name[4] in 'ABCDE' else 'PUT'
            print(f'  {s.name:12s} {typ:4s} K={s.option_strike:6.2f} DTE={dte:2d} last={s.last:6.2f} bid={s.bid:6.2f} ask={s.ask:6.2f} vol={s.session_volume:8.0f}')

# 5. Same for VALE3 and ITUB4
for prefix, cls_name in [('VALE', 'VALE3'), ('ITUB', 'ITUB4')]:
    opts = [s for s in symbols if s.name.startswith(prefix) and s.option_mode > 0]
    opts_with_vol = [s for s in opts if s.session_volume > 0]
    spot_sym = mt5.symbol_info(cls_name)
    sspot = spot_sym.last if spot_sym else 0
    print(f'\n=== {cls_name} ATM straddle candidates (spot={sspot:.2f}) ===')
    print(f'  Total options: {len(opts)}, with volume: {len(opts_with_vol)}')
    for s in opts:
        if s.expiration_time:
            exp = pd.Timestamp(s.expiration_time, unit='s')
            dte = (exp - pd.Timestamp.now()).days
            if 15 <= dte <= 45 and abs(s.option_strike - sspot) < 1.0:
                typ = 'CALL' if s.name[4] in 'ABCDE' else 'PUT'
                print(f'  {s.name:12s} {typ:4s} K={s.option_strike:6.2f} DTE={dte:2d} last={s.last:6.2f} bid={s.bid:6.2f} ask={s.ask:6.2f} spread={s.ask-s.bid:6.4f} vol={s.session_volume:8.0f}')

mt5.shutdown()
