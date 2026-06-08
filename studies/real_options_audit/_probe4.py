"""Probe 4: Check CALL options existence and naming for ALL series"""
import MetaTrader5 as mt5
import pandas as pd

mt5.initialize()
symbols = mt5.symbols_get()

# Check CALL vs PUT by series letter
print('=== PETR4 CALL and PUT by series letter ===')
for sletter in 'ABCDEFGHIJ':
    calls = [s for s in symbols if s.name.startswith('PETR' + sletter) and s.option_mode > 0]
    # Determine if CALL or PUT based on letter
    is_call = sletter in 'ABCDE'
    typ = 'CALL' if is_call else 'PUT'
    if calls:
        exps = set()
        for s in calls:
            if s.expiration_time:
                exps.add(pd.Timestamp(s.expiration_time, unit='s').date())
        total_vol = sum(s.session_volume for s in calls)
        print(f'  Series {sletter} ({typ}): {len(calls)} symbols, vol={total_vol:10.0f}, exps={sorted(exps)}')
    else:
        print(f'  Series {sletter} ({typ}): 0 symbols')

# Now check: which expiration dates have BOTH CALL and PUT?
print('\n=== Expirations with BOTH CALL and PUT for PETR4 ===')
petr_opts = [s for s in symbols if s.name.startswith('PETR') and s.option_mode > 0]
from collections import defaultdict
by_expiry = defaultdict(lambda: {'calls': 0, 'puts': 0, 'call_vol': 0, 'put_vol': 0, 'call_names': [], 'put_names': []})
for s in petr_opts:
    if s.expiration_time:
        exp = pd.Timestamp(s.expiration_time, unit='s')
        is_call = s.name[4] in 'ABCDE'
        key = exp.date()
        if is_call:
            by_expiry[key]['calls'] += 1
            by_expiry[key]['call_vol'] += s.session_volume
            if s.session_volume > 0:
                by_expiry[key]['call_names'].append(s.name)
        else:
            by_expiry[key]['puts'] += 1
            by_expiry[key]['put_vol'] += s.session_volume
            if s.session_volume > 0:
                by_expiry[key]['put_names'].append(s.name)

now = pd.Timestamp.now()
for exp_date in sorted(by_expiry.keys()):
    dte = (pd.Timestamp(exp_date) - now).days
    info = by_expiry[exp_date]
    has_both = info['calls'] > 0 and info['puts'] > 0
    if has_both:
        call_active = [n for n in info['call_names']]
        put_active = [n for n in info['put_names']]
        print(f'  {exp_date} DTE={dte:3d}: {info["calls"]}C/{info["puts"]}P (vol {info["call_vol"]:.0f}C/{info["put_vol"]:.0f}P)')
        print(f'    CALLs: {call_active[:5]}')
        print(f'    PUTs: {put_active[:5]}')

# Same for VALE3
print('\n=== VALE3 CALL/PUT by series letter ===')
for sletter in 'ABCDEFGHIJ':
    opts = [s for s in symbols if s.name.startswith('VALE' + sletter) and s.option_mode > 0]
    is_call = sletter in 'ABCDE'
    typ = 'CALL' if is_call else 'PUT'
    if opts:
        total_vol = sum(s.session_volume for s in opts)
        exps = set()
        for s in opts:
            if s.expiration_time:
                exps.add(pd.Timestamp(s.expiration_time, unit='s').date())
        print(f'  Series {sletter} ({typ}): {len(opts)} symbols, vol={total_vol:10.0f}, exps={sorted(exps)[:3]}')
    else:
        print(f'  Series {sletter} ({typ}): 0 symbols')

# Check what CALL options exist for PETR4 that are not 0
print('\n=== PETR4 CALL options with vol > 0 ===')
petr_calls = [s for s in petr_opts if s.name[4] in 'ABCDE' and s.session_volume > 0]
for s in petr_calls:
    exp = pd.Timestamp(s.expiration_time, unit='s')
    dte = (exp - now).days
    print(f'  {s.name:15s} K={s.option_strike:7.2f} DTE={dte:3d} last={s.last:7.2f} vol={s.session_volume:10.0f}')

# Also check what CALL historical data looks like
print('\n=== PETRA370 historical D1 data check ===')
petra370 = [s for s in petr_opts if s.name == 'PETRA370']
if petra370:
    s = petra370[0]
    rates = mt5.copy_rates_from_pos(s.name, mt5.TIMEFRAME_D1, 0, 500)
    if rates is not None and len(rates) > 0:
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        print(f'  {s.name}: {len(rates)} bars')
        print(f'  Range: {df.time.iloc[0]} to {df.time.iloc[-1]}')
        print(f'  Close range: {df.close.min():.2f} - {df.close.max():.2f}')
        print(f'  Volume range: {df.tick_volume.min()} - {df.tick_volume.max()}')
        print(f'  First 5 rows:')
        print(df.head())

# Check if there is ANY CALL option for VALE and ITUB with volume
print('\n=== VALE3 CALL options with any vol ===')
vale_opts = [s for s in symbols if s.name.startswith('VALE') and s.option_mode > 0]
vale_calls = [s for s in vale_opts if s.name[4] in 'ABCDE' and s.session_volume > 0]
for s in vale_calls:
    exp = pd.Timestamp(s.expiration_time, unit='s')
    dte = (exp - now).days
    print(f'  {s.name:15s} K={s.option_strike:7.2f} DTE={dte:3d} last={s.last:7.2f} vol={s.session_volume:10.0f}')

print('\n=== ITUB4 CALL options with any vol ===')
itub_opts = [s for s in symbols if s.name.startswith('ITUB') and s.option_mode > 0]
itub_calls = [s for s in itub_opts if s.name[4] in 'ABCDE' and s.session_volume > 0]
for s in itub_calls:
    exp = pd.Timestamp(s.expiration_time, unit='s')
    dte = (exp - now).days
    print(f'  {s.name:15s} K={s.option_strike:7.2f} DTE={dte:3d} last={s.last:7.2f} vol={s.session_volume:10.0f}')

mt5.shutdown()
