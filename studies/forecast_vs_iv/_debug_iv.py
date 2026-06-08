"""Debug IV computation for a specific observation"""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from py_vollib.black_scholes.implied_volatility import implied_volatility as bs_iv

mt5.initialize()

# Check what options are available for PETR4
prefix = "PETR"
all_syms = mt5.symbols_get()
opts = [s for s in all_syms if s.name.startswith(prefix) and s.option_mode > 0]

# Find puts with DTE 15-365
now = datetime.now()
puts = []
for s in opts:
    sl = s.name[len(prefix)]
    if sl in 'FGHIJKLMNOPQRSTUVWXYZ' and s.expiration_time:
        exp_dt = datetime.fromtimestamp(s.expiration_time)
        dte = (exp_dt - now).days
        if 15 <= dte <= 365 and s.option_strike:
            spot_info = mt5.symbol_info("PETR4")
            spot = spot_info.bid if spot_info else 0
            moneyness = s.option_strike / spot if spot > 0 else 0
            puts.append({
                'name': s.name,
                'strike': s.option_strike,
                'dte': dte,
                'exp': exp_dt,
                'last': s.last,
                'vol': s.session_volume,
                'moneyness': moneyness,
            })

puts.sort(key=lambda x: abs(x['moneyness'] - 1))
print(f'Total puts in DTE range: {len(puts)}')
print('\nTop 10 ATM puts:')
for p in puts[:10]:
    print(f"  {p['name']:12s} K={p['strike']:6.2f} DTE={p['dte']:3d} last={p['last']:7.2f} vol={p['vol']:8.0f} moneyness={p['moneyness']:.3f}")

# Pick the most ATM put and check its D1 data
atm = puts[0]
print(f'\n=== Testing IV for {atm["name"]} ===')
rates = mt5.copy_rates_from_pos(atm['name'], mt5.TIMEFRAME_D1, 0, 5000)
if rates is not None:
    df = pd.DataFrame(rates)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    print(f'Total D1 bars: {len(df)}')
    print(f'Date range: {df.timestamp.min()} to {df.timestamp.max()}')
    
    # Get underlying data
    und_rates = mt5.copy_rates_from_pos('PETR4', mt5.TIMEFRAME_D1, 0, 5000)
    und_df = pd.DataFrame(und_rates)
    und_df['timestamp'] = pd.to_datetime(und_df['time'], unit='s')
    
    # Check first 5 IV computations
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        opt_price = float(row['close'])
        option_date = row['timestamp']
        
        und_row = und_df[und_df['timestamp'].dt.date == option_date.date()]
        if und_row.empty:
            continue
        und_price = float(und_row.iloc[0]['close'])
        
        tte_years = (atm['exp'] - option_date).days / 365.0
        
        if tte_years < 0.04:
            continue
            
        try:
            iv = bs_iv(opt_price, und_price, atm['strike'], tte_years, 0.1475, 'p')
            print(f"  {option_date.date()}: opt={opt_price:.2f} und={und_price:.2f} strike={atm['strike']:.2f} tte={tte_years:.2f} IV={iv:.2%} moneyness={atm['strike']/und_price:.3f}")
        except Exception as e:
            print(f"  {option_date.date()}: ERROR: {e}")

mt5.shutdown()
