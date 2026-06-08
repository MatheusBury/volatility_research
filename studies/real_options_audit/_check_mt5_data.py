"""Check MT5 underlying D1 data for 2026"""
import MetaTrader5 as mt5
import pandas as pd

mt5.initialize()

for sym in ['PETR4', 'VALE3', 'ITUB4']:
    rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_D1, 0, 2000)
    if rates is not None and len(rates) > 0:
        first = pd.Timestamp(rates[0][0], unit='s')
        last = pd.Timestamp(rates[-1][0], unit='s')
        print(f'{sym}: {len(rates)} D1 bars, {first.date()} to {last.date()}, last close={rates[-1][4]:.2f}')
        
        # Also check M15
        m15_rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 1000)
        if m15_rates is not None and len(m15_rates) > 0:
            m15_first = pd.Timestamp(m15_rates[0][0], unit='s')
            m15_last = pd.Timestamp(m15_rates[-1][0], unit='s')
            print(f'  M15: {len(m15_rates)} bars, {m15_first.date()} to {m15_last.date()}')

mt5.shutdown()
