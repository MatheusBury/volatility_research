from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from studies.market_regime_edge.strategies import BaseStrategy, TradeRecord

N_BARS_PER_YEAR = 252 * 26


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: np.ndarray | None = None
    timestamps: np.ndarray | None = None

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def total_return(self) -> float:
        if not self.trades:
            return 0.0
        return float(np.prod([1 + t.simple_return for t in self.trades]) - 1)

    @property
    def annualized_return(self) -> float:
        if not self.trades:
            return 0.0
        total = self.total_return
        days = (self.trades[-1].timestamp - self.trades[0].timestamp).days
        years = max(days / 365.25, 1 / 365.25)
        return (1 + total) ** (1 / years) - 1

    @property
    def annualized_vol(self) -> float:
        if len(self.trades) < 2:
            return 0.0
        rets = np.array([t.simple_return for t in self.trades])
        return float(np.std(rets, ddof=1) * np.sqrt(N_BARS_PER_YEAR))

    @property
    def sharpe_ratio(self) -> float:
        vol = self.annualized_vol
        if vol < 1e-10:
            return 0.0
        return self.annualized_return / vol

    @property
    def profit_factor(self) -> float:
        if not self.trades:
            return 0.0
        gains = sum(t.simple_return for t in self.trades if t.simple_return > 0)
        losses = abs(sum(t.simple_return for t in self.trades if t.simple_return < 0))
        if losses < 1e-10:
            return float("inf") if gains > 0 else 0.0
        return gains / losses

    @property
    def hit_ratio(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.simple_return > 0)
        return wins / len(self.trades)

    @property
    def max_drawdown(self) -> float:
        if self.equity_curve is None or len(self.equity_curve) < 2:
            return 0.0
        running_max = np.maximum.accumulate(self.equity_curve)
        dd = (self.equity_curve - running_max) / running_max
        return float(np.min(dd))

    @property
    def turnover(self) -> float:
        if not self.trades:
            return 0.0
        total_traded = sum(
            abs(t.simple_return) for t in self.trades
        )
        return total_traded / (len(self.trades) / N_BARS_PER_YEAR)


class BacktestEngine:
    def __init__(
        self,
        cost_per_trade: float = 0.0010,
        slippage: float = 0.0005,
    ):
        self.cost_per_trade = cost_per_trade
        self.slippage = slippage

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        context_df: pd.DataFrame | None = None,
    ) -> BacktestResult:
        signals = strategy.generate_signals(df)
        symbol = df["symbol"].iloc[0]
        trades: list[TradeRecord] = []
        position = 0.0
        equity = [1.0]
        entry_price = 0.0

        for i in range(1, len(df)):
            signal = signals[i]
            price = df["close_price"].iloc[i]
            timestamp = df["timestamp"].iloc[i]

            if position != 0 and (signal == 0 or signal != position):
                exit_price = price * (1 - self.slippage * position)
                ret = np.log(exit_price / entry_price) * position
                simple_ret = (exit_price / entry_price - 1) * position
                cost = self.cost_per_trade
                net_simple_ret = simple_ret - cost
                ctx = {}
                if context_df is not None:
                    row = context_df.iloc[i]
                    ctx = {
                        k: row[k]
                        for k in context_df.columns
                        if k in row.index and k not in ("timestamp", "symbol")
                    }
                trades.append(
                    TradeRecord(
                        timestamp=timestamp,
                        symbol=symbol,
                        strategy=strategy.name,
                        direction=position,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        log_return=ret - cost,
                        simple_return=net_simple_ret,
                        context=ctx,
                    )
                )
                equity.append(equity[-1] * (1 + net_simple_ret))
                position = 0.0

            if signal != 0 and position == 0:
                entry_price = price * (1 + self.slippage * signal)
                position = signal
            else:
                equity.append(equity[-1])

        result = BacktestResult(
            symbol=symbol,
            strategy=strategy.name,
            trades=trades,
            equity_curve=np.array(equity),
            timestamps=df["timestamp"].values[: len(equity)],
        )
        return result
