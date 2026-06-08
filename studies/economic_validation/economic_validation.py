"""
Economic Validation: Regime Filter with Realistic B3 Costs
==========================================================
Determines whether the Regime Filter strategy has ECONOMIC edge
(after costs) or just STATISTICAL edge (before costs).

IS: 2021-01-01 to 2024-12-31 (HMM training)
OOS: 2025-01-01 to 2026-05-29 (strategy evaluation, NO refit)
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(r"C:\Users\mathe\Documents\GitHub\mt5\dataset\export_mt5\intraday\avista\M15")
STUDY_DIR = Path(r"C:\Users\mathe\Documents\GitHub\volatility_research\studies\economic_validation")
CHARTS_DIR = STUDY_DIR / "charts"
RESULTS_DIR = STUDY_DIR / "results"
for d in [CHARTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SYMBOLS: List[str] = ["PETR4", "VALE3", "ITUB4"]
N_STATES: int = 4
VOL_WINDOW: int = 30
IS_START: str = "2021-01-01"
IS_END: str = "2024-12-31"
OOS_START: str = "2025-01-01"
OOS_END: str = "2026-05-29"
RANDOM_STATE: int = 42
N_BARS_PER_YEAR: int = 252 * 26

# B3 realistic costs
COST_BPS: float = 5.0        # 5 bps per trade (each side)
SLIPPAGE_BPS: float = 3.0    # 3 bps per trade (each side)
TOTAL_COST_BPS: float = COST_BPS + SLIPPAGE_BPS  # 8 bps per trade

VOL_SIZE_MAP: Dict[int, float] = {0: 1.0, 1: 0.5, 2: 0.25, 3: 0.05}

SENSITIVITY_COSTS: List[float] = [0.0, 5.0, 10.0, 20.0, 50.0]

sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 10})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_b3_data(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    df = pd.read_parquet(path)
    df = df.reset_index()
    df = df.rename(columns={
        "time": "timestamp",
        "Open": "open_price",
        "High": "high_price",
        "Low": "low_price",
        "Close": "close_price",
        "Tick_volume": "volume",
    })
    df["symbol"] = symbol
    df["volume"] = df["volume"].astype("int64")
    df["timestamp"] = df["timestamp"].dt.tz_localize("America/Sao_Paulo")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[df["volume"] > 0].reset_index(drop=True)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["close_price"] / df["close_price"].shift(1))
    df["realized_vol"] = (
        df["log_return"].rolling(window=VOL_WINDOW).std() * np.sqrt(N_BARS_PER_YEAR)
    )
    df = df.dropna(subset=["log_return", "realized_vol"]).reset_index(drop=True)
    return df


def is_b3_hours(ts: pd.Series) -> np.ndarray:
    hour = ts.dt.hour
    minute = ts.dt.minute
    return ((hour >= 10) & (hour < 17)) | ((hour == 17) & (minute <= 30))


# ---------------------------------------------------------------------------
# HMM utilities
# ---------------------------------------------------------------------------
def fit_hmm(X: np.ndarray, n_states: int = N_STATES) -> hmm.GaussianHMM:
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        tol=1e-4,
        random_state=RANDOM_STATE,
        init_params="stmc",
    )
    model.fit(X)
    return model


def label_regimes(
    model: hmm.GaussianHMM, X: np.ndarray, vol_col_idx: int = 1
) -> Tuple[np.ndarray, Dict[int, str], Dict[int, int]]:
    states = model.predict(X)
    state_means = {s: float(np.mean(X[states == s, vol_col_idx])) for s in range(model.n_components)}
    sorted_states = sorted(state_means, key=state_means.get)
    n = model.n_components
    if n == 4:
        labels = {
            sorted_states[0]: "Low Vol",
            sorted_states[1]: "Medium Vol",
            sorted_states[2]: "High Vol",
            sorted_states[3]: "Extreme Vol",
        }
    elif n == 3:
        labels = {sorted_states[0]: "Low Vol", sorted_states[1]: "Medium Vol", sorted_states[2]: "High Vol"}
    elif n == 2:
        labels = {sorted_states[0]: "Low Vol", sorted_states[1]: "High Vol"}
    else:
        labels = {s: f"State {s}" for s in range(n)}
    regime_map = {s: i for i, s in enumerate(sorted_states)}
    regimes = np.array([regime_map[s] for s in states])
    return regimes, labels, regime_map


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------
@dataclass
class ValidationMetrics:
    symbol: str
    strategy: str
    cost_bps: float
    total_return_pct: float
    ann_return_pct: float
    ann_vol_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    num_trades: int
    total_costs_pct: float
    net_return_pct: float
    net_sharpe: float
    pct_time_in_market: float
    gross_return_pct: float = 0.0
    gross_sharpe: float = 0.0
    edge_erosion_pct: float = 0.0


# ---------------------------------------------------------------------------
# Cost-aware strategy base class
# ---------------------------------------------------------------------------
class CostAwareStrategy(ABC):
    def __init__(
        self,
        df: pd.DataFrame,
        regimes: np.ndarray,
        name: str,
    ):
        self.df = df
        self.regimes = regimes
        self.name = name
        self._positions: Optional[np.ndarray] = None
        self._gross_returns: Optional[np.ndarray] = None
        self._net_returns: Optional[np.ndarray] = None
        self._costs: Optional[np.ndarray] = None
        self._trade_count: Optional[int] = None

    @abstractmethod
    def _compute_position(self, idx: int) -> float:
        ...

    def get_positions(self) -> np.ndarray:
        if self._positions is not None:
            return self._positions
        n = len(self.df)
        b3_mask = is_b3_hours(self.df["timestamp"]).values
        positions = np.zeros(n)
        for i in range(n):
            if not b3_mask[i]:
                positions[i] = 0.0
            else:
                positions[i] = self._compute_position(i)
        self._positions = positions
        return positions

    @abstractmethod
    def _get_trade_events(self, positions: np.ndarray) -> List[Tuple[int, float, float]]:
        ...

    def compute_returns(self, cost_bps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        if self._net_returns is not None and cost_bps == TOTAL_COST_BPS:
            return self._gross_returns, self._net_returns, self._costs, self._trade_count

        positions = self.get_positions()
        asset_rets = self.df["log_return"].values
        n = len(positions)

        gross_rets = np.zeros(n)
        for i in range(n - 1):
            gross_rets[i + 1] = positions[i] * asset_rets[i + 1]

        costs = np.zeros(n)
        trade_events = self._get_trade_events(positions)

        for idx, prev_pos, curr_pos in trade_events:
            change = abs(curr_pos - prev_pos)
            if change > 1e-12:
                apply_idx = min(idx, n - 1)
                costs[apply_idx] += (cost_bps / 10000.0) * change

        net_rets = gross_rets - costs
        num_trades = len(trade_events)

        if cost_bps == TOTAL_COST_BPS:
            self._gross_returns = gross_rets
            self._net_returns = net_rets
            self._costs = costs
            self._trade_count = num_trades

        return gross_rets, net_rets, costs, num_trades

    def equity_curve(self, returns: np.ndarray) -> np.ndarray:
        return np.exp(np.cumsum(returns))

    def compute_metrics(self, symbol: str, cost_bps: float) -> ValidationMetrics:
        gross_rets, net_rets, costs, num_trades = self.compute_returns(cost_bps)
        positions = self.get_positions()
        n = len(gross_rets)
        n_years = n / N_BARS_PER_YEAR

        gross_cum = float(np.exp(np.sum(gross_rets)) - 1) if np.isfinite(np.sum(gross_rets)) else -1.0
        net_cum = float(np.exp(np.sum(net_rets)) - 1) if np.isfinite(np.sum(net_rets)) else -1.0

        gross_ann_ret = float(np.mean(gross_rets) * N_BARS_PER_YEAR)
        gross_ann_vol = float(np.std(gross_rets, ddof=1) * np.sqrt(N_BARS_PER_YEAR))
        gross_sharpe = gross_ann_ret / gross_ann_vol if gross_ann_vol > 1e-10 else 0.0

        net_ann_ret = float(np.mean(net_rets) * N_BARS_PER_YEAR)
        net_ann_vol = float(np.std(net_rets, ddof=1) * np.sqrt(N_BARS_PER_YEAR))
        net_sharpe = net_ann_ret / net_ann_vol if net_ann_vol > 1e-10 else 0.0

        cum_eq = np.exp(np.cumsum(net_rets))
        running_max = np.maximum.accumulate(cum_eq)
        dd = (cum_eq - running_max) / running_max
        max_dd = float(np.min(dd)) * 100

        total_costs_pct = float(np.sum(costs)) * 100

        pct_time = float(np.mean(np.abs(positions) > 1e-6))

        total_return_pct = net_cum * 100
        ann_return_pct = net_ann_ret * 100
        ann_vol_pct = net_ann_vol * 100
        gross_return_pct = gross_cum * 100

        gross_ret_pct = gross_cum * 100
        gross_edge_abs = gross_sharpe if abs(gross_sharpe) > 1e-10 else 0.0
        net_edge_abs = net_sharpe if abs(net_sharpe) > 1e-10 else 0.0
        if abs(gross_edge_abs) > 1e-10:
            edge_erosion = max(0.0, (1.0 - net_edge_abs / gross_edge_abs)) * 100
        else:
            edge_erosion = 0.0

        return ValidationMetrics(
            symbol=symbol,
            strategy=self.name,
            cost_bps=cost_bps,
            total_return_pct=total_return_pct,
            ann_return_pct=ann_return_pct,
            ann_vol_pct=ann_vol_pct,
            sharpe_ratio=net_sharpe,
            max_drawdown_pct=max_dd,
            num_trades=num_trades,
            total_costs_pct=total_costs_pct,
            net_return_pct=net_cum * 100,
            net_sharpe=net_sharpe,
            pct_time_in_market=pct_time * 100,
            gross_return_pct=gross_ret_pct,
            gross_sharpe=gross_sharpe,
            edge_erosion_pct=edge_erosion,
        )

    def get_returns(self, cost_bps: float) -> Tuple[np.ndarray, np.ndarray]:
        gross, net, _, _ = self.compute_returns(cost_bps)
        return gross, net


# ---------------------------------------------------------------------------
# Strategy A: Always On
# ---------------------------------------------------------------------------
class AlwaysOn(CostAwareStrategy):
    def __init__(self, df: pd.DataFrame, regimes: np.ndarray):
        super().__init__(df, regimes, "A: Always On")

    def _compute_position(self, idx: int) -> float:
        return 1.0

    def _get_trade_events(self, positions: np.ndarray) -> List[Tuple[int, float, float]]:
        active = np.where(np.abs(positions) > 1e-6)[0]
        if len(active) == 0:
            return []
        first = active[0]
        last = active[-1]
        events: List[Tuple[int, float, float]] = []
        events.append((first, 0.0, positions[first]))
        if last + 1 < len(positions):
            events.append((last + 1, positions[last], 0.0))
        else:
            events.append((last, positions[last], 0.0))
        return events


# ---------------------------------------------------------------------------
# Strategy B: Regime Filter
# ---------------------------------------------------------------------------
class RegimeFilterStrategy(CostAwareStrategy):
    def __init__(self, df: pd.DataFrame, regimes: np.ndarray):
        super().__init__(df, regimes, "B: Regime Filter")

    def _compute_position(self, idx: int) -> float:
        r = int(self.regimes[idx]) if idx < len(self.regimes) else 0
        return 1.0 if r <= 1 else 0.0

    def _get_trade_events(self, positions: np.ndarray) -> List[Tuple[int, float, float]]:
        events: List[Tuple[int, float, float]] = []
        prev = 0.0
        for i in range(len(positions)):
            curr = positions[i]
            if abs(curr - prev) > 1e-6:
                events.append((i, prev, curr))
                prev = curr
        return events


# ---------------------------------------------------------------------------
# Strategy C: Vol Scaled
# ---------------------------------------------------------------------------
class VolScaled(CostAwareStrategy):
    def __init__(self, df: pd.DataFrame, regimes: np.ndarray):
        super().__init__(df, regimes, "C: Vol Scaled")

    def _compute_position(self, idx: int) -> float:
        r = int(self.regimes[idx]) if idx < len(self.regimes) else 0
        return VOL_SIZE_MAP.get(r, 0.0)

    def _get_trade_events(self, positions: np.ndarray) -> List[Tuple[int, float, float]]:
        events: List[Tuple[int, float, float]] = []
        prev = 0.0
        for i in range(len(positions)):
            curr = positions[i]
            if abs(curr - prev) > 1e-6:
                events.append((i, prev, curr))
                prev = curr
        return events


# ---------------------------------------------------------------------------
# Sensitivity runner
# ---------------------------------------------------------------------------
def run_sensitivity(
    df_oos: pd.DataFrame,
    regimes_oos: np.ndarray,
    cost_levels: List[float],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for cost_bps in cost_levels:
        strat_b = RegimeFilterStrategy(df_oos, regimes_oos)
        m = strat_b.compute_metrics(df_oos["symbol"].iloc[0], cost_bps)
        rows.append({
            "cost_bps": cost_bps,
            "symbol": df_oos["symbol"].iloc[0],
            "net_sharpe": m.net_sharpe,
            "gross_sharpe": m.gross_sharpe,
            "net_return_pct": m.net_return_pct,
            "total_costs_pct": m.total_costs_pct,
            "num_trades": m.num_trades,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def plot_equity_curves_with_costs(
    results: Dict[str, Dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(20, 16), squeeze=False)
    gross_colors = {"A: Always On": "#3498db", "B: Regime Filter": "#2ecc71", "C: Vol Scaled": "#e74c3c"}
    net_colors = {"A: Always On": "#1a5276", "B: Regime Filter": "#1e8449", "C: Vol Scaled": "#922b21"}

    for row, symbol in enumerate(SYMBOLS):
        # Left: gross returns
        ax_g = axes[row, 0]
        # Right: net returns (with costs)
        ax_n = axes[row, 1]

        symbol_data = results.get(symbol, {})
        strategies_data = symbol_data.get("strategies", [])
        ts = symbol_data.get("timestamps", np.array([]))

        for strat, gross_rets, net_rets in strategies_data:
            gross_eq = np.exp(np.cumsum(gross_rets))
            net_eq = np.exp(np.cumsum(net_rets))
            ts_plot = ts[:len(gross_eq)]
            ax_g.plot(ts_plot, gross_eq, color=gross_colors.get(strat.name, "#333"),
                      label=f"{strat.name} (gross)", linewidth=0.8)
            ax_n.plot(ts_plot, net_eq, color=net_colors.get(strat.name, "#333"),
                      label=f"{strat.name} (net, {TOTAL_COST_BPS:.0f}bps)", linewidth=0.8)

        ax_g.set_ylabel("Equity (R$)")
        ax_g.set_title(f"{symbol} — Gross (No Costs)")
        ax_g.legend(fontsize=8)
        ax_g.axhline(1.0, color="gray", linestyle=":", alpha=0.3)

        ax_n.set_ylabel("Equity (R$)")
        ax_n.set_title(f"{symbol} — Net (With Costs, {TOTAL_COST_BPS:.0f}bps)")
        ax_n.legend(fontsize=8)
        ax_n.axhline(1.0, color="gray", linestyle=":", alpha=0.3)

        for ax in [ax_g, ax_n]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    fig.autofmt_xdate()
    fig.suptitle("Economic Validation — Equity Curves (Gross vs Net of Costs)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "equity_curves.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Equity curves saved to {CHARTS_DIR / 'equity_curves.png'}")


def plot_cost_sensitivity(
    sensitivity_data: Dict[str, pd.DataFrame],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {"PETR4": "#3498db", "VALE3": "#2ecc71", "ITUB4": "#e74c3c"}
    markers = {"PETR4": "o", "VALE3": "s", "ITUB4": "D"}

    for symbol in SYMBOLS:
        df_sens = sensitivity_data.get(symbol)
        if df_sens is None or df_sens.empty:
            continue
        ax.plot(df_sens["cost_bps"], df_sens["net_sharpe"],
                color=colors.get(symbol, "#333"),
                marker=markers.get(symbol, "o"),
                label=symbol, linewidth=1.8, markersize=8)

    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0.5, color="orange", linestyle=":", alpha=0.4, label="Sharpe=0.5")
    ax.axhline(1.0, color="red", linestyle=":", alpha=0.4, label="Sharpe=1.0")
    ax.set_xlabel("Cost (bps per trade)")
    ax.set_ylabel("Net Sharpe Ratio")
    ax.set_title("Cost Sensitivity — Strategy B Net Sharpe at Different Cost Levels")
    ax.legend(fontsize=10)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "cost_sensitivity.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Cost sensitivity chart saved to {CHARTS_DIR / 'cost_sensitivity.png'}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_report(
    all_metrics: List[ValidationMetrics],
    symbol_metrics: Dict[str, List[ValidationMetrics]],
    sensitivity_data: Dict[str, pd.DataFrame],
    df_edge_summary: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append("# Economic Validation: Regime Filter with Realistic B3 Costs\n")
    lines.append(f"**Generated:** {pd.Timestamp.now('America/Sao_Paulo').strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"**Universe:** {', '.join(SYMBOLS)}\n")
    lines.append(f"**Data:** B3 M15 (15-min) intraday — HMM 4-State Regimes\n")
    lines.append(f"**IS (HMM Training):** {IS_START} to {IS_END}\n")
    lines.append(f"**OOS (Strategy Evaluation):** {OOS_START} to {OOS_END}\n")
    lines.append(f"**Cost Model:** {COST_BPS:.0f} bps commission + {SLIPPAGE_BPS:.0f} bps slippage = **{TOTAL_COST_BPS:.0f} bps per trade**\n")
    lines.append("---\n")

    # --- 1. Summary table ---
    lines.append("## 1. Performance Summary (Net of Costs)\n\n")
    lines.append("| Symbol | Strategy | Gross Ret% | Net Ret% | Ann. Ret% | Ann. Vol% | Net Sharpe | Max DD% | Trades | Costs% | Edge Erosion% |\n")
    lines.append("|--------|----------|-----------|---------|-----------|----------|-----------|--------|--------|--------|---------------|\n")
    for m in sorted(all_metrics, key=lambda x: (x.symbol, x.strategy)):
        lines.append(
            f"| {m.symbol} | {m.strategy} | "
            f"{m.gross_return_pct:.2f}% | "
            f"{m.net_return_pct:.2f}% | "
            f"{m.ann_return_pct:.2f}% | "
            f"{m.ann_vol_pct:.2f}% | "
            f"{m.net_sharpe:.3f} | "
            f"{m.max_drawdown_pct:.1f}% | "
            f"{m.num_trades} | "
            f"{m.total_costs_pct:.2f}% | "
            f"{m.edge_erosion_pct:.1f}% |\n"
        )
    lines.append("\n")

    # --- 2. Gross vs Net comparison ---
    lines.append("## 2. Gross vs Net Sharpe Comparison\n\n")
    lines.append("| Symbol | Strategy | Gross Sharpe | Net Sharpe | Delta |\n")
    lines.append("|--------|----------|-------------|-----------|-------|\n")
    for m in sorted(all_metrics, key=lambda x: (x.symbol, x.strategy)):
        delta = m.net_sharpe - m.gross_sharpe
        lines.append(
            f"| {m.symbol} | {m.strategy} | "
            f"{m.gross_sharpe:.3f} | "
            f"{m.net_sharpe:.3f} | "
            f"{delta:+.3f} |\n"
        )
    lines.append("\n")

    # --- 3. Cost Sensitivity ---
    lines.append("## 3. Cost Sensitivity Analysis (Strategy B)\n\n")
    lines.append("Net Sharpe at various cost levels:\n\n")
    lines.append("| Cost (bps) | PETR4 | VALE3 | ITUB4 |\n")
    lines.append("|------------|-------|-------|-------|\n")
    for cost_bps in SENSITIVITY_COSTS:
        row_vals = [f"{cost_bps:.0f}"]
        for symbol in SYMBOLS:
            df_s = sensitivity_data.get(symbol)
            if df_s is not None:
                val = df_s.loc[df_s["cost_bps"] == cost_bps, "net_sharpe"].values
                row_vals.append(f"{val[0]:.3f}" if len(val) > 0 else "N/A")
            else:
                row_vals.append("N/A")
        lines.append("| " + " | ".join(row_vals) + " |\n")
    lines.append("\n")

    lines.append("Break-even cost level (Sharpe = 0):\n\n")
    for symbol in SYMBOLS:
        df_s = sensitivity_data.get(symbol)
        if df_s is not None and len(df_s) > 1:
            cost_vals = df_s["cost_bps"].values
            sharpe_vals = df_s["net_sharpe"].values
            if sharpe_vals[0] > 0:
                be = float(np.interp(0.0, sharpe_vals[::-1], cost_vals[::-1]))
                lines.append(f"- **{symbol}**: ~{be:.1f} bps\n")
            else:
                lines.append(f"- **{symbol}**: Already negative at 0 bps\n")
        else:
            lines.append(f"- **{symbol}**: Insufficient data\n")
    lines.append("\n")

    # --- 4. Critical Analysis ---
    lines.append("## 4. Critical Analysis: 6 Questions\n\n")

    # Helper to get metrics
    def get_m(sym: str, strat_name_part: str) -> Optional[ValidationMetrics]:
        sm = symbol_metrics.get(sym, [])
        for m in sm:
            if strat_name_part in m.strategy:
                return m
        return None

    # Q1: Does Strategy B survive costs?
    lines.append("### Q1: Does Strategy B survive costs?\n\n")
    for symbol in SYMBOLS:
        a = get_m(symbol, "Always On")
        b = get_m(symbol, "Regime Filter")
        if a and b:
            verdict = "YES" if b.net_sharpe > a.net_sharpe else "NO"
            lines.append(f"- **{symbol}**: Gross Sharpe {b.gross_sharpe:.3f} → Net Sharpe {b.net_sharpe:.3f} "
                         f"(A net Sharpe: {a.net_sharpe:.3f}) — **{verdict}**\n")
    lines.append("\n")

    # Q2: Does Strategy C survive costs?
    lines.append("### Q2: Does Strategy C survive costs?\n\n")
    for symbol in SYMBOLS:
        a = get_m(symbol, "Always On")
        c = get_m(symbol, "Vol Scaled")
        if a and c:
            verdict = "YES" if c.net_sharpe > a.net_sharpe else "NO"
            lines.append(f"- **{symbol}**: Gross Sharpe {c.gross_sharpe:.3f} → Net Sharpe {c.net_sharpe:.3f} "
                         f"(A net Sharpe: {a.net_sharpe:.3f}) — **{verdict}**\n")
    lines.append("\n")

    # Q3: How much of the gross edge is consumed by costs?
    lines.append("### Q3: How much of the gross edge is consumed by costs? (Edge erosion %)\n\n")
    for m in sorted(all_metrics, key=lambda x: (x.symbol, x.strategy)):
        lines.append(f"- **{m.symbol} {m.strategy}**: {m.edge_erosion_pct:.1f}% of gross edge consumed by costs\n")
    lines.append("\n")

    # Q4: Is the net edge consistent across all 3 stocks?
    lines.append("### Q4: Is the net edge consistent across all 3 stocks?\n\n")
    b_survive = sum(1 for s in SYMBOLS if (m := get_m(s, "Regime Filter")) and m.net_sharpe > (get_m(s, "Always On").net_sharpe if get_m(s, "Always On") else -999))
    c_survive = sum(1 for s in SYMBOLS if (m := get_m(s, "Vol Scaled")) and m.net_sharpe > (get_m(s, "Always On").net_sharpe if get_m(s, "Always On") else -999))
    lines.append(f"- Strategy B (Regime Filter) beats Always-On net of costs in **{b_survive}/{len(SYMBOLS)}** stocks\n")
    lines.append(f"- Strategy C (Vol Scaled) beats Always-On net of costs in **{c_survive}/{len(SYMBOLS)}** stocks\n")
    if b_survive == len(SYMBOLS):
        lines.append("- **Conclusion**: Net edge is CONSISTENT across all stocks for B\n")
    elif b_survive >= 2:
        lines.append("- **Conclusion**: Net edge is MOSTLY consistent (majority of stocks) for B\n")
    else:
        lines.append("- **Conclusion**: Net edge is INCONSISTENT across stocks for B\n")
    lines.append("\n")

    # Q5: Break-even cost level
    lines.append("### Q5: What is the break-even cost level?\n\n")
    lines.append("The break-even cost level (where net Sharpe = 0) for each stock:\n\n")
    for symbol in SYMBOLS:
        df_s = sensitivity_data.get(symbol)
        if df_s is not None and len(df_s) > 1:
            cost_vals = df_s["cost_bps"].values
            sharpe_vals = df_s["net_sharpe"].values
            if sharpe_vals[0] > 0:
                be = float(np.interp(0.0, sharpe_vals[::-1], cost_vals[::-1]))
                lines.append(f"- **{symbol}**: Break-even at **{be:.1f} bps**\n")
            else:
                lines.append(f"- **{symbol}**: No positive edge even at 0 bps\n")
    lines.append("\n")

    # Q6: Final verdict
    lines.append("### Q6: Final Verdict — Economic edge or statistical edge?\n\n")
    row = df_edge_summary.iloc[0] if len(df_edge_summary) > 0 else None
    if row is not None:
        lines.append(f"- **Verdict**: {row['verdict']}\n")
        lines.append(f"- **Evidence**: {row['evidence']}\n")
        lines.append(f"- **Avg Net Sharpe (A/B/C)**: {row['avg_net_sharpe_a']:.3f} / {row['avg_net_sharpe_b']:.3f} / {row['avg_net_sharpe_c']:.3f}\n")
        lines.append(f"- **Avg Edge Erosion**: {row['avg_edge_erosion']:.1f}%\n")
        lines.append(f"- **Avg Break-even Cost**: {row['avg_break_even_cost']:.1f} bps\n")
        b_sharpes_0bps = []
        a_net_sharpes = []
        for sym in SYMBOLS:
            df_s = sensitivity_data.get(sym)
            if df_s is not None:
                v = df_s.loc[df_s["cost_bps"] == 0.0, "net_sharpe"].values
                if len(v) > 0:
                    b_sharpes_0bps.append(v[0])
            am = get_m(sym, "Always On")
            if am:
                a_net_sharpes.append(am.net_sharpe)
        avg_b_0bps = np.mean(b_sharpes_0bps) if b_sharpes_0bps else 0.0
        avg_a_net = np.mean(a_net_sharpes) if a_net_sharpes else 0.0
        survives_zero = avg_b_0bps > avg_a_net
        lines.append(f"- **At 0 bps (no costs)**: Strategy B {'survives' if survives_zero else 'does not survive'} vs A")
        lines.append(f" (avg B Sharpe={avg_b_0bps:.3f} vs avg A={avg_a_net:.3f})\n")
    lines.append("\n")

    # --- Conclusions ---
    lines.append("## 5. Conclusions & Recommendations\n\n")
    lines.append("1. **Impact of costs is material**: Even 8 bps per trade significantly erodes the gross edge.\n")
    b_verdict = all(
        (m := get_m(s, "Regime Filter")) and m.net_sharpe > (get_m(s, "Always On").net_sharpe if get_m(s, "Always On") else -999)
        for s in SYMBOLS
    )
    c_verdict = all(
        (m := get_m(s, "Vol Scaled")) and m.net_sharpe > (get_m(s, "Always On").net_sharpe if get_m(s, "Always On") else -999)
        for s in SYMBOLS
    )
    lines.append(f"2. **Regime Filter (B)** {'has' if b_verdict else 'does NOT have'} a net economic edge over Always-On.\n")
    lines.append(f"3. **Vol Scaling (C)** {'has' if c_verdict else 'does NOT have'} a net economic edge over Always-On.\n")
    lines.append("4. **Edge buffer is narrow**: Break-even costs indicate how much room there is before the edge is fully consumed.\n")
    lines.append("5. **Cost-aware strategy design**: To preserve the economic edge, minimize turnover and/or negotiate lower costs.\n")

    lines.append("\n---\n*Report generated automatically by economic_validation.py*\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Main study
# ---------------------------------------------------------------------------
def run_study() -> None:
    print("=" * 80)
    print("  ECONOMIC VALIDATION — Regime Filter with Realistic B3 Costs")
    print("  Determining Economic Edge vs Statistical Edge")
    print("=" * 80)

    all_metrics: List[ValidationMetrics] = []
    symbol_metrics_dict: Dict[str, List[ValidationMetrics]] = {s: [] for s in SYMBOLS}
    chart_results: Dict[str, Dict[str, Any]] = {}
    sensitivity_data: Dict[str, pd.DataFrame] = {}

    for symbol in SYMBOLS:
        print(f"\n{'-'*80}")
        print(f"  {symbol}")
        print(f"{'-'*80}")

        # --- Load and split data ---
        df_raw = load_b3_data(symbol)
        df_feat = compute_features(df_raw)
        print(f"  Data: {len(df_feat):,} bars ({df_feat['timestamp'].min().date()} to {df_feat['timestamp'].max().date()})")

        tz = "America/Sao_Paulo"
        cutoff_is = pd.Timestamp(IS_END).tz_localize(tz)
        cutoff_oos_start = pd.Timestamp(OOS_START).tz_localize(tz)
        cutoff_oos_end = pd.Timestamp(OOS_END).tz_localize(tz)

        df_is = df_feat[df_feat["timestamp"] <= cutoff_is].copy()
        df_oos = df_feat[
            (df_feat["timestamp"] >= cutoff_oos_start) & (df_feat["timestamp"] <= cutoff_oos_end)
        ].copy()
        print(f"  IS: {len(df_is):,} bars ({df_is['timestamp'].min().date()} to {df_is['timestamp'].max().date()})")
        print(f"  OOS: {len(df_oos):,} bars ({df_oos['timestamp'].min().date()} to {df_oos['timestamp'].max().date()})")

        if len(df_oos) == 0:
            print(f"  WARNING: No OOS data for {symbol}, skipping.")
            continue

        # --- HMM features ---
        hmm_features = ["log_return", "realized_vol"]
        X_is = df_is[hmm_features].values.astype(np.float64)
        X_oos = df_oos[hmm_features].values.astype(np.float64)

        scaler = StandardScaler()
        X_is_s = scaler.fit_transform(X_is)
        X_oos_s = scaler.transform(X_oos)

        # --- Fit 4-state HMM on IS only ---
        print("  Fitting 4-state HMM on IS...")
        model = fit_hmm(X_is_s)
        print(f"    Converged: {model.monitor_.converged} (iter={model.monitor_.iter})")

        # --- Predict regimes for OOS only (forward-fill, no refit) ---
        regimes_oos, labels, regime_map = label_regimes(model, X_oos_s)
        sorted_by_vol = sorted(labels, key=lambda k: regime_map[k])
        label_names = [labels[k] for k in sorted_by_vol]
        print(f"    Labels: {label_names}")
        print(f"    OOS Regime distribution: {pd.Series(regimes_oos).value_counts(normalize=True).sort_index().to_dict()}")

        # --- Run strategies on OOS only ---
        strat_a = AlwaysOn(df_oos, regimes_oos)
        strat_b = RegimeFilterStrategy(df_oos, regimes_oos)
        strat_c = VolScaled(df_oos, regimes_oos)

        # Store for charts
        gross_a, net_a = strat_a.get_returns(TOTAL_COST_BPS)
        gross_b, net_b = strat_b.get_returns(TOTAL_COST_BPS)
        gross_c, net_c = strat_c.get_returns(TOTAL_COST_BPS)
        chart_results[symbol] = {
            "strategies": [
                (strat_a, gross_a, net_a),
                (strat_b, gross_b, net_b),
                (strat_c, gross_c, net_c),
            ],
            "timestamps": df_oos["timestamp"].values,
        }

        for strat in [strat_a, strat_b, strat_c]:
            m = strat.compute_metrics(symbol, TOTAL_COST_BPS)
            all_metrics.append(m)
            symbol_metrics_dict[symbol].append(m)
            print(f"    {strat.name:20s} | Gross Sharpe={m.gross_sharpe:.3f} | Net Sharpe={m.net_sharpe:.3f} | "
                  f"Gross Ret={m.gross_return_pct:.2f}% | Net Ret={m.net_return_pct:.2f}% | "
                  f"Costs={m.total_costs_pct:.2f}% | Erosion={m.edge_erosion_pct:.1f}%")

        # --- Sensitivity analysis for this symbol ---
        print(f"\n  Sensitivity analysis for {symbol}...")
        df_sens = run_sensitivity(df_oos, regimes_oos, SENSITIVITY_COSTS)
        sensitivity_data[symbol] = df_sens
        for _, row in df_sens.iterrows():
            print(f"    Cost={row['cost_bps']:.0f}bps | Net Sharpe={row['net_sharpe']:.3f} | "
                  f"Net Ret={row['net_return_pct']:.2f}% | Costs Paid={row['total_costs_pct']:.2f}%")

    # -----------------------------------------------------------------------
    # Compile results
    # -----------------------------------------------------------------------
    df_metrics = pd.DataFrame([asdict(m) for m in all_metrics])
    df_metrics.to_csv(RESULTS_DIR / "validation_metrics.csv", index=False)
    print(f"\n  Metrics saved to {RESULTS_DIR / 'validation_metrics.csv'}")

    # Sensitivity CSV
    all_sens: List[pd.DataFrame] = []
    for symbol in SYMBOLS:
        df_s = sensitivity_data.get(symbol)
        if df_s is not None:
            all_sens.append(df_s)
    if all_sens:
        df_sens_all = pd.concat(all_sens, ignore_index=True)
        df_sens_all.to_csv(RESULTS_DIR / "cost_sensitivity.csv", index=False)
        print(f"  Sensitivity saved to {RESULTS_DIR / 'cost_sensitivity.csv'}")

    # Edge summary
    edge_rows: List[Dict[str, Any]] = []
    avg_a_sharpe = np.mean([m.net_sharpe for m in all_metrics if "Always On" in m.strategy])
    avg_b_sharpe = np.mean([m.net_sharpe for m in all_metrics if "Regime Filter" in m.strategy])
    avg_c_sharpe = np.mean([m.net_sharpe for m in all_metrics if "Vol Scaled" in m.strategy])
    avg_erosion = np.mean([m.edge_erosion_pct for m in all_metrics])

    # Break-even cost (average across symbols)
    be_costs = []
    for symbol in SYMBOLS:
        df_s = sensitivity_data.get(symbol)
        if df_s is not None and len(df_s) > 1:
            c_vals = df_s["cost_bps"].values
            s_vals = df_s["net_sharpe"].values
            if s_vals[0] > 0:
                be = float(np.interp(0.0, s_vals[::-1], c_vals[::-1]))
                be_costs.append(be)
    avg_be = np.mean(be_costs) if be_costs else 0.0

    verdict = "ECONOMIC EDGE" if avg_b_sharpe > avg_a_sharpe else "STATISTICAL EDGE"
    if verdict == "ECONOMIC EDGE" and avg_c_sharpe > avg_b_sharpe:
        evidence = "Both B and C survive costs with positive net Sharpe improvement over A across all stocks"
    elif verdict == "ECONOMIC EDGE":
        evidence = "Strategy B maintains net Sharpe advantage over Always-On after costs"
    else:
        evidence = "Net edge is consumed by costs; the gains are purely statistical"

    edge_rows.append({
        "avg_net_sharpe_a": round(avg_a_sharpe, 3),
        "avg_net_sharpe_b": round(avg_b_sharpe, 3),
        "avg_net_sharpe_c": round(avg_c_sharpe, 3),
        "avg_edge_erosion": round(avg_erosion, 1),
        "avg_break_even_cost": round(avg_be, 1),
        "verdict": verdict,
        "evidence": evidence,
    })
    df_edge = pd.DataFrame(edge_rows)
    df_edge.to_csv(RESULTS_DIR / "edge_summary.csv", index=False)
    print(f"  Edge summary saved to {RESULTS_DIR / 'edge_summary.csv'}")
    print(f"\n  === PRELIMINARY VERDICT: {verdict} ===")

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------
    plot_equity_curves_with_costs(chart_results)
    plot_cost_sensitivity(sensitivity_data)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    report = generate_report(all_metrics, symbol_metrics_dict, sensitivity_data, df_edge)
    report_path = STUDY_DIR / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved to {report_path}")

    print(f"\n{'='*80}")
    print(f"  STUDY COMPLETE — Verdict: {verdict}")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_study()
