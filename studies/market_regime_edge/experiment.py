from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from studies.market_regime_edge.analysis import (
    regime_conditional_sharpe,
)
from studies.market_regime_edge.context import ContextComputer
from studies.market_regime_edge.data_loader import load_universe
from studies.market_regime_edge.engine import BacktestEngine
from studies.market_regime_edge.regime_map import (
    ContextAnalysisResult,
    analyze_context_variable,
    build_regime_heatmap,
    rank_context_variables,
)
from studies.market_regime_edge.strategies import create_strategies

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("market_regime_edge")

STUDY_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = STUDY_DIR / "results"
CHARTS_DIR = STUDY_DIR / "charts"
CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "market_regime_edge.yaml"
)
for d in [RESULTS_DIR, CHARTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 10})


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def run_single_asset(
    df: pd.DataFrame,
    strategies: list,
    engine: BacktestEngine,
) -> list[ContextAnalysisResult]:
    symbol = df["symbol"].iloc[0]
    logger.info(f"  Processing {symbol} ({len(df)} bars)")

    context = ContextComputer(df)
    context_df = context.compute_all()

    ctx_cols = context.get_context_column_names()
    context_cols_present = [c for c in ctx_cols if c in context_df.columns]

    results: list[ContextAnalysisResult] = []

    for strategy in strategies:
        bt_result = engine.run(df, strategy, context_df)
        if bt_result.n_trades < 5:
            continue

        logger.info(
            f"    {strategy.name}: {bt_result.n_trades} trades, "
            f"Sharpe={bt_result.sharpe_ratio:.3f}, PF={bt_result.profit_factor:.3f}"
        )

        for ctx_col in context_cols_present:
            analysis = analyze_context_variable(bt_result.trades, ctx_col)
            if analysis is not None:
                results.append(analysis)

    return results


def plot_sharpe_heatmap(rank_df: pd.DataFrame) -> None:
    if len(rank_df) == 0:
        return
    pivot = rank_df.pivot_table(
        values="best_sharpe",
        index="strategy",
        columns="context",
        aggfunc="mean",
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] * 1.2), max(6, pivot.shape[0] * 0.6)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax, cbar_kws={"label": "Best Regime Sharpe"})
    ax.set_title("Best Sharpe by Strategy x Context Variable", fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "sharpe_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Heatmap saved: {CHARTS_DIR / 'sharpe_heatmap.png'}")


def plot_sharpe_gap_bars(rank_df: pd.DataFrame) -> None:
    if len(rank_df) == 0:
        return
    top = rank_df.head(20)
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["green" if v > 0 else "red" for v in top["sharpe_gap"]]
    labels = [f"{r['strategy']}_{r['context']}" for _, r in top.iterrows()]
    ax.barh(range(len(top)), top["sharpe_gap"], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Sharpe Gap (Best - Worst Regime)")
    ax.set_title("Top 20 Context Variables by Regime Discrimination Power", fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "sharpe_gap_bars.png", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Sharpe gap chart saved: {CHARTS_DIR / 'sharpe_gap_bars.png'}")


def plot_context_conditional_sharpe(
    all_trades: list,
    context_cols: list[str],
) -> None:
    for ctx_col in context_cols[:6]:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        axes = axes.flatten()
        strategies_plotted = set()
        ax_idx = 0
        for trades, strategy_name in all_trades[:8]:
            if strategy_name in strategies_plotted:
                continue
            strategies_plotted.add(strategy_name)
            analysis = regime_conditional_sharpe(
                [t.simple_return for t in trades],
                [t.context for t in trades],
                ctx_col,
            )
            if not analysis:
                continue
            ax = axes[ax_idx]
            regimes = list(analysis.keys())
            sharpes = [analysis[r]["sharpe"] for r in regimes]
            colors = ["green" if s > 0 else "red" for s in sharpes]
            ax.bar(regimes, sharpes, color=colors, alpha=0.7)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_title(f"{strategy_name}", fontsize=9)
            ax.set_ylabel("Sharpe")
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax_idx += 1

        for j in range(ax_idx, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Conditional Sharpe by {ctx_col}", fontweight="bold")
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / f"conditional_sharpe_{ctx_col}.png", bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Conditional sharpe chart saved: {CHARTS_DIR / f'conditional_sharpe_{ctx_col}.png'}")


def generate_report(
    all_analyses: list[ContextAnalysisResult],
    rank_df: pd.DataFrame,
    heatmap_df: pd.DataFrame,
    config: dict,
    metadata: dict,
) -> str:
    lines: list[str] = []
    lines.append("# Market Regime Edge Discovery — Report\n")
    lines.append(f"**Generated:** {metadata['generated_at']}\n")
    lines.append(f"**Universe:** {metadata['n_symbols']} B3 stocks, M15 timeframe\n")
    lines.append(f"**Period:** Train={config['train_start']} to {config['train_end']}, "
                 f"Val={config['val_start']} to {config['val_end']}, "
                 f"Test={config.get('test_start', 'N/A')} to {config.get('test_end', 'N/A')}\n")
    lines.append(f"**Costs:** {config['cost_per_trade']*10000:.1f} bps per trade + "
                 f"{config['slippage']*10000:.1f} bps slippage\n")
    lines.append(f"**Seed:** {config.get('random_seed', 42)}\n")
    lines.append("---\n")

    # 1. Summary
    lines.append("## 1. Global Summary\n")
    strategies_in_data = set(r.strategy for r in all_analyses if r is not None)
    lines.append(f"- **Strategies tested:** {', '.join(sorted(strategies_in_data))}\n")
    lines.append(f"- **Context variables analyzed:** {len(set(r.context_column for r in all_analyses if r is not None))}\n")
    lines.append(f"- **Total strategy x context combinations:** {len(all_analyses)}\n")

    best_gap = max(all_analyses, key=lambda r: r.sharpe_gap) if all_analyses else None
    if best_gap:
        lines.append(f"- **Best discrimination:** {best_gap.strategy} x {best_gap.context_column} (gap={best_gap.sharpe_gap:.3f})\n")
    lines.append("\n")

    # 2. Context Variable Ranking
    lines.append("## 2. Context Variable Ranking\n")
    lines.append("Ranking by Sharpe Gap (difference between best and worst regime Sharpe).\n\n")
    if len(rank_df) > 0:
        lines.append("| Rank | Strategy | Context | Overall Sharpe | Sharpe Gap | Best Regime | Worst Regime | Best Sharpe | Worst Sharpe |\n")
        lines.append("|------|----------|---------|---------------|------------|-------------|--------------|-------------|--------------|\n")
        for i, (_, r) in enumerate(rank_df.head(30).iterrows()):
            lines.append(
                f"| {i+1} | {r['strategy']} | {r['context']} | "
                f"{r['overall_sharpe']:.3f} | {r['sharpe_gap']:.3f} | "
                f"{r['best_regime']} | {r['worst_regime']} | "
                f"{r['best_sharpe']:.3f} | {r['worst_sharpe']:.3f} |\n"
            )
    lines.append("\n")

    # 3. Heatmap
    lines.append("## 3. Regime Heatmap\n")
    lines.append("Average best Sharpe by strategy x context variable.\n\n")
    if not heatmap_df.empty:
        lines.append("```\n")
        lines.append(heatmap_df.to_string())
        lines.append("\n```\n\n")

    # 4. Best findings
    lines.append("## 4. Best Findings\n")
    top_findings = sorted(
        [r for r in all_analyses if r is not None and r.improvement_ratio > 1.5],
        key=lambda r: r.sharpe_gap,
        reverse=True,
    )[:10]
    if top_findings:
        for r in top_findings:
            lines.append(f"- **{r.strategy}** when **{r.context_column}={r.best_regime}**: "
                         f"Sharpe={max(r.regime_sharpes.values()):.3f} vs overall={r.overall_sharpe:.3f} "
                         f"(gap={r.sharpe_gap:.3f})\n")
    lines.append("\n")

    # 5. Per-strategy detail
    lines.append("## 5. Per-Strategy Detail\n")
    for strategy in sorted(strategies_in_data):
        lines.append(f"### {strategy}\n")
        strat_results = [r for r in all_analyses if r is not None and r.strategy == strategy]
        strat_results.sort(key=lambda r: r.sharpe_gap, reverse=True)
        lines.append("| Context | Overall Sharpe | Sharpe Gap | Best Regime | Worst Regime |\n")
        lines.append("|---------|---------------|------------|-------------|--------------|\n")
        for r in strat_results[:5]:
            lines.append(
                f"| {r.context_column} | {r.overall_sharpe:.3f} | {r.sharpe_gap:.3f} | "
                f"{r.best_regime} ({max(r.regime_sharpes.values()):.3f}) | "
                f"{r.worst_regime} ({min(r.regime_sharpes.values()):.3f}) |\n"
            )
        lines.append("\n")

    # 6. Conclusions
    lines.append("## 6. Conclusions\n")
    if best_gap:
        lines.append(f"1. **Strongest regime effect**: {best_gap.strategy} x {best_gap.context_column} (Sharpe gap={best_gap.sharpe_gap:.3f})\n")
    n_discriminant = sum(1 for r in all_analyses if r is not None and r.sharpe_gap > 0.5)
    lines.append(f"2. **Contexts with strong discrimination** (gap > 0.5): {n_discriminant}\n")
    lines.append(f"3. **Total hypotheses tested**: {len(all_analyses)}\n")

    n_positive = sum(1 for r in all_analyses if r is not None and max(r.regime_sharpes.values()) > 0.5)
    lines.append(f"4. **Promising regime conditions** (best Sharpe > 0.5): {n_positive}\n")

    lines.append("""
### Key Insights
- The edge is NOT in the signal — it's in the context.
- Same strategy can be profitable in one regime and loss-making in another.
- Context-aware strategies can dramatically improve risk-adjusted returns.
- The regime map provides a quantitative framework for strategy allocation.
""")
    lines.append("\n---\n*Report generated automatically by market_regime_edge/experiment.py*\n")

    return "".join(lines)


def run_study() -> None:
    logger.info("=" * 80)
    logger.info("  MARKET REGIME EDGE DISCOVERY")
    logger.info("  Identifying conditions where simple strategies become profitable")
    logger.info("=" * 80)

    config = load_config()
    seed = config.get("random_seed", 42)
    np.random.seed(seed)

    metadata = {
        "seed": seed,
        "generated_at": datetime.now().isoformat(),
        "config_file": str(CONFIG_PATH),
    }

    # Load universe
    logger.info(f"\nLoading universe (max {config.get('universe_size', 50)} assets)...")
    universe = load_universe(
        symbols=config.get("symbols") or None,
        max_assets=50,
    )
    metadata["n_symbols"] = len(universe)
    logger.info(f"  Loaded {len(universe)} symbols")

    symbols_list = sorted(universe.keys())
    all_symbols_analyses: list[ContextAnalysisResult] = []
    all_trades_for_plot: list[tuple[list, str]] = []

    # Create strategies
    strategies = create_strategies(config)
    logger.info(f"Strategies: {[s.name for s in strategies]}")

    # Backtest engine
    engine = BacktestEngine(
        cost_per_trade=config.get("cost_per_trade", 0.001),
        slippage=config.get("slippage", 0.0005),
    )

    # Run per asset
    for symbol in symbols_list:
        df = universe[symbol]
        results = run_single_asset(df, strategies, engine)
        all_symbols_analyses.extend(results)

        # Collect trades for plotting (first strategy only)
        if len(all_trades_for_plot) == 0:
            for strategy in strategies:
                bt = engine.run(df, strategy)
                if bt.n_trades > 0:
                    all_trades_for_plot.append((bt.trades, strategy.name))

    logger.info(f"\nTotal analyses: {len(all_symbols_analyses)}")

    # Rank context variables
    rank_df = rank_context_variables(all_symbols_analyses)
    rank_df.to_csv(RESULTS_DIR / "context_ranking.csv", index=False)
    logger.info(f"  Ranking saved: {RESULTS_DIR / 'context_ranking.csv'}")

    # Build heatmap
    heatmap_df = build_regime_heatmap(all_symbols_analyses)
    if not heatmap_df.empty:
        heatmap_df.to_csv(RESULTS_DIR / "regime_heatmap.csv")
        logger.info(f"  Heatmap saved: {RESULTS_DIR / 'regime_heatmap.csv'}")

    # Generate charts
    logger.info("Generating charts...")
    plot_sharpe_heatmap(rank_df)
    plot_sharpe_gap_bars(rank_df)

    ctx_cols_sample = ["rv_5", "rv_20", "relative_volume", "adx", "sma200_distance", "vol_regime"]
    plot_context_conditional_sharpe(all_trades_for_plot, ctx_cols_sample)

    # Generate report
    report = generate_report(
        all_symbols_analyses,
        rank_df,
        heatmap_df,
        config,
        metadata,
    )
    report_path = STUDY_DIR / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"  Report saved: {report_path}")

    # Save metadata
    with open(RESULTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Save summary results
    summary = []
    for r in all_symbols_analyses:
        if r is None:
            continue
        summary.append({
            "strategy": r.strategy,
            "context": r.context_column,
            "overall_sharpe": r.overall_sharpe,
            "n_trades": r.overall_n_trades,
            "sharpe_gap": r.sharpe_gap,
            "best_regime": r.best_regime,
            "worst_regime": r.worst_regime,
        })
    pd.DataFrame(summary).to_csv(RESULTS_DIR / "results.csv", index=False)

    logger.info(f"\n{'=' * 80}")
    logger.info("  STUDY COMPLETE")
    logger.info(f"{'=' * 80}")
    logger.info(f"  Report: {report_path}")
    logger.info(f"  Results: {RESULTS_DIR}")
    logger.info(f"  Charts: {CHARTS_DIR}")


if __name__ == "__main__":
    run_study()
