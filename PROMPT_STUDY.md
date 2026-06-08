# Estudo de Volatilidade — B3 M15

## Premissa
Volatilidade é a única fonte de edge real em mercados financeiros. Este estudo deve investigar múltiplas abordagens para modelagem, previsão e exploração da volatilidade nos ativos B3 (timeframe M15, 1593 ativos).

## Tópicos obrigatórios

### 1. GARCH e EGARCH
- Ajustar modelos GARCH(1,1) e EGARCH(1,1) para retornos intradiários M15
- Comparar aderência in-sample e previsão out-of-sample
- Avaliar persistência e mean reversion da volatilidade

### 2. Hidden Markov Models (HMM)
- Identificar regimes de volatilidade (baixa, média, alta, extrema)
- Estimar probabilidades de transição entre regimes
- Avaliar estabilidade temporal dos regimes

### 3. Volatility Targeting
- Testar estratégia que dimensiona posição inversamente à volatilidade recente
- Comparar janelas de estimação (15, 30, 60, 120 candles)
- Avaliar Sharpe, drawdown e turnover vs buy-and-hold

### 4. Risk Parity
- Construir carteira com ponderação por vol inversa
- Comparar com alocação igual e com tangency portfolio
- Testar estabilidade das alocações ao longo do tempo

### 5. Volatility Managed Portfolios
- Implementar estratégia que reduz exposição em alta vol e aumenta em baixa vol
- Testar diferentes regras de switching
- Avaliar valor econômico vs abordagem estática

### 6. Regime Switching Models
- Comparar modelos com 2, 3 e 4 regimes
- Estimar parâmetros específicos por regime
- Avaliar poder preditivo da transição de regimes

### 7. Variance Risk Premium
- Calcular spread entre vol realizada e vol implícita (se disponível)
- Analisar poder preditivo do VRP para retornos futuros
- Testar estratégia de timing baseada no VRP

### 8. Opções e Volatilidade Implícita
- Se houver dados de opções no dataset, extrair vol implícita
- Comparar skew e term structure da vol implícita
- Avaliar informações contidas na superfície de volatilidade

## Metodologia
- Separar IS (2021-2024) e OOS (2025-2026)
- Utilizar todos os 1593 ativos do dataset M15 da B3
- Documentar cada hipótese antes de testar
- Reportar resultados negativos como conclusões válidas
- Gerar artefatos em studies/<topicos>/

## Diretrizes
- Seguir RESEARCH_PLAYBOOK.md
- Seguir BACKTEST_GOVERNANCE.md
- Seguir DEFINITION_OF_DONE.md
- Usar DATA_CONTRACTS.md para pipeline de dados
- Código limpo, OOP, type hints, testes
