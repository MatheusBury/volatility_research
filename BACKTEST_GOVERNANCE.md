# Governança de Backtest

## Regras obrigatórias
- Nenhuma estratégia pode ser validada sem separar treino/validação/teste.
- Proibido usar informação futura em qualquer etapa.
- Custos de transação e slippage devem estar habilitados por padrão.

## Checklist de experimento
- Estratégia e hipótese registradas antes da execução.
- Janela temporal e universo definidos.
- Seeds e parâmetros registrados.
- Métricas-alvo e critérios de aceitação definidos.

## Métricas mínimas
- Retorno acumulado e anualizado.
- Volatilidade anualizada.
- Sharpe e Sortino.
- Max drawdown e duração do drawdown.
- Turnover médio.

## Critérios de rejeição
- Sensibilidade extrema a pequenos ajustes de parâmetro.
- Resultado bom apenas em um regime curto.
- Performance explicada por poucos trades outliers.

## Múltiplas Hipóteses

Toda pesquisa que testar múltiplos sinais, parâmetros ou estratégias deve reportar:

- Número total de hipóteses testadas.
- p-value bruto.
- p-value ajustado (Bonferroni).
- p-value ajustado (FDR).
- Deflated Sharpe Ratio (DSR).

Nenhuma estratégia pode ser aprovada apenas pelo p-value bruto.

## Testes de Robustez

Toda estratégia aprovada deve sobreviver a:

- Variação de parâmetros ±10%
- Variação de parâmetros ±20%
- Mudança de timeframe
- Mudança de ativo
- Mudança de período

Estratégias altamente dependentes de parâmetros específicos devem ser rejeitadas.

## Walk Forward Obrigatório

Toda estratégia deve ser validada com:

- In Sample
- Validation
- Out Of Sample

Sempre que possível:

- Walk Forward Rolling
- Walk Forward Expanding

Resultados IS não podem ser usados para aprovação.

## Controle de Overfitting

Reportar obrigatoriamente:

- Probability of Backtest Overfitting (PBO)
- Deflated Sharpe Ratio (DSR)
- Reality Check de White
- SPA Test quando aplicável

Estratégias com:

PBO > 50%

devem ser consideradas suspeitas.

## Dependência de Outliers

Reportar:

- Top 1% trades
- Top 5% trades
- Top 10% trades

Reexecutar métricas removendo:

- Top 1%
- Top 5%

Se a estratégia colapsar, registrar explicitamente.

Estratégias dependentes de poucos trades extremos devem ser classificadas como frágeis.

## Estabilidade por Regime

Reportar desempenho por:

- Baixa volatilidade
- Média volatilidade
- Alta volatilidade
- Volatilidade extrema

Aprovação exige estabilidade em múltiplos regimes ou justificativa econômica para concentração em um regime específico.

## Reprodutibilidade

Todo experimento deve salvar:

- Configuração completa
- Seed
- Universo
- Datas
- Custos
- Features utilizadas
- Versão dos dados

O resultado deve ser reproduzível a partir dos artefatos gerados.

Nenhuma hipótese pode ser aprovada apenas por significância estatística.

Ela deve sobreviver:
- OOS
- Walk Forward
- Custos
- PBO
- DSR
- Cross-Asset Validation
- Teste de Robustez