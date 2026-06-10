# Market Regime Edge Discovery

## Hipótese Central

O edge não está no sinal. O edge está no contexto.

Uma mesma estratégia pode apresentar Sharpe positivo em determinados regimes de mercado e Sharpe negativo em outros. A mistura desses regimes destrói a expectativa total.

## Fenômeno

Estratégias simples de trading (momentum, mean reversion, breakout, etc.) possuem expectativa positiva apenas em condições específicas de mercado. Fora dessas condições, a expectativa é nula ou negativa.

## Direção Esperada

- **Volatilidade**: Estratégias de momentum funcionam melhor em alta volatilidade; mean reversion funciona melhor em baixa volatilidade.
- **Volume**: Breakouts funcionam melhor com volume extremo; mean reversion funciona melhor com volume baixo.
- **Tendência**: Trend following funciona apenas em mercados com tendência (ADX > 25); mean reversion funciona melhor em mercados lateralizados.
- **Estrutura**: Support/Resistance bounce funciona melhor quando o preço está próximo de níveis históricos relevantes.

## Mecanismo Econômico

Diferentes regimes de mercado refletem diferentes participantes e dinâmicas:

1. **Baixa volatilidade + baixo volume**: Mercado sem direção — formadores de mercado dominam. Mean reversion funciona.
2. **Alta volatilidade + alto volume**: Chegada de informação — tendências se formam. Momentum/breakout funciona.
3. **Volatilidade extrema**: Pânico ou euforia — qualquer estratégia tende a falhar por slippage e spreads excessivos.
4. **Tendência forte**: Participantes institucionais posicionados — Trend following funciona.

## Referências

- Lo, A. (2004). "The Adaptive Markets Hypothesis"
- Schwert, G. W. (1989). "Why Does Stock Market Volatility Change Over Time?"
- Ang, A. et al. (2006). "The Cross-Section of Volatility and Expected Returns"

## Perguntas de Pesquisa

1. Momentum funciona melhor em regimes de alta volatilidade?
2. Mean Reversion funciona melhor em regimes de baixa volatilidade?
3. Breakouts funcionam melhor com volume extremo?
4. Gap Fade funciona melhor quando o IBOV está em tendência?
5. VWAP Reversion funciona melhor em mercados sem tendência?
6. Opening Range Breakout funciona melhor em alta volatilidade?
7. Support/Resistance Bounce funciona melhor em mercados em range?
8. Trend Following funciona apenas com ADX > 25?

## Critério de Sucesso

Uma hipótese será considerada promissora apenas se para alguma variável de contexto:

- Sharpe OOS com contexto > 0.5
- Sharpe sem contexto < 0.3 (ou negativo)
- PF > 1.10
- Bootstrap > 95%
- Funciona em múltiplos ativos

## Critério de Rejeição

Rejeitar se:

- Funciona apenas em um ativo
- Funciona apenas em um período
- Depende de parâmetros muito específicos
- Desaparece OOS
- Não sobrevive a custos realistas

## Plano de Execução

1. Carregar universo de ações B3 M15
2. Para cada ativo, computar variáveis de contexto
3. Executar 8 estratégias base
4. Registrar contexto de cada trade
5. Agrupar trades por condição de contexto
6. Avaliar se a condição separa trades com expectativa positiva vs negativa
7. Validar com walk forward, bootstrap, reality check
8. Gerar heatmaps de regimes e ranking de variáveis
9. Documentar conclusões
