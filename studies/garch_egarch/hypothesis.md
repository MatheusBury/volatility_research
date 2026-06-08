# Hipótese: GARCH vs EGARCH para Ações B3 em Dados de 15 Minutos

## Fenômeno
Modelos GARCH(1,1) e EGARCH(1,1) ajustados a retornos logarítmicos de ações da B3
em frequência intraday (M15) apresentam diferenças sistemáticas em:
- Aderência (AIC/BIC)
- Persistência da volatilidade
- Capacidade preditiva fora da amostra

## Direção Esperada
1. **EGARCH terá melhor ajuste (AIC/BIC menor)** que GARCH para a maioria dos ativos,
   pois EGARCH captura assimetria (alavancagem) — quedas elevam volatilidade mais que altas.
2. **Persistência será muito alta** (>0.95) para ambos os modelos em dados M15,
   dada a natureza de alta frequência.
3. **EGARCH terá forecast variance mais realista** (não forçada a ser positiva
   por construção paramétrica, já que modela log-vol).

## Mecanismo Econômico
- Efeito alavancagem (Black, 1976): quedas no preço aumentam a razão
  dívida/patrimônio, elevando o risco percebido e a volatilidade futura.
- Em dados intraday, o efeito assimétrico é mais pronunciado devido a
  microestrutura (ordens de stop-loss, liquidez assimétrica em quedas).
- EGARCH modela o log da variância condicional, permitindo efeitos assimétricos
  sem restrições de não-negatividade dos parâmetros.

## Referências
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
- Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach.
- Black, F. (1976). Studies of stock price volatility changes.
