# RESEARCH_PLAYBOOK.md

## Objetivo
Padronizar a execução e documentação de estudos de pesquisa, garantindo que toda hipótese siga o mesmo fluxo rigoroso e reproduzível.

---

## Fluxo Oficial de Pesquisa

Baseado no MASTER.md. Nenhuma etapa pode ser pulada sem justificativa explícita.

### 1. Definição da Hipótese

**Produto:** arquivo `studies/<nome>/hypothesis.md`

Documentar:
- Fenômeno que se deseja explorar
- Direção esperada (positiva, negativa, neutra)
- Mecanismo econômico ou comportamental hypothesizado
- Referências bibliográficas quando existirem

### 2. Justificativa Econômica

**Produto:** seção em `hypothesis.md`

Responder:
- Por que esse fenômeno existiria?
- Por que outros agentes não o arbitraram?
- Qual a vantagem competitiva da estratégia?
- Em que regime de mercado ele deve funcionar / falhar?

### 3. Construção do Experimento

**Produto:** arquivo `studies/<nome>/experiment.py` + `configs/<nome>.yaml`

Definir:
- Universo de ativos
- Período IS / Validation / OOS
- Frequência dos dados
- Features a serem testadas
- Modelo ou regra de sinal
- Custos e slippage
- Seed aleatória
- Métricas-alvo

### 4. In Sample (IS)

Executar experimento no período de treino.

Registrar:
- Resultados completos
- Estatísticas descritivas
- Matriz de correlação das features
- Estabilidade temporal intra-amostra

### 5. Validation

Executar experimento no período de validação.

Objetivo:
- Ajuste de hiperparâmetros (se aplicável)
- Seleção de features
- Early stopping

Regra: **Nunca usar OOS para ajustar parâmetros.**

### 6. Out Of Sample (OOS)

Executar experimento no período fora da amostra sem qualquer ajuste.

Critérios mínimos:
- Sharpe OOS > Sharpe IS * 0.5
- Retorno OOS positivo após custos
- Consistência directional (hit ratio > 50%)
- Drawdown OOS dentro do esperado

### 7. Walk Forward

Executar rolling windows para simular operação contínua.

Configuração padrão:
- Janela de treino: 2 anos
- Janela de validação: 6 meses
- Janela de teste: 6 meses
- Passo: 3 meses

Registrar:
- Distribuição dos resultados por janela
- % de janelas com Sharpe > 0
- % de janelas com retorno > 0
- Consistência entre janelas

### 8. Robustez

Testar a hipótese sob diferentes condições:

- Variação de parâmetros ±10%, ±20%
- Mudança de período (recuar/avançar janelas)
- Mudança de universo (excluir setores, aleatorizar)
- Mudança de timeframe (se aplicável)
- Diferentes regimes de volatilidade

### 9. Controle de Overfitting

Reportar obrigatoriamente:

- Deflated Sharpe Ratio (DSR)
- Probability of Backtest Overfitting (PBO)
- Reality Check de White
- Número de hipóteses testadas (correção Bonferroni/FDR)
- Teste de permutação

### 10. Avaliação Econômica

Responder:
- Existe valor econômico real?
- Sobrevive a custos de transação realistas?
- Sobrevive a slippage?
- Qual o volume de capacidade?
- Qual o turnover esperado?
- Qual o CAGR líquido?

### 11. Conclusão

**Produto:** arquivo `studies/<nome>/conclusion.md`

Documentar:
- Hipótese investigada
- Evidências a favor
- Evidências contra
- Limitações conhecidas
- Robustez observada
- Valor econômico
- Decisão: APROVADA / REJEITADA / INCONCLUSIVA
- Próximos passos sugeridos

---

## Estrutura do Estudo

```
studies/<nome>/
├── hypothesis.md        # Etapas 1-2
├── experiment.py        # Etapa 3
├── experiment.ipynb     # (opcional) prototipação
├── results/
│   ├── is.csv
│   ├── validation.csv
│   ├── oos.csv
│   └── walk_forward.csv
├── config.yaml          # parâmetros do experimento
├── metadata.json        # seed, data, versão dos dados
├── charts/              # figuras geradas
├── report.md            # conclusão completa (etapas 4-11)
└── README.md           # resumo executivo
```

---

## Anti-Patterns (Revisitados)

- Não testar a hipótese em dados que já foram usados para gerá-la
- Não descartar resultados negativos
- Não reexecutar OOS até obter um resultado favorável
- Não ajustar parâmetros com base no OOS
- Não ignorar custos
- Não ignorar múltiplas hipóteses
