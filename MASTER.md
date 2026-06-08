# MASTER.md

## Missão

Construir um ambiente quantitativo de pesquisa, validação e desenvolvimento de estratégias financeiras para o mercado brasileiro com foco em:

* Robustez estatística.
* Reprodutibilidade.
* Governança de pesquisa.
* Valor econômico real.
* Controle de risco.
* Evolução contínua do conhecimento.

O objetivo principal NÃO é maximizar resultados históricos.

O objetivo principal é identificar fenômenos persistentes que sobrevivam a validações rigorosas e possam gerar valor econômico reproduzível.

---

# Hierarquia de Regras

Em caso de conflito entre documentos, utilizar a seguinte ordem de prioridade:

1. MASTER.md
2. BACKTEST_GOVERNANCE.md
3. DATA_CONTRACTS.md
4. RESEARCH_PLAYBOOK.md
5. DEFINITION_OF_DONE.md
6. PLAYBOOK_CHANGES.md

Documentos de prioridade inferior não podem contradizer documentos de prioridade superior.

---

# Escopo do Projeto

## Mercado Autorizado

Mercado oficial de pesquisa:

* B3
* Mercado à vista

Instrumentos autorizados:

* Ações
* ETFs
* FIIs
* BDRs
* Units

Instrumentos não autorizados:

* Futuros
* Opções
* Forex
* Criptomoedas
* Commodities
* Renda fixa

Novos mercados somente mediante atualização explícita deste documento.

---

## Timeframe Oficial

Timeframe oficial de pesquisa:

* M15 (15 minutos)

Todo desenvolvimento, pesquisa, validação e geração de hipóteses deve assumir M15 como padrão.

Timeframes diferentes devem ser considerados estudos especiais e precisam ser explicitamente justificados.

---

## Universo Oficial

Universo padrão do projeto:

* Dataset oficial definido em DATA_CONTRACTS.md
* Mercado à vista da B3
* Arquivos Parquet
* Aproximadamente 1593 ativos

Nenhum estudo deve utilizar fonte de dados diferente sem documentar explicitamente a alteração.

---

# Filosofia de Pesquisa

Princípios fundamentais:

* Robustez é mais importante que retorno.
* OOS é mais importante que IS.
* Valor econômico é mais importante que significância estatística.
* Explicação econômica é preferível a correlação observada.
* Hipóteses rejeitadas são conhecimento útil.
* Resultados negativos devem ser preservados.
* A ausência de edge é uma conclusão válida.
* Não confundir previsibilidade estatística com lucratividade.
* Não confundir Sharpe histórico com capacidade futura.

---

# Direcionamento Atual de Pesquisa

As linhas prioritárias de pesquisa são:

* Forecast de volatilidade
* Regimes de mercado
* Volatility targeting
* Position sizing
* Gestão de risco
* Estabilidade temporal
* Robustez cross-asset
* Robustez cross-sectional
* Value of Forecasting
* Avaliação econômica da previsibilidade

As linhas abaixo possuem prioridade reduzida:

* Forecast direcional puro
* Estratégias baseadas apenas em autocorrelação simples
* Estratégias sem justificativa econômica
* Estratégias dependentes de poucos trades

---

# Definição de Edge

Uma hipótese só pode ser considerada edge quando:

* Sobrevive OOS.
* Sobrevive custos.
* Sobrevive slippage.
* Sobrevive validação temporal.
* Sobrevive mudança de ativo.
* Sobrevive mudança de regime.
* Possui justificativa econômica plausível.
* Possui valor econômico mensurável.

Nenhuma hipótese pode ser aprovada apenas por:

* Sharpe elevado.
* Retorno elevado.
* p-value baixo.
* Resultado In Sample.
* Backtest visualmente atrativo.

---

# Fluxo Oficial de Pesquisa

Toda pesquisa deve seguir:

1. Definição da hipótese.
2. Justificativa econômica.
3. Construção do experimento.
4. In Sample.
5. Validation.
6. Out Of Sample.
7. Walk Forward.
8. Robustez.
9. Controle de Overfitting.
10. Avaliação Econômica.
11. Conclusão.

Etapas não podem ser puladas sem justificativa explícita.

---

# Critérios Gerais de Aprovação

Uma hipótese somente pode ser aprovada quando:

* Sobrevive OOS.
* Sobrevive custos.
* Sobrevive slippage.
* Sobrevive robustez.
* Sobrevive mudança de regime.
* Possui interpretação econômica.
* Produz valor econômico.
* Apresenta estabilidade razoável.

---

# Critérios Gerais de Rejeição

Rejeitar quando:

* Depende de poucos trades.
* Depende de parâmetros extremamente específicos.
* Funciona apenas em um período.
* Colapsa após custos.
* Colapsa OOS.
* Não possui justificativa econômica plausível.
* Não apresenta estabilidade.
* Não possui valor econômico.

---

# Anti-Padrões

Proibido:

* Utilizar informação futura.
* Utilizar data leakage.
* Ajustar parâmetros usando OOS.
* Escolher apenas o melhor resultado.
* Ignorar custos.
* Ignorar slippage.
* Ignorar múltiplas hipóteses.
* Ignorar resultados negativos.
* Reexecutar experimentos até encontrar resultado positivo.
* Ocultar resultados desfavoráveis.
* Aprovar estratégias apenas por significância estatística.
* Alterar hipóteses após observar resultados.

---

# Princípios de Engenharia

Todo código produzido deve priorizar:

1. Correção.
2. Reprodutibilidade.
3. Clareza.
4. Testabilidade.
5. Observabilidade.
6. Manutenibilidade.
7. Performance.

Performance não justifica perda de clareza sem evidência objetiva.

---

# Princípios de Dados

Todo dado utilizado deve possuir:

* Origem identificável.
* Timezone explícito.
* Schema conhecido.
* Rastreabilidade.
* Versionamento.

Dados sem contrato não devem ser utilizados.

---

# Princípios de Backtest

Todo backtest deve:

* Evitar look-ahead bias.
* Evitar data leakage.
* Considerar custos.
* Considerar slippage.
* Considerar latência quando aplicável.
* Registrar configuração completa.
* Permitir reprodução futura.

---

# Definição de Sucesso

O objetivo do projeto NÃO é:

* Maximizar Sharpe histórico.
* Maximizar retorno histórico.
* Encontrar correlações aleatórias.
* Produzir backtests visualmente atraentes.

O objetivo do projeto é:

* Construir conhecimento acumulativo.
* Encontrar fenômenos persistentes.
* Produzir resultados reproduzíveis.
* Encontrar valor econômico robusto.
* Reduzir o risco de overfitting.
* Melhorar continuamente a qualidade da pesquisa.

---

# Regra Final

Quando existir dúvida entre:

* Retorno e robustez.
* Performance e reprodutibilidade.
* Complexidade e clareza.
* Resultado e evidência.

Escolher sempre a alternativa que maximize robustez, reprodutibilidade e evidência.
