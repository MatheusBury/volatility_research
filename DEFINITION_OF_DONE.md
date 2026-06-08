# Definition of Done (DoD)

Uma tarefa, estudo, modelo ou experimento só é considerado concluído quando TODOS os critérios abaixo forem atendidos.

---

# 1. Qualidade Técnica

* Código executa sem erros.
* Não existem exceções não tratadas.
* Não existem warnings críticos ignorados.
* Não existem dependências quebradas.
* Estrutura do projeto permanece consistente.

---

# 2. Qualidade de Código

Obrigatório:

```bash
ruff check .
```

Resultado:

```text
0 errors
0 warnings
```

Quando aplicável:

```bash
ruff format .
```

Código deve permanecer:

* legível;
* simples;
* modular;
* consistente com a arquitetura do projeto.

---

# 3. Testes

Obrigatório:

```bash
pytest
```

Resultado:

```text
100% dos testes passando
```

Além disso:

* Todo bug corrigido deve possuir teste.
* Toda nova regra deve possuir teste.
* Toda alteração de lógica deve atualizar os testes existentes.

Mudanças sem teste não são consideradas concluídas.

---

# 4. Reprodutibilidade

Todo experimento deve ser reproduzível.

Obrigatório registrar:

* seed utilizada;
* período analisado;
* universo de ativos;
* custos utilizados;
* slippage utilizado;
* parâmetros do modelo;
* configuração completa da execução.

Executar novamente com a mesma configuração deve produzir resultados equivalentes.

---

# 5. Governança de Backtest

Nenhum resultado pode ser considerado válido sem:

* treino;
* validação;
* teste (out-of-sample).

Obrigatório verificar:

* ausência de look-ahead bias;
* ausência de data leakage;
* custos de transação;
* slippage;
* integridade temporal.

---

# 6. Validação Estatística

Toda hipótese aprovada deve possuir evidência estatística.

Quando aplicável:

* p-value;
* intervalo de confiança;
* bootstrap;
* permutation test;
* Deflated Sharpe Ratio;
* Probability of Backtest Overfitting;
* correção para múltiplos testes.

Resultados estatisticamente significativos devem ser acompanhados de interpretação econômica.

---

# 7. Robustez

Toda hipótese deve sobreviver a:

* alteração de parâmetros ±10%;
* alteração de parâmetros ±20%;
* mudança de período;
* mudança de ativo;
* mudança de regime de mercado.

Hipóteses altamente sensíveis devem ser classificadas como frágeis.

---

# 8. Valor Econômico

Não basta existir previsibilidade estatística.

Obrigatório responder:

1. Existe previsibilidade?
2. Existe valor econômico?
3. Sobrevive aos custos?
4. Sobrevive ao slippage?
5. Sobrevive out-of-sample?
6. Sobrevive em múltiplos ativos?

Se a resposta for negativa, a hipótese deve ser registrada como rejeitada.

---

# 9. Artefatos

Todo estudo deve gerar, dentro de `studies/<nome>/`:

```text
studies/<nome>/
├── report.md            # conclusão completa
├── results.csv           # métricas agregadas
├── results.parquet       # resultados detalhados por ativo/período
├── config.yaml           # parâmetros do experimento
├── metadata.json         # seed, data, versão dos dados
├── charts/               # figuras geradas (png, pdf)
├── figures/              # (opcional) figuras para publicação
└── backtests/            # (opcional) resultados de backtest
```

Modelos treinados vão em `models/<nome>.pkl` ou `models/<nome>.onnx`.

Relatórios consolidados vão em `reports/<nome>.md` ou `reports/<nome>.pdf`.

Configurações reutilizáveis vão em `configs/<nome>.yaml`.

---

# 10. Documentação

Sempre atualizar quando houver:

* mudança de arquitetura;
* mudança de contrato de dados;
* mudança de metodologia;
* nova hipótese de pesquisa;
* nova conclusão relevante.

---

# 11. Segurança

Proibido:

* expor credenciais;
* expor tokens;
* expor chaves de API;
* versionar arquivos sensíveis;
* logar informações confidenciais.

---

# 12. Escopo

A tarefa não pode:

* alterar arquivos fora do escopo solicitado;
* modificar comportamento não relacionado;
* introduzir mudanças sem justificativa.

---

# 13. Conclusão de Pesquisa

Um estudo só pode ser considerado concluído quando responder explicitamente:

* Hipótese investigada.
* Evidências a favor.
* Evidências contra.
* Limitações.
* Robustez.
* Valor econômico.
* Conclusão final.

---

# Checklist Final

Antes de concluir qualquer tarefa:

* [ ] Código executa sem erros
* [ ] Ruff aprovado
* [ ] Pytest aprovado
* [ ] Testes atualizados
* [ ] Sem data leakage
* [ ] Sem look-ahead bias
* [ ] Resultados reproduzíveis
* [ ] Custos considerados
* [ ] Artefatos gerados
* [ ] Documentação atualizada
* [ ] Escopo respeitado
* [ ] Conclusão registrada

Somente após todos os itens acima a tarefa pode ser marcada como DONE.
