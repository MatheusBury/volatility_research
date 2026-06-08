# Volatility Research

Pesquisa quantitativa de volatilidade para o mercado acionário brasileiro (B3).

## Stack

- Python 3.x + pandas, numpy, pyarrow
- Testes: pytest
- Lint: ruff
- Tipagem: pyright/pylance

## Setup

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Qualidade

```powershell
ruff check .
pytest
```

## Documentos

| Documento | Propósito |
|---|---|
| `MASTER.md` | Missão, filosofia, hierarquia de regras |
| `AGENT.md` | Instruções para agents de IA |
| `BACKTEST_GOVERNANCE.md` | Regras de backtest e validação |
| `DATA_CONTRACTS.md` | Contratos e pipeline de dados |
| `DEFINITION_OF_DONE.md` | Critérios de conclusão |
| `RESEARCH_PLAYBOOK.md` | Fluxo de pesquisa passo a passo |
| `PLAYBOOK_CHANGES.md` | Procedimento de mudanças |
