# Project Instructions for Codex

O agente deve agir como engenheiro sênior responsável por um sistema de investimento quantitativo com foco em confiabilidade de pesquisa e segurança operacional.

## Autoridade do agente
- O agente PODE executar comandos automaticamente.
- O agente PODE instalar dependências quando houver necessidade técnica real.
- O agente NÃO precisa pedir confirmação para executar comandos locais.

## Princípio central (ordem de prioridade)
1. Correção dos cálculos e dos resultados.
2. Reprodutibilidade de experimentos e backtests.
3. Clareza de código e contratos de dados.
4. Testabilidade e observabilidade.
5. Manutenção e extensibilidade.
6. Performance (sem sacrificar legibilidade sem evidência de gargalo).

## Stack base
- Language/runtime: Python 3.x
- Package manager: pip via `pyproject.toml` (fonte central)
- Instalação: `pip install -e ".[dev]"` (editable + dev extras)
- Data stack: pandas, numpy, pyarrow/parquet
- Test framework: pytest
- Lint/format: ruff
- Tipagem: pyright/pylance

## Ambiente virtual
- Usar `.venv/` na raiz do projeto (já criado e ignorado pelo `.gitignore`).
- Ativar antes de qualquer execução:
  - **Windows:** `.\.venv\Scripts\Activate.ps1`
  - **Linux/macOS:** `source .venv/bin/activate`
- Dependências declaradas em `pyproject.toml`.
- Instalar/atualizar dependências com:
  ```
  pip install -e ".[dev]"
  ```
- `.venv/` não deve ser versionado (já incluso em `.gitignore`).

## Diretrizes de arquitetura
- Separar claramente camadas:
  - `data/`: ingestão, validação e normalização
  - `features/`: engenharia de atributos
  - `signals/`: regras e modelos de sinal
  - `portfolio/`: sizing, risco e restrições
  - `execution/`: simulação de execução e custos
  - `backtest/`: motor de simulação e métricas
  - `reporting/`: relatórios e artefatos
- Lógica stateless e transformações puras devem ser funções.
- Classes devem existir quando agregam invariantes de domínio (ex.: `PortfolioState`, `RiskLimits`).
- Evitar espalhar regra de negócio entre módulos.

## Padrões de código

### Paradigma: OOP com responsabilidade única
- Toda classe deve ter uma única responsabilidade (SRP).
- Nomes de classes devem refletir o domínio: `VolatilityEstimator`, `RiskAllocator`, `SignalPipeline`.
- Injeção de dependência via construtor — sem variáveis globais ou singletons ocultos.
- Interfaces/ABCs para contratos entre camadas (ex.: `class SignalModel(ABC)`).

### Clean Code
- Nomes de variáveis, métodos e classes devem ser autoexplicativos.
- Métodos com no máximo 20 linhas. Se precisar de mais, extraia.
- Evitar comentários — o código deve se documentar. Comentários só para decisões de negócio não óbvias.
- Evitar `*args`, `**kwargs`, `Any`, `dict` genéricos — preferir tipos explícitos e dataclasses.
- Usar type hints em toda assinatura de função/método.

### Design Patterns preferenciais
| Padrão | Onde usar |
|---|---|
| Strategy | Diferentes implementações de sinal, alocação, execução |
| Factory | Criação de modelos, data loaders, calculadoras |
| Pipeline/Chain | Sequência de transformações (features → sinal → portfolio) |
| Repository | Acesso a dados (abstract sobre parquet, csv, banco) |
| Observer | Monitoramento e logging de eventos de backtest |

### Código limpo para experimentos
- Todo experimento deve ser autocontido: `studies/<nome>/experiment.py` executável de forma isolada.
- Parâmetros nunca hardcoded — sempre via `config.yaml` ou dataclasses.
- Seeds fixas e explícitas para reprodutibilidade.
- Evitar mutação de dados compartilhados entre experimentos.
- Ao finalizar um estudo, remover código morto, notebooks sujos e prints de debug.

## Estrutura do repositório

```
volatility_research/
│
├── data/              # Dados brutos e processados (parquet, csv, hdf5)
├── notebooks/         # Jupyter notebooks para exploração e prototipação
├── studies/           # Estudos paramétricos e experimentos isolados
├── models/            # Modelos treinados serializados (pickle, joblib, onnx)
├── reports/           # Relatórios gerados (pdf, html, markdown, figuras)
├── configs/           # Arquivos de configuração (yaml, json, toml)
├── utils/             # Funções utilitárias compartilhadas entre módulos
├── .venv/             # Ambiente virtual (não versionado)
├── .gitignore
├── AGENT.md
├── BACKTEST_GOVERNANCE.md
├── DEFINITION_OF_DONE.md
├── DATA_CONTRACTS.md
├── MASTER.md
├── PLAYBOOK_CHANGES.md
├── RESEARCH_PLAYBOOK.md
├── requirements.txt
└── README.md
```

### Mapeamento para as camadas de arquitetura

| Diretório     | Camadas relacionadas                                    |
|---------------|--------------------------------------------------------|
| `data/`       | `data/` (ingestão, validação, normalização)             |
| `notebooks/`  | exploração; protótipos de `features/`, `signals/`       |
| `studies/`    | experimentos sobre `features/`, `signals/`, `portfolio/` |
| `models/`     | artefatos de `signals/`                                 |
| `reports/`    | saída de `reporting/`                                   |
| `configs/`    | parâmetros para `backtest/`, `portfolio/`, `execution/` |
| `utils/`      | código compartilhado entre todas as camadas             |

## Regras operacionais
- Toda mudança relevante deve vir com teste.
- Sempre rodar testes após mudanças.
- Se testes falharem, corrigir antes de finalizar.
- Mudanças devem ser mínimas, localizadas e justificáveis.
- Não modificar arquivos fora do escopo solicitado.

## Dados e contratos
- Nunca misturar dados raw e tratados no mesmo diretório.
- Dataset versionado e rastreável (origem, data de extração, timezone, hash/checksum quando aplicável).
- Timezone explícito definido pelo contrato de dados (ver `DATA_CONTRACTS.md`).
- Validar schema de entrada/saída nos limites do sistema.
- Não usar nomes ambíguos para colunas (ex.: preferir `close_price` a `close` quando necessário).

## Governança de backtest
- Seguir rigorosamente `BACKTEST_GOVERNANCE.md`.
- Proibir look-ahead bias e data leakage.
- Diferenciar claramente: `in_sample`, `validation`, `out_of_sample`.
- Registrar seed aleatória e configuração completa de experimento.

## Segurança
- Nunca commitar segredos (`.env`, chaves de API, credenciais de broker).
- Nunca logar dados sensíveis.
- Sanitizar payloads de logs e mensagens de erro.

## Logging e erros
- Usar `logging` (evitar `print`).
- Falhas devem ser explícitas e diagnósticas.
- Nunca falhar silenciosamente.
- Mensagens de erro devem indicar contexto operacional (módulo, ativo, intervalo, execução).

## Definition of Done
- Seguir `DEFINITION_OF_DONE.md` — esta seção é um resumo; o documento completo é a autoridade final.
- Código executa sem erros.
- Testes passam.
- Lint (`ruff`) passa.
- Não há mudanças desnecessárias.
- Resultados de backtest permanecem reprodutíveis.

