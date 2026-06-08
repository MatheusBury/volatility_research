# Playbook de Mudanças

## Objetivo
Padronizar alterações para manter qualidade, rastreabilidade e baixo risco de regressão.

## Passo a passo
1. Definir escopo técnico da mudança.
2. Identificar módulos afetados.
3. Implementar mudança mínima.
4. Criar/atualizar testes.
5. Rodar qualidade local:
   - `ruff check .`
   - `pytest`
6. Validar impacto em métricas (quando mudar lógica de estratégia/backtest).
7. Documentar:
   - o que mudou
   - por que mudou
   - impacto esperado

## Regras de review
- Toda alteração de regra de estratégia exige evidência de backtest.
- Toda alteração de contrato de dados exige atualização de documentação.
- Mudanças sem teste ou sem justificativa não devem ser aprovadas.

