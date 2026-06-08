# Hipótese: Motor de Volatilidade para Opções B3

## Fenômeno
A volatilidade implícita (IV) das opções B3 contém um prêmio de risco (VRP) sistemático
em relação à volatilidade realizada futura (RV). Este prêmio varia com:
- Regime de volatilidade do ativo subjacente
- Nível atual do VRP (mean-reversion)
- Strike da opção (ATM vs OTM vs ITM)

## Direção Esperada

### Estudo 1 — Forecast RV vs IV
1. **IV será consistentemente maior que RV** (VRP positivo), confirmando que opções
   embutem prêmio de risco de volatilidade.
2. **Quando VRP é muito positivo** (IV muito cara), vender volatilidade deve gerar
   retorno positivo nos dias seguintes.
3. **Quando VRP é muito negativo** (IV muito barata), comprar volatilidade deve gerar
   retorno positivo nos dias seguintes.

### Estudo 2 — Decis de VRP
4. **Decil 1** (IV mais barata): RV futura será maior que IV — lucro comprando vol.
5. **Decil 10** (IV mais cara): RV futura será menor que IV — lucro vendendo vol.
6. A relação VRP vs retorno futuro deve ser monotônica (quanto mais cara a IV,
   melhor para vender vol).

### Estudo 3 — Regime de Volatilidade
7. **VRP será maior em regimes de alta volatilidade**: investidores pagam mais
   prêmio por proteção em momentos de estresse.
8. **Regimes de baixa volatilidade** devem ter VRP menor ou próximo de zero.
9. Se VRP difere por regime, usar o regime como filtro operacional para
   timing de entrada/saída.

### Estudo 4 — Superfície de Volatilidade
10. **OTM puts terão IV maior** que ATM (skew de cauda).
11. **OTM calls terão IV menor** que ATM (menor demanda por proteção de alta).
12. O forecast de RV deve explicar melhor a IV ATM do que a IV de strikes
    extremos (onde o prêmio de liquidez e cauda dominam).

## Mecanismo Econômico
- **Prêmio de risco de volatilidade**: investidores pagam para se proteger
  contra picos de volatilidade (tail hedge), criando VRP positivo.
- **Assimetria de liquidez**: opções OTM têm menos liquidez, o que aumenta
  o prêmio de liquidez embutido na IV.
- **Efeito alavancagem**: quedas no subjacente elevam a volatilidade futura,
  tornando puts OTM sistematicamente mais caras que calls OTM.
- **Demanda por proteção**: investidores institucionais compram puts para
  hedge de carteira, pressionando a IV para cima.

## Referências
- Carr, P. & Wu, L. (2009). Variance risk premiums.
- Bollerslev, T., Tauchen, G. & Zhou, H. (2009). Expected stock returns
  and variance risk premia.
- Bakshi, G. & Kapadia, N. (2003). Delta-hedged gains and the negative
  market volatility risk premium.
- Black, F. (1976). Studies of stock price volatility changes.
