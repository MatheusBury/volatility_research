# Contratos de Dados

## Fonte de dados

**Origem:** MetaTrader 5 (MT5) — exportado via `C:\Users\mathe\Documents\GitHub\mt5\dataset\export_mt5`

**Escopo autorizado:**
| Campo | Valor |
|---|---|
| Caminho | `C:\Users\mathe\Documents\GitHub\mt5\dataset\export_mt5\intraday\avista\M15\*.parquet` |
| Mercado | B3 (segmento à vista) — ações, BDRs, ETFs, FIIs, units |
| Timeframe | **M15** (15 minutos) apenas |
| Formato | Parquet (1593 arquivos) |
| Permissão | Leitura permitida sem confirmação |

O agente NÃO precisa de autorização para ler essa pasta fora do projeto.

---

## Schema raw (como exportado do MT5)

| Coluna | Tipo Parquet | Descrição |
|---|---|---|
| `time` | `timestamp[ns]` | Abertura do candle no **fuso local B3 (America/Sao_Paulo)**, sem offsets |
| `Open` | `double` | Preço de abertura |
| `High` | `double` | Preço máximo |
| `Low` | `double` | Preço mínimo |
| `Close` | `double` | Preço de fechamento |
| `Tick_volume` | `int64` | Número de ticks no candle |

> ⚠️ **Timezone:** A coluna `time` está em **America/Sao_Paulo**, NÃO em UTC. Todo consumo deve tratar explicitamente esse timezone. O horário de negociação da B3 é ~10:00–17:30 BRT (13:00–20:30 UTC no verão, 12:00–19:30 UTC no inverno).

> ⚠️ **Nomenclatura:** O raw usa `Open`/`High`/`Low`/`Close`/`Tick_volume`/`time`. O contrato padronizado abaixo deve ser usado em camadas de tratamento.

---

## Contrato padronizado (OHLCV limpo)

```python
from datetime import datetime, timezone
import pandas as pd
import pyarrow as pa

CONTRATO_OHLCV = {
    "symbol":      pa.string(),
    "timestamp":   pa.timestamp("ns", tz="America/Sao_Paulo"),
    "open_price":  pa.float64(),
    "high_price":  pa.float64(),
    "low_price":   pa.float64(),
    "close_price": pa.float64(),
    "volume":      pa.int64(),
}
```

| Campo | Tipo | Origem raw |
|---|---|---|
| `symbol` | `string` | Extraído do nome do arquivo (ex.: `PETR4.parquet` → `"PETR4"`) |
| `timestamp` | `timestamp[ns]` com timezone `America/Sao_Paulo` | `time` |
| `open_price` | `float64` | `Open` |
| `high_price` | `float64` | `High` |
| `low_price` | `float64` | `Low` |
| `close_price` | `float64` | `Close` |
| `volume` | `int64` | `Tick_volume` |

---

## Pipeline de leitura padrão

```python
import os, glob
import pandas as pd

DATA_DIR = r"C:\Users\mathe\Documents\GitHub\mt5\dataset\export_mt5\intraday\avista\M15"

def load_b3_m15(symbols=None):
    """Carrega OHLCV M15 de todos (ou dos símbolos informados) e retorna um DataFrame único com as colunas padronizadas."""
    pattern = os.path.join(DATA_DIR, "*.parquet")
    files = glob.glob(pattern)
    if symbols:
        files = [f for f in files if os.path.basename(f).replace(".parquet","") in symbols]

    pieces = []
    for f in files:
        sym = os.path.basename(f).replace(".parquet", "")
        df = pd.read_parquet(f)
        df = df.rename(columns={
            "time": "timestamp",
            "Open": "open_price",
            "High": "high_price",
            "Low": "low_price",
            "Close": "close_price",
            "Tick_volume": "volume",
        })
        df["symbol"] = sym
        df["timestamp"] = df["timestamp"].dt.tz_localize("America/Sao_Paulo")
        pieces.append(df)

    df = pd.concat(pieces, ignore_index=True)
    return df
```

---

## Regras de qualidade

- `timestamp` deve ser monotônico crescente dentro de cada `symbol`.
- Sem timestamps duplicados por `symbol` (cada candle de 15 min é único).
- Sem lacunas não justificadas (feriados B3, after-hours, fim de semana).
- Preços (`open_price`, `high_price`, `low_price`, `close_price`) devem ser ≥ 0.
- `volume` deve ser ≥ 0. `Tick_volume` = 0 indica candle sem negócios.
- Ações listadas em classes múltiplas (ex.: PETR3/PETR4, VALE3/VALE5) são arquivos separados — cada um é um `symbol` distinto.

---

## Versionamento e rastreabilidade

- O dataset é atualizado regularmente via MT5. A data do último candle indica a versão implícita.
- Cada artefato derivado deve apontar para:
  - dataset de origem (caminho da pasta)
  - data da extração
  - versão do código de transformação
  - hash do arquivo parquet fonte (opcional, para auditoria)

---

## Símbolos disponíveis

O dataset contém **1593 ativos** do segmento à vista da B3, incluindo:
- **Ações** (ordinárias ON, preferenciais PN, units): PETR3/4, VALE3, ITUB3/4, BBDC3/4, BBAS3, ABEV3, WEGE3, etc.
- **BDRs** (sufixo 34/35/39): AAPL34, MSFT34, GOOGL34, TSLA34, etc.
- **ETFs** (sufixo 11): BOVA11, IVVB11, SMAL11, XPLG11, etc.
- **FIIs** (sufixo 11): KNRI11, HGLG11, XPLG11, MXRF11, etc.

> A listagem exata pode ser obtida com: `[f.replace(".parquet","") for f in glob.glob(r"C:\\Users\\mathe\\...\\M15\\*.parquet")]`

