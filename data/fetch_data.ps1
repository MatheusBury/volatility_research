$tickers = @(
    "PETR4","VALE3","ITUB4","BBDC4","BBAS3","ABEV3","WEGE3","B3SA3","ELET3","SUZB3",
    "RENT3","RADL3","GGBR4","LREN3","HAPV3","PRIO3","VBBR3","EMBR3","CSNA3","CMIG4",
    "CPLE6","ENGI11","EQTL3","RAIL3","CCRO3","UGPA3","BRFS3","JBSS3","MRFG3","KLBN11",
    "TIMS3","VIVT3","TOTS3","RDOR3","FLRY3","PCAR3","CRFB3","ASAI3","MGLU3","VIIA3",
    "ENEV3","ODPV3","AZUL4","GOLL4","CVCB3","YDUQS3","COGN3","SANB11","BPAC11","NEOE3"
)
$baseUrl = "https://spa.oplab.com.br/v3/market/historical/{0}IVX/1d?amount=252&fill=business_days&smooth=true"
$outputDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$headers = @{
    "access-token" = "1wIzrh60zcnE7WkKXpsBywn755Dlmg%2Fx6vxxiAdGBhzfCofGU6Cj5wY6rx6AOKnx--lPtpvtSCUXgx1o1O%2BY5wfA%3D%3D--OTI3NzJlOTM4ODVhZTQ3MGEzMjRjZmRhNDhjM2IyYjY%3D"
    "x-oplab-client-name" = "pwa"
    "x-oplab-client-version" = "1.0.0"
    "origin" = "https://go.oplab.com.br"
    "referer" = "https://go.oplab.com.br/"
}

foreach ($ticker in $tickers) {
    $url = $baseUrl -f $ticker
    $outputFile = Join-Path $outputDir "${ticker}IVX.json"
    Write-Host "Baixando $ticker..."
    try {
        Invoke-RestMethod -Uri $url -Method Get -Headers $headers -OutFile $outputFile
        Write-Host "  OK -> $outputFile"
    }
    catch {
        Write-Host "  ERRO ao baixar $ticker : $_"
    }
}
