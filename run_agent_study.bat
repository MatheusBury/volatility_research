@echo off
setlocal

set "PROJECT_PATH=C:\Users\mathe\Documents\GitHub\volatility_research"

cd /d "%PROJECT_PATH%" || (
  echo Erro: nao foi possivel acessar a pasta.
  pause
  exit /b 1
)

opencode --model "opencode/deepseek-v4-flash-free" --prompt-file "PROMPT_STUDY.md"

pause
endlocal
