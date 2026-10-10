param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $projectDirectory ".venv\Scripts\python.exe"
$langgraphPath = Join-Path $projectDirectory ".venv\Scripts\langgraph.exe"
$envFile = Join-Path $projectDirectory ".env"

Set-Location -LiteralPath $projectDirectory

# LangGraph CLI reads UTF-8 package data; Windows may otherwise use GBK.
$env:PYTHONUTF8 = "1"

if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Missing .env file. Copy .env.example to .env and add your API keys."
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Missing virtual environment. Run: uv sync --python 3.12.14"
}

if (-not (Test-Path -LiteralPath $langgraphPath)) {
    throw "LangGraph CLI is not installed in .venv. Run: uv sync"
}

if (-not $SkipTests) {
    Write-Host "Running local tests..." -ForegroundColor Cyan

    & $pythonPath -m unittest discover `
        -s tests `
        -p "test_*.py" `
        -v

    if ($LASTEXITCODE -ne 0) {
        throw "Tests failed. LangGraph server was not started."
    }
}

Write-Host "Starting ResearchPilot..." -ForegroundColor Green
Write-Host "Select research_explicit in LangGraph Studio." -ForegroundColor Yellow
Write-Host "No API calls occur until a query is submitted." -ForegroundColor Yellow

& $langgraphPath dev
