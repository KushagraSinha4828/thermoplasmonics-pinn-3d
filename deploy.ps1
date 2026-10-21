$ErrorActionPreference = 'Stop'
$repo = "D:\IISERB\Research\Thermoplasmonics\Thesis\github repository"
Set-Location $repo

Write-Host "[1/5] Installing Git and GitHub CLI (if not present)..." -ForegroundColor Cyan
winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements | Out-Null
winget install --id GitHub.cli -e --source winget --accept-package-agreements --accept-source-agreements | Out-Null

# Refresh Path locally for this session
$env:Path += ";C:\Program Files\Git\cmd;C:\Program Files\GitHub CLI"

Write-Host "\n[2/5] Authenticating with GitHub..." -ForegroundColor Yellow
Write-Host "A browser window will open. Please log in and authorize GitHub CLI." -ForegroundColor Yellow
gh auth login --web

Write-Host "\n[3/5] Initializing Git repository..." -ForegroundColor Cyan
git init
git branch -M main

Write-Host "\n[4/5] Committing all codes, figures, and documentation..." -ForegroundColor Cyan
git add .
git commit -m "Initial release: Physics-Informed 3D U-Net & 3-Layer Green's Function Solver for Thermoplasmonics (V20.101a)"

Write-Host "\n[5/5] Creating public repository and pushing live..." -ForegroundColor Green
gh repo create thermoplasmonics-pinn-3d --public --source=. --remote=origin --push

Write-Host "\nDeployment Complete! Your repository is now live on your GitHub profile." -ForegroundColor Green
Pause
