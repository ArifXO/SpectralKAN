$ErrorActionPreference = "Stop"

$epochs = 200
$outputDir = "results/runs"

function Get-MaxLoggedEpoch {
    param([string]$LogPath)

    if (-not (Test-Path $LogPath)) {
        return 0
    }

    $rows = Import-Csv $LogPath
    if ($null -eq $rows -or $rows.Count -eq 0) {
        return 0
    }

    return ($rows | ForEach-Object { [int]$_.epoch } | Measure-Object -Maximum).Maximum
}

function Invoke-TrainingRun {
    param(
        [string]$Config,
        [string]$RunName,
        [int]$Seed,
        [string[]]$ExtraArgs = @()
    )

    $runDir = Join-Path $outputDir $RunName
    $logPath = Join-Path $runDir "training_log.csv"
    $lastCheckpoint = Join-Path $runDir "checkpoints/last.pt"
    $maxEpoch = Get-MaxLoggedEpoch $logPath

    if ($maxEpoch -ge $epochs) {
        Write-Host "Skipping completed run: $RunName ($maxEpoch epochs logged)"
        return
    }

    $args = @(
        "scripts/train.py",
        "--config", $Config,
        "--epochs", "$epochs",
        "--seed", "$Seed",
        "--save-every", "50",
        "--output-dir", $outputDir,
        "--run-name", $RunName
    ) + $ExtraArgs

    if (Test-Path $lastCheckpoint) {
        Write-Host "Resuming run: $RunName from $lastCheckpoint"
        $args += @("--resume", $lastCheckpoint)
    } elseif (Test-Path $runDir) {
        throw "Run directory exists but no resumable checkpoint was found: $runDir"
    } else {
        Write-Host "Starting run: $RunName"
    }

    & python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for $RunName with exit code $LASTEXITCODE"
    }
}

Invoke-TrainingRun `
    -Config "configs/ptbxl_kan_config.yaml" `
    -RunName "ptbxl_kan_grid5_s42" `
    -Seed 42 `
    -ExtraArgs @("--lr", "1.5e-4", "--kan-grid-size", "5")

Invoke-TrainingRun `
    -Config "configs/ptbxl_kan_config.yaml" `
    -RunName "ptbxl_kan_grid5_s43" `
    -Seed 43 `
    -ExtraArgs @("--lr", "1.5e-4", "--kan-grid-size", "5")

Invoke-TrainingRun `
    -Config "configs/ptbxl_kan_config.yaml" `
    -RunName "ptbxl_kan_grid5_s44" `
    -Seed 44 `
    -ExtraArgs @("--lr", "1.5e-4", "--kan-grid-size", "5")

Invoke-TrainingRun `
    -Config "configs/ptbxl_config.yaml" `
    -RunName "ptbxl_transformer_s42" `
    -Seed 42

Invoke-TrainingRun `
    -Config "configs/ptbxl_config.yaml" `
    -RunName "ptbxl_transformer_s43" `
    -Seed 43

Invoke-TrainingRun `
    -Config "configs/ptbxl_config.yaml" `
    -RunName "ptbxl_transformer_s44" `
    -Seed 44

function Write-Summary {
    param(
        [string]$Prefix,
        [string]$OutputCsv
    )

    $summary = foreach ($seed in @(42, 43, 44)) {
        $logPath = Join-Path $outputDir "$Prefix$seed/training_log.csv"
        if (-not (Test-Path $logPath)) {
            throw "Missing training log: $logPath"
        }

        $rows = Import-Csv $logPath
        if ($null -eq $rows -or $rows.Count -eq 0) {
            throw "Empty training log: $logPath"
        }

        $final = $rows[-1]
        $best = ($rows | Where-Object { $_.val_loss -ne "" } | ForEach-Object { [double]$_.val_loss } | Measure-Object -Minimum).Minimum
        [pscustomobject]@{
            seed = $seed
            final_train_loss = "{0:F6}" -f [double]$final.train_loss
            final_val_loss = "{0:F6}" -f [double]$final.val_loss
            best_val_loss = "{0:F6}" -f [double]$best
        }
    }

    $summary | Export-Csv (Join-Path $outputDir $OutputCsv) -NoTypeInformation
    $summary | Format-Table -AutoSize
}

Write-Host ""
Write-Host "PTB-XL KAN grid5"
Write-Summary -Prefix "ptbxl_kan_grid5_s" -OutputCsv "kan_grid5_ptbxl_summary.csv"

Write-Host ""
Write-Host "PTB-XL transformer"
Write-Summary -Prefix "ptbxl_transformer_s" -OutputCsv "transformer_ptbxl_summary.csv"
