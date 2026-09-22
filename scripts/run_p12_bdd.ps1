[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BddRoot,

    [ValidateSet("dev", "confirmation")]
    [string]$Stage = "dev",

    [ValidateSet("from_scratch", "pretrained", "warm")]
    [string]$Initialization = "from_scratch",

    [string]$Checkpoint,
    [string]$Well,

    [ValidateSet("cpu", "cuda", "hpu")]
    [string]$Device = "cpu",

    [double]$LearningRate,

    [ValidateSet(1, 4)]
    [int]$UpdateEvery,

    [int]$MaxTrainPerRegime = 64,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$entryPoint = Join-Path $repoRoot "scripts\run_adaptation_experiment.py"
$resolvedBdd = [System.IO.Path]::GetFullPath($BddRoot)
$images = Join-Path $resolvedBdd "images\100k\val"
$legacyLabels = Join-Path $resolvedBdd "labels\bdd100k_labels_images_val.json"
$modernLabels = Join-Path $resolvedBdd "labels\det_20\det_val.json"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Python was not found at $python. Run 'uv sync --group analysis' first."
}
if (-not (Test-Path -LiteralPath $images -PathType Container)) {
    throw "BDD validation images were not found at $images."
}
if (-not (Test-Path -LiteralPath $legacyLabels -PathType Leaf) -and
    -not (Test-Path -LiteralPath $modernLabels -PathType Leaf)) {
    throw "BDD validation labels were not found. Expected $legacyLabels or $modernLabels."
}
if ($MaxTrainPerRegime -lt 1) {
    throw "MaxTrainPerRegime must be at least 1."
}

$initArgs = switch ($Initialization) {
    "from_scratch" { @("init=from_scratch", "well=null") }
    "pretrained" {
        if (-not $Checkpoint) {
            throw "-Checkpoint is required when -Initialization pretrained is selected."
        }
        $resolvedCheckpoint = [System.IO.Path]::GetFullPath($Checkpoint)
        if (-not (Test-Path -LiteralPath $resolvedCheckpoint -PathType Leaf)) {
            throw "Pretrained encoder checkpoint was not found at $resolvedCheckpoint."
        }
        @("init=pretrained", "init.checkpoint=$($resolvedCheckpoint.Replace('\', '/'))", "well=null")
    }
    "warm" {
        if (-not $Well) {
            throw "-Well is required when -Initialization warm is selected."
        }
        $resolvedWell = [System.IO.Path]::GetFullPath($Well)
        if (-not (Test-Path -LiteralPath $resolvedWell -PathType Leaf)) {
            throw "Warm method checkpoint was not found at $resolvedWell."
        }
        @("init=from_scratch", "well=$($resolvedWell.Replace('\', '/'))")
    }
}

if ($Stage -eq "dev") {
    $modeArgs = @("--multirun")
    $stageArgs = @("optim.lr=0.0001,0.0003,0.001", "update_every=1,4")
} else {
    $modeArgs = @()
    if (-not $PSBoundParameters.ContainsKey("LearningRate")) {
        throw "-LearningRate is required for confirmation; use the cell chosen on dev seeds."
    }
    if (-not $PSBoundParameters.ContainsKey("UpdateEvery")) {
        throw "-UpdateEvery is required for confirmation; use the cell chosen on dev seeds."
    }
    $stageArgs = @(
        "seeds=[101,103,107,109,113]",
        "seed_split=confirmation",
        "optim.lr=$LearningRate",
        "update_every=$UpdateEvery"
    )
}

$runArgs = @(
    $entryPoint,
    "--config-name", "adaptation_bdd"
) + $modeArgs + @(
    "bdd_root=$($resolvedBdd.Replace('\', '/'))",
    "device=$Device",
    "max_train_per_regime=$MaxTrainPerRegime"
) + $initArgs + $stageArgs

$display = "& `"$python`" " + (($runArgs | ForEach-Object { "`"$_`"" }) -join " ")
Write-Host "BDD layout verified: $resolvedBdd"
Write-Host "Stage: $Stage; initialization: $Initialization; device: $Device"
Write-Host $display
if ($DryRun) {
    return
}

Push-Location $repoRoot
try {
    & $python @runArgs
    if ($LASTEXITCODE -ne 0) {
        throw "P1.2 runner exited with code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
