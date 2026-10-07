param(
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$outerRoot = (Resolve-Path -LiteralPath (Join-Path $projectRoot "..")).Path
if (-not $OutputDirectory) { $OutputDirectory = $outerRoot }
$resolvedOutput = (Resolve-Path -LiteralPath $OutputDirectory).Path
$date = Get-Date -Format "yyyyMMdd"
$archive = Join-Path $resolvedOutput "travel-agent-platform-shareable-$date.zip"
$stageParent = Join-Path $resolvedOutput ".share-package-staging-$date"
$stage = Join-Path $stageParent "travel-agent-platform"
if (Test-Path -LiteralPath $archive) { throw "Output archive already exists: $archive" }
if (Test-Path -LiteralPath $stageParent) { throw "Staging directory already exists: $stageParent" }

function Copy-Tree {
    param([string]$RelativePath, [string[]]$ExcludeDirectories = @(), [string[]]$ExcludeFiles = @())
    $source = Join-Path $projectRoot $RelativePath
    $target = Join-Path $stage $RelativePath
    $arguments = @($source, $target, "/E", "/R:1", "/W:1", "/NFL", "/NDL", "/NJH", "/NJS", "/NP")
    if ($ExcludeDirectories.Count) { $arguments += "/XD"; $arguments += $ExcludeDirectories }
    if ($ExcludeFiles.Count) { $arguments += "/XF"; $arguments += $ExcludeFiles }
    & robocopy.exe @arguments | Out-Null
    if ($LASTEXITCODE -gt 7) { throw "Copy failed for $RelativePath (robocopy $LASTEXITCODE)" }
}

New-Item -ItemType Directory -Path $stage -Force | Out-Null
try {
    foreach ($name in @("README.md", "SHARE_README.md", ".dockerignore", ".gitignore", "docker-compose.yml", "PRODUCT.md")) {
        Copy-Item -LiteralPath (Join-Path $projectRoot $name) -Destination $stage
    }
    foreach ($name in @("pyproject.toml", "uv.lock", ".env.example", "sitecustomize.py")) {
        $backendTarget = Join-Path $stage "backend"
        New-Item -ItemType Directory -Path $backendTarget -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $projectRoot "backend\$name") -Destination $backendTarget
    }
    Copy-Tree "backend\app" @(
        "__pycache__",
        (Join-Path $projectRoot "backend\app\static\place_images\cache")
    ) @("*.pyc")
    Copy-Tree "backend\tests" @("__pycache__", ".pytest_cache") @("*.pyc")
    Copy-Tree "backend\scripts" @("__pycache__") @("*.pyc")
    Copy-Tree "frontend" @("node_modules", "dist", ".npm-cache") @("test-results.log", "*.tsbuildinfo")
    Copy-Tree "data\paddleocr"
    Copy-Tree "data\paddlex"
    Copy-Tree "deploy"
    Copy-Tree "scripts" @("__pycache__") @("*.pyc")
    $docsTarget = Join-Path $stage "docs"
    New-Item -ItemType Directory -Path $docsTarget -Force | Out-Null
    foreach ($name in @("architecture.md", "demo-script.md")) {
        Copy-Item -LiteralPath (Join-Path $projectRoot "docs\$name") -Destination $docsTarget
    }

    # A real but empty file lets Docker bind-mount the database path. Startup creates
    # the schema; no accounts, trips, places, or uploads are copied from this machine.
    New-Item -ItemType File -Path (Join-Path $stage "backend\travel_agent.db") -Force | Out-Null
    tar.exe -a -cf $archive -C $stageParent "travel-agent-platform"
    if ($LASTEXITCODE -ne 0) { throw "Archive creation failed" }
    Write-Output $archive
}
finally {
    if (Test-Path -LiteralPath $stageParent) {
        $resolvedStage = (Resolve-Path -LiteralPath $stageParent).Path
        $safePrefix = $resolvedOutput.TrimEnd('\') + '\'
        if (-not $resolvedStage.StartsWith($safePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove staging outside output directory: $resolvedStage"
        }
        Remove-Item -LiteralPath $resolvedStage -Recurse -Force
    }
}
