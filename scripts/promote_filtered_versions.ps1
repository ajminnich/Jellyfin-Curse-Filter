[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory = $true)]
    [string] $MediaDirectory,

    [int] $ExpectedCount = 0
)

$ErrorActionPreference = 'Stop'
$resolvedDirectory = (Resolve-Path -LiteralPath $MediaDirectory).Path.TrimEnd([IO.Path]::DirectorySeparatorChar)
if (-not (Test-Path -LiteralPath $resolvedDirectory -PathType Container)) {
    throw "Media directory does not exist: $MediaDirectory"
}

$filteredFiles = @(Get-ChildItem -LiteralPath $resolvedDirectory -File -Filter '* - Filtered.mkv' | Sort-Object Name)
if ($ExpectedCount -gt 0 -and $filteredFiles.Count -ne $ExpectedCount) {
    throw "Expected $ExpectedCount filtered files, but found $($filteredFiles.Count)."
}

$operations = foreach ($filteredFile in $filteredFiles) {
    $originalName = $filteredFile.Name -replace ' - Filtered\.mkv$', '.mkv'
    $originalPath = Join-Path $resolvedDirectory $originalName
    $backupPath = "$originalPath.cursefilter-original"
    $sidecarPath = [IO.Path]::ChangeExtension($filteredFile.FullName, '.nfo')

    if (-not (Test-Path -LiteralPath $originalPath -PathType Leaf)) {
        throw "Original media file is missing: $originalPath"
    }
    if (Test-Path -LiteralPath $backupPath) {
        throw "Original-media backup already exists: $backupPath"
    }

    [pscustomobject]@{
        Filtered = $filteredFile.FullName
        Original = $originalPath
        Backup = $backupPath
        Sidecar = $sidecarPath
        SidecarBackup = "$sidecarPath.cursefilter-sidecar"
    }
}

foreach ($operation in $operations) {
    if (-not $PSCmdlet.ShouldProcess($operation.Original, 'Promote filtered media and preserve original')) {
        continue
    }

    Move-Item -LiteralPath $operation.Original -Destination $operation.Backup
    try {
        Move-Item -LiteralPath $operation.Filtered -Destination $operation.Original
        if (Test-Path -LiteralPath $operation.Sidecar) {
            Move-Item -LiteralPath $operation.Sidecar -Destination $operation.SidecarBackup
        }
    }
    catch {
        if (-not (Test-Path -LiteralPath $operation.Original) -and (Test-Path -LiteralPath $operation.Backup)) {
            Move-Item -LiteralPath $operation.Backup -Destination $operation.Original
        }
        throw
    }
}

$operations | Select-Object Original, Backup
