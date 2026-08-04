$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$mediaRoot = Join-Path $projectRoot 'Test Shows'
$captionRoot = Join-Path $projectRoot 'work\captions'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Analysis environment not found at $python"
}

Get-ChildItem -LiteralPath $mediaRoot -File -Filter 'Welcome.to.Wrexham*.mkv' | ForEach-Object {
    $captions = Join-Path $captionRoot ($_.BaseName + '.eng.srt')
    $output = Join-Path $_.DirectoryName ($_.BaseName + '.default.filtered.eng.mka')
    & $python (Join-Path $projectRoot 'scripts\process_media.py') $_.FullName $captions --output $output
    if ($LASTEXITCODE -ne 0) {
        throw "Filtering failed for $($_.Name)"
    }
}

