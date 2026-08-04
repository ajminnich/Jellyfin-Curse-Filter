[CmdletBinding()]
param(
    [string]$Version,
    [string]$Configuration = 'Release',
    [switch]$NoRestore
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pluginProject = Join-Path $projectRoot 'plugin\Jellyfin.Plugin.CurseFilter\Jellyfin.Plugin.CurseFilter.csproj'
$metaPath = Join-Path $projectRoot 'plugin\Jellyfin.Plugin.CurseFilter\meta.json'
$releaseDirectory = Join-Path $projectRoot 'release'
$env:DOTNET_CLI_HOME = Join-Path $projectRoot '.dotnet-cli-home'
$env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = '1'

$meta = Get-Content -Raw $metaPath | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($Version)) {
    $Version = $meta.version
}
if ($Version -ne $meta.version) {
    throw "The requested version ($Version) must match meta.json ($($meta.version))."
}

$publishArguments = @(
    'publish', $pluginProject,
    '--configuration', $Configuration,
    '--output', (Join-Path $releaseDirectory 'plugin'),
    '--nologo'
)
if ($NoRestore) {
    $publishArguments += '--no-restore'
}
& dotnet @publishArguments
if ($LASTEXITCODE -ne 0) {
    throw 'Plugin build failed.'
}

$archiveName = "Jellyfin.Plugin.CurseFilter_$Version.zip"
$archivePath = Join-Path $releaseDirectory $archiveName
if (Test-Path -LiteralPath $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}
Copy-Item -LiteralPath $metaPath -Destination (Join-Path $releaseDirectory 'plugin\meta.json')
Compress-Archive -Path (Join-Path $releaseDirectory 'plugin\*') -DestinationPath $archivePath
$checksum = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()

$repository = @(
    [ordered]@{
        guid = $meta.guid
        name = $meta.name
        overview = $meta.overview
        description = $meta.description
        owner = 'Jellyfin Curse Filter contributors'
        category = $meta.category
        versions = @(
            [ordered]@{
                version = $Version
                changelog = $meta.changelog
                targetAbi = $meta.targetAbi
                sourceUrl = "https://github.com/ajminnich/Jellyfin-Curse-Filter/releases/download/v$Version/$archiveName"
                checksum = $checksum
                timestamp = $meta.timestamp
            }
        )
    }
)
$repository | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $projectRoot 'repository.json') -Encoding utf8

Write-Host "Created $archivePath"
Write-Host "SHA-256: $checksum"
