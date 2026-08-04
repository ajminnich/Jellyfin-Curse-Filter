[CmdletBinding()]
param(
    [string]$MediaRoot = 'C:\TV Shows\Welcome.to.Wrexham.S04.1080p.WEBRip.x265-KONTRAST',
    [string]$CacheRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'work\filtered-audio'),
    [string]$FfmpegPath = 'C:\Program Files\Jellyfin\Server\ffmpeg.exe',
    [string]$FilePattern = '*.mkv'
)

$ErrorActionPreference = 'Stop'
$filteredTitle = 'Filtered English (Bleep)'
$ffprobePath = Join-Path (Split-Path -Parent $FfmpegPath) 'ffprobe.exe'
$resolvedMediaRoot = [System.IO.Path]::GetFullPath($MediaRoot).TrimEnd('\')

if (-not (Test-Path -LiteralPath $resolvedMediaRoot -PathType Container)) {
    throw "Media root not found: $resolvedMediaRoot"
}
if (-not (Test-Path -LiteralPath $FfmpegPath -PathType Leaf) -or
    -not (Test-Path -LiteralPath $ffprobePath -PathType Leaf)) {
    throw 'Jellyfin FFmpeg and FFprobe are required.'
}

New-Item -ItemType Directory -Path $CacheRoot -Force | Out-Null
$results = [System.Collections.Generic.List[object]]::new()

function Activate-FilteredVersion {
    param(
        [Parameter(Mandatory)] [string]$OriginalPath,
        [Parameter(Mandatory)] [string]$FilteredPath
    )

    $backupPath = $OriginalPath + '.cursefilter-original'
    if (Test-Path -LiteralPath $backupPath) {
        throw "Refusing to replace an existing original-media backup: $backupPath"
    }

    Move-Item -LiteralPath $OriginalPath -Destination $backupPath
    try {
        Move-Item -LiteralPath $FilteredPath -Destination $OriginalPath
    }
    catch {
        Move-Item -LiteralPath $backupPath -Destination $OriginalPath
        throw
    }

    return $backupPath
}

Get-ChildItem -LiteralPath $resolvedMediaRoot -File -Filter $FilePattern |
    Where-Object { $_.Name -notlike '*.partial.mkv' -and $_.BaseName -notlike '* - Filtered' } |
    Sort-Object Name |
    ForEach-Object {
        $media = $_.FullName
        $sidecar = Join-Path $_.DirectoryName ($_.BaseName + '.default.filtered.eng.mka')
        $cache = Join-Path $CacheRoot ($_.BaseName + '.filtered.eng.mka')
        $playback = Join-Path $_.DirectoryName ($_.BaseName + ' - Filtered.mkv')
        $temp = Join-Path $_.DirectoryName ('.' + $_.BaseName + '.cursefilter.partial.mkv')

        if (-not ([System.IO.Path]::GetFullPath($temp).StartsWith(
                    $resolvedMediaRoot + '\',
                    [System.StringComparison]::OrdinalIgnoreCase))) {
            throw "Refusing to write outside the media root: $temp"
        }

        $activeTitle = (& $ffprobePath -v error -select_streams a:0 `
                -show_entries stream_tags=title `
                -of 'default=noprint_wrappers=1:nokey=1' $media).Trim()
        if ($activeTitle -eq $filteredTitle) {
            if (Test-Path -LiteralPath $sidecar -PathType Leaf) {
                Move-Item -LiteralPath $sidecar -Destination $cache -Force
            }
            $results.Add([pscustomobject]@{ File = $_.Name; Status = 'already-active' })
            return
        }

        if (Test-Path -LiteralPath $playback -PathType Leaf) {
            $firstTitle = (& $ffprobePath -v error -select_streams a:0 `
                    -show_entries stream_tags=title `
                    -of 'default=noprint_wrappers=1:nokey=1' $playback).Trim()
        }
        else {
            $firstTitle = ''
        }
        if ($firstTitle -eq $filteredTitle) {
            if (Test-Path -LiteralPath $sidecar -PathType Leaf) {
                Move-Item -LiteralPath $sidecar -Destination $cache -Force
            }
            $backup = Activate-FilteredVersion -OriginalPath $media -FilteredPath $playback
            $results.Add([pscustomobject]@{
                    File = $_.Name
                    Status = 'activated'
                    OriginalBackup = [System.IO.Path]::GetFileName($backup)
                })
            return
        }

        if (Test-Path -LiteralPath $playback -PathType Leaf) {
            throw "Refusing to replace existing media file: $playback"
        }

        if (-not (Test-Path -LiteralPath $sidecar -PathType Leaf)) {
            $results.Add([pscustomobject]@{ File = $_.Name; Status = 'no-filtered-audio' })
            return
        }

        Copy-Item -LiteralPath $sidecar -Destination $cache -Force
        if (Test-Path -LiteralPath $temp -PathType Leaf) {
            Remove-Item -LiteralPath $temp -Force
        }

        try {
            & $FfmpegPath -hide_banner -loglevel error -y `
                -i $media -i $sidecar `
                -map '0:v?' -map '1:a:0' -map '0:a?' -map '0:s?' -map '0:t?' `
                -map_metadata 0 -map_chapters 0 -c copy `
                -metadata:s:a:0 language=eng `
                -metadata:s:a:0 "title=$filteredTitle" `
                -disposition:a 0 -disposition:a:0 default `
                $temp
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $temp -PathType Leaf)) {
                throw "FFmpeg remux failed for $($_.Name)"
            }

            $newTitle = (& $ffprobePath -v error -select_streams a:0 `
                    -show_entries stream_tags=title `
                    -of 'default=noprint_wrappers=1:nokey=1' $temp).Trim()
            $sourceDuration = [double](& $ffprobePath -v error `
                    -show_entries format=duration `
                    -of 'default=noprint_wrappers=1:nokey=1' $media)
            $newDuration = [double](& $ffprobePath -v error `
                    -show_entries format=duration `
                    -of 'default=noprint_wrappers=1:nokey=1' $temp)
            $videoCount = [int](& $ffprobePath -v error -select_streams v `
                    -show_entries stream=index -of csv=p=0 $temp |
                    Measure-Object -Line).Lines
            $audioCount = [int](& $ffprobePath -v error -select_streams a `
                    -show_entries stream=index -of csv=p=0 $temp |
                    Measure-Object -Line).Lines
            if ($newTitle -ne $filteredTitle -or $videoCount -lt 1 -or
                $audioCount -lt 2 -or [math]::Abs($sourceDuration - $newDuration) -gt 0.1) {
                throw "Validation failed for remuxed file $($_.Name)"
            }

            Move-Item -LiteralPath $temp -Destination $playback
            Move-Item -LiteralPath $sidecar -Destination $cache -Force
            $backup = Activate-FilteredVersion -OriginalPath $media -FilteredPath $playback
            $results.Add([pscustomobject]@{
                    File = $_.Name
                    Status = 'installed-and-activated'
                    OriginalBackup = [System.IO.Path]::GetFileName($backup)
                    VideoStreams = $videoCount
                    AudioStreams = $audioCount
                    Duration = $newDuration
                })
        }
        finally {
            if (Test-Path -LiteralPath $temp -PathType Leaf) {
                Remove-Item -LiteralPath $temp -Force
            }
        }
    }

$results | ConvertTo-Json -Depth 3
