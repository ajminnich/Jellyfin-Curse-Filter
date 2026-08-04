[CmdletBinding()]
param(
    [string]$MediaRoot = 'C:\TV Shows\Welcome.to.Wrexham.S04.1080p.WEBRip.x265-KONTRAST',
    [string]$FfmpegPath = 'C:\Program Files\Jellyfin\Server\ffmpeg.exe',
    [string]$FilePattern = '* - Filtered.mkv'
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

$results = [System.Collections.Generic.List[object]]::new()
Get-ChildItem -LiteralPath $resolvedMediaRoot -File -Filter $FilePattern |
    Sort-Object Name |
    ForEach-Object {
        $sourcePath = $_.FullName
        $temporaryPath = Join-Path $_.DirectoryName ('.' + $_.BaseName + '.h264.partial.mkv')
        $resolvedTemporaryPath = [System.IO.Path]::GetFullPath($temporaryPath)
        if (-not $resolvedTemporaryPath.StartsWith(
                $resolvedMediaRoot + '\',
                [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to write outside the media root: $temporaryPath"
        }

        $sourceJson = & $ffprobePath -v error -show_entries `
            'stream=index,codec_type,codec_name,channels:stream_tags=title:stream_disposition=default:format=duration' `
            -of json $sourcePath
        $sourceProbe = $sourceJson | ConvertFrom-Json
        $sourceAudio = @($sourceProbe.streams | Where-Object codec_type -eq 'audio')
        $sourceVideo = @($sourceProbe.streams | Where-Object codec_type -eq 'video')
        if ($sourceVideo.Count -ne 1 -or $sourceAudio.Count -lt 2 -or
            $sourceAudio[0].tags.title -ne $filteredTitle) {
            throw "Source is not a validated filtered media version: $sourcePath"
        }
        if ($sourceVideo[0].codec_name -eq 'h264') {
            $results.Add([pscustomobject]@{ File = $_.Name; Status = 'already-h264' })
            return
        }

        if (Test-Path -LiteralPath $temporaryPath -PathType Leaf) {
            Remove-Item -LiteralPath $temporaryPath -Force
        }

        try {
            & $FfmpegPath -hide_banner -loglevel warning -y `
                -hwaccel cuda -hwaccel_output_format cuda -i $sourcePath `
                -map '0:v:0' -map '0:a?' -map '0:s?' -map '0:t?' `
                -map_metadata 0 -map_chapters 0 `
                -vf 'scale_cuda=format=nv12' `
                -c:v h264_nvenc -preset p3 -tune hq -rc vbr -cq 21 -b:v 0 `
                -maxrate 8M -bufsize 16M -spatial-aq 1 -temporal-aq 1 `
                -rc-lookahead 12 -profile:v high -level:v 4.1 `
                -c:a copy -c:s copy -c:t copy `
                -fps_mode passthrough `
                $temporaryPath
            if ($LASTEXITCODE -ne 0 -or
                -not (Test-Path -LiteralPath $temporaryPath -PathType Leaf)) {
                throw "GPU H.264 encode failed for $($_.Name)"
            }

            $outputJson = & $ffprobePath -v error -show_entries `
                'stream=index,codec_type,codec_name,profile,pix_fmt,channels:stream_tags=title:stream_disposition=default:format=duration' `
                -of json $temporaryPath
            $outputProbe = $outputJson | ConvertFrom-Json
            $outputAudio = @($outputProbe.streams | Where-Object codec_type -eq 'audio')
            $outputVideo = @($outputProbe.streams | Where-Object codec_type -eq 'video')
            $sourceDuration = [double]$sourceProbe.format.duration
            $outputDuration = [double]$outputProbe.format.duration
            if ($outputVideo.Count -ne 1 -or $outputVideo[0].codec_name -ne 'h264' -or
                $outputVideo[0].pix_fmt -ne 'yuv420p' -or $outputAudio.Count -lt 2 -or
                $outputAudio[0].tags.title -ne $filteredTitle -or
                $outputAudio[0].disposition.default -ne 1 -or
                [math]::Abs($sourceDuration - $outputDuration) -gt 0.1) {
                throw "Validation failed for H.264 output: $($_.Name)"
            }

            $derivedBackupPath = $sourcePath + '.hevc-backup'
            if (Test-Path -LiteralPath $derivedBackupPath) {
                Remove-Item -LiteralPath $derivedBackupPath -Force
            }
            Move-Item -LiteralPath $sourcePath -Destination $derivedBackupPath
            try {
                Move-Item -LiteralPath $temporaryPath -Destination $sourcePath
            }
            catch {
                Move-Item -LiteralPath $derivedBackupPath -Destination $sourcePath
                throw
            }
            Remove-Item -LiteralPath $derivedBackupPath -Force
            $results.Add([pscustomobject]@{
                    File = $_.Name
                    Status = 'converted'
                    Codec = $outputVideo[0].codec_name
                    Profile = $outputVideo[0].profile
                    PixelFormat = $outputVideo[0].pix_fmt
                    AudioTracks = $outputAudio.Count
                    Duration = $outputDuration
                })
        }
        finally {
            if (Test-Path -LiteralPath $temporaryPath -PathType Leaf) {
                Remove-Item -LiteralPath $temporaryPath -Force
            }
        }
    }

$results | ConvertTo-Json -Depth 3
