[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string[]]$MediaPath,

    [string]$FfmpegPath = 'C:\Program Files\Jellyfin\Server\ffmpeg.exe'
)

$ErrorActionPreference = 'Stop'
$filteredTitle = 'Filtered English (Bleep)'
$ffprobePath = Join-Path (Split-Path -Parent $FfmpegPath) 'ffprobe.exe'
# Existing PCE-encoded AAC is decoded as six unnamed channels. Its samples are
# already in FFmpeg's native six-channel order, so map by position and assign
# the canonical AAC 5.1 labels explicitly.
$layoutFilter = 'channelmap=map=0-FL|1-FR|2-FC|3-LFE|4-BL|5-BR:channel_layout=5.1'

if (-not (Test-Path -LiteralPath $FfmpegPath -PathType Leaf) -or
    -not (Test-Path -LiteralPath $ffprobePath -PathType Leaf)) {
    throw 'Jellyfin FFmpeg and FFprobe are required.'
}

function Get-MediaProbe {
    param([Parameter(Mandatory)] [string]$Path)

    $json = & $ffprobePath -v error -show_entries `
        'stream=index,codec_type,codec_name,profile,pix_fmt,width,height,channels,channel_layout,extradata_size:stream_tags=title,language:stream_disposition=default:format=duration' `
        -of json $Path
    if ($LASTEXITCODE -ne 0) {
        throw "FFprobe failed for $Path"
    }
    return $json | ConvertFrom-Json
}

function Assert-RepairedMedia {
    param(
        [Parameter(Mandatory)] $SourceProbe,
        [Parameter(Mandatory)] $OutputProbe,
        [Parameter(Mandatory)] [string]$Path
    )

    $sourceStreams = @($SourceProbe.streams)
    $outputStreams = @($OutputProbe.streams)
    $sourceAudio = @($sourceStreams | Where-Object codec_type -eq 'audio')
    $outputAudio = @($outputStreams | Where-Object codec_type -eq 'audio')
    $sourceVideo = @($sourceStreams | Where-Object codec_type -eq 'video')
    $outputVideo = @($outputStreams | Where-Object codec_type -eq 'video')

    if ($sourceStreams.Count -ne $outputStreams.Count -or
        $sourceAudio.Count -ne $outputAudio.Count -or
        $sourceVideo.Count -ne $outputVideo.Count) {
        throw "Stream-count validation failed for $Path"
    }
    for ($index = 0; $index -lt $sourceStreams.Count; $index++) {
        if ($sourceStreams[$index].codec_type -ne $outputStreams[$index].codec_type) {
            throw "Stream-order validation failed for $Path at index $index"
        }
    }
    for ($index = 0; $index -lt $sourceVideo.Count; $index++) {
        if ($sourceVideo[$index].codec_name -ne $outputVideo[$index].codec_name -or
            $sourceVideo[$index].pix_fmt -ne $outputVideo[$index].pix_fmt -or
            $sourceVideo[$index].width -ne $outputVideo[$index].width -or
            $sourceVideo[$index].height -ne $outputVideo[$index].height) {
            throw "Video-copy validation failed for $Path"
        }
    }
    for ($index = 1; $index -lt $sourceAudio.Count; $index++) {
        if ($sourceAudio[$index].codec_name -ne $outputAudio[$index].codec_name -or
            $sourceAudio[$index].channels -ne $outputAudio[$index].channels -or
            $sourceAudio[$index].channel_layout -ne $outputAudio[$index].channel_layout) {
            throw "Original-audio validation failed for $Path"
        }
    }

    $filtered = $outputAudio[0]
    if ($filtered.codec_name -ne 'aac' -or
        $filtered.profile -notin @('LC', $null) -or
        $filtered.channels -ne 6 -or
        $filtered.channel_layout -ne '5.1' -or
        $filtered.extradata_size -ne 5 -or
        $filtered.tags.title -ne $filteredTitle -or
        $filtered.disposition.default -ne 1) {
        $actual = [pscustomobject]@{
            codec = $filtered.codec_name
            profile = $filtered.profile
            channels = $filtered.channels
            layout = $filtered.channel_layout
            extradataSize = $filtered.extradata_size
            title = $filtered.tags.title
            default = $filtered.disposition.default
        } | ConvertTo-Json -Compress
        throw "Canonical filtered-audio validation failed for ${Path}: $actual"
    }

    $sourceDuration = [double]$SourceProbe.format.duration
    $outputDuration = [double]$OutputProbe.format.duration
    if ([math]::Abs($sourceDuration - $outputDuration) -gt 0.1) {
        throw "Duration validation failed for $Path"
    }
}

$results = [System.Collections.Generic.List[object]]::new()
foreach ($requestedPath in $MediaPath) {
    $sourcePath = [System.IO.Path]::GetFullPath($requestedPath)
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "Media file not found: $sourcePath"
    }

    $directory = Split-Path -Parent $sourcePath
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($sourcePath)
    $temporaryPath = Join-Path $directory ".$stem.$([guid]::NewGuid().ToString('N')).aac-layout.partial.mkv"
    $rollbackPath = $sourcePath + '.aac-layout-rollback'
    if (Test-Path -LiteralPath $rollbackPath) {
        throw "Refusing to overwrite an existing rollback file: $rollbackPath"
    }

    $sourceProbe = Get-MediaProbe -Path $sourcePath
    $sourceAudio = @($sourceProbe.streams | Where-Object codec_type -eq 'audio')
    if ($sourceAudio.Count -lt 2 -or $sourceAudio[0].tags.title -ne $filteredTitle) {
        throw "Source is not a validated filtered media file: $sourcePath"
    }
    if ($sourceAudio[0].channel_layout -eq '5.1' -and $sourceAudio[0].extradata_size -eq 5) {
        $results.Add([pscustomobject]@{ File = $sourcePath; Status = 'already-canonical' })
        continue
    }
    if ($sourceAudio[0].codec_name -ne 'aac' -or $sourceAudio[0].channels -ne 6) {
        throw "Filtered audio is not repairable six-channel AAC: $sourcePath"
    }

    try {
        & $FfmpegPath -hide_banner -loglevel warning -y `
            -i $sourcePath `
            -map 0 -map_metadata 0 -map_chapters 0 `
            -c copy `
            -filter:a:0 $layoutFilter -c:a:0 aac -profile:a:0 aac_low -b:a:0 320k `
            -metadata:s:a:0 'language=eng' `
            -metadata:s:a:0 "title=$filteredTitle" `
            -disposition:a:0 default `
            -max_muxing_queue_size 2048 `
            $temporaryPath
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $temporaryPath -PathType Leaf)) {
            throw "AAC layout repair failed for $sourcePath"
        }

        $outputProbe = Get-MediaProbe -Path $temporaryPath
        Assert-RepairedMedia -SourceProbe $sourceProbe -OutputProbe $outputProbe -Path $temporaryPath

        Move-Item -LiteralPath $sourcePath -Destination $rollbackPath
        try {
            Move-Item -LiteralPath $temporaryPath -Destination $sourcePath
            $installedProbe = Get-MediaProbe -Path $sourcePath
            Assert-RepairedMedia -SourceProbe $sourceProbe -OutputProbe $installedProbe -Path $sourcePath
        }
        catch {
            if (Test-Path -LiteralPath $sourcePath -PathType Leaf) {
                Remove-Item -LiteralPath $sourcePath -Force
            }
            Move-Item -LiteralPath $rollbackPath -Destination $sourcePath
            throw
        }
        Remove-Item -LiteralPath $rollbackPath -Force

        $results.Add([pscustomobject]@{
                File = $sourcePath
                Status = 'repaired'
                FilteredCodec = $outputProbe.streams[1].codec_name
                FilteredProfile = $outputProbe.streams[1].profile
                FilteredLayout = $outputProbe.streams[1].channel_layout
                Duration = [double]$outputProbe.format.duration
            })
    }
    finally {
        if (Test-Path -LiteralPath $temporaryPath -PathType Leaf) {
            Remove-Item -LiteralPath $temporaryPath -Force
        }
    }
}

$results | ConvertTo-Json -Depth 3
