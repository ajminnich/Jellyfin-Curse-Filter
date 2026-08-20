[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string[]]$MediaPath,

    [string]$CacheRoot = 'E:\JellyFin\filtered-audio',
    [string]$FfmpegPath = 'C:\Program Files\Jellyfin\Server\ffmpeg.exe'
)

$ErrorActionPreference = 'Stop'
$filteredTitle = 'Filtered English (Bleep)'
$originalTitle = 'English - Original'
$ffprobePath = Join-Path (Split-Path -Parent $FfmpegPath) 'ffprobe.exe'

if (-not (Test-Path -LiteralPath $FfmpegPath -PathType Leaf) -or
    -not (Test-Path -LiteralPath $ffprobePath -PathType Leaf)) {
    throw 'Jellyfin FFmpeg and FFprobe are required.'
}

function Get-MediaProbe {
    param([Parameter(Mandatory)] [string]$Path)

    $json = & $ffprobePath -v error -show_entries `
        'stream=index,codec_type,codec_name,pix_fmt,width,height,bit_rate,channels,channel_layout:stream_tags=title,language:stream_disposition=default,forced,hearing_impaired:format=duration,size' `
        -of json $Path
    if ($LASTEXITCODE -ne 0) {
        throw "FFprobe failed for $Path"
    }
    return $json | ConvertFrom-Json
}

function Test-PlaybackSafe {
    param([Parameter(Mandatory)] $Probe)

    $video = @($Probe.streams | Where-Object codec_type -eq 'video')
    $audio = @($Probe.streams | Where-Object codec_type -eq 'audio')
    $subtitles = @($Probe.streams | Where-Object codec_type -eq 'subtitle')
    return (
        $video.Count -eq 1 -and
        $audio.Count -eq 2 -and
        $subtitles.Count -eq 0 -and
        $audio[0].codec_name -eq 'eac3' -and
        $audio[0].channels -eq 6 -and
        $audio[0].channel_layout -eq '5.1(side)' -and
        $audio[0].tags.title -eq $filteredTitle -and
        $audio[0].disposition.default -eq 1
    )
}

function Assert-PlaybackSafe {
    param(
        [Parameter(Mandatory)] $OriginalProbe,
        [Parameter(Mandatory)] $OutputProbe,
        [Parameter(Mandatory)] [string]$Path
    )

    if (-not (Test-PlaybackSafe -Probe $OutputProbe)) {
        throw "Playback-safe stream validation failed for $Path"
    }

    $originalVideo = @($OriginalProbe.streams | Where-Object codec_type -eq 'video')[0]
    $outputVideo = @($OutputProbe.streams | Where-Object codec_type -eq 'video')[0]
    if ($originalVideo.codec_name -ne $outputVideo.codec_name -or
        $originalVideo.pix_fmt -ne $outputVideo.pix_fmt -or
        $originalVideo.width -ne $outputVideo.width -or
        $originalVideo.height -ne $outputVideo.height) {
        throw "Video-copy validation failed for $Path"
    }

    $originalAudio = @($OriginalProbe.streams | Where-Object codec_type -eq 'audio')[0]
    $outputOriginalAudio = @($OutputProbe.streams | Where-Object codec_type -eq 'audio')[1]
    if ($originalAudio.codec_name -ne $outputOriginalAudio.codec_name -or
        $originalAudio.channels -ne $outputOriginalAudio.channels -or
        $originalAudio.channel_layout -ne $outputOriginalAudio.channel_layout) {
        throw "Original-audio validation failed for $Path"
    }

    $originalDuration = [double]$OriginalProbe.format.duration
    $outputDuration = [double]$OutputProbe.format.duration
    if ([math]::Abs($originalDuration - $outputDuration) -gt 0.1) {
        throw "Duration validation failed for $Path"
    }
}

function Find-FilteredAudioSource {
    param(
        [Parameter(Mandatory)] [string]$ActivePath,
        [Parameter(Mandatory)] [string]$Stem,
        [Parameter(Mandatory)] $ActiveProbe
    )

    $cacheMatches = @(
        Get-ChildItem -LiteralPath $CacheRoot -File -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Name.StartsWith($Stem + '.', [System.StringComparison]::OrdinalIgnoreCase) -and
                $_.Name.EndsWith('.filtered.eng.mka', [System.StringComparison]::OrdinalIgnoreCase)
            }
    )
    if ($cacheMatches.Count -gt 1) {
        throw "Multiple filtered-audio caches match $Stem"
    }
    if ($cacheMatches.Count -eq 1) {
        return $cacheMatches[0].FullName
    }

    $activeAudio = @($ActiveProbe.streams | Where-Object codec_type -eq 'audio')
    if ($activeAudio.Count -ge 1 -and $activeAudio[0].tags.title -eq $filteredTitle) {
        return $ActivePath
    }
    throw "No filtered-audio source found for $ActivePath"
}

function Find-EnglishFullSubtitle {
    param([Parameter(Mandatory)] $OriginalProbe)

    $english = @(
        $OriginalProbe.streams |
            Where-Object {
                $_.codec_type -eq 'subtitle' -and
                $_.tags.language -eq 'eng' -and
                $_.disposition.forced -ne 1 -and
                $_.tags.title -notmatch '(?i)forced'
            }
    )
    if ($english.Count -eq 0) {
        throw 'No full English subtitle stream found.'
    }

    $standard = @(
        $english |
            Where-Object {
                $_.disposition.hearing_impaired -ne 1 -and
                $_.tags.title -notmatch '(?i)sdh'
            }
    )
    if ($standard.Count -gt 0) {
        return $standard[0]
    }
    return $english[0]
}

function Install-EnglishSubtitle {
    param(
        [Parameter(Mandatory)] [string]$OriginalPath,
        [Parameter(Mandatory)] $OriginalProbe,
        [Parameter(Mandatory)] [string]$SubtitlePath
    )

    if (Test-Path -LiteralPath $SubtitlePath -PathType Leaf) {
        return 'already-present'
    }

    $subtitle = Find-EnglishFullSubtitle -OriginalProbe $OriginalProbe
    $temporarySubtitle = [System.IO.Path]::ChangeExtension($SubtitlePath, '.partial.srt')
    if (Test-Path -LiteralPath $temporarySubtitle) {
        Remove-Item -LiteralPath $temporarySubtitle -Force
    }
    try {
        & $FfmpegPath -hide_banner -loglevel error -i $OriginalPath `
            -map "0:$($subtitle.index)" -c:s copy $temporarySubtitle
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $temporarySubtitle -PathType Leaf)) {
            throw "English subtitle extraction failed for $OriginalPath"
        }
        $cueCount = @(Get-Content -LiteralPath $temporarySubtitle | Where-Object { $_ -match '^\d+$' }).Count
        if ($cueCount -lt 10) {
            throw "Extracted subtitle has too few cues for $OriginalPath"
        }
        Move-Item -LiteralPath $temporarySubtitle -Destination $SubtitlePath
        return "extracted-$cueCount-cues"
    }
    finally {
        if (Test-Path -LiteralPath $temporarySubtitle) {
            Remove-Item -LiteralPath $temporarySubtitle -Force
        }
    }
}

$results = [System.Collections.Generic.List[object]]::new()
foreach ($requestedPath in $MediaPath) {
    $activePath = [System.IO.Path]::GetFullPath($requestedPath)
    if (-not (Test-Path -LiteralPath $activePath -PathType Leaf)) {
        throw "Media file not found: $activePath"
    }

    $originalPath = $activePath + '.cursefilter-original'
    if (-not (Test-Path -LiteralPath $originalPath -PathType Leaf)) {
        throw "Preserved original is missing: $originalPath"
    }

    $directory = Split-Path -Parent $activePath
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($activePath)
    $subtitlePath = Join-Path $directory ($stem + '.eng.srt')
    $activeProbe = Get-MediaProbe -Path $activePath
    $originalProbe = Get-MediaProbe -Path $originalPath
    $mediaStatus = 'already-safe'

    if (-not (Test-PlaybackSafe -Probe $activeProbe)) {
        $filteredSource = Find-FilteredAudioSource `
            -ActivePath $activePath -Stem $stem -ActiveProbe $activeProbe
        $temporaryPath = Join-Path $directory `
            ('.' + $stem + '.' + [guid]::NewGuid().ToString('N') + '.playback-safe.partial.mkv')
        try {
            & $FfmpegPath -hide_banner -loglevel warning -y `
                -i $originalPath -i $filteredSource `
                -map 0:v:0 -map 1:a:0 -map 0:a:0 `
                -map_metadata 0 -map_chapters 0 `
                -c copy -c:a:0 eac3 -b:a:0 256k `
                -metadata:s:a:0 'language=eng' `
                -metadata:s:a:0 "title=$filteredTitle" `
                -metadata:s:a:1 'language=eng' `
                -metadata:s:a:1 "title=$originalTitle" `
                -disposition:a 0 -disposition:a:0 default `
                -max_interleave_delta 100000 `
                -max_muxing_queue_size 4096 `
                $temporaryPath
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $temporaryPath -PathType Leaf)) {
                throw "Playback-safe rebuild failed for $activePath"
            }

            $outputProbe = Get-MediaProbe -Path $temporaryPath
            Assert-PlaybackSafe -OriginalProbe $originalProbe -OutputProbe $outputProbe -Path $temporaryPath
            Move-Item -LiteralPath $temporaryPath -Destination $activePath -Force
            $installedProbe = Get-MediaProbe -Path $activePath
            Assert-PlaybackSafe -OriginalProbe $originalProbe -OutputProbe $installedProbe -Path $activePath
            $mediaStatus = 'rebuilt'
        }
        finally {
            if (Test-Path -LiteralPath $temporaryPath -PathType Leaf) {
                Remove-Item -LiteralPath $temporaryPath -Force
            }
        }
    }

    $subtitleStatus = Install-EnglishSubtitle `
        -OriginalPath $originalPath `
        -OriginalProbe $originalProbe `
        -SubtitlePath $subtitlePath

    $installed = Get-MediaProbe -Path $activePath
    $audio = @($installed.streams | Where-Object codec_type -eq 'audio')
    $results.Add([pscustomobject]@{
            File = $activePath
            MediaStatus = $mediaStatus
            SubtitleStatus = $subtitleStatus
            FilteredCodec = $audio[0].codec_name
            FilteredLayout = $audio[0].channel_layout
            EmbeddedSubtitles = @($installed.streams | Where-Object codec_type -eq 'subtitle').Count
            Duration = [double]$installed.format.duration
        })
}

$results | ConvertTo-Json -Depth 3
