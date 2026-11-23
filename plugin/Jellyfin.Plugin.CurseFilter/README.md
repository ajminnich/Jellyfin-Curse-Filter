# Jellyfin Curse Filter plugin

The plugin adds the scheduled task **Create profanity-filtered audio**. The
local configuration is enabled, uses CUDA, and processes every movie or episode
under `E:\TV Shows` and `E:\Movies`. The startup run begins after ten seconds
and scans those roots directly, including files not yet indexed by Jellyfin.
Existing filtered tracks and current clean-analysis markers are skipped. In
the default `auto` mode, the task first checks for a matching `.srt` beside the
video, canonicalizing a nonstandard same-folder name as `<video>.eng.srt` for
Jellyfin. It then prefers a full non-forced embedded English track (regular
English before SDH) and uses forced alignment. If no usable subtitles exist,
it uses Whisper `small.en` word timestamps to analyze the audio directly.
`AnalysisMode` can also force `captions` or `audio`.

Generated filtered audio is retained separately in the project's private cache:

```text
work/filtered-audio/Video.<path-hash>.filtered.eng.mka
```

For playback, the plugin creates and verifies an H.264 High/yuv420p Matroska
compatibility version with the filtered track first/default, then promotes that
file to the item's normal `.mkv` filename. The untouched source is retained
with `.cursefilter-original` appended, which Jellyfin ignores. Original audio,
subtitles, attachments, chapters, and metadata remain available in the promoted
copy. MP4 and other supported movie sources are promoted as `.mkv` while the
original keeps its source extension plus `.cursefilter-original`. NVIDIA NVENC
handles the video conversion and avoids the 10-bit HEVC green-screen failure on
incompatible clients.
