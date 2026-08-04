# Jellyfin Curse Filter

> **Experimental and destructive-to-library layout:** this plugin creates a
> filtered playback copy, promotes it to the normal media filename, and keeps
> the original beside it with `.cursefilter-original` appended. Test with a
> small disposable library before using it on an irreplaceable collection.

[Installation and updates](INSTALLATION.md) | [Plugin details](plugin/Jellyfin.Plugin.CurseFilter/README.md)

This project builds a Jellyfin server plugin and a GPU-assisted preprocessing
pipeline that creates default English audio tracks with profanity replaced by a
bleep. It supports both caption-guided alignment and caption-free Whisper audio
analysis. Original media files are preserved as reversible backup files.

In `auto` mode the pipeline uses embedded English captions when available, then
uses Wav2Vec2 CTC forced alignment on the NVIDIA GPU to narrow each censor
interval to the spoken word. When captions are unavailable, Whisper `small.en`
transcribes the first audio track with word timestamps and finds profanity
directly from the audio. Results are cached in JSON before FFmpeg creates a
separate audio file. For Jellyfin playback, the plugin uses the NVIDIA GPU to
create an H.264 High/yuv420p Matroska compatibility version. After validation,
that copy takes the item's normal `.mkv` filename so Jellyfin shows one item.
The untouched source is retained with `.cursefilter-original` appended, an extension
Jellyfin ignores. Filtered audio is first/default, and the original audio remains
selectable as the next track. This avoids green-screen failures on clients whose
10-bit HEVC decoder is incompatible with the source video.

```text
work/filtered-audio/Episode.<path-hash>.filtered.eng.mka
```

The separate filtered audio file is retained privately and is not exposed as a
Jellyfin sidecar. The unmodified source video remains beside the promoted copy
under the ignored backup extension.

## Local pipeline

The checked-in defaults are:

- Automatic caption-guided or caption-free English analysis
- GPU analysis (`cuda`)
- `facebook/wav2vec2-base-960h` alignment model
- `small.en` Whisper transcription model
- 1,000 Hz bleep
- 40 ms timing padding
- NVIDIA H.264 High/yuv420p compatibility video
- AAC-in-Matroska filtered audio, retained separately and embedded for playback

Run `scripts/process_wrexham.ps1` after installing the analysis dependencies.

Caption-free command-line use does not require a subtitle argument:

```powershell
.\.venv\Scripts\python.exe scripts\process_media.py "C:\Movies\Movie.mp4" --mode audio
```

The Jellyfin task processes both `C:\TV Shows` and `C:\Movies` by default. Set
`AnalysisMode` to `auto` (the default), `captions`, or `audio` in the plugin XML
configuration. Non-Matroska movie sources are converted to a promoted `.mkv`;
the untouched source keeps its original extension plus `.cursefilter-original`.

## Publishing a release

The Jellyfin repository feed is [repository.json](repository.json). It is
served from GitHub after the repository is published. Build a release archive
and refresh the feed entry with:

```powershell
.\scripts\build-plugin-release.ps1
```

Then commit the generated `repository.json`, create a matching Git tag, and
push the tag. The GitHub release workflow uploads the generated archive. Full
steps are in [INSTALLATION.md](INSTALLATION.md#publishing-updates).
