# Installation

## Before you begin

Curse Filter is a Windows-focused Jellyfin server plugin. It needs:

- Jellyfin Server 10.11.11 (the plugin targets ABI `10.11.11.0`)
- NVIDIA hardware plus a working CUDA/NVENC FFmpeg installation
- Python 3.10+ and the analysis dependencies
- A local clone of this repository, which supplies the Python processing script

The plugin changes the normal media filename after creating a validated filtered
copy. Keep backups and test first. The original is retained as
`<name>.<ext>.cursefilter-original`; do not delete it unless you deliberately
want to discard the original.

## Step 1: Prepare the processing files

Clone this repository to a stable local path, for example `C:\CurseFilter`:

```powershell
git clone https://github.com/ajminnich/Jellyfin-Curse-Filter.git C:\CurseFilter
Set-Location C:\CurseFilter
py -3 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
```

If you do not have a CUDA-capable NVIDIA GPU, this project is not presently
configured for CPU processing. Its dependency list installs CUDA-enabled
PyTorch.

## Step 2: Add the Jellyfin plugin repository

In Jellyfin, open **Dashboard → Plugins → Repositories** and add this URL:

```text
https://raw.githubusercontent.com/ajminnich/Jellyfin-Curse-Filter/main/repository.json
```

Install **Curse Filter** from the catalog, then restart Jellyfin.

## Step 3: Configure the plugin

Stop Jellyfin. Copy
[`config/Jellyfin.Plugin.CurseFilter.example.xml`](config/Jellyfin.Plugin.CurseFilter.example.xml)
to Jellyfin's plugin configuration directory as
`Jellyfin.Plugin.CurseFilter.xml` (normally
`C:\ProgramData\Jellyfin\Server\config\plugins\`). Set all paths for your
machine, especially the repository clone, Python virtual environment, media
roots, cache folders, and Jellyfin's `ffmpeg.exe`.

Keep `Enabled` set to `false` until you have tested a small library. Restart
Jellyfin, edit the configuration to set `Enabled` to `true`, then restart once
more.

## Step 4: Run and verify

Open **Dashboard → Scheduled Tasks → Curse Filter** and run **Create
profanity-filtered audio**. Watch the Jellyfin server log. On success, refresh
the library and check that the filtered English track is default while the
original audio remains selectable. Confirm that a `.cursefilter-original`
backup exists for every promoted media item.

## Updates

GitHub hosting enables Jellyfin updates only when the repository feed changes:
Jellyfin periodically refreshes `repository.json`, detects a newer version, and
offers the update in **Dashboard → Plugins**. It does not silently update a
plugin while the server is running; install the offered update and restart
Jellyfin.

Plugin updates only replace the .NET plugin files. Because this plugin also
uses the checked-out Python scripts and Python environment, pull the repository
and update dependencies on the server too:

```powershell
Set-Location C:\CurseFilter
git pull
.\venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
```

## Publishing updates

1. Change the version, timestamp, and changelog in `plugin/Jellyfin.Plugin.CurseFilter/meta.json` and the matching version fields in the `.csproj`.
2. Run `./scripts/build-plugin-release.ps1`.
3. Commit the generated `repository.json` with the source changes.
4. Push the commit, then create and push the matching tag, for example `v0.2.1.0`.
5. The GitHub Actions workflow builds the archive, updates `repository.json` with the checksum of that exact archive, and creates the GitHub release. Jellyfin will discover the new feed version on its next refresh.

The GitHub Actions release asset is intentionally not committed. The workflow
commits the final checksum because a GitHub-hosted build has a different archive
checksum from a local build.
