using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using Jellyfin.Data.Enums;
using Jellyfin.Plugin.CurseFilter.Configuration;
using MediaBrowser.Controller.Entities;
using MediaBrowser.Controller.Library;
using MediaBrowser.Model.Tasks;
using Microsoft.Extensions.Logging;

namespace Jellyfin.Plugin.CurseFilter.ScheduledTasks;

/// <summary>Creates profanity-filtered audio for every supported item in the configured library.</summary>
public sealed class ProcessLibraryTask : IScheduledTask
{
    private const string FilteredTrackTitle = "Filtered English (Bleep)";

    private static readonly HashSet<string> VideoExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".mkv", ".mp4", ".m4v", ".avi", ".mov", ".ts", ".m2ts", ".webm"
    };

    private readonly ILibraryManager _libraryManager;
    private readonly ILogger<ProcessLibraryTask> _logger;

    /// <summary>Initializes a new instance of the <see cref="ProcessLibraryTask"/> class.</summary>
    public ProcessLibraryTask(ILibraryManager libraryManager, ILogger<ProcessLibraryTask> logger)
    {
        _libraryManager = libraryManager;
        _logger = logger;
    }

    /// <inheritdoc />
    public string Name => "Create profanity-filtered audio";

    /// <inheritdoc />
    public string Key => "CurseFilterProcessLibrary";

    /// <inheritdoc />
    public string Description => "Analyzes captions or audio and installs a playback-safe default bleeped audio track.";

    /// <inheritdoc />
    public string Category => "Curse Filter";

    /// <inheritdoc />
    public IEnumerable<TaskTriggerInfo> GetDefaultTriggers()
    {
        yield return new TaskTriggerInfo
        {
            Type = TaskTriggerInfoType.StartupTrigger,
            IntervalTicks = TimeSpan.FromMinutes(1).Ticks
        };
        yield return new TaskTriggerInfo
        {
            Type = TaskTriggerInfoType.DailyTrigger,
            TimeOfDayTicks = TimeSpan.FromHours(3).Ticks
        };
    }

    /// <inheritdoc />
    public async Task ExecuteAsync(IProgress<double> progress, CancellationToken cancellationToken)
    {
        PluginConfiguration configuration = Plugin.Instance?.Configuration
            ?? throw new InvalidOperationException("Curse Filter configuration is unavailable.");
        if (!configuration.Enabled)
        {
            _logger.LogInformation("Curse Filter library processing is disabled.");
            progress.Report(100);
            return;
        }

        ValidateConfiguration(configuration);
        Directory.CreateDirectory(configuration.CaptionCachePath);
        Directory.CreateDirectory(configuration.FilteredAudioCachePath);
        Directory.CreateDirectory(configuration.ReportCachePath);
        string[] configuredRoots = GetConfiguredRoots(configuration)
            .Where(Directory.Exists)
            .ToArray();

        IReadOnlyList<BaseItem> items = _libraryManager.GetItemList(new InternalItemsQuery
        {
            Recursive = true,
            IncludeItemTypes = new[] { BaseItemKind.Movie, BaseItemKind.Episode },
            IsVirtualItem = false
        });
        BaseItem[] candidates = items
            .Where(item => configuredRoots.Any(root => IsCandidate(item.Path, root)))
            .OrderBy(item => item.Path, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        string promotionRefreshMarker = Path.Combine(
            configuration.FilteredAudioCachePath,
            ".promotion-index-refreshed");
        bool needsPromotionIndexRefresh = !File.Exists(promotionRefreshMarker)
            && candidates.Any(item => File.Exists(item.Path + ".cursefilter-original"));

        _logger.LogInformation(
            "Curse Filter found {Count} video items under {Roots}.",
            candidates.Length,
            string.Join("; ", configuredRoots));
        if (candidates.Length == 0)
        {
            progress.Report(100);
            return;
        }

        bool createdAny = false;
        for (int index = 0; index < candidates.Length; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            BaseItem item = candidates[index];
            try
            {
                createdAny |= await ProcessItemAsync(item.Path, configuration, cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception exception)
            {
                _logger.LogError(exception, "Curse Filter failed for {Path}; processing will continue.", item.Path);
            }

            progress.Report((index + 1) * 100.0 / candidates.Length);
        }

        if (createdAny || needsPromotionIndexRefresh)
        {
            _logger.LogInformation("Curse Filter changed promoted media; queueing a Jellyfin library scan.");
            _libraryManager.QueueLibraryScan();
            if (needsPromotionIndexRefresh)
            {
                File.WriteAllText(promotionRefreshMarker, DateTimeOffset.UtcNow.ToString("O"));
            }
        }
    }

    private static bool IsCandidate(string? path, string configuredRoot)
    {
        if (string.IsNullOrWhiteSpace(path) || !VideoExtensions.Contains(Path.GetExtension(path)))
        {
            return false;
        }

        if (Path.GetFileNameWithoutExtension(path).EndsWith(" - Filtered", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        string fullPath = Path.GetFullPath(path);
        return fullPath.Equals(configuredRoot, StringComparison.OrdinalIgnoreCase)
            || fullPath.StartsWith(configuredRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase);
    }

    private static void ValidateConfiguration(PluginConfiguration configuration)
    {
        string[] requiredFiles =
        {
            configuration.PythonPath,
            configuration.PipelineScriptPath,
            configuration.FfmpegPath
        };
        foreach (string path in requiredFiles)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("Curse Filter dependency was not found.", path);
            }
        }

        if (!new[] { "auto", "captions", "audio" }.Contains(
                configuration.AnalysisMode,
                StringComparer.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                $"Unsupported analysis mode '{configuration.AnalysisMode}'. Use auto, captions, or audio.");
        }

        string[] roots = GetConfiguredRoots(configuration);
        if (roots.Length == 0 || !roots.Any(Directory.Exists))
        {
            throw new DirectoryNotFoundException(
                $"No configured media root exists: {string.Join("; ", roots)}");
        }
    }

    private static string[] GetConfiguredRoots(PluginConfiguration configuration)
    {
        return new[] { configuration.MediaRoot, configuration.MovieMediaRoot }
            .Where(path => !string.IsNullOrWhiteSpace(path))
            .Select(path => Path.GetFullPath(path)
                .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private async Task<bool> ProcessItemAsync(
        string mediaPath,
        PluginConfiguration configuration,
        CancellationToken cancellationToken)
    {
        string directory = Path.GetDirectoryName(mediaPath)
            ?? throw new InvalidOperationException($"Media item has no parent directory: {mediaPath}");
        string stem = Path.GetFileNameWithoutExtension(mediaPath);
        string ffprobePath = Path.Combine(
            Path.GetDirectoryName(configuration.FfmpegPath) ?? string.Empty,
            "ffprobe.exe");
        if (await HasFilteredTrackAsync(mediaPath, ffprobePath, cancellationToken).ConfigureAwait(false)
            && await HasCompatibleVideoAsync(mediaPath, ffprobePath, cancellationToken).ConfigureAwait(false))
        {
            _logger.LogDebug("The library copy is already filtered and playback-compatible: {Path}.", mediaPath);
            return false;
        }

        string playbackPath = Path.Combine(directory, $"{stem} - Filtered.mkv");
        string promotedPath = Path.Combine(directory, $"{stem}.mkv");
        if (File.Exists(playbackPath)
            && await HasFilteredTrackAsync(playbackPath, ffprobePath, cancellationToken).ConfigureAwait(false)
            && await HasCompatibleVideoAsync(playbackPath, ffprobePath, cancellationToken).ConfigureAwait(false))
        {
            PromoteFilteredVersion(mediaPath, playbackPath, promotedPath);
            ArchiveLegacySidecar(mediaPath, configuration.FilteredAudioCachePath);
            _logger.LogInformation("Promoted the compatible filtered version to {Path}.", mediaPath);
            return true;
        }

        if (File.Exists(playbackPath))
        {
            throw new IOException($"Refusing to replace an existing non-filtered media file: {playbackPath}");
        }

        string cacheSuffix = Convert.ToHexString(
            SHA256.HashData(Encoding.UTF8.GetBytes(Path.GetFullPath(mediaPath))))
            .Substring(0, 12)
            .ToLowerInvariant();
        string outputPath = Path.Combine(
            configuration.FilteredAudioCachePath,
            $"{stem}.{cacheSuffix}.filtered.eng.mka");
        string cleanMarkerPath = outputPath + ".clean";
        string legacySidecarPath = Path.Combine(directory, $"{stem}.default.filtered.eng.mka");
        if (!File.Exists(outputPath) && File.Exists(legacySidecarPath))
        {
            File.Copy(legacySidecarPath, outputPath, true);
        }

        string? captionPath = null;
        string selectedMode = configuration.AnalysisMode.ToLowerInvariant();
        if (selectedMode != "audio")
        {
            string candidateCaptionPath = Path.Combine(
                configuration.CaptionCachePath,
                $"{stem}.{cacheSuffix}.eng.srt");
            if (File.Exists(candidateCaptionPath)
                || await ExtractCaptionAsync(
                    mediaPath,
                    candidateCaptionPath,
                    configuration.FfmpegPath,
                    cancellationToken).ConfigureAwait(false))
            {
                captionPath = candidateCaptionPath;
                selectedMode = "captions";
            }
            else if (selectedMode == "captions")
            {
                _logger.LogWarning("No usable embedded caption track was found for {Path}; skipping caption mode.", mediaPath);
                return false;
            }
            else
            {
                selectedMode = "audio";
                _logger.LogInformation(
                    "No usable embedded captions were found for {Path}; using caption-free audio analysis.",
                    mediaPath);
            }
        }

        if (!File.Exists(outputPath) || configuration.OverwriteExisting)
        {
            string analysisSignature = GetAnalysisSignature(mediaPath, configuration, selectedMode);
            if (!configuration.OverwriteExisting
                && File.Exists(cleanMarkerPath)
                && File.ReadAllText(cleanMarkerPath).Equals(analysisSignature, StringComparison.Ordinal))
            {
                _logger.LogDebug("A current clean analysis already exists for {Path}.", mediaPath);
                return false;
            }

            _logger.LogInformation("Creating filtered audio for {Path} in {Mode} mode.", mediaPath, selectedMode);
            var arguments = new List<string>
            {
                configuration.PipelineScriptPath,
                mediaPath
            };
            if (captionPath is not null)
            {
                arguments.Add(captionPath);
            }

            arguments.AddRange(new[]
            {
                "--output", outputPath,
                "--ffmpeg", configuration.FfmpegPath,
                "--device", configuration.AnalysisDevice,
                "--model", configuration.AlignmentModel,
                "--mode", selectedMode,
                "--transcription-model", configuration.TranscriptionModel,
                "--report-dir", configuration.ReportCachePath
            });
            ProcessResult result = await RunProcessAsync(
                configuration.PythonPath,
                arguments,
                cancellationToken).ConfigureAwait(false);
            if (result.ExitCode != 0)
            {
                throw new InvalidOperationException($"Analysis process exited with {result.ExitCode}: {result.Error}");
            }

            if (!File.Exists(outputPath))
            {
                File.WriteAllText(cleanMarkerPath, analysisSignature);
                _logger.LogInformation("No configured profanity was detected in {Path}; no filtered copy is needed.", mediaPath);
                return false;
            }

            if (File.Exists(cleanMarkerPath))
            {
                File.Delete(cleanMarkerPath);
            }
        }

        await InstallPlaybackSafeTrackAsync(
            mediaPath,
            outputPath,
            playbackPath,
            configuration.FfmpegPath,
            ffprobePath,
            cancellationToken).ConfigureAwait(false);
        PromoteFilteredVersion(mediaPath, playbackPath, promotedPath);
        ArchiveLegacySidecar(mediaPath, configuration.FilteredAudioCachePath);
        _logger.LogInformation("Created and promoted an H.264-compatible filtered media version at {Path}.", mediaPath);
        return true;
    }

    private static string GetAnalysisSignature(
        string mediaPath,
        PluginConfiguration configuration,
        string selectedMode)
    {
        var media = new FileInfo(mediaPath);
        var pipeline = new FileInfo(configuration.PipelineScriptPath);
        return string.Join(
            "|",
            media.Length,
            media.LastWriteTimeUtc.Ticks,
            pipeline.LastWriteTimeUtc.Ticks,
            selectedMode,
            configuration.AlignmentModel,
            configuration.TranscriptionModel);
    }

    private static void PromoteFilteredVersion(string mediaPath, string playbackPath, string promotedPath)
    {
        string backupPath = mediaPath + ".cursefilter-original";
        if (File.Exists(backupPath))
        {
            throw new IOException($"Refusing to replace the existing original-media backup: {backupPath}");
        }

        if (!mediaPath.Equals(promotedPath, StringComparison.OrdinalIgnoreCase) && File.Exists(promotedPath))
        {
            throw new IOException($"Refusing to replace an existing Matroska media file: {promotedPath}");
        }

        File.Move(mediaPath, backupPath);
        try
        {
            File.Move(playbackPath, promotedPath);
        }
        catch
        {
            File.Move(backupPath, mediaPath);
            throw;
        }
    }

    private static async Task<bool> HasFilteredTrackAsync(
        string mediaPath,
        string ffprobePath,
        CancellationToken cancellationToken)
    {
        if (!File.Exists(ffprobePath))
        {
            throw new FileNotFoundException("FFprobe was not found beside FFmpeg.", ffprobePath);
        }

        var arguments = new[]
        {
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream_tags=title",
            "-of", "default=noprint_wrappers=1:nokey=1",
            mediaPath
        };
        ProcessResult result = await RunProcessAsync(ffprobePath, arguments, cancellationToken).ConfigureAwait(false);
        return result.ExitCode == 0
            && result.Output.Trim().Equals(FilteredTrackTitle, StringComparison.OrdinalIgnoreCase);
    }

    private static async Task InstallPlaybackSafeTrackAsync(
        string mediaPath,
        string filteredAudioPath,
        string playbackPath,
        string ffmpegPath,
        string ffprobePath,
        CancellationToken cancellationToken)
    {
        string directory = Path.GetDirectoryName(mediaPath)
            ?? throw new InvalidOperationException($"Media item has no parent directory: {mediaPath}");
        string temporaryPath = Path.Combine(
            directory,
            $".{Path.GetFileNameWithoutExtension(mediaPath)}.{Guid.NewGuid():N}.cursefilter.partial.mkv");
        try
        {
            bool copyCompatibleVideo = await HasCompatibleVideoAsync(
                mediaPath,
                ffprobePath,
                cancellationToken).ConfigureAwait(false);
            var arguments = new List<string>
            {
                "-hide_banner", "-loglevel", "error", "-y"
            };
            if (!copyCompatibleVideo)
            {
                arguments.AddRange(new[]
                {
                    "-hwaccel", "cuda",
                    "-hwaccel_output_format", "cuda"
                });
            }

            arguments.AddRange(new[]
            {
                "-i", mediaPath,
                "-i", filteredAudioPath,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-map", "0:a?",
                "-map", "0:s?",
                "-map", "0:t?",
                "-map_metadata", "0",
                "-map_chapters", "0"
            });
            if (copyCompatibleVideo)
            {
                arguments.AddRange(new[] { "-c:v", "copy" });
            }
            else
            {
                arguments.AddRange(new[]
                {
                    "-vf", "scale_cuda=1920:-2:format=nv12",
                    "-c:v", "h264_nvenc",
                    "-preset", "p3",
                    "-tune", "hq",
                    "-rc", "vbr",
                    "-cq", "22",
                    "-b:v", "0",
                    "-maxrate", "6M",
                    "-bufsize", "12M",
                    "-spatial-aq", "1",
                    "-temporal-aq", "1",
                    "-rc-lookahead", "12",
                    "-profile:v", "high",
                    "-level:v", "4.1",
                    "-fps_mode", "passthrough"
                });
            }

            arguments.AddRange(new[]
            {
                "-c:a", "copy",
                "-c:s", "copy",
                "-c:t", "copy",
                "-metadata:s:a:0", "language=eng",
                "-metadata:s:a:0", $"title={FilteredTrackTitle}",
                "-metadata:s:a:1", "title=Original English",
                "-disposition:a", "0",
                "-disposition:a:0", "default",
                temporaryPath
            });
            ProcessResult result = await RunProcessAsync(ffmpegPath, arguments, cancellationToken).ConfigureAwait(false);
            if (result.ExitCode != 0)
            {
                throw new InvalidOperationException($"Playback-safe remux exited with {result.ExitCode}: {result.Error}");
            }

            if (!await HasFilteredTrackAsync(temporaryPath, ffprobePath, cancellationToken).ConfigureAwait(false))
            {
                throw new InvalidOperationException(
                    "The remuxed file did not contain the filtered track as its first audio stream.");
            }

            if (!await HasCompatibleVideoAsync(temporaryPath, ffprobePath, cancellationToken).ConfigureAwait(false))
            {
                throw new InvalidOperationException(
                    "The generated file did not contain H.264 yuv420p compatibility video.");
            }

            File.Move(temporaryPath, playbackPath);
        }
        finally
        {
            if (File.Exists(temporaryPath))
            {
                File.Delete(temporaryPath);
            }
        }
    }

    private static async Task<bool> HasCompatibleVideoAsync(
        string mediaPath,
        string ffprobePath,
        CancellationToken cancellationToken)
    {
        var arguments = new[]
        {
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt",
            "-of", "default=noprint_wrappers=1",
            mediaPath
        };
        ProcessResult result = await RunProcessAsync(ffprobePath, arguments, cancellationToken).ConfigureAwait(false);
        return result.ExitCode == 0
            && result.Output.Contains("codec_name=h264", StringComparison.OrdinalIgnoreCase)
            && result.Output.Contains("pix_fmt=yuv420p", StringComparison.OrdinalIgnoreCase);
    }

    private static bool ArchiveLegacySidecar(string mediaPath, string cachePath)
    {
        string directory = Path.GetDirectoryName(mediaPath)
            ?? throw new InvalidOperationException($"Media item has no parent directory: {mediaPath}");
        string stem = Path.GetFileNameWithoutExtension(mediaPath);
        string legacyPath = Path.Combine(directory, $"{stem}.default.filtered.eng.mka");
        if (!File.Exists(legacyPath))
        {
            return false;
        }

        Directory.CreateDirectory(cachePath);
        string archivePath = Path.Combine(cachePath, Path.GetFileName(legacyPath));
        File.Move(legacyPath, archivePath, true);
        return true;
    }

    private static async Task<bool> ExtractCaptionAsync(
        string mediaPath,
        string captionPath,
        string ffmpegPath,
        CancellationToken cancellationToken)
    {
        string temporaryPath = captionPath + ".partial.srt";
        var arguments = new[]
        {
            "-hide_banner", "-loglevel", "error", "-y",
            "-i", mediaPath,
            "-map", "0:s:0",
            "-c:s", "srt",
            temporaryPath
        };
        ProcessResult result = await RunProcessAsync(ffmpegPath, arguments, cancellationToken).ConfigureAwait(false);
        if (result.ExitCode != 0 || !File.Exists(temporaryPath))
        {
            if (File.Exists(temporaryPath))
            {
                File.Delete(temporaryPath);
            }

            return false;
        }

        File.Move(temporaryPath, captionPath, true);
        return true;
    }

    private static async Task<ProcessResult> RunProcessAsync(
        string executable,
        IEnumerable<string> arguments,
        CancellationToken cancellationToken)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = executable,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        foreach (string argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using var process = new Process { StartInfo = startInfo };
        if (!process.Start())
        {
            throw new InvalidOperationException($"Could not start {executable}.");
        }

        Task<string> outputTask = process.StandardOutput.ReadToEndAsync(cancellationToken);
        Task<string> errorTask = process.StandardError.ReadToEndAsync(cancellationToken);
        await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
        return new ProcessResult(
            process.ExitCode,
            await outputTask.ConfigureAwait(false),
            await errorTask.ConfigureAwait(false));
    }

    private sealed record ProcessResult(int ExitCode, string Output, string Error);
}
