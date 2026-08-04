using MediaBrowser.Model.Plugins;

namespace Jellyfin.Plugin.CurseFilter.Configuration;

/// <summary>Configuration for library-wide profanity filtering.</summary>
public sealed class PluginConfiguration : BasePluginConfiguration
{
    /// <summary>Gets or sets a value indicating whether scheduled processing is enabled.</summary>
    public bool Enabled { get; set; } = true;

    /// <summary>Gets or sets the TV library root.</summary>
    public string MediaRoot { get; set; } = string.Empty;

    /// <summary>Gets or sets the movie library root.</summary>
    public string MovieMediaRoot { get; set; } = string.Empty;

    /// <summary>Gets or sets the Python executable used by the processing pipeline.</summary>
    public string PythonPath { get; set; } = string.Empty;

    /// <summary>Gets or sets the caption-guided processing script.</summary>
    public string PipelineScriptPath { get; set; } = string.Empty;

    /// <summary>Gets or sets Jellyfin's bundled FFmpeg executable.</summary>
    public string FfmpegPath { get; set; } = string.Empty;

    /// <summary>Gets or sets the directory used for extracted caption files.</summary>
    public string CaptionCachePath { get; set; } = string.Empty;

    /// <summary>Gets or sets the private directory used to retain separate filtered audio files.</summary>
    public string FilteredAudioCachePath { get; set; } = string.Empty;

    /// <summary>Gets or sets the directory used for analysis reports and temporary clips.</summary>
    public string ReportCachePath { get; set; } = string.Empty;

    /// <summary>Gets or sets the GPU device passed to the analysis pipeline.</summary>
    public string AnalysisDevice { get; set; } = "cuda";

    /// <summary>Gets or sets the English forced-alignment model.</summary>
    public string AlignmentModel { get; set; } = "facebook/wav2vec2-base-960h";

    /// <summary>Gets or sets the analysis mode: auto, captions, or audio.</summary>
    public string AnalysisMode { get; set; } = "auto";

    /// <summary>Gets or sets the Whisper model used for caption-free audio analysis.</summary>
    public string TranscriptionModel { get; set; } = "small.en";

    /// <summary>Gets or sets a value indicating whether existing filtered tracks may be replaced.</summary>
    public bool OverwriteExisting { get; set; }
}
